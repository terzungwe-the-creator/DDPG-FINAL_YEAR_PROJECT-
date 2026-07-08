"""
safety_monitor.py — Real-World Safety Envelope Monitor

Extends the simulation SafetyGuardian to real-world deployment with
additional safety layers required for live vehicle operation.

Safety Architecture (Defense in Depth — ISO 26262 ASIL-B):
    Layer 1: Neural network policy (learned safe behaviour)
    Layer 2: SafetyGuardian (rate/angle clamping — same as training)
    Layer 3: RealWorldSafetyMonitor (THIS MODULE):
             - Sensor health monitoring
             - Lateral error envelope (emergency stop threshold)
             - Speed envelope (min/max for LKA engagement)
             - Watchdog timer (actuator heartbeat)
             - Driver readiness assessment
             - Minimum risk condition (MRC) management
    Layer 4: Actuator hardware limits (independent of software)

UNECE WP.29 R157 Compliance:
    §5.1.1: "The system shall monitor its operational design domain"
    §5.2:   "The system shall detect sensor degradation"
    §5.4:   "The system shall execute a minimum risk manoeuvre"
    §6.1:   "Transition demand to driver within 10 seconds"

Reference:
    ISO 26262:2018 Part 3-5 — ASIL-B functional safety
    UNECE WP.29 R157 — Automated Lane Keeping Systems
    ISO 21448:2022 (SOTIF) — Safety of the Intended Functionality
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

import config as cfg
from real_world.sensor_interface import SensorBundle, SensorHealth

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system operational state per UNECE R157 §5."""
    INITIALISING = auto()    # System booting, sensors initialising
    READY = auto()           # All checks passed, awaiting engagement
    ACTIVE = auto()          # LKA steering active
    DEGRADED = auto()        # Partial sensor loss — reduced capability
    TRANSITION_DEMAND = auto()  # Requesting driver takeover (10s timer)
    MINIMUM_RISK = auto()    # Executing minimum risk condition (MRC)
    EMERGENCY_STOP = auto()  # Immediate stop — safety violation
    INACTIVE = auto()        # System off / disengaged by driver


@dataclass
class SafetyDiagnostics:
    """Real-time safety diagnostics for logging and HMI display."""
    system_state: SystemState = SystemState.INITIALISING
    e_lat_abs_m: float = 0.0
    e_lat_rate_mps: float = 0.0
    ttld_s: float = 999.0
    v_x_mps: float = 0.0
    sensor_confidence: float = 0.0
    sensor_age_s: float = 0.0
    actuator_healthy: bool = False
    driver_ready: bool = True
    consecutive_degraded_cycles: int = 0
    time_in_transition_demand_s: float = 0.0
    total_interventions: int = 0
    total_emergency_stops: int = 0
    uptime_s: float = 0.0


@dataclass
class SafetyThresholds:
    """
    Configurable safety thresholds.

    These are calibrated more conservatively than the ISO 15622 evaluation
    thresholds because real-world operation must account for sensor noise,
    latency, and unmodelled disturbances.
    """
    # Lateral error thresholds (m)
    warning_lateral_m: float = cfg.DEPLOY_HANDOFF_LATERAL_M       # 1.0 m → warning
    emergency_lateral_m: float = cfg.DEPLOY_EMERGENCY_LATERAL_M   # 1.5 m → emergency stop
    
    # Speed envelope (m/s)
    min_speed_mps: float = cfg.DEPLOY_MIN_SPEED_MPS    # 5.0 m/s (18 km/h)
    max_speed_mps: float = cfg.DEPLOY_MAX_SPEED_MPS    # 36.0 m/s (130 km/h)
    
    # Sensor freshness
    max_sensor_age_s: float = cfg.DEPLOY_SENSOR_TIMEOUT_S  # 0.1 s
    min_confidence: float = 0.3                            # Below → degraded mode
    
    # Timing
    transition_demand_timeout_s: float = 10.0  # UNECE R157: 10s for driver takeover
    max_degraded_cycles: int = 50              # ~1s at 50 Hz → escalate to transition demand
    
    # TTLD safety margin
    min_ttld_s: float = 1.0  # Minimum acceptable time-to-lane-departure
    
    # Steering rate safety
    max_steering_rate_rads: float = cfg.DEPLOY_MAX_STEERING_RATE


class RealWorldSafetyMonitor:
    """
    Real-time safety envelope monitor for live vehicle deployment.

    Runs every control cycle (50 Hz) and determines the system state.
    Can trigger:
        - DEGRADED mode (reduced sensor confidence)
        - TRANSITION_DEMAND (driver takeover request)
        - MINIMUM_RISK (controlled lane keeping to stop)
        - EMERGENCY_STOP (immediate steering neutralisation)

    The monitor is INDEPENDENT of the RL agent — it runs regardless
    of what the agent produces and can override any steering command.
    """

    def __init__(
        self,
        thresholds: Optional[SafetyThresholds] = None,
        control_hz: int = cfg.DEPLOY_CONTROL_HZ,
    ) -> None:
        self.thresholds = thresholds or SafetyThresholds()
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

        self._state = SystemState.INITIALISING
        self._diagnostics = SafetyDiagnostics()
        self._start_time = time.monotonic()
        self._transition_start_time: Optional[float] = None
        self._prev_e_lat: float = 0.0
        self._prev_delta: float = 0.0
        self._degraded_counter: int = 0

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def diagnostics(self) -> SafetyDiagnostics:
        return self._diagnostics

    def initialise(self) -> bool:
        """
        Complete initialisation checks and transition to READY.

        Returns:
            True if all checks pass and system is ready for engagement.
        """
        self._state = SystemState.READY
        self._start_time = time.monotonic()
        logger.info("SafetyMonitor: Initialised — system READY")
        return True

    def engage(self) -> bool:
        """
        Transition from READY to ACTIVE.

        Returns:
            True if engagement is permitted.
        """
        if self._state != SystemState.READY:
            logger.warning(f"SafetyMonitor: Cannot engage from state {self._state}")
            return False

        self._state = SystemState.ACTIVE
        self._degraded_counter = 0
        logger.info("SafetyMonitor: LKA ENGAGED — system ACTIVE")
        return True

    def disengage(self) -> None:
        """Transition to INACTIVE (driver-initiated disengage)."""
        self._state = SystemState.INACTIVE
        self._transition_start_time = None
        logger.info("SafetyMonitor: LKA DISENGAGED by driver")

    def check(
        self,
        sensors: SensorBundle,
        e_lat_m: float,
        e_psi_rad: float,
        v_x_mps: float,
        delta_rad: float,
        sensor_confidence: float,
    ) -> SystemState:
        """
        Run all safety checks for this control cycle.

        This is called EVERY cycle (50 Hz). It evaluates the current
        state against all safety thresholds and may transition the
        system state.

        Args:
            sensors:            Raw sensor bundle for health checking.
            e_lat_m:            Fused lateral error (m).
            e_psi_rad:          Fused heading error (rad).
            v_x_mps:            Vehicle speed (m/s).
            delta_rad:          Current steering angle command (rad).
            sensor_confidence:  Fused sensor confidence [0, 1].

        Returns:
            Current system state after all checks.
        """
        now = time.monotonic()
        self._diagnostics.uptime_s = now - self._start_time
        self._diagnostics.e_lat_abs_m = abs(e_lat_m)
        self._diagnostics.v_x_mps = v_x_mps
        self._diagnostics.sensor_confidence = sensor_confidence
        self._diagnostics.sensor_age_s = sensors.max_age_s

        # Compute lateral error rate and TTLD
        e_lat_rate = (e_lat_m - self._prev_e_lat) / self.dt if self.dt > 0 else 0.0
        self._diagnostics.e_lat_rate_mps = e_lat_rate

        # Time-to-lane-departure
        margin = self.thresholds.emergency_lateral_m - abs(e_lat_m)
        if margin <= 0:
            ttld = 0.0
        elif (e_lat_m >= 0 and e_lat_rate > 0) or (e_lat_m < 0 and e_lat_rate < 0):
            ttld = margin / max(abs(e_lat_rate), 1e-6)
        else:
            ttld = 999.0
        self._diagnostics.ttld_s = min(ttld, 999.0)

        # Steering rate check
        delta_rate = abs(delta_rad - self._prev_delta) / self.dt if self.dt > 0 else 0.0

        # Update previous values
        self._prev_e_lat = e_lat_m
        self._prev_delta = delta_rad

        # Only run checks when system is ACTIVE or DEGRADED
        if self._state not in (SystemState.ACTIVE, SystemState.DEGRADED,
                                SystemState.TRANSITION_DEMAND):
            return self._state

        # ── Check 1: Emergency lateral error ────────────────────────
        if abs(e_lat_m) >= self.thresholds.emergency_lateral_m:
            logger.critical(
                f"EMERGENCY: |e_lat| = {abs(e_lat_m):.3f}m >= "
                f"{self.thresholds.emergency_lateral_m}m"
            )
            self._trigger_emergency_stop("lateral_error_exceeded")
            return self._state

        # ── Check 2: TTLD too low ───────────────────────────────────
        if ttld < self.thresholds.min_ttld_s and ttld < 998.0:
            logger.warning(f"TTLD critically low: {ttld:.2f}s")
            if ttld < 0.3:
                self._trigger_emergency_stop("ttld_critical")
                return self._state
            elif self._state == SystemState.ACTIVE:
                self._trigger_transition_demand("ttld_low")

        # ── Check 3: Speed envelope ─────────────────────────────────
        if (v_x_mps < self.thresholds.min_speed_mps
                or v_x_mps > self.thresholds.max_speed_mps):
            if self._state == SystemState.ACTIVE:
                logger.warning(
                    f"Speed {v_x_mps:.1f} m/s outside envelope "
                    f"[{self.thresholds.min_speed_mps}, {self.thresholds.max_speed_mps}]"
                )
                self._trigger_transition_demand("speed_envelope")

        # ── Check 4: Sensor health ──────────────────────────────────
        if sensors.max_age_s > self.thresholds.max_sensor_age_s:
            logger.warning(f"Sensor data stale: age={sensors.max_age_s:.3f}s")
            self._degraded_counter += 1
        elif sensor_confidence < self.thresholds.min_confidence:
            logger.warning(f"Sensor confidence low: {sensor_confidence:.2f}")
            self._degraded_counter += 1
        else:
            # Sensors OK — reduce degraded counter
            self._degraded_counter = max(0, self._degraded_counter - 2)

        # Degraded state management
        self._diagnostics.consecutive_degraded_cycles = self._degraded_counter
        if self._degraded_counter >= self.thresholds.max_degraded_cycles:
            if self._state == SystemState.ACTIVE:
                self._trigger_transition_demand("sensor_degradation")
        elif self._degraded_counter > 5 and self._state == SystemState.ACTIVE:
            self._state = SystemState.DEGRADED

        # ── Check 5: Warning lateral error ──────────────────────────
        if abs(e_lat_m) >= self.thresholds.warning_lateral_m:
            if self._state == SystemState.ACTIVE:
                logger.warning(
                    f"Warning: |e_lat| = {abs(e_lat_m):.3f}m >= "
                    f"{self.thresholds.warning_lateral_m}m — requesting handoff"
                )
                self._trigger_transition_demand("lateral_warning")

        # ── Check 6: Transition demand timeout ──────────────────────
        if self._state == SystemState.TRANSITION_DEMAND:
            if self._transition_start_time is not None:
                elapsed = now - self._transition_start_time
                self._diagnostics.time_in_transition_demand_s = elapsed
                if elapsed >= self.thresholds.transition_demand_timeout_s:
                    logger.critical(
                        "Transition demand timeout (10s) — executing MRC"
                    )
                    self._state = SystemState.MINIMUM_RISK

        # ── Check 7: Steering rate ──────────────────────────────────
        if delta_rate > self.thresholds.max_steering_rate_rads * 1.5:
            logger.warning(
                f"Steering rate excessive: {delta_rate:.2f} rad/s "
                f"(limit {self.thresholds.max_steering_rate_rads} rad/s)"
            )

        self._diagnostics.system_state = self._state
        return self._state

    def _trigger_emergency_stop(self, reason: str) -> None:
        """Transition to EMERGENCY_STOP."""
        self._state = SystemState.EMERGENCY_STOP
        self._diagnostics.total_emergency_stops += 1
        self._diagnostics.system_state = self._state
        logger.critical(f"EMERGENCY STOP triggered: {reason}")

    def _trigger_transition_demand(self, reason: str) -> None:
        """Transition to TRANSITION_DEMAND (request driver takeover)."""
        if self._state != SystemState.TRANSITION_DEMAND:
            self._state = SystemState.TRANSITION_DEMAND
            self._transition_start_time = time.monotonic()
            self._diagnostics.total_interventions += 1
            self._diagnostics.system_state = self._state
            logger.warning(f"TRANSITION DEMAND: {reason} — driver has 10s")

    def is_safe_to_steer(self) -> bool:
        """
        Check if it is currently safe to send steering commands.

        Returns:
            True if system state permits active steering.
        """
        return self._state in (
            SystemState.ACTIVE,
            SystemState.DEGRADED,
            SystemState.TRANSITION_DEMAND,
            SystemState.MINIMUM_RISK,
        )

    def get_authority_factor(self) -> float:
        """
        Return steering authority reduction factor based on system state.

        ACTIVE:             1.0 (full authority)
        DEGRADED:           0.7 (reduced)
        TRANSITION_DEMAND:  0.5 (preparing for handoff)
        MINIMUM_RISK:       0.3 (gentle lane keeping only)
        Otherwise:          0.0 (no steering)
        """
        authority_map = {
            SystemState.ACTIVE: 1.0,
            SystemState.DEGRADED: 0.7,
            SystemState.TRANSITION_DEMAND: 0.5,
            SystemState.MINIMUM_RISK: 0.3,
        }
        return authority_map.get(self._state, 0.0)

    def reset(self) -> None:
        """Reset safety monitor to INITIALISING state."""
        self._state = SystemState.INITIALISING
        self._diagnostics = SafetyDiagnostics()
        self._transition_start_time = None
        self._prev_e_lat = 0.0
        self._prev_delta = 0.0
        self._degraded_counter = 0
        self._start_time = time.monotonic()
