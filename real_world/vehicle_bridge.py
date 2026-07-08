"""
vehicle_bridge.py — Real Vehicle ↔ DDPG Agent Bridge

This is the central orchestrator that bridges the gap between:
    Real-world sensors → Perception Pipeline → 8D Observation → DDPG Agent
    DDPG Agent → Normalised Action → Feedforward+Feedback → Steering Command → Actuator

The VehicleBridge mirrors the role of LaneKeepingEnv but for REAL hardware.
It produces the same feedforward + feedback control architecture used during
training, ensuring the agent sees the same observation/action interface
it was trained on.

Key Design Principle:
    The trained policy NEVER knows whether it's running in simulation
    or on a real vehicle. The bridge handles all translation.

Reference:
    ISO 26262:2018 §6 — software architectural design
    UNECE WP.29 R157 §5 — system specification for ALKS
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

import config as cfg
from ddpg.agent import DDPGAgent
from real_world.sensor_interface import SensorBundle, SensorInterface
from real_world.actuator_interface import ActuatorInterface, SteeringCommand, ActuatorState
from real_world.perception_pipeline import PerceptionPipeline, PerceptionState
from real_world.safety_monitor import RealWorldSafetyMonitor, SystemState

logger = logging.getLogger(__name__)


class VehicleBridge:
    """
    Bridges the trained DDPG agent to real vehicle hardware.

    The bridge runs the identical feedforward + feedback architecture
    used during training (see lane_keeping_env.py step()), ensuring
    the agent produces the same quality of corrections on real hardware.

    Control Architecture (mirrors training):
        1. Perception pipeline produces 8D observation
        2. Agent produces normalised action in [-1, 1]
        3. Feedforward: δ_nominal = L·κ + K_us·v²·κ
        4. Feedback: δ_correction = action × correction_authority
        5. Combined: δ_cmd = δ_nominal + δ_correction
        6. Safety guardian clamps rate and angle
        7. Safety monitor checks envelope
        8. Actuator sends to vehicle

    Attributes:
        agent:      Trained DDPG agent (loaded from checkpoint).
        sensors:    Sensor hardware interface.
        actuator:   Steering actuator interface.
        pipeline:   Perception pipeline (sensor fusion → 8D obs).
        monitor:    Real-world safety monitor.
    """

    def __init__(
        self,
        agent: DDPGAgent,
        sensors: SensorInterface,
        actuator: ActuatorInterface,
        control_hz: int = cfg.DEPLOY_CONTROL_HZ,
    ) -> None:
        self.agent = agent
        self.sensors = sensors
        self.actuator = actuator
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

        # Perception pipeline
        self.pipeline = PerceptionPipeline(control_hz=control_hz)

        # Safety monitor
        self.monitor = RealWorldSafetyMonitor(control_hz=control_hz)

        # Control state
        self._delta_prev: float = 0.0
        self._action_prev: float = 0.0
        self._kappa_prev: float = 0.0

        # Feedforward parameters (identical to lane_keeping_env.py)
        self._K_us = self._compute_understeer_gradient()

        # Telemetry
        self._cycle_count: int = 0
        self._total_latency_s: float = 0.0
        self._max_latency_s: float = 0.0

    def _compute_understeer_gradient(self) -> float:
        """
        Compute the understeer gradient K_us for feedforward.

        K_us = (m/L) · (l_r/C_af - l_f/C_ar)

        This is the same formula used in lane_keeping_env.py for
        feedforward steering computation.
        """
        return (cfg.VEHICLE_MASS / cfg.VEHICLE_WHEELBASE) * (
            cfg.VEHICLE_LR / cfg.TYRE_CAF_NOMINAL
            - cfg.VEHICLE_LF / cfg.TYRE_CAR_NOMINAL
        )

    def connect(self) -> bool:
        """
        Establish connections to all hardware interfaces.

        Returns:
            True if all connections successful.
        """
        logger.info("VehicleBridge: Connecting to hardware...")

        if not self.sensors.connect():
            logger.error("VehicleBridge: Sensor connection failed")
            return False

        if not self.actuator.connect():
            logger.error("VehicleBridge: Actuator connection failed")
            self.sensors.close()
            return False

        if not self.monitor.initialise():
            logger.error("VehicleBridge: Safety monitor initialisation failed")
            self.sensors.close()
            self.actuator.close()
            return False

        logger.info("VehicleBridge: All hardware connected")
        return True

    def engage(self) -> bool:
        """
        Engage the LKA system.

        Performs pre-engagement checks and activates steering control.

        Returns:
            True if engagement successful.
        """
        # Read initial sensor state
        sensor_bundle = self.sensors.read()

        if not sensor_bundle.is_valid:
            logger.error("VehicleBridge: Cannot engage — sensors invalid")
            return False

        # Process initial observation
        obs, state = self.pipeline.process(sensor_bundle)

        # Check speed envelope
        if (state.v_x_mps < cfg.DEPLOY_MIN_SPEED_MPS
                or state.v_x_mps > cfg.DEPLOY_MAX_SPEED_MPS):
            logger.error(
                f"VehicleBridge: Cannot engage — speed {state.v_x_mps:.1f} m/s "
                f"outside [{cfg.DEPLOY_MIN_SPEED_MPS}, {cfg.DEPLOY_MAX_SPEED_MPS}]"
            )
            return False

        # Engage actuator
        if not self.actuator.engage():
            logger.error("VehicleBridge: Actuator engagement failed")
            return False

        # Engage safety monitor
        if not self.monitor.engage():
            logger.error("VehicleBridge: Safety monitor engagement failed")
            self.actuator.disengage()
            return False

        # Initialise control state
        self._delta_prev = state.delta_prev_rad
        self._action_prev = 0.0

        logger.info(
            f"VehicleBridge: LKA ENGAGED at {state.v_x_mps * 3.6:.0f} km/h, "
            f"e_lat={state.e_lat_m:.3f}m"
        )
        return True

    def step(self) -> dict:
        """
        Execute one control cycle.

        This is the main control loop body, called at DEPLOY_CONTROL_HZ.

        Returns:
            Dictionary with telemetry:
                - state: SystemState
                - e_lat_m: lateral error
                - delta_cmd: steering command
                - action: agent action
                - latency_s: cycle latency
        """
        cycle_start = time.monotonic()
        telemetry = {}

        # ── 1. Read sensors ────────────────────────────────────────────
        sensor_bundle = self.sensors.read()

        # ── 2. Perception pipeline → 8D observation ────────────────────
        obs, perception_state = self.pipeline.process(sensor_bundle)

        # ── 3. Safety monitor check ────────────────────────────────────
        sys_state = self.monitor.check(
            sensors=sensor_bundle,
            e_lat_m=perception_state.e_lat_m,
            e_psi_rad=perception_state.e_psi_rad,
            v_x_mps=perception_state.v_x_mps,
            delta_rad=self._delta_prev,
            sensor_confidence=perception_state.e_lat_confidence,
        )

        telemetry["state"] = sys_state
        telemetry["e_lat_m"] = perception_state.e_lat_m
        telemetry["e_psi_rad"] = perception_state.e_psi_rad
        telemetry["v_x_mps"] = perception_state.v_x_mps
        telemetry["confidence"] = perception_state.e_lat_confidence

        # ── 4. Handle non-active states ────────────────────────────────
        if sys_state == SystemState.EMERGENCY_STOP:
            self.actuator.emergency_stop()
            telemetry["delta_cmd"] = 0.0
            telemetry["action"] = 0.0
            return telemetry

        if not self.monitor.is_safe_to_steer():
            self.actuator.disengage()
            telemetry["delta_cmd"] = 0.0
            telemetry["action"] = 0.0
            return telemetry

        # ── 5. Agent inference (deterministic — no noise) ──────────────
        action = self.agent.select_action(obs)

        # Action smoothing (identical to training)
        alpha = cfg.ACTION_SMOOTHING_ALPHA
        action_smooth = alpha * float(action.flat[0]) + (1.0 - alpha) * self._action_prev
        self._action_prev = action_smooth

        # ── 6. Feedforward steering ────────────────────────────────────
        v_x = max(perception_state.v_x_mps, 1.0)  # Prevent division by zero
        kappa = perception_state.kappa_ref

        # Use lookahead curvature for preview feedforward
        # Blend current and lookahead based on speed
        preview_blend = min(v_x / 20.0, 1.0)  # Full preview above 20 m/s
        kappa_preview = (
            (1.0 - preview_blend) * kappa
            + preview_blend * perception_state.kappa_la1
        )

        # Feedforward: Ackermann + understeer gradient
        delta_nominal = (
            cfg.VEHICLE_WHEELBASE * kappa_preview
            + self._K_us * (v_x ** 2) * kappa_preview
        )

        # ── 7. Feedback: agent correction ──────────────────────────────
        correction_authority = cfg.CORRECTION_AUTHORITY * cfg.DELTA_MAX
        delta_correction = action_smooth * correction_authority

        # Apply authority factor from safety monitor
        authority_factor = self.monitor.get_authority_factor()
        delta_correction *= authority_factor

        # ── 8. Combined steering command ───────────────────────────────
        delta_cmd = delta_nominal + delta_correction

        # Clamp to physical limits
        delta_cmd = float(np.clip(delta_cmd, -cfg.DELTA_MAX, cfg.DELTA_MAX))

        # Rate limit
        max_delta_change = cfg.DEPLOY_MAX_STEERING_RATE * self.dt
        delta_cmd = float(np.clip(
            delta_cmd,
            self._delta_prev - max_delta_change,
            self._delta_prev + max_delta_change,
        ))

        # ── 9. Send to actuator ────────────────────────────────────────
        steering_rate = (delta_cmd - self._delta_prev) / self.dt if self.dt > 0 else 0.0

        command = SteeringCommand(
            timestamp_s=time.monotonic(),
            delta_rad=delta_cmd,
            delta_rate_rads=steering_rate,
        )

        success = self.actuator.send(command)

        # ── 10. Check for driver override ──────────────────────────────
        if self.actuator.is_driver_override():
            logger.info("VehicleBridge: Driver override detected — disengaging")
            self.actuator.disengage()
            self.monitor.disengage()
            sys_state = SystemState.INACTIVE

        # Update state
        self._delta_prev = delta_cmd
        self._kappa_prev = kappa

        # Telemetry
        cycle_latency = time.monotonic() - cycle_start
        self._cycle_count += 1
        self._total_latency_s += cycle_latency
        self._max_latency_s = max(self._max_latency_s, cycle_latency)

        telemetry["delta_cmd"] = delta_cmd
        telemetry["delta_nominal"] = delta_nominal
        telemetry["delta_correction"] = delta_correction
        telemetry["action"] = action_smooth
        telemetry["authority_factor"] = authority_factor
        telemetry["latency_s"] = cycle_latency
        telemetry["command_sent"] = success

        return telemetry

    def disengage(self) -> None:
        """Smoothly disengage the LKA system."""
        self.actuator.disengage()
        self.monitor.disengage()
        logger.info("VehicleBridge: LKA disengaged")

    def close(self) -> None:
        """Release all hardware resources."""
        self.actuator.disengage()
        self.actuator.close()
        self.sensors.close()

        # Log session statistics
        avg_latency = (
            self._total_latency_s / max(self._cycle_count, 1) * 1000
        )
        logger.info(
            f"VehicleBridge: Session complete — "
            f"{self._cycle_count} cycles, "
            f"avg latency {avg_latency:.1f}ms, "
            f"max latency {self._max_latency_s * 1000:.1f}ms"
        )

    @property
    def session_stats(self) -> dict:
        """Return session statistics."""
        return {
            "total_cycles": self._cycle_count,
            "avg_latency_ms": (
                self._total_latency_s / max(self._cycle_count, 1) * 1000
            ),
            "max_latency_ms": self._max_latency_s * 1000,
            "safety_diagnostics": {
                "total_interventions": self.monitor.diagnostics.total_interventions,
                "total_emergency_stops": self.monitor.diagnostics.total_emergency_stops,
                "uptime_s": self.monitor.diagnostics.uptime_s,
            },
            "perception_stats": self.pipeline.fusion_stats,
        }
