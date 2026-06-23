"""
safety_guardian.py — Rule-Based Safety Supervisor for Lane Keeping

Implements a three-layer safety envelope around the RL agent's steering output:

    Layer 1: Steering rate limiter
        Clamps |δ̇| ≤ MAX_STEER_RATE to prevent jerky control.
        Reference: ISO 15622:2018 §9.2 (comfort requirement).

    Layer 2: Steering angle limiter
        Ensures |δ| ≤ DELTA_MAX (hardware limit at operating speed).

    Layer 3: Handoff trigger
        If |e_lat| > HANDOFF_LAT_THRESHOLD for HANDOFF_SUSTAIN_STEPS
        consecutive steps, signals a driver handoff request.
        Reference: UNECE WP.29 R157 — transition demand.

The guardian wraps the environment's step() output and modifies the
steering command BEFORE it reaches the vehicle dynamics, ensuring
that even a poorly-trained agent cannot produce unsafe vehicle behaviour.

Usage:
    guardian = SafetyGuardian()
    delta_safe = guardian.apply(delta_cmd, delta_prev, dt)
    handoff = guardian.check_handoff(e_lat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Guardian configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Layer 1: Steering rate limit (rad/s)
# 2.5 rad/s is a realistic physical limit for ADAS actuators during evasive maneuvers
# (0.4 rad/s is only for highway comfort cruising)
MAX_STEER_RATE: float = 2.5

# Layer 2: Absolute steering angle limit (rad)
# Same as DELTA_MAX from config — hardware constraint
MAX_STEER_ANGLE: float = cfg.DELTA_MAX

# Layer 3: Handoff trigger thresholds
# UNECE R157 requires transition demand when system reaches limits
HANDOFF_LAT_THRESHOLD: float = 0.85 * (cfg.LANE_WIDTH / 2.0)  # 85% of half-lane
HANDOFF_SUSTAIN_STEPS: int = 50   # 0.5 s at 100 Hz
HANDOFF_COOLDOWN_STEPS: int = 200  # 2.0 s cooldown after handoff


@dataclass
class GuardianStats:
    """Accumulated statistics for guardian interventions."""
    rate_clamp_count: int = 0
    angle_clamp_count: int = 0
    handoff_trigger_count: int = 0
    total_steps: int = 0

    @property
    def rate_intervention_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return 100.0 * self.rate_clamp_count / self.total_steps

    @property
    def angle_intervention_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return 100.0 * self.angle_clamp_count / self.total_steps


class SafetyGuardian:
    """
    Rule-based safety supervisor for lane keeping control.

    Sits between the RL agent and the vehicle actuators, enforcing
    physical and safety constraints on every control command.

    Attributes:
        max_steer_rate:   Maximum allowed steering rate (rad/s).
        max_steer_angle:  Maximum allowed steering angle (rad).
        handoff_active:   Whether a handoff request is currently active.
        stats:            Accumulated intervention statistics.
    """

    def __init__(
        self,
        max_steer_rate: float = MAX_STEER_RATE,
        max_steer_angle: float = MAX_STEER_ANGLE,
        handoff_lat_threshold: float = HANDOFF_LAT_THRESHOLD,
        handoff_sustain_steps: int = HANDOFF_SUSTAIN_STEPS,
    ) -> None:
        self.max_steer_rate = max_steer_rate
        self.max_steer_angle = max_steer_angle
        self.handoff_lat_threshold = handoff_lat_threshold
        self.handoff_sustain_steps = handoff_sustain_steps

        # Internal state
        self._near_boundary_counter: int = 0
        self._cooldown_counter: int = 0
        self.handoff_active: bool = False
        self.stats = GuardianStats()

    def reset(self) -> None:
        """Reset guardian state for a new episode."""
        self._near_boundary_counter = 0
        self._cooldown_counter = 0
        self.handoff_active = False

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self.stats = GuardianStats()

    def apply(
        self,
        delta_cmd: float,
        delta_prev: float,
        dt: float = cfg.SIM_DT,
    ) -> float:
        """
        Apply safety constraints to a steering command.

        Layer 1: Clamp steering rate to ±max_steer_rate.
        Layer 2: Clamp steering angle to ±max_steer_angle.

        Args:
            delta_cmd:  Desired steering angle (rad) from agent + feedforward.
            delta_prev: Previous timestep steering angle (rad).
            dt:         Timestep duration (s).

        Returns:
            Safe steering angle (rad) after all constraints applied.
        """
        self.stats.total_steps += 1
        delta_safe = delta_cmd

        # ── Layer 1: Steering rate limiter ────────────────────────────────
        delta_dot = (delta_safe - delta_prev) / dt
        max_delta_change = self.max_steer_rate * dt

        if abs(delta_safe - delta_prev) > max_delta_change:
            # Clamp the change to the maximum allowed rate
            sign = np.sign(delta_safe - delta_prev)
            delta_safe = delta_prev + sign * max_delta_change
            self.stats.rate_clamp_count += 1

        # ── Layer 2: Steering angle limiter ───────────────────────────────
        if abs(delta_safe) > self.max_steer_angle:
            delta_safe = np.sign(delta_safe) * self.max_steer_angle
            self.stats.angle_clamp_count += 1

        return float(delta_safe)

    def check_handoff(self, e_lat: float) -> bool:
        """
        Check whether a driver handoff should be triggered.

        Layer 3: If the vehicle is persistently near the lane boundary
        (|e_lat| > threshold for sustained_steps consecutive steps),
        trigger a handoff request per UNECE WP.29 R157.

        Args:
            e_lat: Current lateral error (m).

        Returns:
            True if handoff is triggered this step, False otherwise.
        """
        # Cooldown after previous handoff
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False

        # Check if near lane boundary
        if abs(e_lat) > self.handoff_lat_threshold:
            self._near_boundary_counter += 1
        else:
            self._near_boundary_counter = 0

        # Trigger handoff if sustained
        if self._near_boundary_counter >= self.handoff_sustain_steps:
            self.handoff_active = True
            self._near_boundary_counter = 0
            self._cooldown_counter = HANDOFF_COOLDOWN_STEPS
            self.stats.handoff_trigger_count += 1
            logger.warning(
                f"HANDOFF TRIGGERED: |e_lat|={abs(e_lat):.3f}m > "
                f"{self.handoff_lat_threshold:.3f}m for "
                f"{self.handoff_sustain_steps} steps"
            )
            return True

        return False

    def get_info(self) -> dict:
        """Return guardian state as a dictionary for logging."""
        return {
            "guardian_handoff_active": self.handoff_active,
            "guardian_near_boundary_count": self._near_boundary_counter,
            "guardian_rate_clamps": self.stats.rate_clamp_count,
            "guardian_angle_clamps": self.stats.angle_clamp_count,
            "guardian_handoffs": self.stats.handoff_trigger_count,
            "guardian_rate_intervention_pct": self.stats.rate_intervention_pct,
        }
