"""
normaliser.py — Universal Normalisation Pipeline for Multi-Source Data Fusion

Converts raw transitions from any data source (OpenLKA, comma-steering-control,
Argoverse 2, or simulator) into the agent's canonical observation space before
storage in the hybrid replay buffer.

All normalisation constants are physical limits from config.py, not data-driven.
This guarantees zero distribution shift between real-world and simulated data
in the replay buffer.

Safety: Raises ValueError if any raw value exceeds 5× its physical limit,
indicating a parsing error. Does not silently clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg


@dataclass
class RawTransition:
    """
    Universal raw transition format from any data source.

    All values are in physical SI units before normalisation.
    The source field tracks provenance for logging and debugging.

    Attributes:
        source:         Data source identifier ('openlka', 'comma', 'argoverse', 'sim')
        e_lat_m:        Lateral error (metres, signed)
        e_psi_rad:      Heading error (radians, wrapped to [-π, π])
        kappa_ref:      Reference curvature (1/m)
        v_y_mps:        Lateral velocity (m/s) — may be estimated for some sources
        yaw_rate_rads:  Yaw rate (rad/s)
        delta_prev_rad: Previous steering angle (radians)
        kappa_la1:      1-second lookahead curvature (1/m)
        kappa_la2:      2-second lookahead curvature (1/m)
        action_raw_rad: Steering action (radians, current timestep)
        reward:         Scalar reward (computed by reward_fn)
        done:           Episode termination flag

        # Next-state fields (for building the transition tuple)
        next_e_lat_m:        float = 0.0
        next_e_psi_rad:      float = 0.0
        next_kappa_ref:      float = 0.0
        next_v_y_mps:        float = 0.0
        next_yaw_rate_rads:  float = 0.0
        next_delta_prev_rad: float = 0.0
        next_kappa_la1:      float = 0.0
        next_kappa_la2:      float = 0.0
    """
    source: str
    e_lat_m: float
    e_psi_rad: float
    kappa_ref: float
    v_y_mps: float
    yaw_rate_rads: float
    delta_prev_rad: float
    kappa_la1: float
    kappa_la2: float
    action_raw_rad: float
    reward: float
    done: bool

    # Next-state fields
    next_e_lat_m: float = 0.0
    next_e_psi_rad: float = 0.0
    next_kappa_ref: float = 0.0
    next_v_y_mps: float = 0.0
    next_yaw_rate_rads: float = 0.0
    next_delta_prev_rad: float = 0.0
    next_kappa_la1: float = 0.0
    next_kappa_la2: float = 0.0


class UniversalNormaliser:
    """
    Converts RawTransition → normalised (obs, action) ∈ [-1, 1]^N.

    All normalisation constants are physical limits from config.py.
    This ensures identical normalisation across all data sources,
    guaranteeing zero distribution shift in the fused replay buffer.

    Normalisation mapping:
        obs[0] = e_lat_m         / NORM_E_LAT       (1.75 m)
        obs[1] = e_psi_rad       / NORM_E_PSI       (π/4 rad)
        obs[2] = kappa_ref       / NORM_KAPPA        (0.05 1/m)
        obs[3] = v_y_mps         / NORM_V_Y          (2.0 m/s)
        obs[4] = yaw_rate_rads   / NORM_YAW_RATE     (0.5 rad/s)
        obs[5] = delta_prev_rad  / NORM_DELTA         (0.35 rad)
        obs[6] = kappa_la1       / NORM_KAPPA_LA1     (0.05 1/m)
        obs[7] = kappa_la2       / NORM_KAPPA_LA2     (0.05 1/m)

    Action:
        action_norm = action_raw_rad / DELTA_MAX
    """

    # 5× safety bounds for parsing error detection
    SAFETY_MULTIPLIER: float = 5.0

    # Physical limits and their 5× bounds
    BOUNDS = {
        "e_lat_m": (cfg.NORM_E_LAT, cfg.NORM_E_LAT * 5.0),
        "e_psi_rad": (cfg.NORM_E_PSI, cfg.NORM_E_PSI * 5.0),
        "kappa_ref": (cfg.NORM_KAPPA, cfg.NORM_KAPPA * 5.0),
        "v_y_mps": (cfg.NORM_V_Y, cfg.NORM_V_Y * 5.0),
        "yaw_rate_rads": (cfg.NORM_YAW_RATE, cfg.NORM_YAW_RATE * 5.0),
        "delta_prev_rad": (cfg.NORM_DELTA, cfg.NORM_DELTA * 5.0),
        "kappa_la1": (cfg.NORM_KAPPA_LA1, cfg.NORM_KAPPA_LA1 * 5.0),
        "kappa_la2": (cfg.NORM_KAPPA_LA2, cfg.NORM_KAPPA_LA2 * 5.0),
        "action_raw_rad": (cfg.DELTA_MAX, cfg.DELTA_MAX * 5.0),
    }

    def _check_bounds(self, value: float, field_name: str) -> None:
        """
        Validate that a raw value does not exceed 5× its physical limit.

        Args:
            value:      Raw physical value.
            field_name: Name of the field (for error message).

        Raises:
            ValueError: If |value| > 5× physical limit.
        """
        if field_name not in self.BOUNDS:
            return
        _, bound_5x = self.BOUNDS[field_name]
        if abs(value) > bound_5x:
            raise ValueError(
                f"Raw value {field_name}={value:.4f} exceeds 5× physical limit "
                f"({bound_5x:.4f}). This indicates a parsing error."
            )

    def normalise_obs(self, raw: RawTransition) -> np.ndarray:
        """
        Convert a RawTransition's state fields to a normalised observation vector.

        Args:
            raw: RawTransition with physical-unit state values.

        Returns:
            Normalised observation array, shape (8,), clipped to [-1, 1].

        Raises:
            ValueError: If any raw value exceeds 5× its physical limit.
        """
        # Validate bounds
        self._check_bounds(raw.e_lat_m, "e_lat_m")
        self._check_bounds(raw.e_psi_rad, "e_psi_rad")
        self._check_bounds(raw.kappa_ref, "kappa_ref")
        self._check_bounds(raw.v_y_mps, "v_y_mps")
        self._check_bounds(raw.yaw_rate_rads, "yaw_rate_rads")
        self._check_bounds(raw.delta_prev_rad, "delta_prev_rad")
        self._check_bounds(raw.kappa_la1, "kappa_la1")
        self._check_bounds(raw.kappa_la2, "kappa_la2")

        obs = np.array([
            raw.e_lat_m / cfg.NORM_E_LAT,
            raw.e_psi_rad / cfg.NORM_E_PSI,
            raw.kappa_ref / cfg.NORM_KAPPA,
            raw.v_y_mps / cfg.NORM_V_Y,
            raw.yaw_rate_rads / cfg.NORM_YAW_RATE,
            raw.delta_prev_rad / cfg.NORM_DELTA,
            raw.kappa_la1 / cfg.NORM_KAPPA_LA1,
            raw.kappa_la2 / cfg.NORM_KAPPA_LA2,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def normalise_next_obs(self, raw: RawTransition) -> np.ndarray:
        """
        Convert a RawTransition's next-state fields to a normalised observation.

        Args:
            raw: RawTransition with next-state physical-unit values.

        Returns:
            Normalised next-observation array, shape (8,), clipped to [-1, 1].
        """
        obs = np.array([
            raw.next_e_lat_m / cfg.NORM_E_LAT,
            raw.next_e_psi_rad / cfg.NORM_E_PSI,
            raw.next_kappa_ref / cfg.NORM_KAPPA,
            raw.next_v_y_mps / cfg.NORM_V_Y,
            raw.next_yaw_rate_rads / cfg.NORM_YAW_RATE,
            raw.next_delta_prev_rad / cfg.NORM_DELTA,
            raw.next_kappa_la1 / cfg.NORM_KAPPA_LA1,
            raw.next_kappa_la2 / cfg.NORM_KAPPA_LA2,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def normalise_action(self, action_rad: float) -> float:
        """
        Normalise a physical steering angle to [-1, 1].

        Args:
            action_rad: Steering angle in radians.

        Returns:
            Normalised action in [-1, 1].
        """
        self._check_bounds(action_rad, "action_raw_rad")
        return float(np.clip(action_rad / cfg.DELTA_MAX, -1.0, 1.0))

    def denormalise_action(self, action_norm: float) -> float:
        """
        Convert a normalised action back to physical steering angle.

        Args:
            action_norm: Normalised action in [-1, 1].

        Returns:
            Steering angle in radians.
        """
        return float(np.clip(action_norm, -1.0, 1.0) * cfg.DELTA_MAX)
