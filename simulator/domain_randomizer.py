"""
domain_randomizer.py — Domain Randomization for Sim-to-Real Transfer

Randomizes vehicle and environment parameters during training to produce
policies that are robust to real-world variations:

    - Vehicle mass:       ±10% (passenger/cargo load variation)
    - Tyre stiffness:     ±15% (tyre wear, pressure, temperature)
    - Road friction:      0.6–1.0 (dry tarmac to light rain)
    - Wind disturbance:   N(0, 50) N lateral force
    - Sensor latency:     1–3 step delay on observations

Reference: Tobin et al. (2017) — "Domain Randomization for Transferring
           Deep Neural Networks from Simulation to the Real World"

These ranges are conservative and representative of conditions an ADAS
system would encounter in EU motorway driving (ISO 15622 scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class RandomizedParams:
    """Snapshot of randomized parameters for one episode."""
    mass_kg: float = cfg.VEHICLE_MASS
    C_af: float = cfg.TYRE_CAF_NOMINAL
    C_ar: float = cfg.TYRE_CAR_NOMINAL
    friction_mu: float = 1.0
    wind_force_n: float = 0.0
    obs_latency_steps: int = 0


class DomainRandomizer:
    """
    Randomizes simulation parameters each episode for sim-to-real robustness.

    During training, call randomize() at the start of each episode.
    During evaluation, use default (nominal) parameters.

    Attributes:
        enabled:       Whether randomization is active.
        mass_range:    (min_factor, max_factor) for mass.
        tyre_range:    (min_factor, max_factor) for tyre stiffness.
        friction_range: (min_mu, max_mu) for road friction coefficient.
        wind_std:      Standard deviation of lateral wind force (N).
        latency_range: (min_steps, max_steps) for observation delay.
    """

    def __init__(
        self,
        enabled: bool = True,
        mass_range: tuple[float, float] = (0.90, 1.10),
        tyre_range: tuple[float, float] = (0.85, 1.15),
        friction_range: tuple[float, float] = (0.6, 1.0),
        wind_std: float = 50.0,
        latency_range: tuple[int, int] = (0, 3),
    ) -> None:
        self.enabled = enabled
        self.mass_range = mass_range
        self.tyre_range = tyre_range
        self.friction_range = friction_range
        self.wind_std = wind_std
        self.latency_range = latency_range

        # Observation delay buffer
        self._obs_buffer: list[np.ndarray] = []
        self._current_params = RandomizedParams()

    @property
    def params(self) -> RandomizedParams:
        """Current randomized parameters."""
        return self._current_params

    def randomize(self) -> RandomizedParams:
        """
        Generate new randomized parameters for the next episode.

        Returns:
            RandomizedParams with the sampled values.
        """
        if not self.enabled:
            self._current_params = RandomizedParams()
            self._obs_buffer = []
            return self._current_params

        # Vehicle mass: ±10% (passenger/cargo variation)
        mass_factor = np.random.uniform(*self.mass_range)
        mass_kg = cfg.VEHICLE_MASS * mass_factor

        # Tyre stiffness: ±15% (wear, pressure, temperature)
        caf_factor = np.random.uniform(*self.tyre_range)
        car_factor = np.random.uniform(*self.tyre_range)
        C_af = cfg.TYRE_CAF_NOMINAL * caf_factor
        C_ar = cfg.TYRE_CAR_NOMINAL * car_factor

        # Road friction coefficient: 0.6–1.0
        friction_mu = np.random.uniform(*self.friction_range)

        # Lateral wind disturbance force (constant per episode)
        wind_force_n = np.random.normal(0.0, self.wind_std)

        # Observation latency: 0–3 steps (0–30 ms at 100 Hz)
        obs_latency = np.random.randint(
            self.latency_range[0], self.latency_range[1] + 1
        )

        self._current_params = RandomizedParams(
            mass_kg=mass_kg,
            C_af=C_af,
            C_ar=C_ar,
            friction_mu=friction_mu,
            wind_force_n=wind_force_n,
            obs_latency_steps=obs_latency,
        )

        # Reset observation delay buffer
        self._obs_buffer = []

        return self._current_params

    def apply_obs_latency(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply observation latency by buffering observations.

        Args:
            obs: Current true observation.

        Returns:
            Delayed observation (or current if latency=0).
        """
        latency = self._current_params.obs_latency_steps
        if latency == 0 or not self.enabled:
            return obs

        self._obs_buffer.append(obs.copy())

        if len(self._obs_buffer) <= latency:
            # Not enough history yet — return current (bootstrapping)
            return obs

        # Return the delayed observation
        delayed = self._obs_buffer[-latency - 1]
        # Keep buffer bounded
        if len(self._obs_buffer) > latency + 10:
            self._obs_buffer = self._obs_buffer[-(latency + 5):]

        return delayed

    def get_wind_acceleration(self) -> float:
        """
        Compute lateral acceleration from wind force.

        Returns:
            Lateral acceleration (m/s^2) due to wind disturbance.
        """
        if not self.enabled:
            return 0.0
        return self._current_params.wind_force_n / self._current_params.mass_kg

    def get_info(self) -> dict:
        """Return current parameters as a dictionary for logging."""
        p = self._current_params
        return {
            "dr_mass_kg": p.mass_kg,
            "dr_C_af": p.C_af,
            "dr_C_ar": p.C_ar,
            "dr_friction_mu": p.friction_mu,
            "dr_wind_force_n": p.wind_force_n,
            "dr_obs_latency_steps": p.obs_latency_steps,
        }
