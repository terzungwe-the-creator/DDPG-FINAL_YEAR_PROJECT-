"""
lane_keeping_env.py — Gymnasium-Compatible Lane Keeping Environment

Dual-mode environment supporting two physics backends:
    1. "bicycle" — Built-in nonlinear bicycle model with RK4 integration (default)
    2. "carla"   — CARLA 0.9.16 with 4-wheel PhysX vehicle dynamics

Both backends produce the same 8D normalised observation vector, enabling
seamless switching between high-speed offline training (bicycle) and
high-fidelity evaluation (CARLA).

Observation space: 8-dimensional, normalised to [-1, 1]:
    [e_lat_norm, e_psi_norm, kappa_norm, v_y_norm, r_norm,
     delta_prev_norm, kappa_la1_norm, kappa_la2_norm]

Action space: 1-dimensional continuous [-1, 1], mapped to [-DELTA_MAX, DELTA_MAX].

Episode termination:
    - Terminated: |e_lat| >= LANE_WIDTH/2 (lane departure)
    - Truncated: max steps reached (SIM_MAX_STEPS = 3000 → 30 s)

Reference: ISO 15622:2018 — Lane Keeping Assistance Systems
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as cfg
from simulator.vehicle_model import BicycleModel
from simulator.road_profiles import RoadProfile, build_all_profiles
from simulator.reward import compute_reward
from simulator.safety_guardian import SafetyGuardian
from simulator.domain_randomizer import DomainRandomizer

logger = logging.getLogger(__name__)


class LaneKeepingEnv(gym.Env):
    """
    Lane keeping environment with dual-mode physics backend.

    Backend modes:
        "bicycle" — Nonlinear bicycle model (Rajamani 2012), fast, no external
                     dependencies. Used for bulk training and result generation.
        "carla"   — CARLA 0.9.16 with 4-wheel PhysX dynamics. Requires running
                     CARLA server. Used for high-fidelity evaluation.

    Both backends output the same 8D observation and accept the same 1D action,
    allowing trained policies to transfer between them.

    Attributes:
        backend:      Physics backend: "bicycle" or "carla".
        profiles:     Dictionary of road profiles (bicycle mode only).
        vehicle:      BicycleModel instance (bicycle mode only).
        carla_bridge: CarlaBridge instance (carla mode only).
        current_scn:  Active scenario ID.
        step_count:   Current step within the episode.
    """

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self,
        backend: str = "bicycle",
        scenario_id: str = "SCN-01",
        render_mode: Optional[str] = None,
        carla_host: str = cfg.CARLA_HOST,
        carla_port: int = cfg.CARLA_PORT,
        training_mode: bool = True,
        obs_noise_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.backend = backend.lower()
        self.current_scn = scenario_id
        self.render_mode = render_mode
        self.training_mode = training_mode
        self.obs_noise_std = obs_noise_std

        # Observation: 8D normalised vector
        self.observation_space = spaces.Box(
            low=-np.ones(cfg.OBS_DIM, dtype=np.float32),
            high=np.ones(cfg.OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        # Action: 1D normalised steering [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Safety guardian (active in both training and evaluation)
        self.guardian = SafetyGuardian()

        # Domain randomization (training only)
        self.domain_randomizer = DomainRandomizer(enabled=training_mode)

        # Backend-specific initialisation
        if self.backend == "carla":
            from simulator.carla_bridge import CarlaBridge
            self.carla_bridge = CarlaBridge(
                host=carla_host,
                port=carla_port,
            )
            self.carla_bridge.connect()
            self.vehicle = None
            self.profiles = None
            logger.info("LaneKeepingEnv: CARLA backend (4-wheel PhysX)")
        elif self.backend == "bicycle":
            self.carla_bridge = None
            self.vehicle = BicycleModel()
            self.profiles = build_all_profiles()
            logger.info("LaneKeepingEnv: Bicycle model backend (RK4)")
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. Use 'bicycle' or 'carla'."
            )

        # Multi-speed support
        self.episode_speed: float = cfg.V_REFERENCE  # Current episode speed

        # Episode state
        self.step_count: int = 0
        self.arc_length_s: float = 0.0
        self.delta_prev: float = 0.0
        self._episode_data: list[dict] = []

    def update_tyre_params(self, C_af: float, C_ar: float) -> None:
        """
        Update vehicle tyre parameters from DS-02 calibration.

        Dispatches to the appropriate backend.

        Args:
            C_af: Front axle cornering stiffness (N/rad).
            C_ar: Rear axle cornering stiffness (N/rad).
        """
        if self.backend == "carla":
            self.carla_bridge.update_tyre_params(C_af, C_ar)
        else:
            self.vehicle.update_tyre_params(C_af, C_ar)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        scenario_id: Optional[str] = None,
        e_lat_init: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed:         Random seed.
            options:      Additional options (unused).
            scenario_id:  Override scenario. If None, keeps current.
            e_lat_init:   Initial lateral perturbation (m). If None, 0.0.

        Returns:
            (observation, info) tuple per Gymnasium API.
        """
        super().reset(seed=seed)

        if scenario_id is not None:
            self.current_scn = scenario_id

        if e_lat_init is None:
            e_lat_0 = 0.0
        else:
            e_lat_0 = float(e_lat_init)

        if self.backend == "carla":
            return self._reset_carla(e_lat_0)
        else:
            return self._reset_bicycle(e_lat_0)

    def _reset_bicycle(self, e_lat_0: float) -> tuple[np.ndarray, dict]:
        """Reset using bicycle model backend with multi-speed + domain randomization."""
        if self.current_scn not in self.profiles:
            raise ValueError(
                f"Unknown scenario '{self.current_scn}'. "
                f"Available: {list(self.profiles.keys())}"
            )

        profile = self.profiles[self.current_scn]

        initial_psi = float(profile.psi_ref[0])

        # ── Multi-speed curriculum ────────────────────────────────────────
        # During training: randomize speed ±20% around V_REFERENCE
        # This produces agents robust to speed variation (48–72 km/h)
        if self.training_mode:
            speed_factor = np.random.uniform(0.8, 1.2)
            self.episode_speed = cfg.V_REFERENCE * speed_factor
        else:
            self.episode_speed = cfg.V_REFERENCE

        # ── Domain randomization ──────────────────────────────────────────
        dr_params = self.domain_randomizer.randomize()
        if self.training_mode and self.vehicle is not None:
            # Apply randomized tyre parameters scaled by friction coefficient
            # C_effective = C_nominal_randomized * friction_mu
            effective_caf = dr_params.C_af * dr_params.friction_mu
            effective_car = dr_params.C_ar * dr_params.friction_mu
            self.vehicle.update_tyre_params(effective_caf, effective_car)
            # Apply randomized mass (BicycleModel uses 'm' attribute)
            self.vehicle.m = dr_params.mass_kg

        self.vehicle.reset(
            v_x=self.episode_speed,
            e_lat_init=e_lat_0,
            e_psi_init=0.0,
            psi_init=initial_psi,
        )

        # Reset episode state
        self.step_count = 0
        self.arc_length_s = 0.0
        self.delta_prev = 0.0
        self.action_prev = 0.0
        self._episode_data = []
        self.guardian.reset()

        obs = self._get_observation_bicycle()
        info = {
            "scenario_id": self.current_scn,
            "arc_length": 0.0,
            "episode_speed_kmh": self.episode_speed * 3.6,
            "dr_mass_kg": dr_params.mass_kg,
            "dr_friction_mu": dr_params.friction_mu,
        }
        return obs, info

    def _reset_carla(self, e_lat_0: float) -> tuple[np.ndarray, dict]:
        """Reset using CARLA backend."""
        obs, info = self.carla_bridge.reset(
            scenario_id=self.current_scn,
            e_lat_init=e_lat_0,
        )

        self.step_count = 0
        self.arc_length_s = 0.0
        self.delta_prev = 0.0
        self.action_prev = 0.0
        self._episode_data = []

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Uses feedforward + feedback architecture:
            δ_total = δ_nominal(κ) + δ_correction(agent)

        The nominal steering angle is computed from road curvature using the
        steady-state bicycle model (Rajamani, 2012, Eq. 2.26):
            δ_nominal = L * κ + K_us * v² * κ

        where K_us is the understeer gradient. The RL agent outputs only the
        correction term, dramatically reducing the learning burden on curved roads.

        Args:
            action: Normalised steering CORRECTION in [-1, 1], shape (1,) or scalar.

        Returns:
            (observation, reward, terminated, truncated, info) per Gymnasium API.
        """
        raw_action = float(np.clip(action, -1.0, 1.0).flat[0])
        
        # ── Action Smoothing (Low-pass filter) ───────────────────────────
        # Prevents high-frequency oscillations induced by observation noise/latency
        alpha = cfg.ACTION_SMOOTHING_ALPHA
        action_scalar = alpha * raw_action + (1.0 - alpha) * getattr(self, 'action_prev', 0.0)
        self.action_prev = action_scalar

        # ── Feedforward: nominal steering from road curvature ────────────
        if self.backend == "bicycle":
            profile = self.profiles[self.current_scn]
            kappa_ref = profile.get_kappa_at_s(self.arc_length_s)
            v_x = self.vehicle.v_x

            # Preview-point feedforward (production ADAS approach)
            # Look ahead by preview_time seconds and steer towards that curvature.
            # This naturally handles dynamic curvature changes (SCN-03, SCN-04).
            preview_time = cfg.PREVIEW_TIME  # 0.8s — matched to vehicle yaw response lag
            s_preview = self.arc_length_s + v_x * preview_time
            kappa_preview = profile.get_kappa_at_s(s_preview)
        else:
            if self.carla_bridge is not None and self.carla_bridge.vehicle is not None:
                kappa_ref = self.carla_bridge._compute_curvature()
                v_x, _ = self.carla_bridge._get_velocities()
                kappa_la1, _ = self.carla_bridge._get_lookahead_curvature()
                
                v_x = max(v_x, 1.0)
                # Blend current and lookahead based on speed (matches vehicle_bridge.py)
                preview_blend = min(v_x / 20.0, 1.0)
                kappa_preview = (1.0 - preview_blend) * kappa_ref + preview_blend * kappa_la1
            else:
                kappa_ref = getattr(self, '_last_kappa', 0.0)
                v_x = cfg.V_REFERENCE
                kappa_preview = kappa_ref

        # Understeer gradient: K_us = (m/L) * (l_r/C_af - l_f/C_ar)
        # Equivalent to standard Wf/Caf - Wr/Car when used with delta = L*kappa + K_us*v^2*kappa
        # Reference: Rajamani (2012), Eq. 2.26
        K_us = (cfg.VEHICLE_MASS / cfg.VEHICLE_WHEELBASE) * (
            cfg.VEHICLE_LR / cfg.TYRE_CAF_NOMINAL - cfg.VEHICLE_LF / cfg.TYRE_CAR_NOMINAL
        )

        # Use preview curvature for feedforward (anticipatory steering)
        # δ_nom = L*κ_preview + K_us*v²*κ_preview
        delta_nominal = (
            cfg.VEHICLE_WHEELBASE * kappa_preview
            + K_us * (v_x ** 2) * kappa_preview
        )

        # ── Feedback: RL agent correction ────────────────────────────────
        # Agent correction scaled to CORRECTION_AUTHORITY × DELTA_MAX
        # Full authority (1.0) allows agent to use full steering range for corrections
        correction_authority = cfg.CORRECTION_AUTHORITY * cfg.DELTA_MAX
        delta_correction = action_scalar * correction_authority

        # ── Combined steering command ────────────────────────────────────
        delta_cmd = float(np.clip(
            delta_nominal + delta_correction,
            -cfg.DELTA_MAX,
            cfg.DELTA_MAX,
        ))

        if self.backend == "carla":
            return self._step_carla(delta_cmd, action_scalar)
        else:
            return self._step_bicycle(delta_cmd)

    def _step_bicycle(self, delta_cmd: float) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step using bicycle model with safety guardian."""
        profile = self.profiles[self.current_scn]
        kappa_ref = profile.get_kappa_at_s(self.arc_length_s)

        # ── Safety Guardian Layer 1+2: rate and angle clamping ────────────
        delta_safe = self.guardian.apply(delta_cmd, self.delta_prev, cfg.SIM_DT)

        # Integrate vehicle dynamics (one RK4 step) with SAFE command
        friction = self.domain_randomizer.params.friction_mu if self.training_mode else 1.0
        bank = self.domain_randomizer.params.bank_angle_rad if self.training_mode else 0.0
        self.vehicle.step(delta_safe, kappa_ref, friction_mu=friction, bank_angle_rad=bank)

        # ── Wind disturbance (domain randomization) ───────────────────────
        # Apply lateral wind force as a small perturbation to lateral error
        if self.training_mode:
            wind_accel = self.domain_randomizer.get_wind_acceleration()
            # a_lat * dt^2 / 2 gives displacement from wind
            wind_displacement = 0.5 * wind_accel * (cfg.SIM_DT ** 2)
            self.vehicle.state[self.vehicle.IDX_ELAT] += wind_displacement

        # Advance arc length
        self.arc_length_s += self.vehicle.v_x * cfg.SIM_DT
        self.step_count += 1

        # Check termination
        e_lat = self.vehicle.lateral_error
        terminated = abs(e_lat) >= cfg.DEPARTURE_THRESHOLD
        truncated = (
            self.step_count >= cfg.SIM_MAX_STEPS
            or self.arc_length_s >= profile.total_length
        )

        # ── Safety Guardian Layer 3: handoff check ────────────────────────
        handoff = self.guardian.check_handoff(e_lat)
        if handoff:
            # Handoff does not terminate — it signals the driver.
            # In a real system this would trigger HMI alerts.
            pass

        # Reward (uses the SAFE command, not the raw command)
        reward, reward_components = compute_reward(
            e_lat=e_lat,
            e_psi=self.vehicle.heading_error,
            delta_current=delta_safe,
            delta_previous=self.delta_prev,
            v_x=self.vehicle.v_x,
            terminated=terminated,
        )

        delta_dot = (delta_safe - self.delta_prev) / cfg.SIM_DT

        info = {
            "scenario_id": self.current_scn,
            "step": self.step_count,
            "time_s": self.step_count * cfg.SIM_DT,
            "arc_length": self.arc_length_s,
            "e_lat_m": e_lat,
            "e_psi_rad": self.vehicle.heading_error,
            "delta_cmd_raw": delta_cmd,
            "delta_rad": delta_safe,
            "delta_dot": delta_dot,
            "v_x": self.vehicle.v_x,
            "v_y": self.vehicle.v_y,
            "r": self.vehicle.yaw_rate,
            "kappa_ref": kappa_ref,
            "reward": reward,
            "X": self.vehicle.position[0],
            "Y": self.vehicle.position[1],
            "guardian_handoff": handoff,
        }
        info.update(reward_components)
        self._episode_data.append(info)

        self.delta_prev = delta_safe
        obs = self._get_observation_bicycle()

        return obs, reward, terminated, truncated, info

    def _step_carla(self, delta_cmd: float, action_norm: float) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step using CARLA backend."""
        obs, reward, terminated, truncated, info = self.carla_bridge.step(delta_cmd)
        info["scenario_id"] = self.current_scn

        self.step_count += 1
        self.delta_prev = delta_cmd
        self._episode_data.append(info)

        return obs, reward, terminated, truncated, info

    def _get_observation_bicycle(self) -> np.ndarray:
        """
        Build the 8-dimensional normalised observation from bicycle model.

        Observation mapping:
            obs[0] = e_lat / (LANE_WIDTH / 2)        [-1, 1]
            obs[1] = e_psi / (π/4)                    [-1, 1]
            obs[2] = kappa_ref / 0.05                  [-1, 1]
            obs[3] = v_y / 2.0                         [-1, 1]
            obs[4] = r / 0.5                           [-1, 1]
            obs[5] = delta_prev / DELTA_MAX            [-1, 1]
            obs[6] = kappa_lookahead_1 / 0.05          [-1, 1]
            obs[7] = kappa_lookahead_2 / 0.05          [-1, 1]
        """
        profile = self.profiles[self.current_scn]
        kappa_ref = profile.get_kappa_at_s(self.arc_length_s)
        kappa_la1, kappa_la2 = profile.get_lookahead_kappa(
            self.arc_length_s, self.vehicle.v_x
        )

        e_lat_obs = self.vehicle.lateral_error
        e_psi_obs = self.vehicle.heading_error

        if self.training_mode and self.domain_randomizer.enabled:
            # 1. Camera Bias
            e_lat_obs += self.domain_randomizer.params.camera_bias_m

            # 2. Gaussian Noise
            e_lat_obs += np.random.normal(0, self.obs_noise_std)
            e_psi_obs += np.random.normal(0, self.obs_noise_std * 0.5)

            # 3. Random Dropout (1% chance to lose lane lines)
            if np.random.rand() < 0.01:
                e_lat_obs = 0.0
                e_psi_obs = 0.0

        obs = np.array(
            [
                e_lat_obs / cfg.NORM_E_LAT,
                e_psi_obs / cfg.NORM_E_PSI,
                kappa_ref / cfg.NORM_KAPPA,
                self.vehicle.v_y / cfg.NORM_V_Y,
                self.vehicle.yaw_rate / cfg.NORM_YAW_RATE,
                self.delta_prev / cfg.NORM_DELTA,
                kappa_la1 / cfg.NORM_KAPPA_LA1,
                kappa_la2 / cfg.NORM_KAPPA_LA2,
            ],
            dtype=np.float32,
        )

        obs = np.clip(obs, -1.0, 1.0)

        # 4. Apply Stochastic Latency (delayed observation)
        if self.training_mode:
            obs = self.domain_randomizer.apply_obs_latency(obs)

        return obs
    @property
    def episode_data(self) -> list[dict]:
        """Return the list of step info dicts for the current episode."""
        return self._episode_data

    def get_episode_arrays(self) -> dict[str, np.ndarray]:
        """
        Convert episode data to numpy arrays for metrics computation.

        Returns:
            Dictionary mapping field names to numpy arrays.
        """
        if not self._episode_data:
            return {}

        keys = self._episode_data[0].keys()
        result = {}
        for key in keys:
            vals = [step[key] for step in self._episode_data]
            if isinstance(vals[0], (int, float, np.floating, np.integer)):
                result[key] = np.array(vals, dtype=np.float64)
            else:
                result[key] = np.array(vals)
        return result

    def close(self) -> None:
        """Clean up resources."""
        if self.backend == "carla" and self.carla_bridge is not None:
            self.carla_bridge.destroy()
        super().close()
