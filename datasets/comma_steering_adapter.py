"""
comma_steering_adapter.py — DS-02: comma-steering-control Dataset Adapter

Loads comma-steering-control .npy segments and extracts:
    1. Steering command distribution priors for OU noise calibration.
    2. Lateral acceleration vs steering angle relationship for tyre model
       validation (cross-check against linear tyre model F_y = -C_a * alpha).
    3. High-diversity (state, action) pairs for replay buffer pre-population.

Source: comma.ai commaSteeringControl dataset
        https://github.com/commaai/comma-steering-control
Scale:  ~12,500 hours, 300+ vehicle platforms.
License: MIT

Data schema (per segment, NumPy arrays):
    t              — time (s), shape (N,)
    latActive      — openpilot engaged (bool), shape (N,)
    steeringPressed— human override (bool), shape (N,)
    vEgo           — ego speed (m/s), shape (N,)
    aEgo           — longitudinal acceleration (m/s²), shape (N,)
    steeringAngleDeg — steering angle (degrees), shape (N,)
    steeringTorque — steering torque command (Nm), shape (N,)
    latAccelDevice — lateral acceleration (m/s²), shape (N,)
    angleOffsetDeg — camera-to-car angle offset (deg), shape (N,)

Filter: latActive == True AND steeringPressed == False (clean openpilot)
        vEgo > 10.0 m/s (highway speeds, matches V_REFERENCE)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

import config as cfg
from datasets.normaliser import RawTransition, UniversalNormaliser
from ddpg.hybrid_buffer import HybridStratifiedBuffer
from simulator.reward import compute_reward

logger = logging.getLogger(__name__)


class CommaSteeringAdapter:
    """
    Loads comma-steering-control .npy segments.

    Primary use: Tyre model calibration — fit C_af, C_ar from latAccelDevice
    vs steeringAngleDeg at known vEgo values.

    Secondary use: Replay buffer seeding with expert (state, action) pairs
    for training diversity.
    """

    # Expected fields in each .npy segment (structured array or dict of arrays)
    REQUIRED_FIELDS = [
        "latActive", "steeringPressed", "vEgo",
        "steeringAngleDeg", "latAccelDevice",
    ]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.normaliser = UniversalNormaliser()
        self._calibration_result: Optional[Dict] = None
        self._stats = {
            "segments_found": 0,
            "segments_loaded": 0,
            "rows_total": 0,
            "rows_filtered": 0,
            "transitions_created": 0,
        }

    def _load_segments(self) -> List[Dict[str, np.ndarray]]:
        """
        Scan data_dir for .npy files and load them.

        Each .npy file may be a structured array or a dictionary saved via
        np.save with allow_pickle=True. We handle both formats.

        Returns:
            List of dictionaries mapping field names to numpy arrays.
        """
        if not self.data_dir.exists():
            logger.warning(f"Comma steering data directory not found: {self.data_dir}")
            return []

        npy_files = sorted(self.data_dir.glob("**/*.npy"))
        self._stats["segments_found"] = len(npy_files)

        if len(npy_files) == 0:
            logger.warning(f"No .npy files found in {self.data_dir}")
            return []

        logger.info(f"Found {len(npy_files)} comma-steering .npy segments")

        segments = []
        for fpath in npy_files:
            try:
                raw = np.load(fpath, allow_pickle=True)
                # Handle different storage formats
                if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
                    # Structured array
                    seg = {name: raw[name] for name in raw.dtype.names}
                elif isinstance(raw, np.ndarray) and raw.ndim == 0:
                    # Dictionary saved as 0-d object array
                    seg = raw.item()
                else:
                    # Try as dict directly
                    seg = dict(raw) if hasattr(raw, 'keys') else None

                if seg is not None and all(f in seg for f in self.REQUIRED_FIELDS):
                    segments.append(seg)
                    self._stats["segments_loaded"] += 1
            except Exception as e:
                logger.debug(f"Failed to load {fpath.name}: {e}")

        logger.info(f"Successfully loaded {len(segments)} comma-steering segments")
        return segments

    def _filter_segment(
        self, seg: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply quality filters to a segment.

        Filters:
            - latActive == True (openpilot engaged)
            - steeringPressed == False (no human override)
            - vEgo > 10.0 m/s (highway speeds)

        Args:
            seg: Dictionary of arrays for one segment.

        Returns:
            Filtered dictionary with only valid rows.
        """
        n = len(seg["vEgo"])
        self._stats["rows_total"] += n

        mask = np.ones(n, dtype=bool)

        if "latActive" in seg:
            mask &= seg["latActive"].astype(bool)
        if "steeringPressed" in seg:
            mask &= ~seg["steeringPressed"].astype(bool)
        if "vEgo" in seg:
            mask &= seg["vEgo"] > cfg.COMMA_MIN_SPEED_MPS

        self._stats["rows_filtered"] += mask.sum()

        return {key: arr[mask] for key, arr in seg.items()}

    def calibrate_tyre_model(self) -> Tuple[float, float]:
        """
        Fit tyre cornering stiffness (C_af, C_ar) from lateral acceleration data.

        Uses least-squares fit of the lateral acceleration model:
            a_y = v_x² · κ  where κ = δ / L (for small angles)
            F_y_total = m · a_y
            C_eff ≈ m · a_y / δ

        For the split between front and rear:
            C_af_est = C_eff · (l_r / L)  — front carries rear-biased share
            C_ar_est = C_eff · (l_f / L)  — rear carries front-biased share

        This is the standard quasi-static calibration method used in
        vehicle dynamics identification (Rajamani 2012, §3.5).

        Returns:
            (C_af_fitted, C_ar_fitted) in N/rad.

        Side effects:
            Saves calibration results to results/tyre_calibration.json.
            Stores result in self._calibration_result.
        """
        segments = self._load_segments()
        if not segments:
            logger.warning(
                "No comma-steering data — returning nominal tyre parameters"
            )
            self._calibration_result = {
                "C_af": cfg.TYRE_CAF_NOMINAL,
                "C_ar": cfg.TYRE_CAR_NOMINAL,
                "r_squared": 0.0,
                "n_samples": 0,
                "source": "nominal_fallback",
            }
            return cfg.TYRE_CAF_NOMINAL, cfg.TYRE_CAR_NOMINAL

        # Collect (steering_angle_rad, lat_accel, speed) triples
        all_delta = []
        all_ay = []
        all_vx = []

        for seg in segments:
            filtered = self._filter_segment(seg)
            if len(filtered["vEgo"]) < 10:
                continue

            delta_rad = np.deg2rad(filtered["steeringAngleDeg"])
            ay = filtered["latAccelDevice"]
            vx = filtered["vEgo"]

            # Remove samples with very small steering (noise-dominated)
            valid = np.abs(delta_rad) > 0.005  # > 0.3° threshold
            all_delta.append(delta_rad[valid])
            all_ay.append(ay[valid])
            all_vx.append(vx[valid])

        if not all_delta:
            logger.warning("No valid calibration data after filtering")
            self._calibration_result = {
                "C_af": cfg.TYRE_CAF_NOMINAL,
                "C_ar": cfg.TYRE_CAR_NOMINAL,
                "r_squared": 0.0,
                "n_samples": 0,
                "source": "nominal_fallback",
            }
            return cfg.TYRE_CAF_NOMINAL, cfg.TYRE_CAR_NOMINAL

        delta_all = np.concatenate(all_delta)
        ay_all = np.concatenate(all_ay)
        vx_all = np.concatenate(all_vx)

        # Subsample if too large (for computational efficiency)
        max_samples = 500_000
        if len(delta_all) > max_samples:
            rng = np.random.RandomState(cfg.SEED)
            idx = rng.choice(len(delta_all), max_samples, replace=False)
            delta_all = delta_all[idx]
            ay_all = ay_all[idx]
            vx_all = vx_all[idx]

        # Fit: a_y = C_eff / m · δ (linearised model)
        # Using least-squares: C_eff = m · (Σ(a_y · δ)) / (Σ(δ²))
        numerator = np.sum(ay_all * delta_all)
        denominator = np.sum(delta_all ** 2)

        if abs(denominator) < 1e-12:
            logger.warning("Degenerate calibration data — using nominal")
            self._calibration_result = {
                "C_af": cfg.TYRE_CAF_NOMINAL,
                "C_ar": cfg.TYRE_CAR_NOMINAL,
                "r_squared": 0.0,
                "n_samples": len(delta_all),
                "source": "nominal_fallback",
            }
            return cfg.TYRE_CAF_NOMINAL, cfg.TYRE_CAR_NOMINAL

        C_eff = cfg.VEHICLE_MASS * numerator / denominator

        # Clamp to physically plausible range
        C_eff = np.clip(C_eff, 30_000.0, 300_000.0)

        # Split front/rear using wheelbase ratio
        L = cfg.VEHICLE_WHEELBASE
        C_af_fitted = float(C_eff * (cfg.VEHICLE_LR / L))
        C_ar_fitted = float(C_eff * (cfg.VEHICLE_LF / L))

        # Compute R² (coefficient of determination)
        ay_predicted = (C_eff / cfg.VEHICLE_MASS) * delta_all
        ss_res = np.sum((ay_all - ay_predicted) ** 2)
        ss_tot = np.sum((ay_all - np.mean(ay_all)) ** 2)
        r_squared = float(1.0 - ss_res / max(ss_tot, 1e-12))
        r_squared = max(r_squared, 0.0)  # Clamp negative R²

        self._calibration_result = {
            "C_af": C_af_fitted,
            "C_ar": C_ar_fitted,
            "C_eff": float(C_eff),
            "r_squared": r_squared,
            "n_samples": len(delta_all),
            "source": "comma_steering_calibration",
        }

        # Save calibration results
        cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cal_path = cfg.TYRE_CALIBRATION_PATH
        with open(cal_path, "w") as f:
            json.dump(self._calibration_result, f, indent=2)

        logger.info(
            f"Tyre calibration: C_af={C_af_fitted:.0f} N/rad, "
            f"C_ar={C_ar_fitted:.0f} N/rad, "
            f"R²={r_squared:.3f}, n={len(delta_all)}"
        )

        return C_af_fitted, C_ar_fitted

    def load_into_buffer(
        self,
        buffer: HybridStratifiedBuffer,
        max_transitions: int = cfg.COMMA_MAX_TRANSITIONS,
    ) -> int:
        """
        Load comma-steering transitions into the 'comma' sub-buffer.

        For each valid segment, builds (state, action, reward, next_state, done)
        transitions. The state is constructed from available signals:
            - e_lat estimated from lateral acceleration integration
            - e_psi estimated from steering angle and vehicle model
            - kappa from steering angle / wheelbase approximation

        Args:
            buffer:          Hybrid stratified replay buffer.
            max_transitions: Maximum number of transitions to load.

        Returns:
            Number of transitions successfully loaded.
        """
        segments = self._load_segments()
        if not segments:
            logger.warning("No comma-steering data — skipping DS-02 buffer load")
            return 0

        rng = np.random.RandomState(cfg.SEED)
        rng.shuffle(segments)

        loaded = 0
        for seg in segments:
            if loaded >= max_transitions:
                break

            filtered = self._filter_segment(seg)
            n = len(filtered["vEgo"])
            if n < 3:
                continue

            delta_rad = np.deg2rad(filtered["steeringAngleDeg"])
            vx = filtered["vEgo"]
            ay = filtered["latAccelDevice"]

            # Estimate curvature: κ ≈ δ / L (small angle approximation)
            kappa_est = delta_rad / cfg.VEHICLE_WHEELBASE

            # Estimate lateral velocity: v_y ≈ ∫(a_y - v_x·r) dt
            # Simplified: v_y ≈ a_y · dt (reset each segment)
            dt = 0.01  # assume 100 Hz
            if "t" in filtered and len(filtered["t"]) > 1:
                dt_arr = np.diff(filtered["t"])
                dt_arr = np.clip(dt_arr, 0.001, 0.1)
            else:
                dt_arr = np.full(n - 1, dt)

            # Estimate e_lat via double integration of lateral acceleration
            # (rough approximation — comma data lacks direct lane offset)
            v_y_est = np.zeros(n)
            e_lat_est = np.zeros(n)
            for i in range(1, n):
                v_y_est[i] = v_y_est[i - 1] + ay[i] * dt_arr[min(i - 1, len(dt_arr) - 1)]
                v_y_est[i] *= 0.95  # Decay to prevent drift
                e_lat_est[i] = e_lat_est[i - 1] + v_y_est[i] * dt_arr[min(i - 1, len(dt_arr) - 1)]
                e_lat_est[i] *= 0.98  # Decay to prevent drift

            # Estimate heading error from yaw rate (if available) or steering
            e_psi_est = np.zeros(n)
            yaw_rate_est = ay / np.maximum(vx, 1.0)  # r ≈ a_y / v_x

            for i in range(1, n):
                e_psi_dot = yaw_rate_est[i] - kappa_est[i] * vx[i]
                e_psi_est[i] = e_psi_est[i - 1] + e_psi_dot * dt_arr[min(i - 1, len(dt_arr) - 1)]
                e_psi_est[i] = (e_psi_est[i] + np.pi) % (2 * np.pi) - np.pi

            # Build transitions
            for i in range(n - 1):
                if loaded >= max_transitions:
                    break

                delta_prev = delta_rad[i - 1] if i > 0 else delta_rad[i]

                # Lookahead curvature
                la1_idx = min(i + 100, n - 1)
                la2_idx = min(i + 200, n - 1)

                reward, _ = compute_reward(
                    e_lat=e_lat_est[i],
                    e_psi=e_psi_est[i],
                    delta_current=delta_rad[i],
                    delta_previous=delta_prev,
                    v_x=vx[i],
                    terminated=False,
                )

                is_last = (i == n - 2)
                done = is_last or abs(e_lat_est[i + 1]) > cfg.DEPARTURE_THRESHOLD

                try:
                    trans = RawTransition(
                        source="comma",
                        e_lat_m=float(e_lat_est[i]),
                        e_psi_rad=float(e_psi_est[i]),
                        kappa_ref=float(kappa_est[i]),
                        v_y_mps=float(v_y_est[i]),
                        yaw_rate_rads=float(yaw_rate_est[i]),
                        delta_prev_rad=float(delta_prev),
                        kappa_la1=float(kappa_est[la1_idx]),
                        kappa_la2=float(kappa_est[la2_idx]),
                        action_raw_rad=float(delta_rad[i]),
                        reward=float(reward),
                        done=done,
                        next_e_lat_m=float(e_lat_est[i + 1]),
                        next_e_psi_rad=float(e_psi_est[i + 1]),
                        next_kappa_ref=float(kappa_est[i + 1]),
                        next_v_y_mps=float(v_y_est[i + 1]),
                        next_yaw_rate_rads=float(yaw_rate_est[i + 1]),
                        next_delta_prev_rad=float(delta_rad[i]),
                        next_kappa_la1=float(kappa_est[min(la1_idx + 1, n - 1)]),
                        next_kappa_la2=float(kappa_est[min(la2_idx + 1, n - 1)]),
                    )

                    obs = self.normaliser.normalise_obs(trans)
                    next_obs = self.normaliser.normalise_next_obs(trans)
                    action_norm = self.normaliser.normalise_action(trans.action_raw_rad)

                    buffer.push(
                        "comma", obs, action_norm,
                        trans.reward, next_obs, float(trans.done)
                    )
                    loaded += 1
                except ValueError:
                    continue

        self._stats["transitions_created"] = loaded
        logger.info(f"Comma-steering: loaded {loaded} transitions")
        return loaded

    @property
    def calibration_result(self) -> Optional[Dict]:
        """Return the tyre calibration result dictionary."""
        return self._calibration_result

    @property
    def stats(self) -> dict:
        """Return loading statistics."""
        return dict(self._stats)
