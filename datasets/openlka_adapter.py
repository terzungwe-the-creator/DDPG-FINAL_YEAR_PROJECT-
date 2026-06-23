"""
openlka_adapter.py — DS-01: OpenLKA Dataset Adapter

Loads OpenLKA CSV segments and converts them to DDPG-compatible
(state, action, reward, next_state, done) transitions.

Source: Wang et al. (2025), "OpenLKA: An Open Dataset of Lane Keeping Assist
        from Recent Car Models under Real-world Driving Conditions",
        arXiv:2505.09092, §III-D.
Dataset: https://github.com/OpenLKA/OpenLKA
License: MIT

Signals (100 Hz CAN bus data):
    t              — timestamp (s)
    latActive      — LKA system engaged (bool)
    steeringAngle  — front wheel angle (degrees → rad)
    laneOffset     — lateral deviation from lane centre (m) = e_lat
    yawRate        — r (rad/s)
    speed          — v_x (m/s)
    curvature      — κ_ref (1/m)
    lkaStatus      — LKA intervention flag
    jerk           — longitudinal jerk (m/s³)

Filter: Only load rows where latActive == True (expert demonstrations).
Target transitions: 500,000 (upper bound).
Preferred vehicles: Toyota, Ford, Kia, Hyundai (fully decoded CAN).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import config as cfg
from datasets.normaliser import RawTransition, UniversalNormaliser
from ddpg.hybrid_buffer import HybridStratifiedBuffer
from simulator.reward import compute_reward

logger = logging.getLogger(__name__)


class OpenLKAAdapter:
    """
    Loads OpenLKA CSV segments and converts them to DDPG-compatible transitions.

    Processing pipeline:
        1. Scan data_dir for CSV files (one per driving segment).
        2. Load each CSV, filter to latActive == True rows only.
        3. For consecutive row pairs, compute:
           - Observation vector (e_lat, e_psi_estimate, kappa, v_y_est, yawRate, etc.)
           - Action (steeringAngle in radians, normalised)
           - Reward (using the same reward function as the simulator)
           - Done flag (segment end, latActive→False, or |laneOffset| > 1.75 m)
        4. Push normalised transitions to the 'openlka' sub-buffer.

    Heading error estimation:
        OpenLKA does not directly provide heading error (e_psi).
        We estimate it from yaw rate and curvature:
            e_psi ≈ cumulative integral of (r − κ·v_x) over short windows.
        For consecutive rows: Δe_psi ≈ (yawRate − curvature × speed) × dt.
    """

    # Expected CSV column mappings (OpenLKA format)
    COL_MAP = {
        "t": "t",
        "latActive": "latActive",
        "steeringAngle": "steeringAngle",
        "laneOffset": "laneOffset",
        "yawRate": "yawRate",
        "speed": "speed",
        "curvature": "curvature",
    }

    def __init__(
        self,
        data_dir: str,
        max_segments: int = cfg.OPENLKA_MAX_SEGMENTS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_segments = max_segments
        self.normaliser = UniversalNormaliser()
        self._stats = {
            "segments_found": 0,
            "segments_parsed": 0,
            "rows_total": 0,
            "rows_active": 0,
            "transitions_created": 0,
            "filter_reject_rate": 0.0,
        }

    def load_all_segments(self) -> List[pd.DataFrame]:
        """
        Scan data_dir for CSV files and load them.

        Returns:
            List of DataFrames, one per segment.
        """
        if not self.data_dir.exists():
            logger.warning(f"OpenLKA data directory not found: {self.data_dir}")
            return []

        csv_files = sorted(self.data_dir.glob("**/*.csv"))
        self._stats["segments_found"] = len(csv_files)

        if len(csv_files) == 0:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return []

        logger.info(f"Found {len(csv_files)} OpenLKA CSV segments")

        # Limit to max_segments
        if len(csv_files) > self.max_segments:
            rng = np.random.RandomState(cfg.SEED)
            indices = rng.choice(len(csv_files), self.max_segments, replace=False)
            csv_files = [csv_files[i] for i in sorted(indices)]

        segments = []
        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath)
                # Check required columns exist
                required = ["latActive", "steeringAngle", "laneOffset",
                            "yawRate", "speed", "curvature"]
                if all(col in df.columns for col in required):
                    segments.append(df)
                else:
                    missing = [c for c in required if c not in df.columns]
                    logger.debug(f"Skipping {fpath.name}: missing columns {missing}")
            except Exception as e:
                logger.debug(f"Failed to read {fpath.name}: {e}")

        self._stats["segments_parsed"] = len(segments)
        logger.info(f"Successfully loaded {len(segments)} OpenLKA segments")
        return segments

    def segment_to_transitions(
        self, df: pd.DataFrame
    ) -> List[RawTransition]:
        """
        Convert a single CSV segment to a list of RawTransitions.

        Args:
            df: DataFrame with OpenLKA columns.

        Returns:
            List of RawTransition objects.
        """
        # Filter to active LKA rows only
        if "latActive" in df.columns:
            active_mask = df["latActive"].astype(bool)
            df_active = df[active_mask].reset_index(drop=True)
        else:
            df_active = df

        self._stats["rows_total"] += len(df)
        self._stats["rows_active"] += len(df_active)

        if len(df_active) < 3:
            return []

        # Filter by minimum speed
        if "speed" in df_active.columns:
            df_active = df_active[
                df_active["speed"] >= cfg.OPENLKA_MIN_SPEED_MPS
            ].reset_index(drop=True)

        if len(df_active) < 3:
            return []

        transitions: List[RawTransition] = []

        # Compute heading error estimate via incremental integration
        e_psi_estimate = np.zeros(len(df_active))
        for i in range(1, len(df_active)):
            yaw_rate = df_active["yawRate"].iloc[i]
            kappa = df_active["curvature"].iloc[i]
            speed = df_active["speed"].iloc[i]

            # Determine dt from timestamps if available
            if "t" in df_active.columns:
                dt = df_active["t"].iloc[i] - df_active["t"].iloc[i - 1]
                dt = max(dt, 0.001)  # Clamp to prevent zero dt
            else:
                dt = 0.01  # Default 100 Hz

            # ė_psi = r − κ·v_x
            e_psi_dot = yaw_rate - kappa * speed
            e_psi_estimate[i] = e_psi_estimate[i - 1] + e_psi_dot * dt

            # Wrap to [-π, π]
            e_psi_estimate[i] = (e_psi_estimate[i] + np.pi) % (2 * np.pi) - np.pi

        # Build transitions from consecutive rows
        for i in range(len(df_active) - 1):
            row_curr = df_active.iloc[i]
            row_next = df_active.iloc[i + 1]

            e_lat = row_curr["laneOffset"]
            e_lat_next = row_next["laneOffset"]
            speed_curr = row_curr["speed"]
            speed_next = row_next["speed"]
            kappa_curr = row_curr["curvature"]
            kappa_next = row_next["curvature"]

            # Steering angle: degrees → radians
            delta_curr_rad = np.deg2rad(row_curr["steeringAngle"])
            delta_next_rad = np.deg2rad(row_next["steeringAngle"])

            # Previous steering (for smoothness) — use i-1 if available
            if i > 0:
                delta_prev_rad = np.deg2rad(df_active.iloc[i - 1]["steeringAngle"])
            else:
                delta_prev_rad = delta_curr_rad

            # Estimate lateral velocity from v_x * sin(e_psi)
            v_y_est = speed_curr * np.sin(e_psi_estimate[i])
            v_y_next_est = speed_next * np.sin(e_psi_estimate[i + 1])

            yaw_rate_curr = row_curr["yawRate"]
            yaw_rate_next = row_next["yawRate"]

            # Lookahead curvature — approximate using future rows
            # 1s lookahead at 100 Hz = ~100 rows ahead
            la1_idx = min(i + 100, len(df_active) - 1)
            la2_idx = min(i + 200, len(df_active) - 1)
            kappa_la1 = df_active["curvature"].iloc[la1_idx]
            kappa_la2 = df_active["curvature"].iloc[la2_idx]

            kappa_la1_next = df_active["curvature"].iloc[min(la1_idx + 1, len(df_active) - 1)]
            kappa_la2_next = df_active["curvature"].iloc[min(la2_idx + 1, len(df_active) - 1)]

            # Compute reward using the same function as the simulator
            reward, _ = compute_reward(
                e_lat=e_lat,
                e_psi=e_psi_estimate[i],
                delta_current=delta_curr_rad,
                delta_previous=delta_prev_rad,
                v_x=speed_curr,
                terminated=False,
            )

            # Done conditions
            is_last = (i == len(df_active) - 2)
            departed = abs(e_lat_next) > cfg.DEPARTURE_THRESHOLD
            done = is_last or departed

            try:
                trans = RawTransition(
                    source="openlka",
                    e_lat_m=float(e_lat),
                    e_psi_rad=float(e_psi_estimate[i]),
                    kappa_ref=float(kappa_curr),
                    v_y_mps=float(v_y_est),
                    yaw_rate_rads=float(yaw_rate_curr),
                    delta_prev_rad=float(delta_prev_rad),
                    kappa_la1=float(kappa_la1),
                    kappa_la2=float(kappa_la2),
                    action_raw_rad=float(delta_curr_rad),
                    reward=float(reward),
                    done=done,
                    next_e_lat_m=float(e_lat_next),
                    next_e_psi_rad=float(e_psi_estimate[i + 1]),
                    next_kappa_ref=float(kappa_next),
                    next_v_y_mps=float(v_y_next_est),
                    next_yaw_rate_rads=float(yaw_rate_next),
                    next_delta_prev_rad=float(delta_curr_rad),
                    next_kappa_la1=float(kappa_la1_next),
                    next_kappa_la2=float(kappa_la2_next),
                )
                transitions.append(trans)
            except ValueError as e:
                # Bounds violation — skip this transition
                logger.debug(f"OpenLKA transition skipped (bounds): {e}")
                continue

        return transitions

    def load_into_buffer(
        self,
        buffer: HybridStratifiedBuffer,
        max_transitions: int = cfg.OPENLKA_MAX_TRANSITIONS,
    ) -> int:
        """
        Load OpenLKA transitions into the 'openlka' sub-buffer.

        Randomises segment order before loading to prevent temporal bias.

        Args:
            buffer:          Hybrid stratified replay buffer.
            max_transitions: Maximum number of transitions to load.

        Returns:
            Number of transitions successfully loaded.
        """
        segments = self.load_all_segments()
        if not segments:
            logger.warning("No OpenLKA segments available — skipping DS-01")
            return 0

        # Randomise segment order
        rng = np.random.RandomState(cfg.SEED)
        indices = rng.permutation(len(segments))

        loaded = 0
        for idx in indices:
            if loaded >= max_transitions:
                break

            transitions = self.segment_to_transitions(segments[idx])
            for trans in transitions:
                if loaded >= max_transitions:
                    break
                try:
                    obs = self.normaliser.normalise_obs(trans)
                    next_obs = self.normaliser.normalise_next_obs(trans)
                    action_norm = self.normaliser.normalise_action(trans.action_raw_rad)

                    buffer.push(
                        "openlka", obs, action_norm,
                        trans.reward, next_obs, float(trans.done)
                    )
                    loaded += 1
                except ValueError:
                    continue

        self._stats["transitions_created"] = loaded
        if self._stats["rows_total"] > 0:
            self._stats["filter_reject_rate"] = 1.0 - (
                self._stats["rows_active"] / self._stats["rows_total"]
            )

        logger.info(
            f"OpenLKA: loaded {loaded} transitions from "
            f"{self._stats['segments_parsed']} segments "
            f"(reject rate: {self._stats['filter_reject_rate']:.3f})"
        )
        return loaded

    @property
    def stats(self) -> dict:
        """Return loading statistics."""
        return dict(self._stats)
