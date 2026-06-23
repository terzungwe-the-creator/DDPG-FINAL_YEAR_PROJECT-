"""
preloader.py — Dataset Pre-Load Pipeline Orchestrator

Orchestrates download checking, parsing, normalisation, and buffer
pre-population for all three external datasets.

Execution order (dependencies exist):
    1. Load DS-02 (comma-steering) → run tyre calibration
    2. Update tyre parameters with calibrated values (if R² > 0.85)
    3. Load DS-01 (OpenLKA) → push to 'openlka' sub-buffer
    4. Load DS-03 (Argoverse 2) → push to 'argoverse' sub-buffer
    5. Load DS-02 transitions → push to 'comma' sub-buffer
    6. Log preload statistics to results/dataset_preload_stats.json

Fallback behaviour:
    If a dataset directory does not exist (--skip-ds01, --skip-ds02,
    --skip-ds03 flags), skip that dataset gracefully and log a WARNING.
    The system trains and evaluates with available data.
    Pure simulation-only mode is valid as a fallback.

Caching:
    Serialises preprocessed transitions to results/cache/{source}_transitions.npy
    after first parse. Subsequent runs load from cache.
    Cache is invalidated if --reload-datasets is passed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import config as cfg
from ddpg.hybrid_buffer import HybridStratifiedBuffer

logger = logging.getLogger(__name__)


@dataclass
class PreloadStats:
    """Statistics from the dataset preload pipeline."""
    openlka_transitions_loaded: int = 0
    openlka_segments_parsed: int = 0
    openlka_filter_reject_rate: float = 0.0
    comma_transitions_loaded: int = 0
    comma_calibrated_caf: float = cfg.TYRE_CAF_NOMINAL
    comma_calibrated_car: float = cfg.TYRE_CAR_NOMINAL
    comma_calibration_r2: float = 0.0
    argoverse_transitions_loaded: int = 0
    argoverse_scenarios_processed: int = 0
    argoverse_reject_rate: float = 0.0
    total_real_transitions: int = 0
    buffer_utilisation_pct: float = 0.0
    preload_wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialisation."""
        return {
            "openlka_transitions_loaded": self.openlka_transitions_loaded,
            "openlka_segments_parsed": self.openlka_segments_parsed,
            "openlka_filter_reject_rate": round(self.openlka_filter_reject_rate, 4),
            "comma_transitions_loaded": self.comma_transitions_loaded,
            "comma_calibrated_caf": round(self.comma_calibrated_caf, 1),
            "comma_calibrated_car": round(self.comma_calibrated_car, 1),
            "comma_calibration_r2": round(self.comma_calibration_r2, 4),
            "argoverse_transitions_loaded": self.argoverse_transitions_loaded,
            "argoverse_scenarios_processed": self.argoverse_scenarios_processed,
            "argoverse_reject_rate": round(self.argoverse_reject_rate, 4),
            "total_real_transitions": self.total_real_transitions,
            "buffer_utilisation_pct": round(self.buffer_utilisation_pct, 1),
            "preload_wall_time_s": round(self.preload_wall_time_s, 1),
        }


class DatasetPreloader:
    """
    Orchestrates the complete dataset pre-load pipeline.

    This runs BEFORE the training loop starts. It populates the hybrid
    stratified buffer with real-world transitions and optionally calibrates
    the tyre model using DS-02 data.

    Attributes:
        buffer:          Hybrid stratified replay buffer to populate.
        skip_ds01:       If True, skip OpenLKA dataset.
        skip_ds02:       If True, skip comma-steering-control dataset.
        skip_ds03:       If True, skip Argoverse 2 dataset.
        reload_datasets: If True, ignore cache and re-parse all datasets.
    """

    def __init__(
        self,
        buffer: HybridStratifiedBuffer,
        skip_ds01: bool = False,
        skip_ds02: bool = False,
        skip_ds03: bool = False,
        reload_datasets: bool = False,
    ) -> None:
        self.buffer = buffer
        self.skip_ds01 = skip_ds01
        self.skip_ds02 = skip_ds02
        self.skip_ds03 = skip_ds03
        self.reload_datasets = reload_datasets
        self.stats = PreloadStats()

    def _check_dataset_availability(self) -> Dict[str, bool]:
        """
        Check which datasets are available on disk.

        Returns:
            Dictionary mapping source name to availability flag.
        """
        availability = {
            "openlka": not self.skip_ds01 and cfg.OPENLKA_DATA_DIR.exists(),
            "comma": not self.skip_ds02 and cfg.COMMA_DATA_DIR.exists(),
            "argoverse": not self.skip_ds03 and cfg.ARGOVERSE_DATA_DIR.exists(),
        }

        for source, available in availability.items():
            if not available:
                skip_flag = (
                    (source == "openlka" and self.skip_ds01) or
                    (source == "comma" and self.skip_ds02) or
                    (source == "argoverse" and self.skip_ds03)
                )
                if skip_flag:
                    logger.info(f"Dataset '{source}' skipped by user flag")
                else:
                    logger.warning(
                        f"Dataset '{source}' directory not found — skipping"
                    )

        return availability

    def _get_cache_path(self, source: str) -> Path:
        """Get the cache file path for a given source."""
        return cfg.CACHE_DIR / f"{source}_transitions.npy"

    def _load_from_cache(self, source: str) -> Optional[int]:
        """
        Attempt to load pre-parsed transitions from cache.

        The cache stores transitions as a structured numpy array with fields:
        (state, action, reward, next_state, done).

        Args:
            source: Data source name.

        Returns:
            Number of transitions loaded, or None if cache miss.
        """
        cache_path = self._get_cache_path(source)

        if self.reload_datasets or not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=True)
            if not isinstance(data, np.ndarray) or data.ndim == 0:
                data = data.item()

            states = data["states"]
            actions = data["actions"]
            rewards = data["rewards"]
            next_states = data["next_states"]
            dones = data["dones"]

            n = len(states)
            for i in range(n):
                self.buffer.push(
                    source,
                    states[i],
                    actions[i],
                    float(rewards[i]),
                    next_states[i],
                    float(dones[i]),
                )

            logger.info(f"Loaded {n} cached transitions for '{source}'")
            return n

        except Exception as e:
            logger.warning(f"Cache load failed for '{source}': {e}")
            return None

    def _save_to_cache(self, source: str) -> None:
        """
        Save the current sub-buffer contents to cache.

        Args:
            source: Data source name.
        """
        cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(source)

        sub_buf = self.buffer.sub_buffers[source]
        n = sub_buf.size

        if n == 0:
            return

        data = {
            "states": sub_buf.states[:n].copy(),
            "actions": sub_buf.actions[:n].copy(),
            "rewards": sub_buf.rewards[:n].copy(),
            "next_states": sub_buf.next_states[:n].copy(),
            "dones": sub_buf.dones[:n].copy(),
        }

        np.save(cache_path, data, allow_pickle=True)
        logger.info(f"Cached {n} transitions for '{source}' to {cache_path}")

    def run(self) -> PreloadStats:
        """
        Execute the full dataset pre-load pipeline.

        Order:
            1. DS-02 → tyre calibration
            2. DS-01 → expert demonstrations
            3. DS-03 → trajectory diversity
            4. DS-02 → buffer diversity seeding
            5. Statistics logging

        Returns:
            PreloadStats with counts and calibration results.
        """
        start_time = time.time()
        cfg.ensure_directories()

        availability = self._check_dataset_availability()

        # ── Step 1: DS-02 — Tyre calibration ─────────────────────────────────
        if availability["comma"]:
            logger.info("=" * 60)
            logger.info("Step 1/4: DS-02 comma-steering — Tyre calibration")
            logger.info("=" * 60)

            from datasets.comma_steering_adapter import CommaSteeringAdapter
            comma_adapter = CommaSteeringAdapter(str(cfg.COMMA_DATA_DIR))

            caf, car = comma_adapter.calibrate_tyre_model()
            cal = comma_adapter.calibration_result

            if cal is not None:
                self.stats.comma_calibrated_caf = cal["C_af"]
                self.stats.comma_calibrated_car = cal["C_ar"]
                self.stats.comma_calibration_r2 = cal["r_squared"]
        else:
            logger.info("DS-02 not available — using nominal tyre parameters")
            comma_adapter = None

        # ── Step 2: DS-01 — OpenLKA expert demonstrations ───────────────────
        if availability["openlka"]:
            logger.info("=" * 60)
            logger.info("Step 2/4: DS-01 OpenLKA — Expert LKA demonstrations")
            logger.info("=" * 60)

            cached = self._load_from_cache("openlka")
            if cached is not None:
                self.stats.openlka_transitions_loaded = cached
            else:
                from datasets.openlka_adapter import OpenLKAAdapter
                openlka = OpenLKAAdapter(str(cfg.OPENLKA_DATA_DIR))
                n_loaded = openlka.load_into_buffer(self.buffer)
                self.stats.openlka_transitions_loaded = n_loaded
                self.stats.openlka_segments_parsed = openlka.stats.get(
                    "segments_parsed", 0
                )
                self.stats.openlka_filter_reject_rate = openlka.stats.get(
                    "filter_reject_rate", 0.0
                )
                self._save_to_cache("openlka")
        else:
            logger.info("DS-01 not available — skipping OpenLKA")

        # ── Step 3: DS-03 — Argoverse 2 trajectory diversity ────────────────
        if availability["argoverse"]:
            logger.info("=" * 60)
            logger.info("Step 3/4: DS-03 Argoverse 2 — Trajectory diversity")
            logger.info("=" * 60)

            cached = self._load_from_cache("argoverse")
            if cached is not None:
                self.stats.argoverse_transitions_loaded = cached
            else:
                from datasets.argoverse2_adapter import Argoverse2Adapter
                av2 = Argoverse2Adapter(str(cfg.ARGOVERSE_DATA_DIR))
                n_loaded = av2.load_into_buffer(self.buffer)
                self.stats.argoverse_transitions_loaded = n_loaded
                self.stats.argoverse_scenarios_processed = av2.stats.get(
                    "scenarios_processed", 0
                )
                self.stats.argoverse_reject_rate = av2.stats.get(
                    "reject_rate", 0.0
                )
                self._save_to_cache("argoverse")
        else:
            logger.info("DS-03 not available — skipping Argoverse 2")

        # ── Step 4: DS-02 — Buffer diversity seeding ─────────────────────────
        if availability["comma"] and comma_adapter is not None:
            logger.info("=" * 60)
            logger.info("Step 4/4: DS-02 comma-steering — Buffer diversity")
            logger.info("=" * 60)

            cached = self._load_from_cache("comma")
            if cached is not None:
                self.stats.comma_transitions_loaded = cached
            else:
                n_loaded = comma_adapter.load_into_buffer(self.buffer)
                self.stats.comma_transitions_loaded = n_loaded
                self._save_to_cache("comma")
        else:
            logger.info("DS-02 buffer seeding skipped")

        # ── Finalise statistics ──────────────────────────────────────────────
        self.stats.total_real_transitions = (
            self.stats.openlka_transitions_loaded
            + self.stats.comma_transitions_loaded
            + self.stats.argoverse_transitions_loaded
        )

        total_capacity = sum(cfg.BUFFER_CAPACITIES.values())
        if total_capacity > 0:
            self.stats.buffer_utilisation_pct = (
                self.stats.total_real_transitions / total_capacity * 100.0
            )

        self.stats.preload_wall_time_s = time.time() - start_time

        # Save stats
        stats_path = cfg.PRELOAD_STATS_PATH
        with open(stats_path, "w") as f:
            json.dump(self.stats.to_dict(), f, indent=2)

        # Log summary
        logger.info("=" * 60)
        logger.info("Dataset Preload Complete")
        logger.info(f"  OpenLKA:    {self.stats.openlka_transitions_loaded:>8d} transitions")
        logger.info(f"  Comma:      {self.stats.comma_transitions_loaded:>8d} transitions")
        logger.info(f"  Argoverse:  {self.stats.argoverse_transitions_loaded:>8d} transitions")
        logger.info(f"  Total real: {self.stats.total_real_transitions:>8d} transitions")
        logger.info(f"  Buffer use: {self.stats.buffer_utilisation_pct:.1f}%")
        logger.info(f"  Wall time:  {self.stats.preload_wall_time_s:.1f}s")
        logger.info("=" * 60)

        return self.stats
