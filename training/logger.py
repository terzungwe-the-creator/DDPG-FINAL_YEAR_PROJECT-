"""
logger.py — Training and Evaluation Logging

Provides CSV and JSON logging for the training loop and evaluation pipeline.

Training log: 28 columns per episode (results/training_log.csv)
Evaluation log: per-timestep and per-scenario summary CSVs.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


TRAINING_LOG_COLUMNS = [
    "episode", "scenario_id", "curriculum_phase",
    "total_reward",
    "mean_e_lat_abs", "rmse_e_lat", "max_e_lat_abs",
    "mean_e_psi_abs", "rmse_e_psi",
    "lane_departure_flag", "lksr_episode",
    "delta_dot_rms", "control_effort",
    "settling_time_s", "overshoot_pct",
    "ttld_mean", "sbvr_pct",
    "episode_steps", "lap_complete",
    "critic_loss_mean", "actor_loss_mean", "q_mean",
    "noise_sigma",
    "buf_openlka_size", "buf_comma_size", "buf_argoverse_size", "buf_sim_size",
    "real_sim_ratio",
]

EVAL_RAW_COLUMNS = [
    "episode_id", "scenario_id", "timestep", "time_s",
    "e_lat_m", "e_psi_rad", "delta_rad", "delta_dot",
    "v_x", "v_y", "r", "reward", "ttld_s",
]

EVAL_SUMMARY_COLUMNS = [
    "scenario_id", "mean_e_lat", "rmse_e_lat", "max_e_lat", "rmse_e_psi",
    "lksr", "ldr", "settling_s", "overshoot_pct", "control_effort", "steer_rms",
    "ttld_p5", "sbvr_pct", "iso15622_pass",
]


class TrainingLogger:
    """
    CSV logger for the training loop.

    Writes one row per episode to results/training_log.csv with 28 columns.
    Maintains in-memory history for convergence checking and plotting.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path or cfg.TRAINING_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV header
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRAINING_LOG_COLUMNS)

        # In-memory history
        self.reward_history: List[float] = []
        self.rmse_history: List[float] = []
        self.lksr_history: List[float] = []
        self.critic_loss_history: List[float] = []
        self.actor_loss_history: List[float] = []

        # Per-step accumulator within an episode
        self._step_metrics: List[Dict] = []

    def log_step(self, update_info: Dict[str, float]) -> None:
        """Accumulate per-step training metrics within an episode."""
        self._step_metrics.append(update_info)

    def log_episode(
        self,
        episode: int,
        scenario_id: str,
        ep_metrics: Dict,
        buffer_sizes: Dict[str, int],
        noise_sigma: float = 0.0,
    ) -> None:
        """
        Log one complete episode to CSV and update in-memory history.

        Args:
            episode:      Episode number.
            scenario_id:  Scenario used in this episode.
            ep_metrics:   Dictionary with computed episode metrics.
            buffer_sizes: Current sub-buffer sizes.
            noise_sigma:  Current noise sigma.
        """
        # Compute mean training losses from step accumulator
        if self._step_metrics:
            critic_loss_mean = float(np.mean(
                [m.get("critic_loss", 0.0) for m in self._step_metrics]
            ))
            actor_loss_mean = float(np.mean(
                [m.get("actor_loss", 0.0) for m in self._step_metrics]
            ))
            q_mean = float(np.mean(
                [m.get("q_mean", 0.0) for m in self._step_metrics]
            ))
        else:
            critic_loss_mean = 0.0
            actor_loss_mean = 0.0
            q_mean = 0.0

        self._step_metrics = []

        # Determine curriculum phase
        phase = cfg.get_buffer_phase(episode)

        # Compute real-sim ratio
        n_real = sum(buffer_sizes.get(k, 0) for k in ["openlka", "comma", "argoverse"])
        n_sim = max(buffer_sizes.get("sim", 0), 1)
        rsr = n_real / n_sim

        row = [
            episode,
            scenario_id,
            phase,
            ep_metrics.get("total_reward", 0.0),
            ep_metrics.get("mean_e_lat_abs", 0.0),
            ep_metrics.get("rmse_e_lat", 0.0),
            ep_metrics.get("max_e_lat_abs", 0.0),
            ep_metrics.get("mean_e_psi_abs", 0.0),
            ep_metrics.get("rmse_e_psi", 0.0),
            ep_metrics.get("lane_departure_flag", 0),
            ep_metrics.get("lksr_episode", 0.0),
            ep_metrics.get("delta_dot_rms", 0.0),
            ep_metrics.get("control_effort", 0.0),
            ep_metrics.get("settling_time_s", 0.0),
            ep_metrics.get("overshoot_pct", 0.0),
            ep_metrics.get("ttld_mean", 0.0),
            ep_metrics.get("sbvr_pct", 0.0),
            ep_metrics.get("episode_steps", 0),
            ep_metrics.get("lap_complete", 0),
            critic_loss_mean,
            actor_loss_mean,
            q_mean,
            noise_sigma,
            buffer_sizes.get("openlka", 0),
            buffer_sizes.get("comma", 0),
            buffer_sizes.get("argoverse", 0),
            buffer_sizes.get("sim", 0),
            rsr,
        ]

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])

        # Update history
        self.reward_history.append(ep_metrics.get("total_reward", 0.0))
        self.rmse_history.append(ep_metrics.get("rmse_e_lat", 0.0))
        self.lksr_history.append(ep_metrics.get("lksr_episode", 0.0))
        self.critic_loss_history.append(critic_loss_mean)
        self.actor_loss_history.append(actor_loss_mean)


class EvalLogger:
    """
    Logger for evaluation results.

    Writes per-timestep raw data and per-scenario summaries.
    """

    def __init__(
        self,
        raw_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
    ) -> None:
        self.raw_path = raw_path or cfg.EVAL_RAW_PATH
        self.summary_path = summary_path or cfg.EVAL_SUMMARY_PATH
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialise raw CSV
        with open(self.raw_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(EVAL_RAW_COLUMNS)

    def log_timestep(
        self,
        episode_id: int,
        scenario_id: str,
        timestep: int,
        time_s: float,
        e_lat_m: float,
        e_psi_rad: float,
        delta_rad: float,
        delta_dot: float,
        v_x: float,
        v_y: float,
        r: float,
        reward: float,
        ttld_s: float,
    ) -> None:
        """Write one timestep to the raw evaluation CSV."""
        row = [
            episode_id, scenario_id, timestep, f"{time_s:.4f}",
            f"{e_lat_m:.6f}", f"{e_psi_rad:.6f}", f"{delta_rad:.6f}",
            f"{delta_dot:.6f}", f"{v_x:.4f}", f"{v_y:.6f}",
            f"{r:.6f}", f"{reward:.6f}", f"{ttld_s:.4f}",
        ]
        with open(self.raw_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def write_summary(self, summaries: List[Dict]) -> None:
        """
        Write per-scenario evaluation summaries.

        Args:
            summaries: List of dictionaries, one per scenario.
        """
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_COLUMNS)
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    k: f"{v:.6f}" if isinstance(v, float) else v
                    for k, v in s.items()
                    if k in EVAL_SUMMARY_COLUMNS
                })
