"""
trainer.py — Main DDPG Training Loop

Implements the 600-episode training pipeline with:
    - Dataset preloading (Phase 0)
    - Tyre model calibration update
    - M-22 pretrain-only evaluation
    - Curriculum-based scenario progression
    - Phase-aware hybrid buffer sampling
    - OU noise annealing
    - Checkpoint saving
    - Convergence monitoring

Reference: Lillicrap et al. (2016), arXiv:1509.02971 — DDPG algorithm.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import config as cfg
from ddpg.agent import DDPGAgent
from ddpg.hybrid_buffer import HybridStratifiedBuffer
from ddpg.noise import OUNoise
from datasets.preloader import DatasetPreloader, PreloadStats
from metrics.iso15622 import compute_mean_lat_error, compute_rmse_lat, compute_max_lat_error, compute_heading_rmse, compute_lksr
from metrics.ieee2846 import compute_steering_rate_rms, compute_control_effort, compute_settling_time, compute_overshoot
from metrics.safety import compute_ttld_series, compute_sbvr
from metrics.dataset_quality import compute_pretrain_performance
from simulator.lane_keeping_env import LaneKeepingEnv
from training.curriculum import get_curriculum_profile, get_curriculum_phase_name
from training.logger import TrainingLogger

logger = logging.getLogger(__name__)


def compute_episode_metrics(episode_data: list[dict]) -> Dict:
    """
    Compute all metrics for a completed episode.

    Args:
        episode_data: List of step info dicts from the environment.

    Returns:
        Dictionary with all episode-level metrics.
    """
    if not episode_data:
        return {
            "total_reward": 0.0, "mean_e_lat_abs": 0.0, "rmse_e_lat": 0.0,
            "max_e_lat_abs": 0.0, "mean_e_psi_abs": 0.0, "rmse_e_psi": 0.0,
            "lane_departure_flag": 0, "lksr_episode": 1.0,
            "delta_dot_rms": 0.0, "control_effort": 0.0,
            "settling_time_s": 0.0, "overshoot_pct": 0.0,
            "ttld_mean": 999.0, "sbvr_pct": 0.0,
            "episode_steps": 0, "lap_complete": 0,
        }

    e_lat = np.array([s["e_lat_m"] for s in episode_data])
    e_psi = np.array([s["e_psi_rad"] for s in episode_data])
    delta = np.array([s["delta_rad"] for s in episode_data])
    rewards = np.array([s["reward"] for s in episode_data])

    ttld = compute_ttld_series(e_lat)
    departed = bool(np.any(np.abs(e_lat) >= cfg.DEPARTURE_THRESHOLD))

    return {
        "total_reward": float(np.sum(rewards)),
        "mean_e_lat_abs": compute_mean_lat_error(e_lat),
        "rmse_e_lat": compute_rmse_lat(e_lat),
        "max_e_lat_abs": compute_max_lat_error(e_lat),
        "mean_e_psi_abs": float(np.mean(np.abs(e_psi))),
        "rmse_e_psi": compute_heading_rmse(e_psi),
        "lane_departure_flag": int(departed),
        "lksr_episode": compute_lksr(e_lat),
        "delta_dot_rms": compute_steering_rate_rms(delta),
        "control_effort": compute_control_effort(delta),
        "settling_time_s": compute_settling_time(e_lat),
        "overshoot_pct": compute_overshoot(e_lat),
        "ttld_mean": float(np.mean(ttld[ttld < 998.0])) if np.any(ttld < 998.0) else 999.0,
        "sbvr_pct": compute_sbvr(e_lat),
        "episode_steps": len(episode_data),
        "lap_complete": int(not departed and len(episode_data) >= cfg.SIM_MAX_STEPS * 0.9),
    }


def check_convergence_warning(
    episode: int,
    reward_history: list[float],
    window: int = cfg.CONVERGENCE_WINDOW,
) -> None:
    """
    Log a warning if training appears stalled.

    Checks if the rolling mean reward has not improved over the last
    `window` episodes.
    """
    if len(reward_history) < window * 2:
        return

    recent = np.mean(reward_history[-window:])
    previous = np.mean(reward_history[-2 * window:-window])

    if recent <= previous * 1.01:
        logger.warning(
            f"Episode {episode}: Training may be stalled. "
            f"Recent avg reward: {recent:.2f}, Previous: {previous:.2f}"
        )


class Trainer:
    """
    Main training orchestrator.

    Manages the complete training pipeline from dataset preloading
    through 600-episode DDPG training with curriculum scheduling.
    """

    def __init__(
        self,
        device: str = "cpu",
        skip_ds01: bool = False,
        skip_ds02: bool = False,
        skip_ds03: bool = False,
        reload_datasets: bool = False,
        n_episodes: int = cfg.N_EPISODES,
        seed: int = cfg.SEED,
    ) -> None:
        self.device = device
        self.skip_ds01 = skip_ds01
        self.skip_ds02 = skip_ds02
        self.skip_ds03 = skip_ds03
        self.reload_datasets = reload_datasets
        self.n_episodes = n_episodes
        self.seed = seed

        # Set seeds
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

        # Create components
        self.env = LaneKeepingEnv()
        self.agent = DDPGAgent(device=device)
        self.buffer = HybridStratifiedBuffer(device=device)
        self.noise = OUNoise()
        self.training_logger = TrainingLogger()

        # Preload stats
        self.preload_stats: Optional[PreloadStats] = None
        self.pretrain_rmse: float = float("nan")

        # Sim-only mode detection
        self.sim_only_mode: bool = skip_ds01 and skip_ds02 and skip_ds03
        if self.sim_only_mode:
            logger.info("SIM-ONLY MODE: All datasets skipped — using adapted hyperparameters")
            self._warmup_steps = cfg.SIM_ONLY_WARMUP_STEPS
            self._updates_per_step = cfg.SIM_ONLY_UPDATES_PER_STEP
            self._noise_sigma_init = cfg.SIM_ONLY_NOISE_SIGMA_INIT
        else:
            self._warmup_steps = cfg.WARMUP_STEPS
            self._updates_per_step = cfg.UPDATES_PER_STEP
            self._noise_sigma_init = cfg.NOISE_SIGMA_INIT

    def run(self) -> Dict:
        """
        Execute the complete training pipeline.

        Returns:
            Dictionary with training summary statistics.
        """
        cfg.ensure_directories()
        start_time = time.time()

        # ── Phase 0: Dataset Preloading ──────────────────────────────────────
        logger.info("=" * 70)
        logger.info("PHASE 0: Dataset Preloading")
        logger.info("=" * 70)

        preloader = DatasetPreloader(
            buffer=self.buffer,
            skip_ds01=self.skip_ds01,
            skip_ds02=self.skip_ds02,
            skip_ds03=self.skip_ds03,
            reload_datasets=self.reload_datasets,
        )
        self.preload_stats = preloader.run()

        # Tyre model calibration update
        if self.preload_stats.comma_calibration_r2 > cfg.COMMA_TYRE_CALIBRATION_R2_THRESHOLD:
            self.env.update_tyre_params(
                C_af=self.preload_stats.comma_calibrated_caf,
                C_ar=self.preload_stats.comma_calibrated_car,
            )
            logger.info(
                f"Tyre model updated: C_af={self.preload_stats.comma_calibrated_caf:.0f} "
                f"C_ar={self.preload_stats.comma_calibrated_car:.0f} "
                f"R²={self.preload_stats.comma_calibration_r2:.3f}"
            )
        else:
            logger.info("Using nominal tyre parameters (calibration R² < 0.85 or no data)")

        # M-22: Pretrain-only evaluation (skip cleanly in sim-only mode)
        if self.sim_only_mode:
            logger.info("SIM-ONLY MODE: Skipping M-22 pretrain evaluation (no expert data)")
        elif self.buffer.total_size >= cfg.BATCH_SIZE:
            logger.info("Computing M-22: Pretraining performance...")
            self.pretrain_rmse = compute_pretrain_performance(
                agent=self.agent,
                buffer=self.buffer,
                env=self.env,
                pretrain_steps=cfg.PRETRAIN_GRADIENT_STEPS,
            )
            logger.info(f"M-22 Pretraining RMSE e_lat (SCN-01): {self.pretrain_rmse:.4f} m")
        else:
            logger.info("Buffer too small for M-22 pretrain evaluation — skipping")

        # ── Phase 1–3: RL Training Loop ─────────────────────────────────────
        logger.info("=" * 70)
        logger.info(f"TRAINING: {self.n_episodes} episodes")
        logger.info("=" * 70)

        total_steps = 0
        best_rolling_rmse = float("inf")
        best_checkpoint_path: Optional[Path] = None
        convergence_window = cfg.CONVERGENCE_WINDOW  # 50

        for episode in range(self.n_episodes):
            # Select scenario based on curriculum
            scenario_id = get_curriculum_profile(episode)
            phase_name = get_curriculum_phase_name(episode)

            # Reset environment
            state, _ = self.env.reset(scenario_id=scenario_id,
                                       e_lat_init=np.random.uniform(-0.3, 0.3) if episode > 10 else 0.0)
            self.noise.reset()
            self.noise.set_sigma(cfg.get_noise_sigma(episode))

            done = False
            while not done:
                # Action selection
                if total_steps < self._warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)
                    noise_val = self.noise.sample()
                    action = np.clip(action + noise_val, -1.0, 1.0)

                # Environment step
                next_state, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated

                # Push to 'sim' sub-buffer
                self.buffer.push(
                    "sim", state, action, reward, next_state, float(done)
                )

                # Agent update (with configurable UTD ratio)
                if total_steps >= self._warmup_steps:
                    for _ in range(self._updates_per_step):
                        update_info = self.agent.update(self.buffer, episode)
                        self.training_logger.log_step(update_info)

                state = next_state
                total_steps += 1

            # Episode complete — compute metrics
            ep_metrics = compute_episode_metrics(self.env.episode_data)

            # Log episode
            self.training_logger.log_episode(
                episode=episode,
                scenario_id=scenario_id,
                ep_metrics=ep_metrics,
                buffer_sizes=self.buffer.sizes,
                noise_sigma=self.noise.current_sigma,
            )

            # Progress logging
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Ep {episode + 1:4d}/{self.n_episodes} | "
                    f"Phase: {phase_name} | SCN: {scenario_id} | "
                    f"Reward: {ep_metrics['total_reward']:8.2f} | "
                    f"RMSE_lat: {ep_metrics['rmse_e_lat']:.4f} m | "
                    f"LKSR: {ep_metrics['lksr_episode']:.3f} | "
                    f"Steps: {ep_metrics['episode_steps']} | "
                    f"sig: {self.noise.current_sigma:.3f}"
                )

            # Periodic checkpoint
            if (episode + 1) % cfg.CHECKPOINT_INTERVAL == 0:
                ckpt_path = self.agent.save_checkpoint(episode + 1)
                logger.info(f"Checkpoint saved: {ckpt_path}")

            # ── Best-checkpoint tracking ──────────────────────────────────
            if len(self.training_logger.rmse_history) >= convergence_window:
                rolling_rmse = np.mean(
                    self.training_logger.rmse_history[-convergence_window:]
                )
                if rolling_rmse < best_rolling_rmse:
                    best_rolling_rmse = rolling_rmse
                    best_checkpoint_path = self.agent.save_checkpoint(
                        episode + 1, suffix="_best"
                    )
                    logger.info(
                        f"New best checkpoint at ep {episode + 1}: "
                        f"rolling RMSE = {best_rolling_rmse:.4f} m "
                        f"-> {best_checkpoint_path}"
                    )

            # ── Learning rate decay (prevent late-training divergence) ────
            # Decay LR by 0.5x at 75% and 90% of training
            decay_points = [int(self.n_episodes * 0.75), int(self.n_episodes * 0.90)]
            if (episode + 1) in decay_points:
                for param_group in self.agent.actor_optim.param_groups:
                    param_group["lr"] *= 0.5
                for param_group in self.agent.critic_optim.param_groups:
                    param_group["lr"] *= 0.5
                logger.info(
                    f"Learning rate decayed at ep {episode + 1}: "
                    f"actor_lr={self.agent.actor_optim.param_groups[0]['lr']:.2e}, "
                    f"critic_lr={self.agent.critic_optim.param_groups[0]['lr']:.2e}"
                )

            # Convergence warning
            check_convergence_warning(
                episode, self.training_logger.reward_history
            )

        # Save final checkpoint
        final_ckpt = self.agent.save_checkpoint(self.n_episodes)
        logger.info(f"Final checkpoint: {final_ckpt}")

        if best_checkpoint_path:
            logger.info(f"Best checkpoint: {best_checkpoint_path} (RMSE={best_rolling_rmse:.4f} m)")
        else:
            best_checkpoint_path = final_ckpt

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        return {
            "total_episodes": self.n_episodes,
            "total_steps": total_steps,
            "wall_time_s": elapsed,
            "final_rmse_lat": self.training_logger.rmse_history[-1] if self.training_logger.rmse_history else float("nan"),
            "final_lksr": self.training_logger.lksr_history[-1] if self.training_logger.lksr_history else 0.0,
            "pretrain_rmse_m22": self.pretrain_rmse,
            "best_checkpoint": str(best_checkpoint_path),
            "best_rolling_rmse": best_rolling_rmse,
        }

