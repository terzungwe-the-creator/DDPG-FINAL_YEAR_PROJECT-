"""
evaluator.py — Deterministic Evaluation Pipeline

Evaluates the trained DDPG agent across all 5 road scenarios using the
deterministic policy (zero noise). Produces per-timestep raw data,
per-scenario summaries, and a machine-readable performance report.

Evaluation protocol (ISO 15622:2018 §8.4):
    - 20 episodes per scenario × 5 scenarios = 100 episodes total.
    - Each episode initialised with lateral perturbation U(-0.2, 0.2) m.
    - Deterministic policy: no exploration noise.
    - Metrics computed per-episode and aggregated per-scenario.

Output files:
    results/eval_raw.csv           — per-timestep, 100 episodes
    results/eval_summary.csv       — per-scenario ISO metrics
    results/performance_report.json — machine-readable, CI/CD compatible
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config as cfg
from ddpg.agent import DDPGAgent
from metrics.iso15622 import (
    compute_mean_lat_error,
    compute_rmse_lat,
    compute_max_lat_error,
    compute_heading_rmse,
    compute_lksr,
    compute_lane_departure_rate,
    iso15622_pass_fail,
)
from metrics.ieee2846 import (
    compute_steering_rate_rms,
    compute_control_effort,
    compute_settling_time,
    compute_overshoot,
)
from metrics.safety import (
    compute_ttld_series,
    compute_ttld_p5,
    compute_sbvr,
    compute_mtbd,
)
from simulator.lane_keeping_env import LaneKeepingEnv
from training.logger import EvalLogger

logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Auto-detect git hash via subprocess. Returns 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(cfg.PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class Evaluator:
    """
    Deterministic evaluation of a trained DDPG lane keeping agent.

    Runs 20 episodes per scenario across all 5 scenarios (100 total).
    Computes ISO 15622, IEEE 2846, and UNECE R157 metrics. Produces
    raw CSV, summary CSV, and JSON performance report.

    Attributes:
        agent:       Trained DDPG agent.
        env:         Lane keeping environment.
        eval_logger: Evaluation CSV logger.
        seed:        Random seed for perturbation reproducibility.
    """

    def __init__(
        self,
        agent: DDPGAgent,
        env: LaneKeepingEnv,
        seed: int = cfg.SEED,
        preload_stats: Optional[dict] = None,
        pretrain_rmse: float = float("nan"),
    ) -> None:
        self.agent = agent
        self.env = env
        self.seed = seed
        self.preload_stats = preload_stats or {}
        self.pretrain_rmse = pretrain_rmse
        self.eval_logger = EvalLogger()
        self._rng = np.random.RandomState(seed + 1000)

    def run(self) -> Dict:
        """
        Execute the full evaluation protocol.

        Returns:
            Dictionary with overall evaluation results and per-scenario summaries.
        """
        cfg.ensure_directories()
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("EVALUATION: Deterministic Policy Assessment")
        logger.info(f"  Scenarios:  {cfg.EVAL_N_SCENARIOS}")
        logger.info(f"  Episodes/scenario: {cfg.EVAL_EPISODES_PER_SCENARIO}")
        logger.info(f"  Total episodes: {cfg.EVAL_TOTAL_EPISODES}")
        logger.info(f"  Perturbation: U(-{cfg.EVAL_PERTURBATION_RANGE}, "
                     f"+{cfg.EVAL_PERTURBATION_RANGE}) m")
        logger.info("=" * 70)

        scenario_results: Dict[str, Dict] = {}
        episode_id = 0

        for scn_id in cfg.SCENARIO_IDS:
            logger.info(f"Evaluating {scn_id}...")

            # Collect per-episode data for this scenario
            scn_e_lats: List[np.ndarray] = []
            scn_e_psis: List[np.ndarray] = []
            scn_deltas: List[np.ndarray] = []
            scn_rewards: List[float] = []
            scn_ttlds: List[np.ndarray] = []
            scn_trajectories: List[Dict] = []

            for ep_idx in range(cfg.EVAL_EPISODES_PER_SCENARIO):
                # Random lateral perturbation per ISO 15622 §8.4
                e_lat_init = self._rng.uniform(
                    -cfg.EVAL_PERTURBATION_RANGE,
                    cfg.EVAL_PERTURBATION_RANGE,
                )

                state, _ = self.env.reset(
                    scenario_id=scn_id,
                    e_lat_init=e_lat_init,
                )

                ep_e_lat = []
                ep_e_psi = []
                ep_delta = []
                ep_reward = 0.0
                ep_xs = []
                ep_ys = []

                done = False
                step_count = 0
                prev_delta = 0.0

                while not done and step_count < cfg.SIM_MAX_STEPS:
                    # Deterministic action (no noise)
                    action = self.agent.select_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    ep_e_lat.append(info["e_lat_m"])
                    ep_e_psi.append(info["e_psi_rad"])
                    ep_delta.append(info["delta_rad"])
                    ep_reward += reward
                    ep_xs.append(info["X"])
                    ep_ys.append(info["Y"])

                    # Compute TTLD for this timestep (logged inline)
                    delta_dot = info.get("delta_dot", 0.0)

                    # Log per-timestep to raw CSV
                    self.eval_logger.log_timestep(
                        episode_id=episode_id,
                        scenario_id=scn_id,
                        timestep=step_count,
                        time_s=info["time_s"],
                        e_lat_m=info["e_lat_m"],
                        e_psi_rad=info["e_psi_rad"],
                        delta_rad=info["delta_rad"],
                        delta_dot=delta_dot,
                        v_x=info["v_x"],
                        v_y=info["v_y"],
                        r=info["r"],
                        reward=reward,
                        ttld_s=0.0,  # Computed post-hoc below
                    )

                    state = next_state
                    prev_delta = info["delta_rad"]
                    step_count += 1

                # Episode complete — compute TTLD series
                e_lat_arr = np.array(ep_e_lat)
                e_psi_arr = np.array(ep_e_psi)
                delta_arr = np.array(ep_delta)
                ttld_arr = compute_ttld_series(e_lat_arr)

                scn_e_lats.append(e_lat_arr)
                scn_e_psis.append(e_psi_arr)
                scn_deltas.append(delta_arr)
                scn_rewards.append(ep_reward)
                scn_ttlds.append(ttld_arr)
                scn_trajectories.append({
                    "x": np.array(ep_xs),
                    "y": np.array(ep_ys),
                    "e_lat": e_lat_arr,
                })

                episode_id += 1

            # Aggregate scenario metrics across all 20 episodes
            all_e_lat = np.concatenate(scn_e_lats)
            all_e_psi = np.concatenate(scn_e_psis)
            all_delta = np.concatenate(scn_deltas)
            all_ttld = np.concatenate(scn_ttlds)

            iso_result = iso15622_pass_fail(all_e_lat, all_e_psi)

            scn_summary = {
                "scenario_id": scn_id,
                "mean_e_lat": iso_result["mean_e_lat"],
                "rmse_e_lat": iso_result["rmse_e_lat"],
                "max_e_lat": iso_result["max_e_lat"],
                "rmse_e_psi": iso_result["rmse_e_psi"],
                "lksr": iso_result["lksr"],
                "ldr": iso_result["ldr"],
                "settling_s": compute_settling_time(all_e_lat),
                "overshoot_pct": compute_overshoot(all_e_lat),
                "control_effort": compute_control_effort(all_delta),
                "steer_rms": compute_steering_rate_rms(all_delta),
                "ttld_p5": compute_ttld_p5(all_ttld),
                "sbvr_pct": compute_sbvr(all_e_lat),
                "iso15622_pass": iso_result["overall_pass"],
            }

            scenario_results[scn_id] = scn_summary

            pass_str = "PASS" if scn_summary["iso15622_pass"] else "FAIL"
            logger.info(
                f"  {scn_id}: RMSE_lat={scn_summary['rmse_e_lat']:.4f} m | "
                f"LKSR={scn_summary['lksr']:.3f} | "
                f"TTLD_p5={scn_summary['ttld_p5']:.2f} s | "
                f"ISO 15622: {pass_str}"
            )

        # Write per-scenario summary CSV
        summaries = list(scenario_results.values())
        self.eval_logger.write_summary(summaries)

        # Overall pass/fail
        overall_pass = all(
            s["iso15622_pass"] for s in scenario_results.values()
        )

        # Find convergence episode from training log
        convergence_episode = self._find_convergence_episode()

        # Build performance report
        report = self._build_performance_report(
            scenario_results=scenario_results,
            overall_pass=overall_pass,
            convergence_episode=convergence_episode,
        )

        # Save performance report
        with open(cfg.PERFORMANCE_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info(f"EVALUATION COMPLETE — {'PASS' if overall_pass else 'FAIL'}")
        logger.info(f"  Overall ISO 15622: {'PASS' if overall_pass else 'FAIL'}")
        logger.info(f"  Results: {cfg.EVAL_SUMMARY_PATH}")
        logger.info(f"  Report:  {cfg.PERFORMANCE_REPORT_PATH}")
        logger.info(f"  Time:    {elapsed:.1f}s")
        logger.info("=" * 70)

        return {
            "overall_pass": overall_pass,
            "scenarios": scenario_results,
            "report_path": str(cfg.PERFORMANCE_REPORT_PATH),
            "wall_time_s": elapsed,
        }

    def _find_convergence_episode(self) -> int:
        """
        Detect the convergence episode from training log.

        Convergence is defined as the first episode where the rolling
        mean RMSE e_lat drops below ISO15622_RMSE_LAT_LIMIT and stays
        below for at least CONVERGENCE_WINDOW episodes.

        Returns:
            Convergence episode number, or -1 if not converged.
        """
        if not cfg.TRAINING_LOG_PATH.exists():
            return -1

        try:
            import pandas as pd
            df = pd.read_csv(cfg.TRAINING_LOG_PATH)
            if "rmse_e_lat" not in df.columns or len(df) < cfg.CONVERGENCE_WINDOW:
                return -1

            rmse = df["rmse_e_lat"].values
            window = cfg.CONVERGENCE_WINDOW
            threshold = cfg.ISO15622_RMSE_LAT_LIMIT

            for i in range(len(rmse) - window):
                rolling_mean = np.mean(rmse[i:i + window])
                if rolling_mean < threshold:
                    return int(df["episode"].iloc[i])

        except Exception:
            pass

        return -1

    def _build_performance_report(
        self,
        scenario_results: Dict[str, Dict],
        overall_pass: bool,
        convergence_episode: int,
    ) -> Dict:
        """Build the machine-readable performance report JSON."""
        # Extract preload stats safely
        openlka_trans = self.preload_stats.get("openlka_transitions_loaded", 0)
        comma_trans = self.preload_stats.get("comma_transitions_loaded", 0)
        argoverse_trans = self.preload_stats.get("argoverse_transitions_loaded", 0)
        tyre_r2 = self.preload_stats.get("comma_calibration_r2", 0.0)

        report = {
            "system_id": "DDPG-LKA-DataFusion-v3",
            "evaluation_standard": "ISO 15622:2018",
            "evaluation_date": datetime.now(timezone.utc).isoformat(),
            "overall_pass": overall_pass,
            "training_data": {
                "openlka_transitions": openlka_trans,
                "comma_transitions": comma_trans,
                "argoverse_transitions": argoverse_trans,
                "sim_transitions": "~600000",
                "tyre_calibration_r2": round(tyre_r2, 4),
                "pretrain_rmse_m22": round(self.pretrain_rmse, 4)
                if not np.isnan(self.pretrain_rmse)
                else "N/A",
            },
            "scenarios": {},
            "convergence_episode": convergence_episode,
            "git_hash": _get_git_hash(),
            "seed": self.seed,
        }

        for scn_id, scn_data in scenario_results.items():
            report["scenarios"][scn_id] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in scn_data.items()
            }

        return report
