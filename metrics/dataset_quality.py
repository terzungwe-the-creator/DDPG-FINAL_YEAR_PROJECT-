"""
dataset_quality.py — Dataset Quality Metrics for Hybrid Training Validation

Implements metrics M-19 through M-22 from the agreed metrics framework.
These metrics are unique to the hybrid training approach and validate
the quality and contribution of external dataset integration.

Metrics:
    M-19: Real-to-Sim Transition Ratio  RSR       — n_real / n_sim in buffer
    M-20: Tyre Model Calibration R²     R²_tyre   — goodness of fit
    M-21: Expert Data Coverage           EDC       — fraction of obs-space cells visited
    M-22: Pretraining Performance        PTP       — RMSE e_lat after preload-only training
"""

from __future__ import annotations

from typing import Dict

import numpy as np

import config as cfg
from ddpg.hybrid_buffer import HybridStratifiedBuffer


def compute_real_sim_ratio(buffer: HybridStratifiedBuffer) -> float:
    """
    M-19: Real-to-Sim Transition Ratio.

    Computes the ratio of real-world transitions to simulated transitions
    currently in the buffer. This ratio should decrease over training
    as simulation data accumulates.

    RSR = n_real / max(n_sim, 1)

    Where n_real = n_openlka + n_comma + n_argoverse.

    Args:
        buffer: Hybrid stratified replay buffer.

    Returns:
        RSR ratio (dimensionless).
    """
    sizes = buffer.sizes
    n_real = sizes.get("openlka", 0) + sizes.get("comma", 0) + sizes.get("argoverse", 0)
    n_sim = max(sizes.get("sim", 0), 1)
    return float(n_real / n_sim)


def compute_tyre_calibration_r2(calibration_result: dict) -> float:
    """
    M-20: Tyre Model Calibration R².

    Reports the coefficient of determination from DS-02 tyre model fitting.
    R² > 0.85 indicates calibration is reliable enough to override nominal
    tyre parameters.

    Args:
        calibration_result: Dictionary from CommaSteeringAdapter.calibration_result.
                           Must contain key 'r_squared'.

    Returns:
        R² value in [0, 1]. Returns 0.0 if calibration was not performed.
    """
    if calibration_result is None:
        return 0.0
    return float(calibration_result.get("r_squared", 0.0))


def compute_obs_space_coverage(
    buffer: HybridStratifiedBuffer,
    source: str = "openlka",
    n_bins: int = 10,
) -> float:
    """
    M-21: Expert Data Coverage.

    Computes the fraction of observation space cells visited by expert
    data (DS-01). The observation space is discretised into a grid with
    n_bins per dimension. Coverage = (cells visited) / (total cells).

    Since the full 8D grid would have n_bins^8 cells (too sparse), we
    compute coverage on 2D projections of the most important pairs:
        (e_lat, e_psi), (kappa, yaw_rate), (delta_prev, kappa_la1)
    and report the average.

    Args:
        buffer: Hybrid stratified replay buffer.
        source: Data source to measure coverage for.
        n_bins: Number of bins per dimension.

    Returns:
        Coverage fraction in [0, 1].
    """
    sub_buf = buffer.sub_buffers.get(source)
    if sub_buf is None or sub_buf.size == 0:
        return 0.0

    states = sub_buf.states[:sub_buf.size]

    # 2D projection pairs (dimension indices)
    projection_pairs = [
        (0, 1),  # e_lat vs e_psi
        (2, 4),  # kappa vs yaw_rate
        (5, 6),  # delta_prev vs kappa_la1
    ]

    coverages = []
    for dim_a, dim_b in projection_pairs:
        # Values are already normalised to [-1, 1]
        a_vals = states[:, dim_a]
        b_vals = states[:, dim_b]

        # Bin edges in [-1, 1]
        edges = np.linspace(-1.0, 1.0, n_bins + 1)

        # Compute 2D histogram
        hist, _, _ = np.histogram2d(a_vals, b_vals, bins=[edges, edges])

        total_cells = n_bins * n_bins
        occupied_cells = int(np.sum(hist > 0))
        coverages.append(occupied_cells / total_cells)

    return float(np.mean(coverages))


def compute_pretrain_performance(
    agent,
    buffer: HybridStratifiedBuffer,
    env,
    pretrain_steps: int = cfg.PRETRAIN_GRADIENT_STEPS,
) -> float:
    """
    M-22: Pretraining Performance.

    After preloading all three datasets into the buffer, runs a single
    gradient update pass of `pretrain_steps` steps on the actor/critic
    using only real-world data. Then evaluates the resulting policy on
    SCN-01. Returns the RMSE e_lat.

    This quantifies how much the real-world data alone teaches the agent
    before any simulation rollout begins.

    Args:
        agent:          DDPG agent (will be modified in-place).
        buffer:         Hybrid stratified replay buffer (must contain data).
        env:            Lane keeping environment.
        pretrain_steps: Number of gradient steps. Default: 1000.

    Returns:
        RMSE e_lat (m) on SCN-01 after pretraining. Returns NaN if
        buffer is empty or agent fails.
    """
    import torch

    if buffer.total_size < cfg.BATCH_SIZE:
        return float("nan")

    # Pretrain: run gradient updates using buffer data
    # Use episode=0 to sample with Phase 1 weights (expert-heavy)
    for step in range(pretrain_steps):
        try:
            agent.update(buffer, episode=0)
        except RuntimeError:
            break

    # Evaluate on SCN-01
    try:
        state, _ = env.reset(scenario_id="SCN-01", e_lat_init=0.0)
        e_lats = []

        done = False
        step_count = 0
        while not done and step_count < cfg.SIM_MAX_STEPS:
            action = agent.select_action(state)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            e_lats.append(info["e_lat_m"])
            step_count += 1

        if len(e_lats) > 0:
            e_lat_arr = np.array(e_lats)
            return float(np.sqrt(np.mean(e_lat_arr ** 2)))
    except Exception:
        pass

    return float("nan")


def compute_all_dataset_metrics(
    buffer: HybridStratifiedBuffer,
    calibration_result: dict | None = None,
) -> Dict[str, float]:
    """
    Compute all dataset quality metrics.

    Args:
        buffer:             Hybrid stratified replay buffer.
        calibration_result: Tyre calibration result dict (optional).

    Returns:
        Dictionary with M-19, M-20, M-21 values.
    """
    return {
        "real_sim_ratio_m19": compute_real_sim_ratio(buffer),
        "tyre_calibration_r2_m20": compute_tyre_calibration_r2(calibration_result),
        "expert_coverage_m21": compute_obs_space_coverage(buffer, source="openlka"),
    }
