"""
reward.py — Five-Component Reward Function for Lane Keeping Control

Each component is a standalone, named function with a citation in its docstring.
The same functions are used to compute reward on both simulated and real-world
transitions (dataset adapters), ensuring consistent reward signal across the
fused replay buffer.

Components:
    1. reward_lateral   — Lane keeping accuracy (dominant)
    2. reward_heading   — Heading alignment
    3. reward_smoothness — Steering smoothness (ISO 15622 §9.2 comfort)
    4. reward_progress  — Forward progress incentive
    5. reward_terminal  — Lane departure penalty

Weights (from config.py):
    w_lateral  = 5.0
    w_heading  = 2.0
    w_smooth   = 1.0
    w_progress = 0.3

Total reward: r = w_lat·r_lat + w_head·r_head + w_smooth·r_smooth + w_prog·r_prog
              + r_terminal (if applicable)
"""

from __future__ import annotations

import numpy as np

import config as cfg


def reward_lateral(e_lat: float) -> float:
    """
    Reward component for lateral displacement accuracy.

    Penalises lateral deviation from lane centre using a Gaussian-like
    exponential decay. This is the dominant reward component (w=2.5)
    since lane keeping is the primary control objective.

    Reference: ISO 15622:2018 §6 — lateral displacement metric.

    Formulation:
        r_lat = exp(-5.0 · (e_lat / (LANE_WIDTH/2))²)
        Range: [0, 1]. Maximum at e_lat = 0.

    The divisor LANE_WIDTH/2 normalises the error so that the reward
    drops to ~exp(-5) ≈ 0.007 at the lane boundary (full departure).

    Args:
        e_lat: Lateral deviation from lane centre (m), signed.

    Returns:
        Scalar reward in [0, 1].
    """
    normalised = e_lat / cfg.LANE_WIDTH_HALF
    return float(np.exp(-5.0 * normalised ** 2))


def reward_heading(e_psi: float) -> float:
    """
    Reward component for heading alignment.

    Penalises heading error relative to road tangent. Heading alignment
    is critical for preventing overshoot during curve entry/exit.

    Reference: ISO 15622:2018 §6 — heading error metric.

    Formulation:
        r_head = exp(-3.0 · (e_psi / (π/4))²)
        Range: [0, 1]. Maximum at e_psi = 0.

    The normalisation by π/4 (45°) means reward drops to ~exp(-3) ≈ 0.05
    at 45° heading error, which is an extreme misalignment.

    Args:
        e_psi: Heading error (rad), signed, wrapped to [-π, π].

    Returns:
        Scalar reward in [0, 1].
    """
    normalised = e_psi / cfg.NORM_E_PSI
    return float(np.exp(-3.0 * normalised ** 2))


def reward_smoothness(delta_current: float, delta_previous: float) -> float:
    """
    Reward component for steering smoothness (comfort).

    Penalises large steering rate (change in steering angle per timestep).
    This enforces the ISO 15622:2018 §9.2 comfort requirement and
    promotes smooth, human-like steering behaviour.

    Reference: ISO 15622:2018 §9.2 — comfort assessment criterion.
               IEEE 2846-2022 — steering rate RMS target < 0.20 rad/s.

    Formulation:
        Δδ = (δ_current − δ_previous) / dt
        r_smooth = exp(-0.5 · (Δδ / 0.2)²)
        Range: [0, 1].

    The reference value 0.2 rad/s is the IEEE 2846-2022 target for steering
    rate RMS, used here as the Gaussian width.

    Args:
        delta_current:  Current steering angle (rad).
        delta_previous: Previous steering angle (rad).

    Returns:
        Scalar reward in [0, 1].
    """
    delta_dot = (delta_current - delta_previous) / cfg.SIM_DT
    normalised = delta_dot / cfg.IEEE2846_STEER_RATE_RMS_LIMIT
    return float(np.exp(-0.5 * normalised ** 2))


def reward_progress(v_x: float, e_psi: float) -> float:
    """
    Reward component for forward progress.

    Provides a small positive reward proportional to the longitudinal
    component of velocity projected along the road tangent. This prevents
    the degenerate zero-speed policy that minimises all other penalties.

    Formulation:
        r_prog = (v_x · cos(e_psi)) / V_REFERENCE
        Clamped to [0, 1].

    Args:
        v_x:   Longitudinal velocity (m/s).
        e_psi: Heading error (rad).

    Returns:
        Scalar reward in [0, 1].
    """
    progress = (v_x * np.cos(e_psi)) / cfg.V_REFERENCE
    return float(np.clip(progress, 0.0, 1.0))


def reward_terminal(e_lat: float) -> float:
    """
    Terminal penalty for lane departure.

    Applied when |e_lat| exceeds the lane boundary (LANE_WIDTH/2 = 1.75 m).
    The episode ends with a large negative reward to strongly discourage
    lane departure.

    Reference: ISO 15622:2018 §7.1 — lane departure definition.

    Args:
        e_lat: Lateral deviation from lane centre (m), signed.

    Returns:
        REWARD_TERMINAL_PENALTY (-10.0) if departed, else 0.0.
    """
    if abs(e_lat) >= cfg.DEPARTURE_THRESHOLD:
        return cfg.REWARD_TERMINAL_PENALTY
    return 0.0


def reward_boundary_proximity(e_lat: float) -> float:
    """
    Continuous penalty that increases as the vehicle approaches the lane boundary.

    Activates when |e_lat| > 50% of half-lane-width (i.e. beyond 0.875 m).
    Provides gradient signal BEFORE departure, encouraging the agent to learn
    recovery maneuvers rather than only learning "don't depart."

    Reference: Code review §3.5 — terminal-proximity reward shaping.

    Formulation:
        margin = LANE_WIDTH_HALF - |e_lat|
        If margin > 50% of LANE_WIDTH_HALF: 0.0 (not in danger zone)
        Else: -(1 - margin / (0.5 * LANE_WIDTH_HALF))^2

    Args:
        e_lat: Lateral deviation from lane centre (m), signed.

    Returns:
        Scalar penalty in [-1, 0].
    """
    margin = cfg.LANE_WIDTH_HALF - abs(e_lat)
    if margin > cfg.LANE_WIDTH_HALF * 0.5:  # Not in danger zone
        return 0.0
    # Quadratic penalty approaching boundary
    danger_frac = 1.0 - (margin / (cfg.LANE_WIDTH_HALF * 0.5))
    return -danger_frac ** 2  # Max penalty: -1.0 at boundary


def compute_reward(
    e_lat: float,
    e_psi: float,
    delta_current: float,
    delta_previous: float,
    v_x: float,
    terminated: bool,
) -> tuple[float, dict]:
    """
    Compute the total composite reward from all six components.

    This function is used by:
    1. The simulator environment (lane_keeping_env.py) for RL rollouts.
    2. Dataset adapters (openlka_adapter.py, etc.) for real-world transitions.
    Using the same function ensures consistent reward signal across the
    fused replay buffer.

    Total: r = w_lat·r_lat + w_head·r_head + w_smooth·r_smooth + w_prog·r_prog
              + w_boundary·r_boundary + r_terminal

    Args:
        e_lat:          Lateral error (m).
        e_psi:          Heading error (rad).
        delta_current:  Current steering angle (rad).
        delta_previous: Previous steering angle (rad).
        v_x:            Longitudinal velocity (m/s).
        terminated:     Whether the episode has terminated due to departure.

    Returns:
        Tuple of (total_reward, component_dict) where component_dict contains
        individual reward values for logging.
    """
    r_lat = reward_lateral(e_lat)
    r_head = reward_heading(e_psi)
    r_smooth = reward_smoothness(delta_current, delta_previous)
    r_prog = reward_progress(v_x, e_psi)
    r_boundary = reward_boundary_proximity(e_lat)
    r_term = reward_terminal(e_lat) if terminated else 0.0

    total = (
        cfg.REWARD_W_LATERAL * r_lat
        + cfg.REWARD_W_HEADING * r_head
        + cfg.REWARD_W_SMOOTH * r_smooth
        + cfg.REWARD_W_PROGRESS * r_prog
        + cfg.REWARD_W_BOUNDARY * r_boundary
        + r_term
    )

    components = {
        "r_lateral": r_lat,
        "r_heading": r_head,
        "r_smoothness": r_smooth,
        "r_progress": r_prog,
        "r_boundary": r_boundary,
        "r_terminal": r_term,
        "r_total": total,
    }

    return total, components
