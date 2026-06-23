"""
iso15622.py — ISO 15622:2018 Lane Keeping Assistance Metrics

Implements metrics M-01 through M-06 from the agreed metrics framework.
Each function is standalone, unit-testable, and cites its source clause.

Standard: ISO 15622:2018 — Intelligent transport systems —
          Adaptive cruise control systems — Performance requirements and
          test procedures (extended to LKA per Annex D).

Metrics:
    M-01: Mean Lateral Displacement Error  |e_lat|       — §6
    M-02: RMS Lateral Error                RMSE_lat      — §6, Eq.1
    M-03: Maximum Lateral Deviation        e_lat_max     — §6
    M-04: Heading Error RMS                RMSE_e_psi    — §6
    M-05: Lane Departure Rate              LDR           — §7.2
    M-06: Lane Keeping Success Rate        LKSR          — §7.1
"""

from __future__ import annotations

from typing import Dict

import numpy as np

import config as cfg


def compute_mean_lat_error(e_lat: np.ndarray) -> float:
    """
    M-01: Mean absolute lateral displacement error.

    ISO 15622:2018 §6: "The mean absolute lateral displacement of the
    vehicle from the lane centre shall be computed over the evaluation
    interval."

    Args:
        e_lat: Array of lateral errors (m), shape (N,).

    Returns:
        Mean |e_lat| in metres.
    """
    return float(np.mean(np.abs(e_lat)))


def compute_rmse_lat(e_lat: np.ndarray) -> float:
    """
    M-02: Root-mean-square lateral error.

    ISO 15622:2018 §6, Eq. 1:
        RMSE_lat = sqrt(1/N · Σ e_lat_i²)

    Args:
        e_lat: Array of lateral errors (m), shape (N,).

    Returns:
        RMSE of e_lat in metres.
    """
    return float(np.sqrt(np.mean(e_lat ** 2)))


def compute_max_lat_error(e_lat: np.ndarray) -> float:
    """
    M-03: Maximum absolute lateral deviation.

    ISO 15622:2018 §6: "The maximum absolute lateral deviation observed
    during the evaluation interval."

    Args:
        e_lat: Array of lateral errors (m), shape (N,).

    Returns:
        max(|e_lat|) in metres.
    """
    return float(np.max(np.abs(e_lat)))


def compute_heading_rmse(e_psi: np.ndarray) -> float:
    """
    M-04: Heading error RMS.

    ISO 15622:2018 §6: "RMS heading angle error between the vehicle
    heading and the lane tangent direction."

    Args:
        e_psi: Array of heading errors (rad), shape (N,).

    Returns:
        RMSE of e_psi in radians.
    """
    return float(np.sqrt(np.mean(e_psi ** 2)))


def compute_lksr(e_lat: np.ndarray, threshold: float = cfg.ISO15622_DEPARTURE_THR) -> float:
    """
    M-06: Lane Keeping Success Rate.

    ISO 15622:2018 §7.1: "The fraction of evaluation time during which
    the vehicle remains within the departure boundary."

    LKSR = (number of timesteps where |e_lat| < threshold) / N

    Args:
        e_lat:     Array of lateral errors (m), shape (N,).
        threshold: Departure boundary (m). Default: 0.75 m per ISO 15622.

    Returns:
        LKSR as a fraction in [0, 1].
    """
    n = len(e_lat)
    if n == 0:
        return 0.0
    n_within = int(np.sum(np.abs(e_lat) < threshold))
    return float(n_within / n)


def compute_lane_departure_rate(
    e_lat: np.ndarray,
    dt: float = cfg.SIM_DT,
    threshold: float = cfg.ISO15622_DEPARTURE_THR,
) -> float:
    """
    M-05: Lane Departure Rate.

    ISO 15622:2018 §7.2: "The rate of lane departures per unit time."

    A departure event is detected when |e_lat| crosses the threshold
    from below (rising edge detection).

    Args:
        e_lat:     Array of lateral errors (m), shape (N,).
        dt:        Timestep (s).
        threshold: Departure boundary (m).

    Returns:
        LDR as percentage of timesteps with departure events.
    """
    n = len(e_lat)
    if n < 2:
        return 0.0

    abs_elat = np.abs(e_lat)
    # Detect rising edge crossings of threshold
    departures = 0
    for i in range(1, n):
        if abs_elat[i] >= threshold and abs_elat[i - 1] < threshold:
            departures += 1

    total_time = n * dt
    if total_time <= 0:
        return 0.0

    # Return as events per second, expressed as percentage
    return float(departures / total_time * 100.0)


def iso15622_pass_fail(
    e_lat: np.ndarray,
    e_psi: np.ndarray,
) -> Dict[str, any]:
    """
    ISO 15622:2018 composite pass/fail assessment.

    Thresholds (from config.py, sourced from §6.3 and §7.1):
        - Mean |e_lat| < 0.30 m
        - RMSE e_lat  < 0.40 m
        - RMSE e_psi  < 0.087 rad (5°)
        - LKSR        > 0.95 (95%)

    Args:
        e_lat: Array of lateral errors (m).
        e_psi: Array of heading errors (rad).

    Returns:
        Dictionary with individual metric values, pass/fail flags, and
        overall pass/fail status.
    """
    mean_lat = compute_mean_lat_error(e_lat)
    rmse_lat = compute_rmse_lat(e_lat)
    max_lat = compute_max_lat_error(e_lat)
    rmse_psi = compute_heading_rmse(e_psi)
    lksr = compute_lksr(e_lat)
    ldr = compute_lane_departure_rate(e_lat)

    result = {
        "mean_e_lat": round(mean_lat, 4),
        "rmse_e_lat": round(rmse_lat, 4),
        "max_e_lat": round(max_lat, 4),
        "rmse_e_psi": round(rmse_psi, 4),
        "lksr": round(lksr, 4),
        "ldr": round(ldr, 4),
        "pass_mean_lat": mean_lat < cfg.ISO15622_LAT_ERROR_LIMIT,
        "pass_rmse_lat": rmse_lat < cfg.ISO15622_RMSE_LAT_LIMIT,
        "pass_heading": rmse_psi < cfg.ISO15622_HEADING_LIMIT,
        "pass_lksr": lksr >= cfg.ISO15622_MIN_LKSR,
    }

    result["overall_pass"] = all([
        result["pass_mean_lat"],
        result["pass_rmse_lat"],
        result["pass_heading"],
        result["pass_lksr"],
    ])

    return result
