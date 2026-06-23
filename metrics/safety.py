"""
safety.py — UNECE WP.29 R157 Safety Margin Metrics

Implements metrics M-11 through M-13 from the agreed metrics framework.

Standard: UNECE WP.29 R157 — Automated Lane Keeping Systems (ALKS)
          UN Regulation No. 157 — Uniform provisions concerning the
          approval of vehicles with regard to ALKS.

Metrics:
    M-11: Time-to-Lane-Departure  TTLD  — (dep_thr − |e_lat|) / max(ė_lat, ε)
    M-12: Safety Boundary Violation Rate  SBVR — % timesteps |e_lat| > dep_thr
    M-13: Mean Time Between Departures    MTBD — mean inter-departure interval
"""

from __future__ import annotations

import numpy as np

import config as cfg


def compute_ttld_series(
    e_lat: np.ndarray,
    dt: float = cfg.SIM_DT,
    departure_threshold: float = cfg.ISO15622_DEPARTURE_THR,
    epsilon: float = cfg.UNECE_R157_TTLD_EPSILON,
) -> np.ndarray:
    """
    M-11: Time-to-Lane-Departure time series.

    UNECE WP.29 R157: "The time remaining before the vehicle would depart
    the lane if the current lateral velocity were maintained."

    TTLD_i = (dep_thr − |e_lat_i|) / max(|ė_lat_i|, ε)

    Where:
        ė_lat_i ≈ (e_lat_{i+1} − e_lat_i) / dt (finite difference estimate)
        ε = 1e-6 (prevents division by zero when lateral velocity ≈ 0)

    TTLD is only meaningful when the vehicle is approaching the lane boundary
    (ė_lat pointing outward). When moving away from boundary, TTLD is set
    to a large sentinel value (999.0 s).

    Args:
        e_lat:              Array of lateral errors (m), shape (N,).
        dt:                 Timestep (s).
        departure_threshold: Lane departure boundary (m).
        epsilon:            Division-by-zero guard (s).

    Returns:
        TTLD array (s), shape (N,). Large values indicate safe state.
    """
    n = len(e_lat)
    if n < 2:
        return np.full(n, 999.0)

    ttld = np.full(n, 999.0, dtype=np.float64)

    for i in range(n - 1):
        abs_elat = abs(e_lat[i])
        margin = departure_threshold - abs_elat

        if margin <= 0:
            # Already departed
            ttld[i] = 0.0
            continue

        # Lateral velocity (rate of change of e_lat)
        e_lat_dot = (e_lat[i + 1] - e_lat[i]) / dt

        # Only compute TTLD when moving towards boundary
        # If e_lat > 0 and e_lat_dot > 0 → moving towards right boundary
        # If e_lat < 0 and e_lat_dot < 0 → moving towards left boundary
        approaching = (e_lat[i] >= 0 and e_lat_dot > 0) or \
                      (e_lat[i] < 0 and e_lat_dot < 0)

        if not approaching:
            # Moving away from boundary → safe
            ttld[i] = 999.0
            continue

        abs_e_lat_dot = abs(e_lat_dot)
        ttld[i] = margin / max(abs_e_lat_dot, epsilon)

        # Cap at sentinel value
        ttld[i] = min(ttld[i], 999.0)

    # Last timestep: copy from previous
    ttld[-1] = ttld[-2] if n >= 2 else 999.0

    return ttld


def compute_ttld_p5(ttld: np.ndarray) -> float:
    """
    5th percentile TTLD — worst-case safety margin.

    UNECE WP.29 R157: The 5th percentile represents the near-worst-case
    time available for corrective action before lane departure.

    Args:
        ttld: TTLD array (s), shape (N,).

    Returns:
        5th percentile TTLD in seconds.
    """
    if len(ttld) == 0:
        return 0.0

    # Filter out sentinel values for percentile computation
    valid = ttld[ttld < 998.0]
    if len(valid) == 0:
        # All timesteps have sentinel TTLD → vehicle is always safe
        return 999.0

    return float(np.percentile(valid, 5))


def compute_sbvr(
    e_lat: np.ndarray,
    departure_threshold: float = cfg.ISO15622_DEPARTURE_THR,
) -> float:
    """
    M-12: Safety Boundary Violation Rate.

    UNECE WP.29 R157: "The percentage of evaluation timesteps during which
    the vehicle exceeds the departure boundary."

    SBVR = (Σ 𝟙[|e_lat_i| > dep_thr]) / N × 100

    Args:
        e_lat:              Array of lateral errors (m), shape (N,).
        departure_threshold: Lane departure boundary (m).

    Returns:
        SBVR as a percentage (%).
    """
    n = len(e_lat)
    if n == 0:
        return 0.0

    violations = int(np.sum(np.abs(e_lat) > departure_threshold))
    return float(violations / n * 100.0)


def compute_mtbd(
    e_lat: np.ndarray,
    dt: float = cfg.SIM_DT,
    departure_threshold: float = cfg.ISO15622_DEPARTURE_THR,
) -> float:
    """
    M-13: Mean Time Between Departures.

    UNECE WP.29 R157: "The average time interval between successive
    lane departure events."

    A departure event is a rising-edge crossing of |e_lat| past the
    departure threshold.

    Args:
        e_lat:              Array of lateral errors (m), shape (N,).
        dt:                 Timestep (s).
        departure_threshold: Lane departure boundary (m).

    Returns:
        MTBD in seconds. Returns total episode time if 0 or 1 departures.
    """
    n = len(e_lat)
    if n < 2:
        return 0.0

    abs_elat = np.abs(e_lat)
    total_time = n * dt

    # Detect departure events (rising edge crossings)
    departure_times = []
    for i in range(1, n):
        if abs_elat[i] >= departure_threshold and abs_elat[i - 1] < departure_threshold:
            departure_times.append(i * dt)

    if len(departure_times) < 2:
        return total_time

    # Compute mean inter-departure interval
    intervals = np.diff(departure_times)
    return float(np.mean(intervals))
