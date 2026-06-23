"""
ieee2846.py — IEEE 2846-2022 Control Quality Metrics

Implements metrics M-07 through M-10 from the agreed metrics framework.

Standard: IEEE 2846-2022 — Standard for Assumptions in Safety-Related Models
          for Automated Driving Systems.

Metrics:
    M-07: Steering Rate RMS    δ̇_rms  — RMS of dδ/dt; target < 0.20 rad/s
    M-08: Control Effort       CE      — ∫δ²dt (accumulated steering energy)
    M-09: Settling Time        t_settle — Time to |e_lat| < 0.10 m (sustained)
    M-10: Overshoot            OS      — (e_peak/e_init − 1) × 100%
"""

from __future__ import annotations

import numpy as np

import config as cfg


def compute_steering_rate_rms(
    delta: np.ndarray, dt: float = cfg.SIM_DT
) -> float:
    """
    M-07: Steering Rate RMS.

    IEEE 2846-2022: "The RMS of the steering angle rate of change provides
    a measure of control smoothness."

    δ̇_rms = sqrt(1/(N-1) · Σ((δ_{i+1} − δ_i) / dt)²)

    Target: < 0.20 rad/s (IEEE2846_STEER_RATE_RMS_LIMIT).

    Args:
        delta: Array of steering angles (rad), shape (N,).
        dt:    Timestep (s).

    Returns:
        RMS steering rate in rad/s.
    """
    if len(delta) < 2:
        return 0.0

    delta_dot = np.diff(delta) / dt
    return float(np.sqrt(np.mean(delta_dot ** 2)))


def compute_control_effort(
    delta: np.ndarray, dt: float = cfg.SIM_DT
) -> float:
    """
    M-08: Control Effort.

    IEEE 2846-2022: "Accumulated steering energy over the evaluation interval."

    CE = ∫δ²dt ≈ Σ δ_i² · dt

    Unit: rad²·s. Lower values indicate less aggressive control.

    Args:
        delta: Array of steering angles (rad), shape (N,).
        dt:    Timestep (s).

    Returns:
        Control effort in rad²·s.
    """
    return float(np.sum(delta ** 2) * dt)


def compute_settling_time(
    e_lat: np.ndarray,
    dt: float = cfg.SIM_DT,
    threshold: float = cfg.IEEE2846_SETTLING_THRESHOLD,
    sustain_time: float = cfg.IEEE2846_SETTLING_SUSTAIN,
) -> float:
    """
    M-09: Settling Time.

    IEEE 2846-2022: "Time from the start of the evaluation until |e_lat|
    first enters and remains within the settling threshold for a sustained
    period."

    The settling criterion requires |e_lat| < threshold for at least
    `sustain_time` seconds continuously.

    Args:
        e_lat:        Array of lateral errors (m), shape (N,).
        dt:           Timestep (s).
        threshold:    Settling threshold (m). Default: 0.10 m.
        sustain_time: Duration to sustain below threshold (s). Default: 0.5 s.

    Returns:
        Settling time in seconds. Returns total episode time if never settles.
    """
    n = len(e_lat)
    if n == 0:
        return 0.0

    sustain_steps = int(sustain_time / dt)
    total_time = n * dt

    for i in range(n):
        # Check if all subsequent steps within sustain window are below threshold
        end_idx = min(i + sustain_steps, n)
        if end_idx - i < sustain_steps:
            break

        window = np.abs(e_lat[i:end_idx])
        if np.all(window < threshold):
            return float(i * dt)

    return float(total_time)


def compute_overshoot(e_lat: np.ndarray) -> float:
    """
    M-10: Overshoot.

    IEEE 2846-2022: "The percentage by which the peak lateral error exceeds
    the initial lateral error during a disturbance response."

    OS = (e_peak / e_init − 1) × 100%

    Where:
        e_init = |e_lat[0]| (initial perturbation magnitude)
        e_peak = max(|e_lat|) over the episode

    If e_init ≈ 0 (no perturbation), overshoot is defined as:
        OS = e_peak / 0.01 × 100  (normalised to 1 cm reference)

    Args:
        e_lat: Array of lateral errors (m), shape (N,).

    Returns:
        Overshoot as a percentage (%). Returns 0.0 if no overshoot.
    """
    if len(e_lat) == 0:
        return 0.0

    e_init = abs(e_lat[0])
    e_peak = float(np.max(np.abs(e_lat)))

    if e_init < 0.01:
        # No meaningful initial perturbation
        # Report peak error normalised to 1 cm reference
        return float(e_peak / 0.01 * 100.0) if e_peak > 0.01 else 0.0

    if e_peak <= e_init:
        return 0.0

    return float((e_peak / e_init - 1.0) * 100.0)
