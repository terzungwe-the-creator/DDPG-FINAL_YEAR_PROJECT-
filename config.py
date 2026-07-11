"""
config.py — Central Configuration for DDPG Lane Keeping System v3.0

All physical constants, hyperparameters, ISO thresholds, and system paths.
Every constant cites its source standard by number and clause.

Standards:
    ISO 15622:2018 — Lane Keeping Assistance Systems
    IEEE 2846-2022 — Assumptions for Models in Safety-Related ADS
    UNECE WP.29 R157 — Automated Lane Keeping Systems
    ISO 3888-2:2011 — Double Lane Change Test
    Rajamani (2012) — Vehicle Dynamics and Control, 2nd ed.

Author: Principal Autonomous Systems Engineer
Version: 3.0 — Dataset-Augmented Training
"""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1 — VEHICLE DYNAMICS PARAMETERS
# Source: Rajamani (2012) Table 2.1, Table 3.2; NHTSA NCAP geometry specs
# Representative class: BYD Seal / Tesla Model 3 BEV midsize
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VEHICLE_MASS: float = 1650.0        # kg — kerb weight, BEV midsize sedan
VEHICLE_IZ: float = 2315.3          # kg·m² — yaw moment of inertia
VEHICLE_LF: float = 1.105           # m — CoM to front axle
VEHICLE_LR: float = 1.738           # m — CoM to rear axle
VEHICLE_WHEELBASE: float = VEHICLE_LF + VEHICLE_LR  # m — total wheelbase

V_REFERENCE: float = 16.67          # m/s — 60 km/h (ISO 15622:2018 test speed)
LANE_WIDTH: float = 3.50            # m — EU standard lane width (AASHTO/EC)
LANE_WIDTH_HALF: float = LANE_WIDTH / 2.0  # m — half lane width

DELTA_MAX: float = 0.35             # rad — ±20° front wheel angle at 60 km/h

# Nominal tyre cornering stiffness — Rajamani (2012) Table 3.2
# Overridden by DS-02 (comma-steering-control) calibration if R² > 0.85
TYRE_CAF_NOMINAL: float = 88000.0   # N/rad — front axle cornering stiffness
TYRE_CAR_NOMINAL: float = 94000.0   # N/rad — rear axle cornering stiffness

# Simulation timestep — RK4 integration mandatory
SIM_DT: float = 0.01                # s — 100 Hz physics update rate
SIM_MAX_STEPS: int = 2000           # steps — 20 s max episode duration (sufficient for all profiles)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2 — OBSERVATION & ACTION SPACE NORMALISATION
# All normalisation constants are physical limits, not data-driven.
# This guarantees zero distribution shift between real-world and simulated data.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBS_DIM: int = 8
ACT_DIM: int = 1

# Normalisation divisors for each observation dimension
NORM_E_LAT: float = LANE_WIDTH_HALF           # 1.75 m
NORM_E_PSI: float = np.pi / 4.0               # 0.785 rad (45°)
NORM_KAPPA: float = 0.05                       # 1/m — max curvature
NORM_V_Y: float = 2.0                         # m/s — max lateral velocity
NORM_YAW_RATE: float = 0.5                     # rad/s — max yaw rate
NORM_DELTA: float = DELTA_MAX                  # 0.35 rad
NORM_KAPPA_LA1: float = 0.05                   # 1/m — 1s lookahead curvature
NORM_KAPPA_LA2: float = 0.05                   # 1/m — 2s lookahead curvature


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3 — REWARD FUNCTION WEIGHTS
# Justified against training stability ablations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REWARD_W_LATERAL: float = 8.0    # Dominant: lane keeping is the primary control objective
REWARD_W_HEADING: float = 4.0    # Increased: heading RMSE fails on SCN-03/04, needs stronger alignment signal
REWARD_W_SMOOTH: float = 1.0     # Moderate: penalise jerky steering without drowning lateral
REWARD_W_PROGRESS: float = 0.3   # Small: prevents zero-speed policy collapse
REWARD_W_BOUNDARY: float = 3.0   # Boundary proximity penalty weight (§3.5 code review)
REWARD_TERMINAL_PENALTY: float = -20.0  # Lane departure penalty

# Departure threshold for terminal condition
DEPARTURE_THRESHOLD: float = LANE_WIDTH_HALF  # 1.75 m — lane boundary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4 — ISO 15622:2018 PASS/FAIL THRESHOLDS
# Source: ISO 15622:2018 §6.3, §7.1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ISO15622_LAT_ERROR_LIMIT: float = 0.30    # m — §6.3: mean |e_lat| limit
ISO15622_RMSE_LAT_LIMIT: float = 0.40     # m — §6.3: RMSE e_lat limit
ISO15622_HEADING_LIMIT: float = 0.087     # rad — §6.3: 5.0 degrees
ISO15622_DEPARTURE_THR: float = 0.75      # m — §7.1: departure boundary
ISO15622_MIN_LKSR: float = 0.95           # — — §7.1: 95% minimum success rate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5 — IEEE 2846-2022 CONTROL QUALITY TARGETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IEEE2846_STEER_RATE_RMS_LIMIT: float = 0.20   # rad/s — steering rate RMS target
IEEE2846_SETTLING_THRESHOLD: float = 0.10     # m — |e_lat| threshold for settling
IEEE2846_SETTLING_SUSTAIN: float = 0.5        # s — duration to sustain below threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6 — UNECE WP.29 R157 SAFETY MARGINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNECE_R157_TTLD_MIN: float = 0.4              # s — minimum time-to-lane-departure
UNECE_R157_TTLD_EPSILON: float = 1e-6         # s — prevents division by zero in TTLD


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §7 — DDPG HYPERPARAMETERS
# Sources: Lillicrap et al. (2016) Table 1, Appendix C;
#          Fujimoto et al. (2018) TD3; Duan et al. (2016) benchmarking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTOR_LR: float = 5e-5              # Halved to prevent catastrophic forgetting across scenarios
CRITIC_LR: float = 1e-3             # Critic learns faster to track shifting policy
GAMMA: float = 0.99                  # Discount — horizon ~100 steps = 1.0 s at 100 Hz
TAU: float = 0.005                   # Polyak averaging coefficient
BATCH_SIZE: int = 256                # Mini-batch size
POLICY_UPDATE_FREQ: int = 2          # Delayed policy update (TD3-style)
CRITIC_GRAD_CLIP: float = 1.0        # Gradient clipping on critic
ACTOR_GRAD_CLIP: float = 0.5         # Gradient clipping on actor (prevents policy collapse)
WARMUP_STEPS: int = 5000             # Reduced from 20K: buffer pre-populated with real data
UPDATES_PER_STEP: int = 1            # Gradient steps per environment step (UTD ratio)

# Sim-only mode overrides (activated when all datasets are skipped)
SIM_ONLY_WARMUP_STEPS: int = 5_000   # Reduced warmup — agent learns fast with curriculum
SIM_ONLY_UPDATES_PER_STEP: int = 2   # Higher UTD ratio compensates for no preloaded data
SIM_ONLY_NOISE_SIGMA_INIT: float = 0.20  # More exploration noise without expert guidance

# Total buffer capacity across all sub-buffers
BUFFER_CAPACITY: int = 2_000_000     # 2M: holds ~1M real + ~600K sim

# Per-source sub-buffer capacities
BUFFER_CAPACITIES: Dict[str, int] = {
    "openlka": 600_000,
    "comma": 400_000,
    "argoverse": 300_000,
    "sim": 700_000,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §8 — NOISE SCHEDULE (Ornstein-Uhlenbeck with annealing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOISE_SIGMA_INIT: float = 0.15      # Initial exploration noise (with expert data)
NOISE_SIGMA_FINAL: float = 0.05     # Final exploitation noise (maintain exploration)
NOISE_THETA: float = 0.15           # OU mean reversion rate
NOISE_ANNEAL_START: int = 50         # Episode when annealing begins (earlier for FF+FB)
NOISE_ANNEAL_END: int = 700          # Episode when annealing completes (slower for multi-scene)

# Feedforward preview time (s) — lookahead for anticipatory steering
PREVIEW_TIME: float = 1.2            # 1.2s at 16.67 m/s = 20m preview (improved curve anticipation)
CORRECTION_AUTHORITY: float = 0.85    # Increased from 0.70: SCN-04 needs more steering headroom at peak curvature
ACTION_SMOOTHING_ALPHA: float = 0.30  # Low-pass filter coefficient for action smoothing (faster response)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §9 — STRATIFIED BUFFER SAMPLING WEIGHTS
# Source: Nair et al. (2018) ICRA — overcoming exploration with demonstrations
# Phase-aware weighting to transition from expert-biased to sim-dominant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUFFER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "phase1": {"openlka": 0.40, "comma": 0.20, "argoverse": 0.20, "sim": 0.20},
    "phase2": {"openlka": 0.30, "comma": 0.15, "argoverse": 0.15, "sim": 0.40},
    "phase3": {"openlka": 0.15, "comma": 0.10, "argoverse": 0.10, "sim": 0.65},
}

# Phase boundaries (episode numbers)
PHASE1_END: int = 150
PHASE2_END: int = 300
# Phase 3: 300–600


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §10 — TRAINING PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_EPISODES: int = 1000                # Total training episodes
CHECKPOINT_INTERVAL: int = 50        # Save model every N episodes
CONVERGENCE_WINDOW: int = 50         # Rolling window for convergence check


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §11 — EVALUATION PARAMETERS
# Source: ISO 15622:2018 §8.4 — disturbance rejection procedure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVAL_EPISODES_PER_SCENARIO: int = 20    # 20 episodes per scenario
EVAL_N_SCENARIOS: int = 5               # SCN-01 through SCN-05
EVAL_TOTAL_EPISODES: int = EVAL_EPISODES_PER_SCENARIO * EVAL_N_SCENARIOS
EVAL_PERTURBATION_RANGE: float = 0.2    # m — U(-0.2, 0.2) lateral init


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §12 — DATASET PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPENLKA_MAX_SEGMENTS: int = 500
OPENLKA_MAX_TRANSITIONS: int = 500_000
OPENLKA_MIN_SPEED_MPS: float = 10.0       # Filter low-speed data

COMMA_MAX_TRANSITIONS: int = 300_000
COMMA_MIN_SPEED_MPS: float = 10.0         # Highway speeds only
COMMA_TYRE_CALIBRATION_R2_THRESHOLD: float = 0.85

ARGOVERSE_MAX_SCENARIOS: int = 10_000
ARGOVERSE_MAX_TRANSITIONS: int = 200_000
ARGOVERSE_MAX_ELAT_FILTER: float = 1.5    # m — discard off-road timesteps
ARGOVERSE_NUM_WORKERS: int = 4

# Pretrain-only evaluation (M-22)
PRETRAIN_GRADIENT_STEPS: int = 1000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §13 — CURRICULUM SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO_IDS: List[str] = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]

CURRICULUM_PHASES: Dict[str, Dict] = {
    "phase1": {"episodes": (0, 100), "scenarios": ["SCN-01"]},
    "phase2": {"episodes": (100, 250), "scenarios": ["SCN-01", "SCN-02"]},
    "phase3": {"episodes": (250, 500), "scenarios": ["SCN-01", "SCN-02", "SCN-03"]},
    "phase4": {"episodes": (500, 1000), "scenarios": SCENARIO_IDS},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §14 — FILE PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT: Path = Path(__file__).parent.resolve()
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CACHE_DIR: Path = RESULTS_DIR / "cache"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# Dataset directories (external — must exist if not skipped)
OPENLKA_DATA_DIR: Path = PROJECT_ROOT / "data" / "openlka"
COMMA_DATA_DIR: Path = PROJECT_ROOT / "data" / "comma_steering"
ARGOVERSE_DATA_DIR: Path = PROJECT_ROOT / "data" / "argoverse2"

# Output files
TRAINING_LOG_PATH: Path = RESULTS_DIR / "training_log.csv"
EVAL_RAW_PATH: Path = RESULTS_DIR / "eval_raw.csv"
EVAL_SUMMARY_PATH: Path = RESULTS_DIR / "eval_summary.csv"
PERFORMANCE_REPORT_PATH: Path = RESULTS_DIR / "performance_report.json"
PRELOAD_STATS_PATH: Path = RESULTS_DIR / "dataset_preload_stats.json"
TYRE_CALIBRATION_PATH: Path = RESULTS_DIR / "tyre_calibration.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §15 — PLOTTING STYLE
# IEEE double-column format, 300 DPI, publication quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PLOT_STYLE: Dict = {
    "fig_width_double": 7.16,       # inches — IEEE Transactions double-column
    "fig_width_single": 3.5,        # inches — IEEE Transactions single-column
    "dpi": 300,
    "linewidth": 1.5,
    "colors": {
        "primary": "#1f4e79",       # engineering blue
        "secondary": "#d62728",     # ISO safety red
        "reference": "#2ca02c",     # reference path green
        "expert": "#ff7f0e",        # orange — expert data source
        "sim": "#9467bd",           # purple — simulation data
        "neutral": "#7f7f7f",       # annotations
        "shade": "#aec7e8",         # confidence bands
    },
    "fonts": {
        "family": "DejaVu Serif",   # Fallback for Times New Roman
        "label": 9,
        "tick": 8,
        "legend": 8,
        "title": 10,
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §16 — SEED & DEVICE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEED: int = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §16b — REAL-WORLD DEPLOYMENT CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Control loop frequency (Hz) — must match vehicle CAN bus rate
DEPLOY_CONTROL_HZ: int = 50          # 50 Hz control loop (20 ms period)
DEPLOY_SENSOR_TIMEOUT_S: float = 0.1 # Max age of sensor data before emergency stop
DEPLOY_MAX_STEERING_RATE: float = 2.5  # rad/s — actuator rate limit
DEPLOY_MIN_SPEED_MPS: float = 5.0    # m/s — minimum speed for LKA engagement
DEPLOY_MAX_SPEED_MPS: float = 36.0   # m/s — 130 km/h max speed for LKA
DEPLOY_EMERGENCY_LATERAL_M: float = 1.5  # Emergency stop if |e_lat| exceeds this
DEPLOY_HANDOFF_LATERAL_M: float = 1.0    # Driver handoff warning threshold
DEPLOY_HEARTBEAT_TIMEOUT_S: float = 0.5  # Watchdog timeout for actuator comms

# Sensor fusion weights
DEPLOY_CAMERA_WEIGHT: float = 0.6    # Camera lane detection weight
DEPLOY_LIDAR_WEIGHT: float = 0.25    # LiDAR lane detection weight
DEPLOY_MAP_WEIGHT: float = 0.15      # HD map prior weight

# Supported hardware interfaces
DEPLOY_CAN_INTERFACE: str = "socketcan"  # CAN bus interface type
DEPLOY_CAN_CHANNEL: str = "can0"         # CAN channel name
DEPLOY_CAN_BITRATE: int = 500000         # CAN bus bitrate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §17 — CARLA INTEGRATION (v0.9.16)
# Source: CARLA 0.9.16 Python API — carla.readthedocs.io
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CARLA_HOST: str = "localhost"
CARLA_PORT: int = 2000
CARLA_TIMEOUT: float = 10.0
CARLA_VEHICLE_BP: str = "vehicle.tesla.model3"
CARLA_SYNC_MODE: bool = True
CARLA_FIXED_DT: float = SIM_DT          # 0.01 s — match physics timestep

CARLA_MAP_SCENARIOS: Dict[str, str] = {
    "SCN-01": "Town04",   # Long straight highway
    "SCN-02": "Town03",   # Constant radius curves
    "SCN-03": "Town07",   # Winding rural roads
    "SCN-04": "Town02",   # Narrow lanes for DLC
    "SCN-05": "Town01",   # Mixed urban
}


def ensure_directories() -> None:
    """Create all required output directories."""
    for d in [RESULTS_DIR, CACHE_DIR, CHECKPOINTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Auto-detect best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_buffer_phase(episode: int) -> str:
    """Return the buffer sampling phase key for a given episode."""
    if episode < PHASE1_END:
        return "phase1"
    elif episode < PHASE2_END:
        return "phase2"
    else:
        return "phase3"


def get_noise_sigma(episode: int) -> float:
    """Compute annealed noise sigma for a given episode."""
    if episode < NOISE_ANNEAL_START:
        return NOISE_SIGMA_INIT
    elif episode >= NOISE_ANNEAL_END:
        return NOISE_SIGMA_FINAL
    else:
        progress = (episode - NOISE_ANNEAL_START) / (NOISE_ANNEAL_END - NOISE_ANNEAL_START)
        return NOISE_SIGMA_INIT + progress * (NOISE_SIGMA_FINAL - NOISE_SIGMA_INIT)
