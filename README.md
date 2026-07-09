# DDPG Lane Keeping System v3.0

A dual-backend autonomous lane keeping system using Deep Deterministic Policy Gradient (DDPG) with TD3 extensions. The agent is trained on a fast nonlinear bicycle model and validated on CARLA 0.9.16 high-fidelity simulation, with performance evaluated against ISO 15622:2018, IEEE 2846-2022, and UNECE WP.29 R157 safety standards.

## Key Features

* **Dual-Backend Architecture** — Train on a nonlinear bicycle model (RK4, 100 Hz) for speed, validate on CARLA 0.9.16 (PhysX 4-wheel dynamics) for fidelity. Both backends produce an identical 8D normalised observation vector, enabling zero-shot policy transfer.
* **Feedforward + Feedback Control** — Physics-based nominal steering from road curvature plus a learned RL correction, limiting the agent's burden to residual disturbance rejection.
* **Hybrid Data Fusion** — Stratified replay buffer fuses online RL rollouts with expert demonstrations from OpenLKA, comma-steering-control, and Argoverse 2 datasets.
* **Multi-Scenario Generalisation** — RMSE-weighted exponential curriculum across 5 road geometries with adaptive oversampling of struggling scenarios.
* **Domain Randomisation** — Mass, tyre stiffness, friction, wind, observation noise, sensor latency, and camera bias perturbations for sim-to-real robustness.
* **Safety Envelope** — Three-layer SafetyGuardian (steering rate limiter, angle limiter, UNECE R157 handoff trigger) wraps all agent outputs in both backends.
* **Standards Compliance** — Automated ISO 15622 pass/fail evaluation with per-scenario reporting.

## Requirements

* Python 3.10+
* NumPy, SciPy, Gymnasium
* PyTorch (for GPU-accelerated training)
* Optional: CARLA 0.9.16 (for high-fidelity validation)
* Optional: Matplotlib, Pandas (for figure generation and analysis)

## Installation

```bash
git clone https://github.com/terzungwe-the-creator/DDPG-FINAL_YEAR_PROJECT-.git
cd DDPG-FINAL_YEAR_PROJECT-
pip install -r requirements.txt
```

For CARLA validation, install CARLA 0.9.16 separately and ensure the `carla` Python package is on your path.

## Usage

### Training

```bash
# Full pipeline: dataset preloading + training + evaluation + plots
python main.py --all

# Training only (bicycle backend, default)
python main.py --train

# Training with CARLA backend
python main.py --train --backend carla

# Evaluation only (requires trained checkpoint)
python main.py --eval
python main.py --eval --backend carla   # Evaluate on CARLA

# Generate figures from existing results
python main.py --plot
```

### Standalone Training Scripts

```bash
# PyTorch training (GPU-accelerated, recommended)
python run_training_pytorch.py

# NumPy training (CPU-optimised, no PyTorch dependency)
python run_training_numpy.py
```

### Cloud Training (Kaggle)

For GPU-accelerated training on Kaggle's free Tesla T4:

```python
!git clone https://github.com/terzungwe-the-creator/DDPG-FINAL_YEAR_PROJECT-.git
%cd DDPG-FINAL_YEAR_PROJECT-
!pip install gymnasium
!python run_training_pytorch.py
```

Expected output:
```
Training 1500 episodes on cuda, warmup=5000, batch=256
Architecture: TD3 Actor 8->256->128->1, Hybrid PER, RoundRobin Curriculum
PyTorch 2.10.0+cu128, CUDA available: True
GPU: Tesla T4
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--backend [bicycle\|carla]` | Physics backend (default: `bicycle`) |
| `--episodes N` | Number of training episodes (default: 1500) |
| `--seed N` | Random seed for reproducibility |
| `--device [cpu\|cuda]` | Override automatic device detection |
| `--skip-ds01` / `--skip-ds02` / `--skip-ds03` | Skip specific external datasets |
| `--reload-datasets` | Force re-parsing datasets from source |
| `--verbose` | Enable debug-level logging |

## Architecture

### Dual-Backend Design

```
                    +---------------------+
                    |   DDPG Agent        |
                    |   Actor: 8->256->   |
                    |   128->1 (tanh)     |
                    +--------+------------+
                             |
                     action (normalised)
                             |
                    +--------v------------+
                    |  FF + FB Controller  |
                    |  delta = delta_nom   |
                    |      + delta_corr    |
                    +--------+------------+
                             |
                    +--------v------------+
                    |  Safety Guardian     |
                    |  (rate + angle +     |
                    |   handoff limiter)   |
                    +--------+------------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +------------v-----------+
    |  Bicycle Model     |     |  CARLA 0.9.16          |
    |  (RK4, 100 Hz)     |     |  (PhysX, sync mode)    |
    |  2-DoF dynamics    |     |  4-wheel dynamics      |
    +--------------------+     +------------------------+
              |                             |
              +-------------+---------------+
                            |
                  8D normalised observation
                  [e_lat, e_psi, kappa, v_y,
                   r, delta_prev, kappa_la1,
                   kappa_la2]
```

Both backends produce the same observation and accept the same action, enabling a policy trained on the bicycle model to be evaluated on CARLA without retraining.

### Training Pipeline

| Phase | Episodes | Scenarios | Key Behaviour |
|-------|----------|-----------|---------------|
| Phase 1: Warmup | 0-10 | SCN-01, SCN-02 | Bootstrap Q-values on straight + curve |
| Phase 2: All Scenes | 10-300 | All 5 | Full geometry diversity, RMSE-weighted sampling |
| Phase 3: Refinement | 300-750 | All 5 | Adaptive curriculum, sum-based scoring |
| Phase 4: Polish | 750-1500 | All 5 | Low noise, cosine LR decay |

### Mini-Eval & Regression Guard

Every 50 episodes, a deterministic mini-evaluation runs 8 trials per scenario. The best model is saved based on a sum-based score across all 5 scenarios. Rollback to the best checkpoint only occurs on significant overall score regression (> 0.3 drop), allowing the agent to freely explore trade-offs between scenarios without being trapped at an early checkpoint.

### Scenarios

| ID | Road Type | Geometry | Key Challenge |
|----|-----------|----------|---------------|
| SCN-01 | Straight road | 300 m, zero curvature | Baseline tracking |
| SCN-02 | Constant radius curve | R=80 m with clothoid transitions | Steady-state cornering |
| SCN-03 | Sinusoidal winding | Peak curvature 0.02 1/m, 100 m wavelength | Continuous steering adaptation |
| SCN-04 | Double lane change | ISO 3888-2 geometry, 3.5 m lateral offset | Transient manoeuvres |
| SCN-05 | Combined urban | R=60 m curve + R=40 m S-bend, variable speed | Mixed geometry with speed changes |

### Domain Randomisation

Randomised at the start of each training episode with cosine difficulty scaling:

| Parameter | Range | Justification |
|-----------|-------|---------------|
| Vehicle mass | +/-10% | Passenger/cargo variation |
| Tyre stiffness (front/rear) | +/-15% | Tyre wear, pressure, temperature |
| Road friction | 0.6 - 1.0 | Dry tarmac to light rain |
| Lateral wind force | N(0, 50) N | Crosswind disturbance |
| Road bank angle | +/-5 degrees | Camber/superelevation |
| Camera bias | +/-0.1 m | Lane detection calibration error |
| Observation latency | 0-3 steps | Processing pipeline delay |
| Curvature-proportional noise | Up to 7x on tight curves | Prevents curvature over-reliance |

In CARLA mode, friction randomisation is mapped to weather presets (clear/light rain/moderate rain).

### CARLA Integration

The CARLA backend (via `carla_bridge.py`) provides:

| Feature | Implementation |
|---------|---------------|
| Vehicle | Tesla Model 3 blueprint, mass/wheelbase matched to bicycle model |
| Physics | Synchronous mode, fixed 0.01 s timestep, PhysX 4-wheel dynamics |
| Lane errors | Waypoint projection with Menger curvature estimation |
| Speed control | Proportional throttle/brake controller targeting 60 km/h |
| Traffic | Traffic Manager with autopilot NPCs, hybrid physics mode |
| Weather | Friction-to-weather mapping for domain randomisation parity |
| Safety | SafetyGuardian applied in both backends identically |
| Maps | Town01-04, Town07 mapped to SCN-01 through SCN-05 |

## Project Structure

```
preacher/
├── config.py                  # Global hyperparameters and physical constants
├── main.py                    # CLI entry point (full pipeline)
├── run_training_pytorch.py    # PyTorch training backend (GPU-accelerated)
├── run_training_numpy.py      # NumPy training backend (CPU-optimised)
├── test_system.py             # Comprehensive test suite
├── plot_results.py            # Publication-quality figure generation
│
├── simulator/                 # Physics and environment
│   ├── vehicle_model.py       # Nonlinear bicycle model (RK4, Pacejka tyres)
│   ├── carla_bridge.py        # CARLA 0.9.16 interface (PhysX, waypoint projection)
│   ├── lane_keeping_env.py    # Gymnasium env with backend dispatch
│   ├── road_profiles.py       # 5 road scenarios with O(1) curvature LUT
│   ├── reward.py              # 6-component composite reward function
│   ├── safety_guardian.py     # 3-layer safety envelope (rate/angle/handoff)
│   └── domain_randomizer.py   # Parameter perturbation engine
│
├── ddpg/                      # Reinforcement learning
│   ├── agent.py               # TD3 agent (twin critics, delayed policy updates)
│   ├── networks.py            # Actor/Critic MLPs with LayerNorm (8->256->128->1)
│   ├── hybrid_buffer.py       # 4-sub-buffer stratified PER replay buffer
│   └── noise.py               # Ornstein-Uhlenbeck exploration noise
│
├── training/                  # Training orchestration
│   ├── trainer.py             # Main training loop (dual-backend aware)
│   ├── curriculum.py          # Round-robin scenario scheduling
│   ├── evaluator.py           # Deterministic 100-episode evaluation
│   └── logger.py              # Structured CSV/JSON logging
│
├── datasets/                  # External data integration
│   └── preloader.py           # OpenLKA, comma, Argoverse 2 adapters
│
├── metrics/                   # Performance measurement
│   ├── iso15622.py            # ISO 15622:2018 compliance metrics
│   ├── ieee2846.py            # IEEE 2846-2022 control quality metrics
│   ├── safety.py              # TTLD, SBVR, MTBD computation
│   └── dataset_quality.py     # M-22 pretrain performance metric
│
├── real_world/                # Deployment pipeline
│   ├── deployment_runner.py   # 50 Hz control loop
│   ├── perception_pipeline.py # Sensor fusion + normalisation
│   ├── safety_monitor.py      # Runtime safety state machine
│   ├── sensor_interface.py    # Abstract sensor API
│   └── actuator_interface.py  # Abstract actuator API
│
└── results/                   # Output directory (git-ignored)
    ├── system.log             # Training log
    ├── training_log.csv       # Per-episode metrics
    ├── eval_raw.csv           # Per-timestep evaluation data
    ├── eval_summary.csv       # Per-scenario evaluation results
    ├── performance_report.json# ISO 15622 pass/fail report
    └── figures/               # Generated plots
```

## Standards Compliance

| Standard | Coverage |
|----------|----------|
| ISO 15622:2018 | Lateral error, heading error, LKSR thresholds, automated pass/fail |
| IEEE 2846-2022 | Steering rate RMS, control effort, settling time, overshoot |
| UNECE WP.29 R157 | Handoff trigger via SafetyGuardian Layer 3, TTLD monitoring |
| ISO 3888-2:2011 | SCN-04 double lane change exact geometry |
| ISO 26262:2018 | ASIL-B safety envelope design principles |

## Test Suite

```bash
python -m pytest test_system.py -v --tb=short
```

Covers: physics integration, road geometry, feedforward controller, domain randomisation, safety guardian, reward function, ISO/IEEE/UNECE metrics, perception pipeline, and deployment loop.
