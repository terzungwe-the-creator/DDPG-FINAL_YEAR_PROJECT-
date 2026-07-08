# DDPG Lane Keeping System v3.0

Industrial-grade simulation-based performance evaluation of a lane keeping system utilizing Deep Deterministic Policy Gradient (DDPG) with TD3 extensions. The system integrates external datasets and a data fusion pipeline to train and evaluate an autonomous steering agent across five road scenarios, meeting established safety standards (ISO 15622:2018, IEEE 2846-2022, UNECE WP.29 R157).

## Key Features

* **Multi-Scenario Generalization** — Round-robin curriculum with failure-weighted oversampling trains the agent across all 5 road geometries (straight, constant curve, sinusoidal, double lane change, S-bend with clothoid transitions).
* **Domain Randomization** — Active from episode 1: mass, tyre stiffness, friction, wind gusts, observation noise, and sensor latency perturbations ensure robust transfer.
* **Dual Training Backends** — Pure NumPy backend for CPU-only systems; PyTorch backend with automatic CUDA detection for GPU acceleration.
* **Safety Envelope** — Three-layer SafetyGuardian (steering rate limiter, angle limiter, handoff trigger) wraps all agent outputs.
* **Real-World Deployment Pipeline** — Perception pipeline with sensor fusion, safety state machine, and 50 Hz CAN bus control loop.

## Requirements

* Python 3.10 or higher
* NumPy, SciPy, Gymnasium
* Optional: PyTorch (for GPU-accelerated training)
* Optional: Matplotlib (for figure generation)

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

The codebase provides two distinct training backends. Both use the same simulator physics, reward function, and curriculum.

* **NumPy Training (CPU-optimized)**
  High-performance pure-NumPy backend with vectorized operations. Recommended for CPU-only systems or environments where PyTorch has DLL issues.
  ```bash
  python run_training_numpy.py
  ```

* **PyTorch Training (GPU-accelerated)**
  Standard deep learning pipeline with automatic CUDA detection. Recommended when a GPU is available (e.g., Kaggle T4, Colab).
  ```bash
  python run_training_pytorch.py
  ```

* **Full Pipeline via main.py**
  Orchestrates dataset preloading, tyre calibration, training, evaluation, and figure generation:
  ```bash
  python main.py --all
  python main.py --train              # Training only
  python main.py --eval               # Evaluation only (requires checkpoint)
  python main.py --plot               # Generate figures from existing results
  python main.py --deploy --simulated # Simulated deployment test
  ```

### Cloud Training (Kaggle — Free GPU)

For faster training without local hardware constraints:

1. Create a new [Kaggle Notebook](https://www.kaggle.com/code) and set the accelerator to **GPU T4 ×2**.
2. Upload the project as a dataset or zip file.
3. Run:
   ```python
   !cp -r /kaggle/input/preacher/* /kaggle/working/
   %cd /kaggle/working
   !pip install gymnasium
   !python run_training_pytorch.py
   ```

The PyTorch backend auto-detects CUDA and logs the GPU device. Expect 3–5× speedup over CPU for gradient computation.

### Advanced Options

Append these flags to `main.py` commands:

| Flag | Description |
|------|-------------|
| `--backend [bicycle\|carla]` | Physics backend (default: `bicycle`) |
| `--episodes [int]` | Number of training episodes (default: 1000) |
| `--seed [int]` | Random seed for reproducibility |
| `--device [cpu\|cuda]` | Override automatic device detection |
| `--skip-ds01` / `--skip-ds02` / `--skip-ds03` | Skip specific external datasets |
| `--reload-datasets` | Force re-parsing datasets from source |
| `--verbose` | Enable debug-level logging |

## Training Architecture

### Curriculum

The system uses a round-robin curriculum with failure-weighted oversampling:

| Phase | Episodes | Scenarios | Purpose |
|-------|----------|-----------|---------|
| Warmup | 0–30 | SCN-01, SCN-02 | Bootstrap critic Q-values on straight + curve |
| All Scenes | 30–200 | All 5 | Expose agent to full geometry diversity |
| Refinement | 200–500 | All 5 (failure-weighted) | Struggling scenes get 3× sampling weight |
| Polish | 500–1000 | All 5 (failure-weighted) | Fine-tune with full domain randomization |

### Scenarios

| ID | Road Type | Key Challenge |
|----|-----------|---------------|
| SCN-01 | Straight road | Baseline tracking |
| SCN-02 | Constant radius curve | Steady-state cornering |
| SCN-03 | Sinusoidal curvature | Continuous steering adaptation |
| SCN-04 | Double lane change (ISO 3888-2) | Transient manoeuvres |
| SCN-05 | S-bend with clothoid transitions | Curvature discontinuity handling |

### Domain Randomization

Active from episode 1 via linear difficulty schedule (0.3 → 1.0):
- Vehicle mass ±15%
- Tyre cornering stiffness ±20%
- Road friction coefficient 0.7–1.0
- Lateral wind gusts
- Observation noise (Gaussian)
- Sensor latency simulation
- Camera bias offset
- Initial lateral perturbation ±0.3m

## Test Suite

Run the full test suite (38 tests covering physics, road geometry, feedforward, domain randomization, safety, reward, metrics, perception, and deployment):

```bash
python -m pytest test_system.py -v --tb=short
```

## Project Structure

```
preacher/
├── config.py               # Global hyperparameters and constants
├── main.py                 # CLI entry point (full pipeline)
├── run_training_numpy.py   # NumPy training backend (CPU-optimized)
├── run_training_pytorch.py # PyTorch training backend (GPU-accelerated)
├── test_system.py          # Comprehensive test suite (38 tests)
├── plot_results.py         # Publication-quality figure generation
│
├── simulator/              # Physics and environment
│   ├── vehicle_model.py    # Nonlinear bicycle model (RK4 integration)
│   ├── lane_keeping_env.py # Gymnasium environment wrapper
│   ├── road_profiles.py    # 5 road scenarios with O(1) curvature LUT
│   ├── reward.py           # Composite reward function
│   ├── safety_guardian.py  # 3-layer safety envelope
│   └── domain_randomizer.py# Parameter perturbation engine
│
├── ddpg/                   # Reinforcement learning
│   ├── agent.py            # TD3 agent (delayed policy updates)
│   ├── networks.py         # Actor/Critic MLPs (8→256→128→1)
│   ├── hybrid_buffer.py    # Stratified PER replay buffer
│   └── noise.py            # Ornstein-Uhlenbeck exploration noise
│
├── training/               # Training orchestration
│   ├── trainer.py          # Main training loop
│   ├── curriculum.py       # Round-robin scenario scheduling
│   ├── evaluator.py        # Deterministic evaluation (100 episodes)
│   └── logger.py           # Structured CSV/JSON logging
│
├── datasets/               # External data integration
│   └── preloader.py        # OpenLKA, comma, Argoverse 2 adapters
│
├── metrics/                # Performance measurement
│   ├── iso15622.py         # ISO 15622:2018 compliance metrics
│   ├── ieee2846.py         # IEEE 2846-2022 safety metrics
│   └── safety.py           # TTLD, SBVR computation
│
├── real_world/             # Deployment pipeline
│   ├── deployment_runner.py# 50 Hz control loop
│   ├── perception_pipeline.py # Sensor fusion + normalisation
│   ├── safety_monitor.py   # Runtime safety state machine
│   ├── sensor_interface.py # Abstract sensor API
│   └── actuator_interface.py # Abstract actuator API
│
└── results/                # Output directory (git-ignored)
    ├── system.log          # Training log
    ├── training_log.csv    # Per-episode metrics
    ├── eval_summary.csv    # Per-scenario evaluation results
    ├── performance_report.json # ISO 15622 pass/fail report
    └── figures/            # Generated plots
```

## Standards Compliance

| Standard | Coverage |
|----------|----------|
| ISO 15622:2018 | Lateral error, heading error, LKSR thresholds |
| IEEE 2846-2022 | Steering rate RMS, control effort, settling time |
| UNECE WP.29 R157 | Handoff trigger (SafetyGuardian Layer 3) |
| ISO 3888-2:2011 | SCN-04 double lane change geometry |
| ISO 26262:2018 | ASIL-B safety envelope design |
