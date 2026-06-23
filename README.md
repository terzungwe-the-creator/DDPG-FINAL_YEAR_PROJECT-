# DDPG Lane Keeping System v3.0

Industrial grade simulation based performance evaluation of a lane keeping system utilizing Deep Deterministic Policy Gradient (DDPG). The system integrates external datasets and a data fusion pipeline to train and evaluate an autonomous agent to meet established safety standards (ISO 15622:2018, IEEE 2846-2022, UNECE WP.29 R157).

## Requirements

* Python 3.10 or higher
* Recommended: NVIDIA GPU with CUDA support for accelerated training

## Installation

1. Clone the repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The primary entry point is `main.py`. This script orchestrates dataset preloading, tyre model calibration, DDPG training, deterministic evaluation across multiple scenarios, and publication quality figure generation.

### Core Commands

* **Full Pipeline**: Run the complete sequence from training to evaluation and plotting.
  ```bash
  python main.py --all
  ```

* **Training Only**: Run the dataset preloading and DDPG training sequence.
  ```bash
  python main.py --train
  ```

* **Evaluation Only**: Evaluate an existing trained agent across all test scenarios. This requires a saved checkpoint.
  ```bash
  python main.py --eval
  ```

* **Figure Generation**: Generate plots from existing results files.
  ```bash
  python main.py --plot
  ```

### Advanced Options

You can append these flags to the commands above to customize the execution:

* `--backend [bicycle|carla]`: Select the physics backend. The default is `bicycle`. The `carla` option requires a running CARLA 0.9.16 server.
* `--episodes [int]`: Set the number of training episodes.
* `--seed [int]`: Set the random seed for reproducible results.
* `--device [cpu|cuda]`: Override automatic device detection and force CPU or GPU usage.
* `--skip-ds01`, `--skip-ds02`, `--skip-ds03`: Skip the loading of specific external datasets to save time or memory.
* `--reload-datasets`: Force the system to re-parse datasets from their source files instead of using local cache.
* `--verbose`: Enable debug level logging to the console.

## Project Structure

* `simulator/`: Contains the physics backend implementations, OpenAI Gym environment wrapper, domain randomizer, and safety guardian.
* `ddpg/`: Contains the reinforcement learning components including the agent, neural networks, and replay buffer.
* `training/`: Contains the curriculum generator, evaluator, trainer, and structured logging tools.
* `datasets/`: Contains the preloader and adapters for parsing external driving datasets (OpenLKA, comma steering control, Argoverse 2).
* `metrics/`: Contains specialized scripts for extracting and calculating performance metrics.
* `results/`: The output directory. All training logs, network checkpoints, and generated figures are saved here automatically. Generated files are excluded from version control to maintain repository health.
