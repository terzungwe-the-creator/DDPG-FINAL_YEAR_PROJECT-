"""
main.py — CLI Entry Point for DDPG Lane Keeping System v3.0

Orchestrates the complete pipeline:
    1. Dataset preloading (DS-01, DS-02, DS-03)
    2. Tyre model calibration
    3. DDPG training (600 episodes, curriculum-based)
    4. Deterministic evaluation (100 episodes, 5 scenarios)
    5. Publication-quality figure generation (8 PNGs)

Usage:
    python main.py --all                   # Full pipeline
    python main.py --train                 # Training only
    python main.py --eval                  # Evaluation only (requires checkpoint)
    python main.py --plot                  # Plot from existing results
    python main.py --all --skip-ds01       # Skip OpenLKA dataset
    python main.py --all --skip-ds02       # Skip comma-steering dataset
    python main.py --all --skip-ds03       # Skip Argoverse 2 dataset

Standards:
    ISO 15622:2018, IEEE 2846-2022, UNECE WP.29 R157,
    ISO 3888-2:2011, ISO 26262:2018 ASIL-B

Version: 3.0 — Dataset-Augmented Training
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

import config as cfg


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging to console and file."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # File handler
    cfg.ensure_directories()
    file_handler = logging.FileHandler(
        cfg.RESULTS_DIR / "system.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ddpg-lka-v3",
        description=(
            "DDPG Lane Keeping System v3.0 — "
            "Industrial-Grade Simulation-Based Performance Evaluation "
            "with External Dataset Integration & Data Fusion Pipeline"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Standards: ISO 15622:2018, IEEE 2846-2022, UNECE WP.29 R157\n"
            "Examples:\n"
            "  python main.py --all                 Full pipeline\n"
            "  python main.py --train               Training only\n"
            "  python main.py --eval                Evaluation only\n"
            "  python main.py --plot                Plotting only\n"
            "  python main.py --all --skip-ds01     Skip OpenLKA\n"
        ),
    )

    # Pipeline stages
    stage_group = parser.add_argument_group("Pipeline stages")
    stage_group.add_argument(
        "--all", action="store_true",
        help="Run complete pipeline: train → evaluate → plot",
    )
    stage_group.add_argument(
        "--train", action="store_true",
        help="Run training only (includes dataset preloading)",
    )
    stage_group.add_argument(
        "--eval", action="store_true",
        help="Run evaluation only (requires trained checkpoint)",
    )
    stage_group.add_argument(
        "--plot", action="store_true",
        help="Generate figures from existing results files",
    )

    # Dataset controls
    ds_group = parser.add_argument_group("Dataset controls")
    ds_group.add_argument(
        "--skip-ds01", action="store_true",
        help="Skip DS-01 (OpenLKA) dataset loading",
    )
    ds_group.add_argument(
        "--skip-ds02", action="store_true",
        help="Skip DS-02 (comma-steering-control) dataset loading",
    )
    ds_group.add_argument(
        "--skip-ds03", action="store_true",
        help="Skip DS-03 (Argoverse 2) dataset loading",
    )
    ds_group.add_argument(
        "--reload-datasets", action="store_true",
        help="Force re-parsing of datasets (ignore cache)",
    )

    # Training controls
    train_group = parser.add_argument_group("Training controls")
    train_group.add_argument(
        "--backend", type=str, default="bicycle",
        choices=["bicycle", "carla"],
        help="Physics backend: 'bicycle' (default, no CARLA needed) or 'carla' (requires CARLA 0.9.16 server)",
    )
    train_group.add_argument(
        "--episodes", type=int, default=cfg.N_EPISODES,
        help=f"Number of training episodes (default: {cfg.N_EPISODES})",
    )
    train_group.add_argument(
        "--seed", type=int, default=cfg.SEED,
        help=f"Random seed (default: {cfg.SEED})",
    )
    train_group.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cpu' or 'cuda' (auto-detect if not set)",
    )

    # Evaluation controls
    eval_group = parser.add_argument_group("Evaluation controls")
    eval_group.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint .pt file for evaluation (uses latest if not set)",
    )

    # Misc
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Validate: at least one stage must be selected
    if not any([args.all, args.train, args.eval, args.plot]):
        parser.print_help()
        print("\nError: Please specify at least one pipeline stage "
              "(--all, --train, --eval, --plot)")
        sys.exit(1)

    return args


def find_latest_checkpoint() -> Path:
    """Find the most recent checkpoint file."""
    checkpoint_dir = cfg.CHECKPOINTS_DIR
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint directory found: {checkpoint_dir}"
        )

    checkpoints = sorted(checkpoint_dir.glob("ddpg_checkpoint_ep*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}"
        )

    return checkpoints[-1]


def run_training(args: argparse.Namespace) -> dict:
    """Execute the training pipeline."""
    logger = logging.getLogger("main.train")
    logger.info("=" * 70)
    logger.info("DDPG-LKA v3.0 — TRAINING PIPELINE")
    logger.info("=" * 70)

    from training.trainer import Trainer

    device = args.device or cfg.get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Skip DS-01: {args.skip_ds01}")
    logger.info(f"Skip DS-02: {args.skip_ds02}")
    logger.info(f"Skip DS-03: {args.skip_ds03}")

    trainer = Trainer(
        device=device,
        skip_ds01=args.skip_ds01,
        skip_ds02=args.skip_ds02,
        skip_ds03=args.skip_ds03,
        reload_datasets=args.reload_datasets,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    result = trainer.run()

    logger.info("Training pipeline complete")
    logger.info(f"  Total episodes: {result['total_episodes']}")
    logger.info(f"  Total steps:    {result['total_steps']}")
    logger.info(f"  Wall time:      {result['wall_time_s']:.1f}s")
    logger.info(f"  Final RMSE lat: {result['final_rmse_lat']:.4f} m")
    logger.info(f"  Final LKSR:     {result['final_lksr']:.3f}")

    return result


def run_evaluation(args: argparse.Namespace, trainer_result: dict = None) -> dict:
    """Execute the evaluation pipeline."""
    logger = logging.getLogger("main.eval")
    logger.info("=" * 70)
    logger.info("DDPG-LKA v3.0 — EVALUATION PIPELINE")
    logger.info("=" * 70)

    import torch
    from ddpg.agent import DDPGAgent
    from simulator.lane_keeping_env import LaneKeepingEnv
    from training.evaluator import Evaluator

    device = args.device or cfg.get_device()

    # Create agent and load checkpoint
    agent = DDPGAgent(device=device)
    env = LaneKeepingEnv(training_mode=False)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint()

    logger.info(f"Loading checkpoint: {ckpt_path}")
    agent.load_checkpoint(ckpt_path)

    # Retrieve preload stats if available
    preload_stats = {}
    pretrain_rmse = float("nan")

    if cfg.PRELOAD_STATS_PATH.exists():
        import json
        with open(cfg.PRELOAD_STATS_PATH) as f:
            preload_stats = json.load(f)

    if trainer_result is not None:
        pretrain_rmse = trainer_result.get("pretrain_rmse_m22", float("nan"))

    # Update tyre parameters if calibration was performed
    if cfg.TYRE_CALIBRATION_PATH.exists():
        import json
        with open(cfg.TYRE_CALIBRATION_PATH) as f:
            cal = json.load(f)
        r2 = cal.get("r_squared", 0.0)
        if r2 > cfg.COMMA_TYRE_CALIBRATION_R2_THRESHOLD:
            env.update_tyre_params(
                C_af=cal["C_af"],
                C_ar=cal["C_ar"],
            )
            logger.info(
                f"Tyre model from calibration: C_af={cal['C_af']:.0f}, "
                f"C_ar={cal['C_ar']:.0f}, R²={r2:.3f}"
            )

    evaluator = Evaluator(
        agent=agent,
        env=env,
        seed=args.seed,
        preload_stats=preload_stats,
        pretrain_rmse=pretrain_rmse,
    )

    result = evaluator.run()

    logger.info("Evaluation pipeline complete")
    logger.info(f"  Overall pass: {result['overall_pass']}")
    logger.info(f"  Report:       {result['report_path']}")

    return result


def run_plotting(args: argparse.Namespace) -> None:
    """Generate publication-quality figures."""
    logger = logging.getLogger("main.plot")
    logger.info("=" * 70)
    logger.info("DDPG-LKA v3.0 — FIGURE GENERATION")
    logger.info("=" * 70)

    from plot_results import generate_all_figures

    generate_all_figures()

    logger.info(f"Figures saved to {cfg.FIGURES_DIR}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("  DDPG Lane Keeping System v3.0")
    logger.info("  Dataset-Augmented Training -- Industrial Grade")
    logger.info("  ISO 15622:2018 | IEEE 2846-2022 | UNECE WP.29 R157")
    logger.info("=" * 60)

    start_time = time.time()

    trainer_result = None

    if args.all or args.train:
        trainer_result = run_training(args)

    if args.all or args.eval:
        run_evaluation(args, trainer_result=trainer_result)

    if args.all or args.plot:
        run_plotting(args)

    elapsed = time.time() - start_time
    logger.info(f"Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
