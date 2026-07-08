"""
deployment_runner.py — Real-Time Deployment Control Loop

The top-level orchestrator for deploying the trained DDPG lane keeping
agent on a real vehicle. Manages the real-time control loop, logging,
and graceful shutdown.

Usage:
    # From command line:
    python main.py --deploy --checkpoint results/checkpoints/best.pth

    # Programmatic:
    runner = DeploymentRunner(
        checkpoint_path="results/checkpoints/best.pth",
        sensor_interface=MyCANSensorInterface(),
        actuator_interface=MyDBWActuator(),
    )
    runner.run()

Control Loop Architecture:
    50 Hz fixed-rate loop (20 ms period):
        1. Read sensors (< 1 ms)
        2. Perception pipeline (< 2 ms)
        3. Safety check (< 0.1 ms)
        4. Agent inference (< 1 ms on GPU, < 5 ms on CPU)
        5. Feedforward + feedback (< 0.1 ms)
        6. Send actuator command (< 1 ms)
        7. Log telemetry (< 0.5 ms)
    Total budget: < 10 ms — well within 20 ms period

Reference:
    ISO 26262:2018 Part 6 — software unit design
    UNECE WP.29 R157 §5.4 — real-time performance requirements
"""

from __future__ import annotations

import csv
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config as cfg
from ddpg.agent import DDPGAgent
from real_world.sensor_interface import (
    SensorInterface,
    SimulatedSensorInterface,
)
from real_world.actuator_interface import (
    ActuatorInterface,
    SimulatedActuator,
)
from real_world.vehicle_bridge import VehicleBridge
from real_world.safety_monitor import SystemState

logger = logging.getLogger(__name__)


class DeploymentRunner:
    """
    Real-time deployment controller for the DDPG lane keeping agent.

    Manages the complete lifecycle:
        1. Load trained agent from checkpoint
        2. Connect to sensor and actuator hardware
        3. Run fixed-rate control loop
        4. Log telemetry to CSV
        5. Handle graceful shutdown (SIGINT/SIGTERM)

    Supports two modes:
        - Real hardware: Connect to actual CAN bus sensors/actuators
        - Simulated: Use SimulatedSensorInterface + SimulatedActuator
          for end-to-end pipeline testing without hardware

    Attributes:
        checkpoint_path: Path to trained agent checkpoint.
        bridge:          VehicleBridge connecting agent to hardware.
        control_hz:      Control loop frequency (Hz).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        sensor_interface: Optional[SensorInterface] = None,
        actuator_interface: Optional[ActuatorInterface] = None,
        control_hz: int = cfg.DEPLOY_CONTROL_HZ,
        scenario_id: str = "SCN-01",
        max_steps: Optional[int] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.scenario_id = scenario_id
        self.max_steps = max_steps
        self.log_dir = log_dir or cfg.RESULTS_DIR / "deployment"

        # Load trained agent
        self.agent = self._load_agent()

        # Default to simulated hardware if none provided
        if sensor_interface is None:
            logger.info("No sensor interface provided — using simulated sensors")
            self._simulated = True
            sensor_interface = SimulatedSensorInterface()
        else:
            self._simulated = False

        if actuator_interface is None:
            logger.info("No actuator interface provided — using simulated actuator")
            actuator_interface = SimulatedActuator()

        # Create vehicle bridge
        self.bridge = VehicleBridge(
            agent=self.agent,
            sensors=sensor_interface,
            actuator=actuator_interface,
            control_hz=control_hz,
        )

        # Telemetry logging
        self._telemetry_log: list[dict] = []
        self._running = False
        self._shutdown_requested = False

    def _load_agent(self) -> DDPGAgent:
        """
        Load trained DDPG agent from checkpoint.

        Returns:
            DDPGAgent with loaded weights in eval mode.
        """
        agent = DDPGAgent(
            state_dim=cfg.OBS_DIM,
            action_dim=cfg.ACTION_DIM,
        )

        if self.checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=cfg.DEVICE,
                weights_only=False,
            )
            agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(
                f"Checkpoint not found: {self.checkpoint_path} — "
                "using untrained agent (for testing only)"
            )

        # Set to eval mode (no dropout, batchnorm in eval)
        agent.actor.eval()
        agent.critic.eval()

        return agent

    def run(self) -> dict:
        """
        Execute the deployment control loop.

        This is the main entry point for real-world deployment.
        Runs until:
            - max_steps reached (if set)
            - Ctrl+C (SIGINT) received
            - Emergency stop triggered
            - Driver disengages

        Returns:
            Session statistics dictionary.
        """
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Setup
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 70)
        logger.info("DEPLOYMENT RUNNER — DDPG Lane Keeping System")
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Control Hz: {self.control_hz}")
        logger.info(f"  Mode: {'SIMULATED' if self._simulated else 'REAL HARDWARE'}")
        logger.info(f"  Session ID: {session_id}")
        logger.info("=" * 70)

        # Connect to hardware
        if not self.bridge.connect():
            logger.error("Hardware connection failed — aborting deployment")
            return {"success": False, "reason": "connection_failed"}

        # If simulated, reset the sim environment
        if self._simulated and isinstance(self.bridge.sensors, SimulatedSensorInterface):
            self.bridge.sensors.reset(self.scenario_id)

        # Engage LKA
        if not self.bridge.engage():
            logger.error("LKA engagement failed — aborting deployment")
            self.bridge.close()
            return {"success": False, "reason": "engagement_failed"}

        # ── Main control loop ──────────────────────────────────────────
        self._running = True
        step = 0
        loop_overruns = 0

        logger.info("Control loop STARTED")

        try:
            while self._running:
                loop_start = time.monotonic()

                # Execute one control cycle
                telemetry = self.bridge.step()

                # If simulated, also step the environment
                if self._simulated and isinstance(self.bridge.sensors, SimulatedSensorInterface):
                    # Extract action from bridge state and step sim
                    action = np.array([telemetry.get("action", 0.0)])
                    sim_info = self.bridge.sensors.step(action)

                # Log telemetry
                telemetry["step"] = step
                telemetry["timestamp_s"] = time.monotonic()
                self._telemetry_log.append(telemetry)

                # Check termination conditions
                state = telemetry.get("state", SystemState.ACTIVE)

                if state == SystemState.EMERGENCY_STOP:
                    logger.critical("EMERGENCY STOP — deployment terminated")
                    self._running = False
                    break

                if state == SystemState.INACTIVE:
                    logger.info("System INACTIVE (driver disengage) — deployment ended")
                    self._running = False
                    break

                if self.max_steps is not None and step >= self.max_steps:
                    logger.info(f"Max steps ({self.max_steps}) reached — stopping")
                    self._running = False
                    break

                if self._shutdown_requested:
                    logger.info("Shutdown requested — disengaging")
                    self.bridge.disengage()
                    self._running = False
                    break

                # Periodic status logging (every 5 seconds)
                if step > 0 and step % (self.control_hz * 5) == 0:
                    self._log_periodic_status(step, telemetry)

                # Fixed-rate timing
                step += 1
                elapsed = time.monotonic() - loop_start
                sleep_time = self.dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    loop_overruns += 1
                    if loop_overruns % 100 == 1:
                        logger.warning(
                            f"Control loop overrun: {elapsed * 1000:.1f}ms "
                            f"(budget: {self.dt * 1000:.1f}ms)"
                        )

        except Exception as e:
            logger.critical(f"Unhandled exception in control loop: {e}")
            self.bridge.disengage()
            raise
        finally:
            # Always clean up
            self.bridge.close()
            self._running = False

        # ── Save results ───────────────────────────────────────────────
        results = self._save_session(session_id, step, loop_overruns)

        logger.info("=" * 70)
        logger.info("DEPLOYMENT SESSION COMPLETE")
        logger.info(f"  Steps: {step}")
        logger.info(f"  Duration: {step * self.dt:.1f}s")
        logger.info(f"  Loop overruns: {loop_overruns}")
        logger.info(f"  Results: {self.log_dir / session_id}")
        logger.info("=" * 70)

        return results

    def _signal_handler(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Signal {signum} received — requesting shutdown")
        self._shutdown_requested = True

    def _log_periodic_status(self, step: int, telemetry: dict) -> None:
        """Log periodic status update."""
        elapsed_s = step * self.dt
        e_lat = telemetry.get("e_lat_m", 0.0)
        v_x = telemetry.get("v_x_mps", 0.0)
        delta = telemetry.get("delta_cmd", 0.0)
        state = telemetry.get("state", SystemState.ACTIVE)
        latency = telemetry.get("latency_s", 0.0) * 1000

        logger.info(
            f"[{elapsed_s:.0f}s] state={state.name} | "
            f"e_lat={e_lat:+.3f}m | v={v_x * 3.6:.0f}km/h | "
            f"δ={delta:+.3f}rad | lat={latency:.1f}ms"
        )

    def _save_session(self, session_id: str, total_steps: int,
                      loop_overruns: int) -> dict:
        """
        Save deployment session data.

        Creates:
            - deployment/{session_id}/telemetry.csv
            - deployment/{session_id}/session_report.json
        """
        session_dir = self.log_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save telemetry CSV
        csv_path = session_dir / "telemetry.csv"
        if self._telemetry_log:
            columns = [
                "step", "timestamp_s", "e_lat_m", "e_psi_rad", "v_x_mps",
                "delta_cmd", "delta_nominal", "delta_correction", "action",
                "authority_factor", "confidence", "latency_s",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=columns, extrasaction="ignore"
                )
                writer.writeheader()
                for row in self._telemetry_log:
                    # Convert enums and other non-serializable types
                    clean_row = {}
                    for k in columns:
                        v = row.get(k, 0.0)
                        if isinstance(v, (SystemState,)):
                            v = v.name
                        clean_row[k] = v
                    writer.writerow(clean_row)

        # Session report
        bridge_stats = self.bridge.session_stats
        monitor_diag = self.bridge.monitor.diagnostics

        report = {
            "session_id": session_id,
            "checkpoint": str(self.checkpoint_path),
            "mode": "simulated" if self._simulated else "real_hardware",
            "scenario_id": self.scenario_id if self._simulated else "N/A",
            "control_hz": self.control_hz,
            "total_steps": total_steps,
            "duration_s": round(total_steps * self.dt, 1),
            "loop_overruns": loop_overruns,
            "avg_latency_ms": round(bridge_stats["avg_latency_ms"], 2),
            "max_latency_ms": round(bridge_stats["max_latency_ms"], 2),
            "safety": {
                "total_interventions": monitor_diag.total_interventions,
                "total_emergency_stops": monitor_diag.total_emergency_stops,
            },
            "perception": bridge_stats["perception_stats"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        report_path = session_dir / "session_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


class SimulatedDeploymentRunner(DeploymentRunner):
    """
    Convenience class for testing the full deployment pipeline
    using the simulated environment.

    Runs through all 5 scenarios and validates that the deployed
    agent + safety monitor + perception pipeline work end-to-end.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        scenarios: Optional[list[str]] = None,
        steps_per_scenario: int = 1000,
        **kwargs,
    ) -> None:
        self.scenarios = scenarios or list(cfg.SCENARIO_IDS)
        self.steps_per_scenario = steps_per_scenario

        super().__init__(
            checkpoint_path=checkpoint_path,
            sensor_interface=SimulatedSensorInterface(),
            actuator_interface=SimulatedActuator(),
            max_steps=steps_per_scenario,
            **kwargs,
        )

    def run_all_scenarios(self) -> dict:
        """
        Run the deployment pipeline through all 5 scenarios.

        Returns:
            Dictionary with per-scenario results.
        """
        logger.info("=" * 70)
        logger.info("SIMULATED DEPLOYMENT: All Scenarios")
        logger.info("=" * 70)

        results = {}
        for scn_id in self.scenarios:
            logger.info(f"\n--- Scenario: {scn_id} ---")
            self.scenario_id = scn_id
            self.max_steps = self.steps_per_scenario

            # Reset simulated sensor environment
            if isinstance(self.bridge.sensors, SimulatedSensorInterface):
                self.bridge.sensors.reset(scn_id)

            # Reset pipeline and monitor
            self.bridge.pipeline.reset()
            self.bridge.monitor.reset()
            self.bridge._delta_prev = 0.0
            self.bridge._action_prev = 0.0

            # Re-initialise
            self.bridge.monitor.initialise()
            self._telemetry_log = []
            self._running = False
            self._shutdown_requested = False

            result = self.run()
            results[scn_id] = result

            # Extract key metrics
            if self._telemetry_log:
                e_lats = [t.get("e_lat_m", 0.0) for t in self._telemetry_log]
                max_elat = max(abs(e) for e in e_lats)
                rmse = float(np.sqrt(np.mean(np.array(e_lats) ** 2)))
                logger.info(
                    f"  {scn_id}: RMSE={rmse:.4f}m, max_elat={max_elat:.4f}m"
                )

        return results
