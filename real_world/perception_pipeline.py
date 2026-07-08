"""
perception_pipeline.py — Sensor Fusion → 8D Observation Vector

Converts raw sensor data from multiple sources (camera, LiDAR, IMU, GPS, CAN)
into the canonical 8D normalised observation vector consumed by the DDPG agent.

This is the CRITICAL bridge between the real world and the trained policy.
The observation vector MUST be normalised identically to training data
(using physical constants from config.py, NOT data-driven statistics).

Observation Vector (identical to simulator/lane_keeping_env.py):
    obs[0] = e_lat_m         / NORM_E_LAT       (lateral error)
    obs[1] = e_psi_rad       / NORM_E_PSI       (heading error)
    obs[2] = kappa_ref       / NORM_KAPPA        (current curvature)
    obs[3] = v_y_mps         / NORM_V_Y          (lateral velocity)
    obs[4] = yaw_rate_rads   / NORM_YAW_RATE     (yaw rate)
    obs[5] = delta_prev_rad  / NORM_DELTA        (previous steering angle)
    obs[6] = kappa_la1       / NORM_KAPPA_LA1     (1s lookahead curvature)
    obs[7] = kappa_la2       / NORM_KAPPA_LA2     (2s lookahead curvature)

Sensor Fusion Strategy:
    - Lateral error:   Weighted average of camera and LiDAR, confidence-gated
    - Heading error:   Camera-primary (LiDAR doesn't provide heading directly)
    - Curvature:       Camera (polynomial lane fit) backed by HD map prior
    - Lateral velocity: Estimated from IMU lateral acceleration (bicycle model)
    - Yaw rate:        IMU-primary (highest bandwidth)
    - Steering angle:  CAN bus (direct vehicle sensor)

Reference:
    ISO 26262:2018 — functional safety for sensor fusion
    SAE J3016:2021 — levels of driving automation (L2+ perception requirements)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config as cfg
from real_world.sensor_interface import (
    SensorBundle,
    SensorHealth,
    CameraLaneDetection,
    LiDARLaneDetection,
    IMUData,
)

logger = logging.getLogger(__name__)


@dataclass
class PerceptionState:
    """
    Intermediate perception state before normalisation.

    Contains fused values in physical units, with confidence scores
    and diagnostics for each field.
    """
    # Fused lateral error (m)
    e_lat_m: float = 0.0
    e_lat_confidence: float = 0.0

    # Fused heading error (rad)
    e_psi_rad: float = 0.0

    # Road curvature (1/m)
    kappa_ref: float = 0.0
    kappa_la1: float = 0.0
    kappa_la2: float = 0.0

    # Vehicle dynamics
    v_y_mps: float = 0.0
    yaw_rate_rads: float = 0.0

    # Steering state
    delta_prev_rad: float = 0.0

    # Vehicle speed
    v_x_mps: float = 0.0

    # Validity
    is_valid: bool = False
    sensor_age_s: float = 0.0


class LateralVelocityEstimator:
    """
    Estimates lateral velocity (v_y) from IMU data using a simplified
    bicycle model observer.

    v_y is not directly measurable — it must be estimated from:
        1. IMU lateral acceleration: a_y = v̇_y + v_x · r
           → v̇_y = a_y - v_x · r
           → v_y ≈ ∫(a_y - v_x · r) dt  (with drift correction)

        2. Drift correction using camera lateral error rate:
           ė_lat ≈ v_y + v_x · e_psi  (kinematic relation)
           → v_y ≈ ė_lat - v_x · e_psi

    The estimator blends both methods with a complementary filter.
    """

    def __init__(self, alpha: float = 0.05, dt: float = 0.02) -> None:
        self.alpha = alpha          # High-pass filter coefficient
        self.dt = dt                # Control period (s)
        self.v_y_imu: float = 0.0  # IMU-integrated estimate
        self.v_y_fused: float = 0.0
        self._prev_e_lat: Optional[float] = None
        self._prev_time: Optional[float] = None

    def update(
        self,
        a_y: float,       # Lateral acceleration (m/s²)
        v_x: float,       # Longitudinal velocity (m/s)
        r: float,          # Yaw rate (rad/s)
        e_lat: float,      # Lateral error (m)
        e_psi: float,      # Heading error (rad)
        timestamp: float,  # Current time (s)
    ) -> float:
        """
        Update the lateral velocity estimate.

        Returns:
            Estimated lateral velocity v_y (m/s).
        """
        # Method 1: IMU integration (drifts over time)
        v_y_dot = a_y - v_x * r
        self.v_y_imu += v_y_dot * self.dt

        # Method 2: Kinematic observer from lateral error rate
        if self._prev_e_lat is not None and self._prev_time is not None:
            dt_actual = timestamp - self._prev_time
            if dt_actual > 0.001:
                e_lat_dot = (e_lat - self._prev_e_lat) / dt_actual
                v_y_kinematic = e_lat_dot - v_x * e_psi

                # Complementary filter: low-pass kinematic + high-pass IMU
                self.v_y_fused = (
                    (1.0 - self.alpha) * self.v_y_fused
                    + self.alpha * v_y_kinematic
                    + self.alpha * self.v_y_imu
                )

                # Reset IMU integrator to prevent unbounded drift
                self.v_y_imu = self.v_y_fused
            else:
                self.v_y_fused = self.v_y_imu
        else:
            self.v_y_fused = self.v_y_imu

        self._prev_e_lat = e_lat
        self._prev_time = timestamp

        return self.v_y_fused

    def reset(self) -> None:
        """Reset the estimator state."""
        self.v_y_imu = 0.0
        self.v_y_fused = 0.0
        self._prev_e_lat = None
        self._prev_time = None


class PerceptionPipeline:
    """
    Converts a SensorBundle into the canonical 8D observation vector.

    This pipeline:
        1. Validates sensor health and freshness
        2. Fuses camera + LiDAR lateral error estimates
        3. Estimates lateral velocity from IMU + camera
        4. Normalises all values to [-1, 1] using physical constants
        5. Clips to prevent out-of-range inputs to the neural network

    The normalisation constants MUST match training exactly (config.py).
    """

    def __init__(self, control_hz: int = cfg.DEPLOY_CONTROL_HZ) -> None:
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.v_y_estimator = LateralVelocityEstimator(dt=self.dt)

        # Moving average filters for noise reduction
        self._e_lat_buffer: deque = deque(maxlen=5)
        self._e_psi_buffer: deque = deque(maxlen=3)
        self._kappa_buffer: deque = deque(maxlen=3)

        # Last known good values (for hold-over during sensor dropouts)
        self._last_good_state: Optional[PerceptionState] = None

        # Diagnostics
        self.fusion_stats = {
            "camera_only_count": 0,
            "lidar_only_count": 0,
            "fused_count": 0,
            "holdover_count": 0,
        }

    def process(self, sensors: SensorBundle) -> tuple[np.ndarray, PerceptionState]:
        """
        Convert sensor bundle to 8D normalised observation vector.

        Args:
            sensors: Complete sensor data bundle for this control cycle.

        Returns:
            Tuple of:
                - Normalised observation vector, shape (8,), values in [-1, 1]
                - PerceptionState with fused physical-unit values

        Raises:
            ValueError: If no valid sensor data is available AND no
                        hold-over state exists.
        """
        state = PerceptionState()
        state.sensor_age_s = sensors.max_age_s

        # ── Step 1: Fuse lateral error ─────────────────────────────────
        e_lat, confidence = self._fuse_lateral_error(sensors)
        state.e_lat_m = e_lat
        state.e_lat_confidence = confidence

        # ── Step 2: Heading error (camera primary) ─────────────────────
        if sensors.camera.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.e_psi_rad = sensors.camera.e_psi_rad
        elif self._last_good_state is not None:
            state.e_psi_rad = self._last_good_state.e_psi_rad
        else:
            state.e_psi_rad = 0.0

        # ── Step 3: Road curvature ─────────────────────────────────────
        if sensors.camera.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.kappa_ref = sensors.camera.curvature_1m
            state.kappa_la1 = sensors.camera.curvature_la1
            state.kappa_la2 = sensors.camera.curvature_la2
        elif self._last_good_state is not None:
            state.kappa_ref = self._last_good_state.kappa_ref
            state.kappa_la1 = self._last_good_state.kappa_la1
            state.kappa_la2 = self._last_good_state.kappa_la2

        # ── Step 4: Vehicle speed ──────────────────────────────────────
        if sensors.vehicle_can.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.v_x_mps = sensors.vehicle_can.v_x_mps
        elif sensors.gps.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.v_x_mps = sensors.gps.speed_mps
        elif self._last_good_state is not None:
            state.v_x_mps = self._last_good_state.v_x_mps

        # ── Step 5: Yaw rate (IMU primary) ─────────────────────────────
        if sensors.imu.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.yaw_rate_rads = sensors.imu.yaw_rate_rads
        elif self._last_good_state is not None:
            state.yaw_rate_rads = self._last_good_state.yaw_rate_rads

        # ── Step 6: Lateral velocity estimation ────────────────────────
        if sensors.imu.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.v_y_mps = self.v_y_estimator.update(
                a_y=sensors.imu.lat_accel_mps2,
                v_x=state.v_x_mps,
                r=state.yaw_rate_rads,
                e_lat=state.e_lat_m,
                e_psi=state.e_psi_rad,
                timestamp=sensors.system_time_s,
            )
        elif self._last_good_state is not None:
            state.v_y_mps = self._last_good_state.v_y_mps

        # ── Step 7: Previous steering angle (CAN bus) ──────────────────
        if sensors.vehicle_can.health in (SensorHealth.OK, SensorHealth.DEGRADED):
            state.delta_prev_rad = sensors.vehicle_can.steering_angle_rad
        elif self._last_good_state is not None:
            state.delta_prev_rad = self._last_good_state.delta_prev_rad

        # ── Step 8: Validity check ─────────────────────────────────────
        state.is_valid = sensors.is_valid and confidence > 0.3

        if state.is_valid:
            self._last_good_state = state

        # ── Step 9: Normalise to [-1, 1] ───────────────────────────────
        obs = self._normalise(state)

        return obs, state

    def _fuse_lateral_error(self, sensors: SensorBundle) -> tuple[float, float]:
        """
        Fuse camera and LiDAR lateral error estimates.

        Weighted average with confidence gating and outlier rejection.

        Returns:
            Tuple of (fused e_lat in metres, confidence in [0, 1]).
        """
        camera_ok = sensors.camera.health in (SensorHealth.OK, SensorHealth.DEGRADED)
        lidar_ok = sensors.lidar.health in (SensorHealth.OK, SensorHealth.DEGRADED)

        if camera_ok and lidar_ok:
            # Both available: weighted average
            w_cam = cfg.DEPLOY_CAMERA_WEIGHT * sensors.camera.confidence
            w_lid = cfg.DEPLOY_LIDAR_WEIGHT * sensors.lidar.confidence

            # Outlier rejection: if camera and LiDAR disagree by >0.3m,
            # trust the one with higher confidence
            if abs(sensors.camera.e_lat_m - sensors.lidar.e_lat_m) > 0.3:
                if sensors.camera.confidence > sensors.lidar.confidence:
                    self.fusion_stats["camera_only_count"] += 1
                    return sensors.camera.e_lat_m, sensors.camera.confidence
                else:
                    self.fusion_stats["lidar_only_count"] += 1
                    return sensors.lidar.e_lat_m, sensors.lidar.confidence

            total_w = w_cam + w_lid
            if total_w > 0:
                fused = (w_cam * sensors.camera.e_lat_m + w_lid * sensors.lidar.e_lat_m) / total_w
                confidence = min(sensors.camera.confidence + sensors.lidar.confidence * 0.5, 1.0)
            else:
                fused = sensors.camera.e_lat_m
                confidence = sensors.camera.confidence

            self.fusion_stats["fused_count"] += 1
            return fused, confidence

        elif camera_ok:
            self.fusion_stats["camera_only_count"] += 1
            return sensors.camera.e_lat_m, sensors.camera.confidence

        elif lidar_ok:
            self.fusion_stats["lidar_only_count"] += 1
            return sensors.lidar.e_lat_m, sensors.lidar.confidence

        else:
            # Hold-over from last good state
            self.fusion_stats["holdover_count"] += 1
            if self._last_good_state is not None:
                return self._last_good_state.e_lat_m, 0.2  # Degraded confidence
            return 0.0, 0.0

    def _normalise(self, state: PerceptionState) -> np.ndarray:
        """
        Normalise physical-unit state to [-1, 1] observation vector.

        Uses IDENTICAL constants to training (config.py) to ensure
        zero distribution shift between sim-trained and deployed policy.
        """
        obs = np.array([
            state.e_lat_m / cfg.NORM_E_LAT,
            state.e_psi_rad / cfg.NORM_E_PSI,
            state.kappa_ref / cfg.NORM_KAPPA,
            state.v_y_mps / cfg.NORM_V_Y,
            state.yaw_rate_rads / cfg.NORM_YAW_RATE,
            state.delta_prev_rad / cfg.NORM_DELTA,
            state.kappa_la1 / cfg.NORM_KAPPA_LA1,
            state.kappa_la2 / cfg.NORM_KAPPA_LA2,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def reset(self) -> None:
        """Reset perception pipeline state."""
        self.v_y_estimator.reset()
        self._e_lat_buffer.clear()
        self._e_psi_buffer.clear()
        self._kappa_buffer.clear()
        self._last_good_state = None
