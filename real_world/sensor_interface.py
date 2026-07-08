"""
sensor_interface.py — Abstract Sensor Interfaces for Real-World Deployment

Defines the data structures and abstract base classes for all sensor inputs
required by the lane keeping system. Concrete implementations connect to
actual hardware (cameras, LiDAR, IMU, GPS) via CAN bus, ROS, or direct APIs.

The sensor interface ensures that ALL real-world perception data flows through
a single, well-defined API before reaching the perception pipeline, which
converts it into the canonical 8D observation vector.

Reference:
    ISO 26262:2018 §6 — sensor data integrity requirements
    UNECE WP.29 R157 §5.2 — sensor specification for ALKS

Author: Principal Autonomous Systems Engineer
Version: 3.0 — Real-World Deployment
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


class SensorHealth(Enum):
    """Sensor operational state per ISO 26262 diagnostic coverage."""
    OK = auto()
    DEGRADED = auto()      # Partial data loss — reduced confidence
    STALE = auto()         # Data older than DEPLOY_SENSOR_TIMEOUT_S
    FAULT = auto()         # Hardware fault — sensor offline
    NOT_AVAILABLE = auto() # Sensor not configured


@dataclass
class CameraLaneDetection:
    """
    Lane detection output from forward-facing camera system.

    This is the PRIMARY sensor for lateral error estimation.
    Typically produced by a CNN lane detector (e.g., LaneNet, CLRNet,
    TuSimple-style) running on the vehicle's compute platform.

    Attributes:
        timestamp_s:    Sensor timestamp (monotonic clock, seconds).
        e_lat_m:        Estimated lateral offset from lane centre (m, signed).
                        Positive = right of centre.
        e_psi_rad:      Estimated heading error relative to lane tangent (rad).
        confidence:     Detection confidence in [0, 1]. Below 0.5 triggers
                        degraded mode.
        lane_width_m:   Detected lane width (m). Used for plausibility check.
        curvature_1m:   Estimated road curvature at current position (1/m).
        curvature_la1:  Curvature at 1-second lookahead (1/m).
        curvature_la2:  Curvature at 2-second lookahead (1/m).
        left_marking:   Left lane marking type ('solid', 'dashed', 'none').
        right_marking:  Right lane marking type.
        health:         Sensor health status.
    """
    timestamp_s: float = 0.0
    e_lat_m: float = 0.0
    e_psi_rad: float = 0.0
    confidence: float = 0.0
    lane_width_m: float = 3.5
    curvature_1m: float = 0.0
    curvature_la1: float = 0.0
    curvature_la2: float = 0.0
    left_marking: str = "solid"
    right_marking: str = "solid"
    health: SensorHealth = SensorHealth.NOT_AVAILABLE


@dataclass
class LiDARLaneDetection:
    """
    Lane boundary detection from LiDAR point cloud processing.

    SECONDARY sensor — fused with camera for robustness.
    LiDAR provides better geometric accuracy than cameras in adverse
    lighting but has lower lane marking detection confidence.

    Attributes:
        timestamp_s:    Sensor timestamp (s).
        e_lat_m:        Lateral offset estimate from point cloud lane fit (m).
        confidence:     Detection confidence in [0, 1].
        left_boundary:  Left lane boundary as (x, y) polyline in vehicle frame.
        right_boundary: Right lane boundary as (x, y) polyline.
        health:         Sensor health status.
    """
    timestamp_s: float = 0.0
    e_lat_m: float = 0.0
    confidence: float = 0.0
    left_boundary: Optional[np.ndarray] = None    # shape (N, 2)
    right_boundary: Optional[np.ndarray] = None   # shape (N, 2)
    health: SensorHealth = SensorHealth.NOT_AVAILABLE


@dataclass
class IMUData:
    """
    Inertial Measurement Unit data for vehicle dynamics estimation.

    Provides yaw rate, lateral acceleration, and longitudinal acceleration
    at high frequency (typically 100–400 Hz). This is the PRIMARY source
    for yaw rate (r) and lateral velocity estimation (v_y via integration).

    Attributes:
        timestamp_s:    Sensor timestamp (s).
        yaw_rate_rads:  Yaw rate about vertical axis (rad/s, positive = left turn).
        lat_accel_mps2: Lateral acceleration (m/s², positive = left).
        lon_accel_mps2: Longitudinal acceleration (m/s², positive = forward).
        roll_rate_rads: Roll rate (rad/s) — used for road bank compensation.
        health:         Sensor health status.
    """
    timestamp_s: float = 0.0
    yaw_rate_rads: float = 0.0
    lat_accel_mps2: float = 0.0
    lon_accel_mps2: float = 0.0
    roll_rate_rads: float = 0.0
    health: SensorHealth = SensorHealth.NOT_AVAILABLE


@dataclass
class GPSData:
    """
    GNSS position and velocity for map matching and speed reference.

    Attributes:
        timestamp_s:    Sensor timestamp (s).
        latitude_deg:   WGS84 latitude (degrees).
        longitude_deg:  WGS84 longitude (degrees).
        altitude_m:     Altitude above WGS84 ellipsoid (m).
        speed_mps:      Ground speed (m/s).
        heading_deg:    True heading (degrees, 0 = North, clockwise).
        hdop:           Horizontal dilution of precision.
        num_satellites: Number of satellites in fix.
        health:         Sensor health status.
    """
    timestamp_s: float = 0.0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    hdop: float = 99.0
    num_satellites: int = 0
    health: SensorHealth = SensorHealth.NOT_AVAILABLE


@dataclass
class VehicleCANData:
    """
    Vehicle CAN bus data — steering angle, wheel speeds, etc.

    This data comes from the vehicle's own sensors via OBD-II or
    manufacturer CAN bus.

    Attributes:
        timestamp_s:        Sensor timestamp (s).
        steering_angle_rad: Current steering wheel angle (rad, at the wheel).
        steering_rate_rads: Steering rate (rad/s).
        wheel_speed_fl:     Front-left wheel speed (m/s).
        wheel_speed_fr:     Front-right wheel speed (m/s).
        wheel_speed_rl:     Rear-left wheel speed (m/s).
        wheel_speed_rr:     Rear-right wheel speed (m/s).
        v_x_mps:            Longitudinal velocity from wheel speeds (m/s).
        brake_pressure_bar: Brake pressure (bar) — for safety monitoring.
        gear:               Current gear (0=P, 1=R, 2=N, 3=D).
        health:             Sensor health status.
    """
    timestamp_s: float = 0.0
    steering_angle_rad: float = 0.0
    steering_rate_rads: float = 0.0
    wheel_speed_fl: float = 0.0
    wheel_speed_fr: float = 0.0
    wheel_speed_rl: float = 0.0
    wheel_speed_rr: float = 0.0
    v_x_mps: float = 0.0
    brake_pressure_bar: float = 0.0
    gear: int = 3
    health: SensorHealth = SensorHealth.NOT_AVAILABLE


@dataclass
class SensorBundle:
    """
    Complete sensor data bundle for one control cycle.

    Aggregates all sensor inputs into a single timestamped snapshot.
    The perception pipeline consumes this to produce the 8D observation.
    """
    system_time_s: float = 0.0
    camera: CameraLaneDetection = field(default_factory=CameraLaneDetection)
    lidar: LiDARLaneDetection = field(default_factory=LiDARLaneDetection)
    imu: IMUData = field(default_factory=IMUData)
    gps: GPSData = field(default_factory=GPSData)
    vehicle_can: VehicleCANData = field(default_factory=VehicleCANData)

    @property
    def is_valid(self) -> bool:
        """Check if minimum required sensors are healthy."""
        # Minimum: camera OR lidar for lane detection, plus IMU for dynamics
        has_lane = (
            self.camera.health in (SensorHealth.OK, SensorHealth.DEGRADED)
            or self.lidar.health in (SensorHealth.OK, SensorHealth.DEGRADED)
        )
        has_dynamics = self.imu.health in (SensorHealth.OK, SensorHealth.DEGRADED)
        has_speed = (
            self.vehicle_can.health in (SensorHealth.OK, SensorHealth.DEGRADED)
            or self.gps.health in (SensorHealth.OK, SensorHealth.DEGRADED)
        )
        return has_lane and has_dynamics and has_speed

    @property
    def max_age_s(self) -> float:
        """Return the age of the oldest sensor reading."""
        now = self.system_time_s
        ages = []
        if self.camera.health != SensorHealth.NOT_AVAILABLE:
            ages.append(now - self.camera.timestamp_s)
        if self.imu.health != SensorHealth.NOT_AVAILABLE:
            ages.append(now - self.imu.timestamp_s)
        if self.vehicle_can.health != SensorHealth.NOT_AVAILABLE:
            ages.append(now - self.vehicle_can.timestamp_s)
        return max(ages) if ages else float("inf")


class SensorInterface(ABC):
    """
    Abstract base class for real-world sensor hardware interfaces.

    Concrete implementations handle the actual hardware communication
    (CAN bus polling, ROS subscriber, camera frame grabber, etc.).

    Subclasses must implement:
        - connect()  — establish hardware connection
        - read()     — poll latest sensor data
        - close()    — release hardware resources
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to sensor hardware.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    def read(self) -> SensorBundle:
        """
        Read the latest sensor data from all configured sensors.

        Returns:
            SensorBundle with the latest readings from all sensors.
            Unavailable sensors will have health = NOT_AVAILABLE.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all sensor hardware resources."""
        ...

    @abstractmethod
    def get_health(self) -> dict[str, SensorHealth]:
        """
        Return health status of all sensors.

        Returns:
            Dictionary mapping sensor name to SensorHealth enum.
        """
        ...


class SimulatedSensorInterface(SensorInterface):
    """
    Simulated sensor interface for testing the deployment pipeline.

    Wraps the bicycle model environment to produce realistic sensor
    data without requiring actual hardware. This allows end-to-end
    testing of the full deployment pipeline on a development machine.
    """

    def __init__(self, env=None) -> None:
        from simulator.lane_keeping_env import LaneKeepingEnv
        self.env = env or LaneKeepingEnv(training_mode=False)
        self._connected = False
        self._last_obs = None
        self._step_count = 0

    def connect(self) -> bool:
        """Initialise the simulated environment."""
        try:
            self._last_obs, _ = self.env.reset(scenario_id="SCN-01")
            self._connected = True
            logger.info("SimulatedSensorInterface: Connected to bicycle model")
            return True
        except Exception as e:
            logger.error(f"SimulatedSensorInterface: Connection failed: {e}")
            return False

    def read(self) -> SensorBundle:
        """Read simulated sensor data from the environment state."""
        if not self._connected:
            return SensorBundle()

        now = time.monotonic()
        vehicle = self.env.vehicle
        profile = self.env.profiles[self.env.current_scn]

        # Camera-like lane detection from ground truth
        kappa_ref = profile.get_kappa_at_s(self.env.arc_length_s)
        kappa_la1, kappa_la2 = profile.get_lookahead_kappa(
            self.env.arc_length_s, vehicle.v_x
        )

        camera = CameraLaneDetection(
            timestamp_s=now,
            e_lat_m=vehicle.lateral_error,
            e_psi_rad=vehicle.heading_error,
            confidence=0.95,
            lane_width_m=cfg.LANE_WIDTH,
            curvature_1m=kappa_ref,
            curvature_la1=kappa_la1,
            curvature_la2=kappa_la2,
            health=SensorHealth.OK,
        )

        imu = IMUData(
            timestamp_s=now,
            yaw_rate_rads=vehicle.yaw_rate,
            lat_accel_mps2=0.0,
            health=SensorHealth.OK,
        )

        can = VehicleCANData(
            timestamp_s=now,
            steering_angle_rad=self.env.delta_prev,
            v_x_mps=vehicle.v_x,
            health=SensorHealth.OK,
        )

        gps = GPSData(
            timestamp_s=now,
            speed_mps=vehicle.v_x,
            health=SensorHealth.OK,
        )

        return SensorBundle(
            system_time_s=now,
            camera=camera,
            imu=imu,
            vehicle_can=can,
            gps=gps,
        )

    def step(self, action: np.ndarray) -> dict:
        """Step the simulated environment (used by deployment runner)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._step_count += 1
        return info

    def reset(self, scenario_id: str = "SCN-01") -> None:
        """Reset the simulated environment."""
        self._last_obs, _ = self.env.reset(scenario_id=scenario_id)
        self._step_count = 0

    def close(self) -> None:
        """Close the simulated environment."""
        self.env.close()
        self._connected = False

    def get_health(self) -> dict[str, SensorHealth]:
        return {
            "camera": SensorHealth.OK,
            "lidar": SensorHealth.NOT_AVAILABLE,
            "imu": SensorHealth.OK,
            "gps": SensorHealth.OK,
            "vehicle_can": SensorHealth.OK,
        }
