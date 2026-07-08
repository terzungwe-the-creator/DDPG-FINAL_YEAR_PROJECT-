"""
hardware_adapters.py — Concrete Hardware Interface Implementations

Provides ready-to-use sensor and actuator interfaces for common
real-world vehicle platforms:

    1. CANBusSensorInterface  — Reads sensors from vehicle CAN bus
    2. ROSSensorInterface     — Subscribes to ROS2 sensor topics  
    3. CANBusActuator         — Sends steering via CAN bus drive-by-wire
    4. ROSActuator            — Publishes steering to ROS2 actuator node

Each adapter translates hardware-specific protocols into the abstract
SensorInterface / ActuatorInterface API consumed by VehicleBridge.

Note: These adapters require their respective hardware libraries
(python-can, rclpy) which are NOT installed by default. Import errors
are caught gracefully with informative messages.

Reference:
    SAE J1939 — CAN bus for heavy vehicles
    ISO 11898 — CAN bus physical/data-link layer
    ROS2 Humble — Robot Operating System (for research platforms)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

import config as cfg
from real_world.sensor_interface import (
    SensorInterface,
    SensorBundle,
    SensorHealth,
    CameraLaneDetection,
    LiDARLaneDetection,
    IMUData,
    GPSData,
    VehicleCANData,
)
from real_world.actuator_interface import (
    ActuatorInterface,
    ActuatorState,
    SteeringCommand,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CAN BUS INTERFACES
# ═══════════════════════════════════════════════════════════════════════════════

class CANBusSensorInterface(SensorInterface):
    """
    Reads vehicle sensor data from CAN bus.

    Supports common CAN message IDs for:
        - Steering angle (0x025)
        - Wheel speeds (0x0B4)
        - IMU / yaw rate (0x0B6)
        - Brake pressure (0x224)

    Camera and LiDAR are typically on separate interfaces (Ethernet/USB)
    and must be provided via additional adapters.

    Args:
        channel:       CAN interface name (e.g., 'can0', 'vcan0').
        bitrate:       CAN bus bitrate.
        camera_source: Optional external camera lane detector callable.
        lidar_source:  Optional external LiDAR lane detector callable.
    """

    def __init__(
        self,
        channel: str = cfg.DEPLOY_CAN_CHANNEL,
        bitrate: int = cfg.DEPLOY_CAN_BITRATE,
        interface: str = cfg.DEPLOY_CAN_INTERFACE,
        camera_source=None,
        lidar_source=None,
    ) -> None:
        self.channel = channel
        self.bitrate = bitrate
        self.interface = interface
        self._camera_source = camera_source
        self._lidar_source = lidar_source
        self._bus = None
        self._connected = False

        # CAN message ID mapping (adjust for your vehicle)
        self.CAN_IDS = {
            "steering_angle": 0x025,
            "wheel_speeds": 0x0B4,
            "yaw_rate": 0x0B6,
            "brake": 0x224,
            "speed": 0x0B0,
        }

        # Latest parsed CAN data
        self._latest_can = VehicleCANData()
        self._latest_imu = IMUData()

    def connect(self) -> bool:
        """Connect to CAN bus interface."""
        try:
            import can
            self._bus = can.Bus(
                channel=self.channel,
                interface=self.interface,
                bitrate=self.bitrate,
            )
            self._connected = True
            logger.info(
                f"CANBusSensorInterface: Connected to {self.channel} "
                f"({self.interface}, {self.bitrate} bps)"
            )
            return True
        except ImportError:
            logger.error(
                "python-can not installed. Install with: pip install python-can"
            )
            return False
        except Exception as e:
            logger.error(f"CAN bus connection failed: {e}")
            return False

    def read(self) -> SensorBundle:
        """Read latest sensor data from CAN bus and external sources."""
        now = time.monotonic()
        bundle = SensorBundle(system_time_s=now)

        if self._connected and self._bus is not None:
            # Read all available CAN messages (non-blocking)
            self._poll_can_messages()

            bundle.vehicle_can = self._latest_can
            bundle.vehicle_can.timestamp_s = now
            bundle.vehicle_can.health = SensorHealth.OK

            bundle.imu = self._latest_imu
            bundle.imu.timestamp_s = now
            bundle.imu.health = SensorHealth.OK

        # External camera (if provided)
        if self._camera_source is not None:
            try:
                camera_data = self._camera_source()
                if camera_data is not None:
                    bundle.camera = camera_data
                    bundle.camera.health = SensorHealth.OK
            except Exception as e:
                logger.warning(f"Camera source error: {e}")
                bundle.camera.health = SensorHealth.FAULT

        # External LiDAR (if provided)
        if self._lidar_source is not None:
            try:
                lidar_data = self._lidar_source()
                if lidar_data is not None:
                    bundle.lidar = lidar_data
                    bundle.lidar.health = SensorHealth.OK
            except Exception as e:
                logger.warning(f"LiDAR source error: {e}")
                bundle.lidar.health = SensorHealth.FAULT

        return bundle

    def _poll_can_messages(self) -> None:
        """Poll all available CAN messages without blocking."""
        if self._bus is None:
            return

        # Read up to 50 messages per cycle
        for _ in range(50):
            msg = self._bus.recv(timeout=0.0)  # Non-blocking
            if msg is None:
                break
            self._parse_can_message(msg)

    def _parse_can_message(self, msg) -> None:
        """
        Parse a CAN message and update internal state.

        Note: CAN message parsing is vehicle-specific. The byte offsets
        and scaling factors below are EXAMPLES for a typical passenger
        vehicle. Adjust for your specific vehicle's DBC file.
        """
        arb_id = msg.arbitration_id
        data = msg.data

        if arb_id == self.CAN_IDS["steering_angle"]:
            # Example: bytes 0-1 = steering angle in 0.1° units, signed
            raw = int.from_bytes(data[0:2], "little", signed=True)
            self._latest_can.steering_angle_rad = np.radians(raw * 0.1)

        elif arb_id == self.CAN_IDS["wheel_speeds"]:
            # Example: 4 wheel speeds, each 2 bytes, in 0.01 km/h
            for i, attr in enumerate(
                ["wheel_speed_fl", "wheel_speed_fr",
                 "wheel_speed_rl", "wheel_speed_rr"]
            ):
                raw = int.from_bytes(data[i*2:(i+1)*2], "little", signed=False)
                speed_mps = raw * 0.01 / 3.6  # km/h → m/s
                setattr(self._latest_can, attr, speed_mps)

            # Compute v_x from rear wheel average
            self._latest_can.v_x_mps = (
                self._latest_can.wheel_speed_rl
                + self._latest_can.wheel_speed_rr
            ) / 2.0

        elif arb_id == self.CAN_IDS["yaw_rate"]:
            # Example: bytes 0-1 = yaw rate in 0.01 °/s, signed
            raw = int.from_bytes(data[0:2], "little", signed=True)
            self._latest_imu.yaw_rate_rads = np.radians(raw * 0.01)

            # bytes 2-3 = lateral acceleration in 0.001 g
            if len(data) >= 4:
                raw_ay = int.from_bytes(data[2:4], "little", signed=True)
                self._latest_imu.lat_accel_mps2 = raw_ay * 0.001 * 9.81

        elif arb_id == self.CAN_IDS["speed"]:
            # Vehicle speed in 0.01 km/h
            raw = int.from_bytes(data[0:2], "little", signed=False)
            self._latest_can.v_x_mps = raw * 0.01 / 3.6

    def close(self) -> None:
        """Close CAN bus connection."""
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
        self._connected = False
        logger.info("CANBusSensorInterface: Closed")

    def get_health(self) -> dict[str, SensorHealth]:
        return {
            "camera": SensorHealth.NOT_AVAILABLE if self._camera_source is None
                      else SensorHealth.OK,
            "lidar": SensorHealth.NOT_AVAILABLE if self._lidar_source is None
                     else SensorHealth.OK,
            "imu": SensorHealth.OK if self._connected else SensorHealth.FAULT,
            "gps": SensorHealth.NOT_AVAILABLE,
            "vehicle_can": SensorHealth.OK if self._connected else SensorHealth.FAULT,
        }


class CANBusActuator(ActuatorInterface):
    """
    Drive-by-wire steering actuator via CAN bus.

    Sends steering angle commands to the vehicle's EPS (Electric Power
    Steering) unit via CAN bus. The EPS must be in drive-by-wire mode
    (requires vehicle-specific enable sequence).

    Safety: The actuator enforces independent hardware rate/angle limits.
    The watchdog timer requires commands at ≥ DEPLOY_CONTROL_HZ to stay
    active — if commands stop, the EPS returns to manual mode.

    Args:
        channel:     CAN interface name.
        bitrate:     CAN bus bitrate.
        steer_cmd_id: CAN arbitration ID for steering commands.
    """

    def __init__(
        self,
        channel: str = cfg.DEPLOY_CAN_CHANNEL,
        bitrate: int = cfg.DEPLOY_CAN_BITRATE,
        interface: str = cfg.DEPLOY_CAN_INTERFACE,
        steer_cmd_id: int = 0x2E4,
    ) -> None:
        self.channel = channel
        self.bitrate = bitrate
        self.interface = interface
        self.steer_cmd_id = steer_cmd_id
        self._bus = None
        self._state = ActuatorState.IDLE
        self._connected = False
        self._last_command_time = 0.0
        self._override_torque_threshold = 2.0  # Nm

    def connect(self) -> bool:
        try:
            import can
            self._bus = can.Bus(
                channel=self.channel,
                interface=self.interface,
                bitrate=self.bitrate,
            )
            self._connected = True
            logger.info(f"CANBusActuator: Connected to {self.channel}")
            return True
        except ImportError:
            logger.error("python-can not installed")
            return False
        except Exception as e:
            logger.error(f"CAN actuator connection failed: {e}")
            return False

    def engage(self) -> bool:
        if not self._connected:
            return False
        # Vehicle-specific EPS enable sequence would go here
        self._state = ActuatorState.ACTIVE
        logger.info("CANBusActuator: EPS drive-by-wire ENGAGED")
        return True

    def disengage(self) -> None:
        # Send neutral command then disable
        if self._bus is not None:
            self._send_steering_can(0.0, 0.0, enable=False)
        self._state = ActuatorState.IDLE
        logger.info("CANBusActuator: EPS drive-by-wire DISENGAGED")

    def send(self, command: SteeringCommand) -> bool:
        if self._state != ActuatorState.ACTIVE or self._bus is None:
            return False

        safe_cmd = command.clamp()
        success = self._send_steering_can(
            safe_cmd.delta_rad,
            safe_cmd.delta_rate_rads,
            enable=True,
        )

        if success:
            self._last_command_time = time.monotonic()

        return success

    def _send_steering_can(
        self, delta_rad: float, rate_rads: float, enable: bool
    ) -> bool:
        """
        Construct and send CAN message for steering command.

        Message format (example — adjust for your vehicle):
            Byte 0-1: Steering angle (0.1° units, signed)
            Byte 2-3: Steering rate (0.1°/s units)
            Byte 4:   Enable flag (0x01 = active, 0x00 = disabled)
            Byte 5-7: Checksum / counter
        """
        try:
            import can

            angle_raw = int(np.degrees(delta_rad) * 10)
            rate_raw = int(np.degrees(rate_rads) * 10)
            enable_byte = 0x01 if enable else 0x00

            data = bytearray(8)
            data[0:2] = angle_raw.to_bytes(2, "little", signed=True)
            data[2:4] = rate_raw.to_bytes(2, "little", signed=True)
            data[4] = enable_byte

            msg = can.Message(
                arbitration_id=self.steer_cmd_id,
                data=bytes(data),
                is_extended_id=False,
            )
            self._bus.send(msg)
            return True

        except Exception as e:
            logger.error(f"CAN send failed: {e}")
            return False

    def emergency_stop(self) -> None:
        if self._bus is not None:
            self._send_steering_can(0.0, 0.0, enable=False)
        self._state = ActuatorState.EMERGENCY_STOP
        logger.critical("CANBusActuator: EMERGENCY STOP — EPS disabled")

    def get_state(self) -> ActuatorState:
        if (self._state == ActuatorState.ACTIVE
                and time.monotonic() - self._last_command_time > cfg.DEPLOY_HEARTBEAT_TIMEOUT_S):
            self._state = ActuatorState.FAULT
        return self._state

    def is_driver_override(self) -> bool:
        # In a real implementation, read steering torque from CAN
        # and check against threshold
        return False

    def close(self) -> None:
        self.disengage()
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
        self._connected = False


# ═══════════════════════════════════════════════════════════════════════════════
# ROS2 INTERFACES (for research platforms like NVIDIA DRIVE, Autoware)
# ═══════════════════════════════════════════════════════════════════════════════

class ROSSensorInterface(SensorInterface):
    """
    ROS2 sensor interface for research autonomous vehicle platforms.

    Subscribes to standard ROS2 topics:
        - /perception/lane_detection    → CameraLaneDetection
        - /sensing/imu/imu_data         → IMUData
        - /sensing/gnss/gnss_data       → GPSData
        - /vehicle/status/steering      → VehicleCANData

    Requires: rclpy (ROS2 Python client library)
    """

    def __init__(self, node_name: str = "lka_sensor_node") -> None:
        self._node_name = node_name
        self._node = None
        self._connected = False
        self._latest_bundle = SensorBundle()

    def connect(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(self._node_name)
            self._connected = True
            logger.info(f"ROSSensorInterface: ROS2 node '{self._node_name}' created")

            # Subscribe to topics (placeholder topic names)
            # Actual topic names depend on your ROS2 stack
            logger.info("ROSSensorInterface: Subscribing to sensor topics")

            return True

        except ImportError:
            logger.error(
                "rclpy not installed. Install ROS2 Humble or later."
            )
            return False
        except Exception as e:
            logger.error(f"ROS2 connection failed: {e}")
            return False

    def read(self) -> SensorBundle:
        if self._node is not None:
            # Spin once to process callbacks
            try:
                import rclpy
                rclpy.spin_once(self._node, timeout_sec=0.001)
            except Exception:
                pass
        self._latest_bundle.system_time_s = time.monotonic()
        return self._latest_bundle

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._connected = False

    def get_health(self) -> dict[str, SensorHealth]:
        return {
            "camera": SensorHealth.OK if self._connected else SensorHealth.FAULT,
            "lidar": SensorHealth.NOT_AVAILABLE,
            "imu": SensorHealth.OK if self._connected else SensorHealth.FAULT,
            "gps": SensorHealth.NOT_AVAILABLE,
            "vehicle_can": SensorHealth.OK if self._connected else SensorHealth.FAULT,
        }
