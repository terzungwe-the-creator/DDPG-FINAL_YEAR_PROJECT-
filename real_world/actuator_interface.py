"""
actuator_interface.py — Abstract Actuator Interface for Real-World Steering

Defines the abstract base class for steering actuator hardware and the
SteeringCommand data structure. Concrete implementations communicate with
real drive-by-wire systems via CAN bus, ROS, or vendor-specific APIs.

Safety Architecture:
    The actuator interface enforces a HARDWARE-LEVEL safety envelope
    independent of the software safety guardian. This provides defense
    in depth per ISO 26262 ASIL-B:
        - Maximum steering angle clamp (hardware limit)
        - Maximum steering rate clamp (actuator physical limit)
        - Watchdog heartbeat (commands must arrive within timeout)
        - Emergency zero-torque mode (fail-safe default)

Reference:
    ISO 26262:2018 §8 — hardware-software interface specification
    UNECE WP.29 R79 — steering equipment requirements
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


class ActuatorState(Enum):
    """Actuator operational state."""
    IDLE = auto()           # Not engaged — driver has full control
    ACTIVE = auto()         # LKA steering active
    OVERRIDE = auto()       # Driver override detected (torque sensor)
    EMERGENCY_STOP = auto() # Emergency — steering neutralised
    FAULT = auto()          # Hardware fault


@dataclass
class SteeringCommand:
    """
    Steering command sent to the drive-by-wire actuator.

    All commands are rate-limited and angle-clamped before reaching
    the actuator. The actuator should ALSO enforce its own limits
    as a hardware safety layer.

    Attributes:
        timestamp_s:        Command timestamp (monotonic clock, seconds).
        delta_rad:          Desired front wheel steering angle (rad).
                            Positive = left turn (ISO convention).
        delta_rate_rads:    Desired steering rate (rad/s). Informational —
                            the actuator enforces its own rate limit.
        torque_nm:          Optional steering torque (Nm). Used by
                            torque-overlay systems instead of angle control.
        mode:               Command mode: 'angle' or 'torque'.
        priority:           Command priority: 'normal' or 'emergency'.
    """
    timestamp_s: float = 0.0
    delta_rad: float = 0.0
    delta_rate_rads: float = 0.0
    torque_nm: float = 0.0
    mode: str = "angle"
    priority: str = "normal"

    def clamp(self, max_angle: float = cfg.DELTA_MAX,
              max_rate: float = cfg.DEPLOY_MAX_STEERING_RATE,
              dt: float = 0.02) -> 'SteeringCommand':
        """
        Apply safety clamping to the steering command.

        Args:
            max_angle: Maximum allowed steering angle (rad).
            max_rate:  Maximum allowed steering rate (rad/s).
            dt:        Control period (s).

        Returns:
            New SteeringCommand with clamped values.
        """
        clamped_delta = float(np.clip(self.delta_rad, -max_angle, max_angle))
        clamped_rate = float(np.clip(self.delta_rate_rads, -max_rate, max_rate))

        return SteeringCommand(
            timestamp_s=self.timestamp_s,
            delta_rad=clamped_delta,
            delta_rate_rads=clamped_rate,
            torque_nm=self.torque_nm,
            mode=self.mode,
            priority=self.priority,
        )


class ActuatorInterface(ABC):
    """
    Abstract base class for drive-by-wire steering actuator.

    Concrete implementations handle the actual hardware communication
    (CAN bus message sending, ROS topic publishing, vendor API calls).

    The actuator interface provides:
        1. connect()    — establish actuator connection
        2. send()       — send a rate-limited steering command
        3. engage()     — enable LKA steering control
        4. disengage()  — release steering control to driver
        5. emergency()  — immediate steering neutralisation
        6. close()      — release hardware resources

    ISO 26262 ASIL-B requires that the actuator can be disengaged
    at any time by the driver applying steering torque (override).
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the steering actuator.

        Returns:
            True if connection successful.
        """
        ...

    @abstractmethod
    def engage(self) -> bool:
        """
        Enable LKA steering mode.

        The actuator should only engage if:
            - Connection is established
            - Vehicle speed is within [MIN_SPEED, MAX_SPEED]
            - No driver override detected
            - No active faults

        Returns:
            True if engagement successful.
        """
        ...

    @abstractmethod
    def disengage(self) -> None:
        """
        Smoothly release steering control back to the driver.

        The steering angle should ramp to zero over ~0.5s to avoid
        abrupt control transitions.
        """
        ...

    @abstractmethod
    def send(self, command: SteeringCommand) -> bool:
        """
        Send a steering command to the actuator.

        The command should be rate-limited and angle-clamped before
        being sent to the hardware. Returns False if the command
        could not be sent (connection lost, fault, etc.).

        Args:
            command: SteeringCommand to send.

        Returns:
            True if command was sent successfully.
        """
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """
        Immediately neutralise steering and disengage.

        This is the highest-priority command. It must:
            1. Set steering torque/angle to zero
            2. Disengage LKA mode
            3. Alert the driver (HMI signal)
        """
        ...

    @abstractmethod
    def get_state(self) -> ActuatorState:
        """Return current actuator state."""
        ...

    @abstractmethod
    def is_driver_override(self) -> bool:
        """
        Check if the driver is actively overriding the steering.

        Detected via steering torque sensor: if the driver applies
        torque above a threshold (~2 Nm), the system must immediately
        disengage per UNECE R79.

        Returns:
            True if driver override detected.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release actuator hardware resources."""
        ...


class SimulatedActuator(ActuatorInterface):
    """
    Simulated steering actuator for testing the deployment pipeline.

    Applies the steering command to the simulated environment without
    requiring any real hardware. Supports driver override simulation.
    """

    def __init__(self) -> None:
        self._state = ActuatorState.IDLE
        self._connected = False
        self._last_command: Optional[SteeringCommand] = None
        self._last_command_time: float = 0.0
        self._override_active = False
        self._command_history: list[SteeringCommand] = []

    def connect(self) -> bool:
        self._connected = True
        self._state = ActuatorState.IDLE
        logger.info("SimulatedActuator: Connected")
        return True

    def engage(self) -> bool:
        if not self._connected:
            return False
        self._state = ActuatorState.ACTIVE
        logger.info("SimulatedActuator: LKA engaged")
        return True

    def disengage(self) -> None:
        self._state = ActuatorState.IDLE
        logger.info("SimulatedActuator: LKA disengaged")

    def send(self, command: SteeringCommand) -> bool:
        if self._state != ActuatorState.ACTIVE:
            return False
        if self._override_active:
            self._state = ActuatorState.OVERRIDE
            return False

        # Apply safety clamping
        safe_cmd = command.clamp()
        self._last_command = safe_cmd
        self._last_command_time = time.monotonic()
        self._command_history.append(safe_cmd)

        # Keep bounded history
        if len(self._command_history) > 1000:
            self._command_history = self._command_history[-500:]

        return True

    def emergency_stop(self) -> None:
        self._state = ActuatorState.EMERGENCY_STOP
        self._last_command = SteeringCommand(
            timestamp_s=time.monotonic(),
            delta_rad=0.0,
            priority="emergency",
        )
        logger.warning("SimulatedActuator: EMERGENCY STOP")

    def get_state(self) -> ActuatorState:
        # Check watchdog timeout
        if (self._state == ActuatorState.ACTIVE
                and self._last_command_time > 0
                and time.monotonic() - self._last_command_time > cfg.DEPLOY_HEARTBEAT_TIMEOUT_S):
            logger.warning("SimulatedActuator: Watchdog timeout — disengaging")
            self._state = ActuatorState.FAULT
        return self._state

    def is_driver_override(self) -> bool:
        return self._override_active

    def simulate_driver_override(self, active: bool = True) -> None:
        """Simulate driver steering override for testing."""
        self._override_active = active
        if active:
            self._state = ActuatorState.OVERRIDE
            logger.info("SimulatedActuator: Driver override simulated")

    def close(self) -> None:
        self.disengage()
        self._connected = False
        logger.info("SimulatedActuator: Closed")
