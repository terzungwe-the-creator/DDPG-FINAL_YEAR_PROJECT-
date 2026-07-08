"""real_world/__init__.py — Real-World Deployment Package.

Provides the complete hardware abstraction layer for deploying the trained
DDPG lane keeping agent on real vehicles with real sensors and actuators.

Architecture:
    Sensors (Camera/LiDAR/IMU/GPS) → PerceptionPipeline → 8D Observation
    8D Observation → DDPGAgent → Normalised Action [-1, 1]
    Normalised Action → VehicleBridge → Steering Actuator (CAN/ROS)

Safety Layers:
    1. SafetyMonitor — real-time safety envelope (builds on SafetyGuardian)
    2. Watchdog timer — actuator heartbeat monitoring
    3. Emergency stop — immediate steering neutralisation
    4. Driver handoff — UNECE R157 transition demand

Note: Heavy imports (VehicleBridge, DeploymentRunner) are deferred to avoid
importing torch at package init time. Use explicit imports when needed.

Reference: ISO 26262:2018 ASIL-B — functional safety for ADAS
"""

from real_world.sensor_interface import (
    SensorInterface,
    CameraLaneDetection,
    IMUData,
    GPSData,
    LiDARLaneDetection,
)
from real_world.actuator_interface import ActuatorInterface, SteeringCommand
from real_world.perception_pipeline import PerceptionPipeline
from real_world.safety_monitor import RealWorldSafetyMonitor

# VehicleBridge and DeploymentRunner import DDPGAgent which requires torch.
# Lazy import to avoid torch DLL issues at package init time.
# Use: from real_world.vehicle_bridge import VehicleBridge
# Use: from real_world.deployment_runner import DeploymentRunner

__all__ = [
    "SensorInterface",
    "CameraLaneDetection",
    "IMUData",
    "GPSData",
    "LiDARLaneDetection",
    "ActuatorInterface",
    "SteeringCommand",
    "PerceptionPipeline",
    "RealWorldSafetyMonitor",
]
