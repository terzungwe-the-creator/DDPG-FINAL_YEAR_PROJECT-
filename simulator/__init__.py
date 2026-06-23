"""simulator/__init__.py — Simulator package."""

from simulator.vehicle_model import BicycleModel
from simulator.road_profiles import RoadProfile, build_all_profiles
from simulator.reward import compute_reward
from simulator.lane_keeping_env import LaneKeepingEnv

__all__ = [
    "BicycleModel",
    "RoadProfile",
    "build_all_profiles",
    "compute_reward",
    "LaneKeepingEnv",
]

# CarlaBridge is imported lazily by LaneKeepingEnv when backend="carla"
# to avoid requiring the carla package unless actually used.
