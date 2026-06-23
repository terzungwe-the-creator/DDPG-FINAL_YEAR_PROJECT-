"""
carla_bridge.py — CARLA Simulator Bridge for Lane Keeping Environment

Interfaces with CARLA 0.9.16 to provide a high-fidelity 4-wheel vehicle
simulation backend. Extracts the same 8D observation vector as the bicycle
model fallback, enabling seamless backend switching.

CARLA provides PhysX-based 4-wheel vehicle dynamics including:
    - Independent tyre forces per wheel (front-left, front-right, rear-left, rear-right)
    - Suspension spring/damper model
    - Ackermann steering geometry
    - Aerodynamic drag and rolling resistance

Vehicle: Tesla Model 3 blueprint (vehicle.tesla.model3)
    Mass:       1650 kg (matched to config.py)
    Wheelbase:  2.843 m (matched to config.py)

Reference:
    CARLA 0.9.16 Python API — carla.readthedocs.io
    VehiclePhysicsControl, WheelPhysicsControl classes

Version: 3.0 — Dataset-Augmented Training with CARLA Backend
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg

logger = logging.getLogger(__name__)


class CarlaBridge:
    """
    CARLA simulator bridge providing 4-wheel vehicle dynamics.

    Connects to a running CARLA server, spawns an ego vehicle, and translates
    CARLA state into the canonical 8D observation vector used by the DDPG agent.

    Attributes:
        client:      carla.Client instance.
        world:       carla.World instance.
        vehicle:     carla.Vehicle actor (ego vehicle).
        map:         carla.Map instance for waypoint queries.
        waypoints:   List of waypoints defining the current reference path.
        connected:   Whether the bridge is connected to CARLA.
    """

    # CARLA town mapping for each scenario
    SCENARIO_MAPS: Dict[str, str] = {
        "SCN-01": "Town04",   # Long straight highway segments
        "SCN-02": "Town03",   # Constant radius curves available
        "SCN-03": "Town07",   # Winding rural road geometry
        "SCN-04": "Town02",   # Narrow lanes for double lane change
        "SCN-05": "Town01",   # Mixed urban with curves and straights
    }

    # Spawn point indices per scenario (selected for appropriate road geometry)
    SCENARIO_SPAWN_INDICES: Dict[str, int] = {
        "SCN-01": 0,
        "SCN-02": 10,
        "SCN-03": 5,
        "SCN-04": 20,
        "SCN-05": 15,
    }

    def __init__(
        self,
        host: str = cfg.CARLA_HOST,
        port: int = cfg.CARLA_PORT,
        timeout: float = cfg.CARLA_TIMEOUT,
        vehicle_blueprint: str = cfg.CARLA_VEHICLE_BP,
        sync_mode: bool = cfg.CARLA_SYNC_MODE,
        fixed_dt: float = cfg.CARLA_FIXED_DT,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.vehicle_blueprint = vehicle_blueprint
        self.sync_mode = sync_mode
        self.fixed_dt = fixed_dt

        # CARLA objects (initialised on connect)
        self.client = None
        self.world = None
        self.vehicle = None
        self.map = None

        # State tracking
        self.connected: bool = False
        self._current_map: str = ""
        self._waypoint_cache: List = []
        self._waypoint_distances: np.ndarray = np.array([])
        self._delta_prev: float = 0.0
        self._step_count: int = 0
        self._episode_data: list = []

        # Traffic Manager
        self._traffic_manager = None
        self._npc_vehicles: List = []
        self._tm_port: int = 8000
        self.traffic_density: int = 10  # Number of NPC vehicles

    def connect(self) -> None:
        """
        Connect to CARLA server and configure synchronous mode.

        Raises:
            RuntimeError: If CARLA server is unreachable.
        """
        try:
            import carla
        except ImportError:
            raise ImportError(
                "CARLA Python API not installed. Install with:\n"
                "  pip install carla==0.9.16\n"
                "Or add CARLA PythonAPI to sys.path."
            )

        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            server_version = self.client.get_server_version()
            logger.info(f"Connected to CARLA server v{server_version} "
                        f"at {self.host}:{self.port}")

            self.world = self.client.get_world()
            self.map = self.world.get_map()

            # Configure synchronous mode with fixed timestep
            if self.sync_mode:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_dt
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
                logger.info(f"Synchronous mode: dt={self.fixed_dt}s")

            # Initialize Traffic Manager
            try:
                self._traffic_manager = self.client.get_trafficmanager(self._tm_port)
                self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
                self._traffic_manager.set_synchronous_mode(self.sync_mode)
                self._traffic_manager.set_hybrid_physics_mode(True)
                self._traffic_manager.set_hybrid_physics_radius(50.0)
                logger.info(f"Traffic Manager initialised on port {self._tm_port}")
            except Exception as tm_err:
                logger.warning(f"Traffic Manager init failed: {tm_err}")
                self._traffic_manager = None

            self.connected = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to CARLA at {self.host}:{self.port}: {e}"
            )

    def _load_map(self, map_name: str) -> None:
        """Load a CARLA map if not already loaded."""
        if self._current_map == map_name:
            return

        logger.info(f"Loading map: {map_name}")
        self.client.load_world(map_name)
        # Wait for map to load
        time.sleep(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Re-apply sync mode after map change
        if self.sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_dt
            self.world.apply_settings(settings)

        self._current_map = map_name

    def spawn_vehicle(self) -> None:
        """
        Spawn the ego vehicle with configured physics.

        Uses the Tesla Model 3 blueprint and applies physics control
        to match the vehicle parameters in config.py.
        """
        import carla

        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find(self.vehicle_blueprint)
        if bp is None:
            # Fallback to any sedan
            bp = bp_library.filter("vehicle.tesla.*")[0]
            logger.warning(f"Blueprint '{self.vehicle_blueprint}' not found, "
                           f"using '{bp.id}'")

        # Get a spawn point
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on current map")

        spawn_point = spawn_points[0]

        self.vehicle = self.world.try_spawn_actor(bp, spawn_point)
        if self.vehicle is None:
            # Try alternative spawn points
            for sp in spawn_points[1:5]:
                self.vehicle = self.world.try_spawn_actor(bp, sp)
                if self.vehicle is not None:
                    break

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        logger.info(f"Spawned vehicle: {bp.id} (id={self.vehicle.id})")

        # Configure 4-wheel physics to match our vehicle parameters
        self._apply_physics_control()

        # Tick once to register the vehicle
        if self.sync_mode:
            self.world.tick()

    def _apply_physics_control(self) -> None:
        """
        Apply physics control matching config.py vehicle parameters.

        Configures all 4 wheels with appropriate tyre friction,
        steering angles, and suspension parameters.
        """
        import carla

        physics = self.vehicle.get_physics_control()

        # Chassis parameters
        physics.mass = int(cfg.VEHICLE_MASS)
        physics.drag_coefficient = 0.30  # Cd for sedan

        # Torque curve for constant-speed operation at V_REFERENCE
        # Sufficient torque to maintain 60 km/h
        physics.torque_curve = [
            carla.Vector2D(x=0, y=500),
            carla.Vector2D(x=1500, y=500),
            carla.Vector2D(x=5000, y=500),
        ]
        physics.use_gear_autobox = True
        physics.gear_switch_time = 0.3
        physics.final_ratio = 4.0

        # 4-wheel configuration
        # Front wheels: steering, higher cornering stiffness
        max_steer_deg = float(np.degrees(cfg.DELTA_MAX))
        tyre_friction_front = cfg.TYRE_CAF_NOMINAL / 50000.0  # Scale to CARLA units
        tyre_friction_rear = cfg.TYRE_CAR_NOMINAL / 50000.0

        wheel_fl = carla.WheelPhysicsControl(
            tire_friction=max(tyre_friction_front, 2.0),
            damping_rate=1.5,
            max_steer_angle=max_steer_deg,
            radius=33.0,  # cm — typical sedan tyre radius
            max_brake_torque=1500.0,
            max_handbrake_torque=0.0,
            position=carla.Vector3D(x=cfg.VEHICLE_LF * 100, y=-75.0, z=30.0),
        )
        wheel_fr = carla.WheelPhysicsControl(
            tire_friction=max(tyre_friction_front, 2.0),
            damping_rate=1.5,
            max_steer_angle=max_steer_deg,
            radius=33.0,
            max_brake_torque=1500.0,
            max_handbrake_torque=0.0,
            position=carla.Vector3D(x=cfg.VEHICLE_LF * 100, y=75.0, z=30.0),
        )
        # Rear wheels: no steering
        wheel_rl = carla.WheelPhysicsControl(
            tire_friction=max(tyre_friction_rear, 2.0),
            damping_rate=1.5,
            max_steer_angle=0.0,
            radius=33.0,
            max_brake_torque=1500.0,
            max_handbrake_torque=3000.0,
            position=carla.Vector3D(x=-cfg.VEHICLE_LR * 100, y=-75.0, z=30.0),
        )
        wheel_rr = carla.WheelPhysicsControl(
            tire_friction=max(tyre_friction_rear, 2.0),
            damping_rate=1.5,
            max_steer_angle=0.0,
            radius=33.0,
            max_brake_torque=1500.0,
            max_handbrake_torque=3000.0,
            position=carla.Vector3D(x=-cfg.VEHICLE_LR * 100, y=75.0, z=30.0),
        )

        physics.wheels = [wheel_fl, wheel_fr, wheel_rl, wheel_rr]

        # Suspension
        physics.damping_rate_full_throttle = 0.15
        physics.damping_rate_zero_throttle_clutch_engaged = 2.0
        physics.damping_rate_zero_throttle_clutch_disengaged = 0.35

        self.vehicle.apply_physics_control(physics)
        logger.info(f"Applied 4-wheel physics: mass={physics.mass}kg, "
                     f"max_steer={max_steer_deg:.1f}°, "
                     f"tyre_friction_f={tyre_friction_front:.2f}, "
                     f"tyre_friction_r={tyre_friction_rear:.2f}")

    def update_tyre_params(self, C_af: float, C_ar: float) -> None:
        """
        Update tyre friction from DS-02 calibration results.

        Args:
            C_af: Front axle cornering stiffness (N/rad).
            C_ar: Rear axle cornering stiffness (N/rad).
        """
        import carla

        if self.vehicle is None:
            logger.warning("Cannot update tyre params — no vehicle spawned")
            return

        physics = self.vehicle.get_physics_control()
        friction_front = max(C_af / 50000.0, 1.5)
        friction_rear = max(C_ar / 50000.0, 1.5)

        for i, wheel in enumerate(physics.wheels):
            if i < 2:  # Front wheels
                wheel.tire_friction = friction_front
            else:       # Rear wheels
                wheel.tire_friction = friction_rear

        self.vehicle.apply_physics_control(physics)
        logger.info(f"Applied tyre friction: front={friction_front:.2f}, "
                     f"rear={friction_rear:.2f}")

    def spawn_traffic(self, n_vehicles: int = None) -> int:
        """
        Spawn NPC vehicles with Traffic Manager autopilot.

        Creates realistic traffic around the ego vehicle. NPCs use CARLA's
        built-in Traffic Manager for lane-following and collision avoidance.

        Args:
            n_vehicles: Number of NPC vehicles. Default: self.traffic_density.

        Returns:
            Number of NPCs successfully spawned.
        """
        if self._traffic_manager is None:
            logger.warning("Traffic Manager not available — skipping NPC spawn")
            return 0

        import carla

        if n_vehicles is None:
            n_vehicles = self.traffic_density

        # Destroy existing NPCs first
        self.destroy_traffic()

        # Get available spawn points (exclude ego vehicle's location)
        spawn_points = self.map.get_spawn_points()
        np.random.shuffle(spawn_points)

        # Get vehicle blueprints (sedans and small cars only)
        bp_library = self.world.get_blueprint_library()
        vehicle_bps = bp_library.filter("vehicle.*")
        # Filter to sedans for realistic highway/urban traffic
        safe_bps = [bp for bp in vehicle_bps
                     if int(bp.get_attribute("number_of_wheels")) == 4]
        if not safe_bps:
            safe_bps = list(vehicle_bps)

        spawned = 0
        for i in range(min(n_vehicles, len(spawn_points))):
            bp = np.random.choice(safe_bps)
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                bp.set_attribute("color", np.random.choice(colors))

            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                # Enable autopilot via Traffic Manager
                npc.set_autopilot(True, self._tm_port)
                # Set safe driving behaviour
                self._traffic_manager.vehicle_percentage_speed_difference(npc, 10.0)
                self._traffic_manager.distance_to_leading_vehicle(npc, 3.0)
                self._traffic_manager.auto_lane_change(npc, False)
                self._npc_vehicles.append(npc)
                spawned += 1

        if spawned > 0:
            logger.info(f"Spawned {spawned}/{n_vehicles} NPC vehicles with TM autopilot")

        return spawned

    def destroy_traffic(self) -> None:
        """Destroy all NPC vehicles."""
        for npc in self._npc_vehicles:
            try:
                if npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        self._npc_vehicles.clear()

    def reset(
        self,
        scenario_id: str = "SCN-01",
        e_lat_init: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        Loads the appropriate map, teleports the vehicle to the scenario
        start position, and builds the reference waypoint path.

        Args:
            scenario_id: Scenario identifier (SCN-01 through SCN-05).
            e_lat_init:  Initial lateral perturbation (m).
            seed:        Random seed (unused, CARLA is deterministic in sync mode).

        Returns:
            (observation, info) tuple.
        """
        import carla

        # Load map for this scenario
        map_name = self.SCENARIO_MAPS.get(scenario_id, "Town04")
        self._load_map(map_name)

        # Destroy existing vehicle and respawn
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        self.spawn_vehicle()

        # Get spawn point for this scenario
        spawn_points = self.map.get_spawn_points()
        spawn_idx = self.SCENARIO_SPAWN_INDICES.get(scenario_id, 0)
        spawn_idx = min(spawn_idx, len(spawn_points) - 1)
        spawn_transform = spawn_points[spawn_idx]

        # Apply initial lateral perturbation
        if abs(e_lat_init) > 1e-6:
            yaw_rad = math.radians(spawn_transform.rotation.yaw)
            spawn_transform.location.x -= e_lat_init * math.sin(yaw_rad)
            spawn_transform.location.y += e_lat_init * math.cos(yaw_rad)

        # Teleport vehicle to spawn
        self.vehicle.set_transform(spawn_transform)

        # Set initial velocity (V_REFERENCE forward)
        yaw_rad = math.radians(spawn_transform.rotation.yaw)
        self.vehicle.set_target_velocity(carla.Vector3D(
            x=cfg.V_REFERENCE * math.cos(yaw_rad),
            y=cfg.V_REFERENCE * math.sin(yaw_rad),
            z=0.0,
        ))

        # Build reference waypoint path
        self._build_waypoint_path(spawn_transform.location)

        # Reset state
        self._delta_prev = 0.0
        self._step_count = 0
        self._episode_data = []

        # Tick to stabilise physics
        if self.sync_mode:
            for _ in range(5):
                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=0.5, steer=0.0, brake=0.0
                ))
                self.world.tick()

        obs = self._get_observation()
        info = {"scenario_id": scenario_id, "arc_length": 0.0}
        return obs, info

    def _build_waypoint_path(self, start_location, path_length: float = 600.0) -> None:
        """
        Build a dense waypoint reference path from the start location.

        Generates waypoints at 0.5 m intervals for the specified path length.

        Args:
            start_location: carla.Location of the path start.
            path_length:    Total path length to generate (m).
        """
        start_wp = self.map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=self._get_lane_type(),
        )

        waypoints = [start_wp]
        current_wp = start_wp
        total_dist = 0.0
        step_size = 0.5  # m between waypoints

        while total_dist < path_length:
            next_wps = current_wp.next(step_size)
            if not next_wps:
                break
            current_wp = next_wps[0]
            waypoints.append(current_wp)
            total_dist += step_size

        self._waypoint_cache = waypoints

        # Pre-compute waypoint positions for fast nearest-neighbour search
        positions = np.array([
            [wp.transform.location.x, wp.transform.location.y]
            for wp in waypoints
        ], dtype=np.float64)
        self._waypoint_positions = positions

        logger.debug(f"Built waypoint path: {len(waypoints)} points, "
                      f"{total_dist:.0f} m")

    def _get_lane_type(self):
        """Get CARLA lane type enum."""
        import carla
        return carla.LaneType.Driving

    def step(self, delta_cmd: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one simulation step in CARLA.

        Applies the steering command, advances the simulation by one tick,
        and extracts the updated state.

        Args:
            delta_cmd: Steering angle in radians, will be normalised to [-1, 1]
                       for CARLA's VehicleControl.steer.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        import carla
        from simulator.reward import compute_reward

        # Normalise steering to CARLA range [-1, 1]
        steer_normalised = float(np.clip(delta_cmd / cfg.DELTA_MAX, -1.0, 1.0))

        # Speed controller: simple proportional throttle/brake
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_error = cfg.V_REFERENCE - speed

        if speed_error > 0:
            throttle = min(0.8, 0.3 + 0.1 * speed_error)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.5, 0.1 * abs(speed_error))

        # Apply control
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer_normalised,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(control)

        # Advance simulation
        if self.sync_mode:
            self.world.tick()

        self._step_count += 1

        # Extract state
        e_lat, e_psi = self._compute_lane_errors()
        kappa_ref = self._compute_curvature()
        v_x, v_y = self._get_velocities()
        r = self._get_yaw_rate()

        # Termination conditions
        terminated = abs(e_lat) >= cfg.DEPARTURE_THRESHOLD
        truncated = (
            self._step_count >= cfg.SIM_MAX_STEPS
            or self._get_arc_length() >= self._get_path_length()
        )

        # Compute reward
        delta_dot = (delta_cmd - self._delta_prev) / cfg.SIM_DT
        reward, reward_components = compute_reward(
            e_lat=e_lat,
            e_psi=e_psi,
            delta_current=delta_cmd,
            delta_previous=self._delta_prev,
            v_x=v_x,
            terminated=terminated,
        )

        # Build info dict
        transform = self.vehicle.get_transform()
        info = {
            "scenario_id": "",
            "step": self._step_count,
            "time_s": self._step_count * cfg.SIM_DT,
            "arc_length": self._get_arc_length(),
            "e_lat_m": e_lat,
            "e_psi_rad": e_psi,
            "delta_rad": delta_cmd,
            "delta_dot": delta_dot,
            "v_x": v_x,
            "v_y": v_y,
            "r": r,
            "kappa_ref": kappa_ref,
            "reward": reward,
            "X": transform.location.x,
            "Y": transform.location.y,
        }
        info.update(reward_components)
        self._episode_data.append(info)

        self._delta_prev = delta_cmd
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _compute_lane_errors(self) -> Tuple[float, float]:
        """
        Compute lateral error and heading error from CARLA state.

        Projects the vehicle position onto the nearest reference waypoint
        and computes the signed perpendicular distance (e_lat) and heading
        difference (e_psi).

        Returns:
            (e_lat, e_psi) in metres and radians respectively.
        """
        transform = self.vehicle.get_transform()
        veh_loc = np.array([transform.location.x, transform.location.y])
        veh_yaw = math.radians(transform.rotation.yaw)

        # Find nearest waypoint
        if len(self._waypoint_positions) == 0:
            return 0.0, 0.0

        diffs = self._waypoint_positions - veh_loc
        dists_sq = np.sum(diffs**2, axis=1)
        nearest_idx = int(np.argmin(dists_sq))

        wp = self._waypoint_cache[nearest_idx]
        wp_yaw = math.radians(wp.transform.rotation.yaw)
        wp_loc = np.array([wp.transform.location.x, wp.transform.location.y])

        # Signed lateral error: positive = right of centre
        # e_lat = (veh - wp) · n, where n is the left-pointing normal
        dx = veh_loc[0] - wp_loc[0]
        dy = veh_loc[1] - wp_loc[1]
        # Normal vector (perpendicular to waypoint heading, pointing left)
        nx = -math.sin(wp_yaw)
        ny = math.cos(wp_yaw)
        e_lat = dx * nx + dy * ny

        # Heading error
        e_psi = veh_yaw - wp_yaw
        # Wrap to [-π, π]
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi

        return float(e_lat), float(e_psi)

    def _compute_curvature(self) -> float:
        """
        Compute road curvature at the nearest waypoint using the
        circumscribed circle method (3-point curvature estimation).

        Returns:
            Curvature κ in 1/m.
        """
        transform = self.vehicle.get_transform()
        veh_loc = np.array([transform.location.x, transform.location.y])

        if len(self._waypoint_positions) < 3:
            return 0.0

        diffs = self._waypoint_positions - veh_loc
        dists_sq = np.sum(diffs**2, axis=1)
        nearest_idx = int(np.argmin(dists_sq))

        # Use 3 points: nearest-1, nearest, nearest+1
        idx_prev = max(0, nearest_idx - 2)
        idx_curr = nearest_idx
        idx_next = min(len(self._waypoint_positions) - 1, nearest_idx + 2)

        if idx_prev == idx_curr or idx_curr == idx_next:
            return 0.0

        p1 = self._waypoint_positions[idx_prev]
        p2 = self._waypoint_positions[idx_curr]
        p3 = self._waypoint_positions[idx_next]

        # Menger curvature: κ = 4·Area / (|p1-p2|·|p2-p3|·|p3-p1|)
        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        d12 = np.linalg.norm(p2 - p1)
        d23 = np.linalg.norm(p3 - p2)
        d31 = np.linalg.norm(p1 - p3)
        denom = d12 * d23 * d31

        if denom < 1e-10:
            return 0.0

        kappa = 4.0 * area / denom

        # Determine sign from cross product (positive = left turn)
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            kappa = -kappa

        return float(np.clip(kappa, -0.1, 0.1))

    def _get_velocities(self) -> Tuple[float, float]:
        """
        Get longitudinal and lateral velocities in the vehicle frame.

        Returns:
            (v_x, v_y) in m/s.
        """
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)

        # Rotate world velocity to vehicle frame
        v_x = velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)
        v_y = -velocity.x * math.sin(yaw) + velocity.y * math.cos(yaw)

        return float(v_x), float(v_y)

    def _get_yaw_rate(self) -> float:
        """Get yaw rate from CARLA angular velocity."""
        angular_vel = self.vehicle.get_angular_velocity()
        # CARLA angular velocity is in deg/s, convert to rad/s
        return float(math.radians(angular_vel.z))

    def _get_arc_length(self) -> float:
        """Estimate arc length from step count and average speed."""
        return self._step_count * cfg.V_REFERENCE * cfg.SIM_DT

    def _get_path_length(self) -> float:
        """Get total path length from waypoint cache."""
        if len(self._waypoint_positions) < 2:
            return float("inf")
        diffs = np.diff(self._waypoint_positions, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def _get_lookahead_curvature(self) -> Tuple[float, float]:
        """
        Compute curvature at 1s and 2s lookahead positions.

        Returns:
            (kappa_la1, kappa_la2) — curvature at lookahead points.
        """
        transform = self.vehicle.get_transform()
        veh_loc = np.array([transform.location.x, transform.location.y])
        v_x, _ = self._get_velocities()

        if len(self._waypoint_positions) < 3:
            return 0.0, 0.0

        # Find nearest waypoint index
        diffs = self._waypoint_positions - veh_loc
        dists_sq = np.sum(diffs**2, axis=1)
        nearest_idx = int(np.argmin(dists_sq))

        # Lookahead distances
        la1_dist = v_x * 1.0  # 1 second ahead
        la2_dist = v_x * 2.0  # 2 seconds ahead

        # Convert to waypoint indices (waypoints are ~0.5m apart)
        step = 0.5
        la1_idx = min(nearest_idx + int(la1_dist / step), len(self._waypoint_cache) - 3)
        la2_idx = min(nearest_idx + int(la2_dist / step), len(self._waypoint_cache) - 3)

        kappa_la1 = self._curvature_at_index(la1_idx)
        kappa_la2 = self._curvature_at_index(la2_idx)

        return kappa_la1, kappa_la2

    def _curvature_at_index(self, idx: int) -> float:
        """Compute curvature at a given waypoint index using 3-point method."""
        idx = max(1, min(idx, len(self._waypoint_positions) - 2))
        p1 = self._waypoint_positions[idx - 1]
        p2 = self._waypoint_positions[idx]
        p3 = self._waypoint_positions[idx + 1]

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        d12 = np.linalg.norm(p2 - p1)
        d23 = np.linalg.norm(p3 - p2)
        d31 = np.linalg.norm(p1 - p3)
        denom = d12 * d23 * d31

        if denom < 1e-10:
            return 0.0

        kappa = 4.0 * area / denom
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            kappa = -kappa

        return float(np.clip(kappa, -0.1, 0.1))

    def _get_observation(self) -> np.ndarray:
        """
        Build the 8D normalised observation from CARLA state.

        Same observation space as the bicycle model backend:
            [e_lat_norm, e_psi_norm, kappa_norm, v_y_norm, r_norm,
             delta_prev_norm, kappa_la1_norm, kappa_la2_norm]

        Returns:
            Observation array, shape (8,), clipped to [-1, 1].
        """
        e_lat, e_psi = self._compute_lane_errors()
        kappa_ref = self._compute_curvature()
        _, v_y = self._get_velocities()
        r = self._get_yaw_rate()
        kappa_la1, kappa_la2 = self._get_lookahead_curvature()

        obs = np.array([
            e_lat / cfg.NORM_E_LAT,
            e_psi / cfg.NORM_E_PSI,
            kappa_ref / cfg.NORM_KAPPA,
            v_y / cfg.NORM_V_Y,
            r / cfg.NORM_YAW_RATE,
            self._delta_prev / cfg.NORM_DELTA,
            kappa_la1 / cfg.NORM_KAPPA_LA1,
            kappa_la2 / cfg.NORM_KAPPA_LA2,
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    @property
    def episode_data(self) -> list:
        """Return episode data for metrics computation."""
        return self._episode_data

    def get_episode_arrays(self) -> dict:
        """Convert episode data to numpy arrays."""
        if not self._episode_data:
            return {}
        keys = self._episode_data[0].keys()
        result = {}
        for key in keys:
            vals = [step[key] for step in self._episode_data]
            if isinstance(vals[0], (int, float)):
                result[key] = np.array(vals, dtype=np.float64)
            else:
                result[key] = np.array(vals)
        return result

    def destroy(self) -> None:
        """Destroy all CARLA actors and reset world settings."""
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
            logger.info("Destroyed ego vehicle")

        if self.world is not None and self.sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        self.connected = False

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.destroy()
        except Exception:
            pass
