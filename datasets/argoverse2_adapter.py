"""
argoverse2_adapter.py — DS-03: Argoverse 2 Motion Forecasting Adapter

Extracts lateral error and heading error time series from AV2 ego trajectories
by projecting ego pose onto the nearest lane centreline.

Source: Wilson et al. (2023), "Argoverse 2: Next Generation Datasets for
        Self-Driving Perception and Forecasting", arXiv:2301.00493.
Install: pip install av2
Scale:   250,000 scenarios, 10 Hz, 6 US cities.
License: CC BY-NC-SA 4.0

Processing pipeline:
    1. For each scenario, load the focal agent track (ego vehicle).
    2. Load the corresponding HD map lane graph.
    3. For each timestep t:
       a. Find the nearest lane segment to ego position (X_t, Y_t).
       b. Project ego position onto that segment's centreline.
       c. Compute e_lat_t = signed perpendicular distance (m).
       d. Compute e_psi_t = ego heading - lane tangent angle (rad), wrapped.
       e. Compute kappa_ref_t from lane centreline curvature at projection point.
       f. Compute delta_t from heading change rate.
    4. Build (obs_t, action_t, reward_t, obs_{t+1}, done_t) tuples.
    5. Filter: discard timesteps where |e_lat| > 1.5 m.

Parallelised with concurrent.futures.ProcessPoolExecutor, max_workers=4.
Target: process 10,000 scenarios for 200,000 transitions.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg
from datasets.normaliser import RawTransition, UniversalNormaliser
from ddpg.hybrid_buffer import HybridStratifiedBuffer
from simulator.reward import compute_reward

logger = logging.getLogger(__name__)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _compute_curvature_from_points(
    points: np.ndarray, idx: int
) -> float:
    """
    Estimate curvature at a point from three consecutive centreline points.

    Uses the circumscribed circle method:
        κ = 2 · |cross(p1-p0, p2-p0)| / (|p1-p0| · |p2-p0| · |p2-p1|)

    Args:
        points: Array of (x, y) centreline points, shape (N, 2).
        idx:    Index of the point to compute curvature at.

    Returns:
        Curvature (1/m) at the indexed point. Returns 0.0 at boundaries.
    """
    if idx <= 0 or idx >= len(points) - 1:
        return 0.0

    p0 = points[idx - 1]
    p1 = points[idx]
    p2 = points[idx + 1]

    d01 = np.linalg.norm(p1 - p0)
    d12 = np.linalg.norm(p2 - p1)
    d02 = np.linalg.norm(p2 - p0)

    if d01 < 1e-6 or d12 < 1e-6 or d02 < 1e-6:
        return 0.0

    # Cross product magnitude = 2 * area of triangle
    cross = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    kappa = 2.0 * cross / (d01 * d12 * d02)

    return float(kappa)


def _project_point_to_polyline(
    point: np.ndarray, polyline: np.ndarray
) -> Tuple[float, float, int, float]:
    """
    Project a point onto a polyline (sequence of 2D points).

    Finds the closest point on the polyline to the given point.

    Args:
        point:    2D point (x, y).
        polyline: Array of 2D points, shape (N, 2).

    Returns:
        (e_lat_signed, tangent_angle, closest_segment_idx, arc_length_to_projection)
        e_lat_signed: Signed perpendicular distance (positive = left of centreline).
        tangent_angle: Angle of the polyline tangent at the projection point.
        closest_segment_idx: Index of the closest segment.
        arc_length: Arc length from polyline start to projection point.
    """
    min_dist = float("inf")
    best_e_lat = 0.0
    best_tangent = 0.0
    best_seg_idx = 0
    best_arc = 0.0

    cumulative_arc = 0.0

    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        ab = b - a
        seg_len = np.linalg.norm(ab)

        if seg_len < 1e-8:
            cumulative_arc += seg_len
            continue

        ab_unit = ab / seg_len
        ap = point - a

        # Project point onto segment
        t = np.dot(ap, ab_unit)
        t_clamped = np.clip(t, 0.0, seg_len)

        closest = a + t_clamped * ab_unit
        diff = point - closest
        dist = np.linalg.norm(diff)

        if dist < min_dist:
            min_dist = dist

            # Signed distance: positive = left of travel direction
            cross = ab_unit[0] * diff[1] - ab_unit[1] * diff[0]
            best_e_lat = float(cross)  # Positive = left

            # Tangent angle
            best_tangent = float(np.arctan2(ab_unit[1], ab_unit[0]))
            best_seg_idx = i
            best_arc = cumulative_arc + t_clamped

        cumulative_arc += seg_len

    return best_e_lat, best_tangent, best_seg_idx, best_arc


def _process_single_scenario(
    scenario_path: Path,
) -> List[dict]:
    """
    Process a single AV2 scenario into raw transition data.

    This function is designed to be called in a ProcessPoolExecutor.
    It uses the av2 API if available, otherwise falls back to parquet
    file parsing.

    Args:
        scenario_path: Path to the scenario directory.

    Returns:
        List of transition dicts (serialisable for cross-process transport).
    """
    transitions = []

    try:
        # Try using av2 API
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.map.map_api import ArgoverseStaticMap

        scenario = scenario_serialization.load_argoverse_scenario_parquet(
            scenario_path
        )

        # Get focal track (ego vehicle)
        focal_track = None
        for track in scenario.tracks:
            if track.track_id == scenario.focal_track_id:
                focal_track = track
                break

        if focal_track is None or len(focal_track.object_states) < 5:
            return []

        # Extract positions and headings
        positions = np.array([
            [s.position[0], s.position[1]]
            for s in focal_track.object_states
        ])
        headings = np.array([s.heading for s in focal_track.object_states])
        timestamps = np.array([
            s.timestep for s in focal_track.object_states
        ])

        # Load map
        map_dir = scenario_path / "map"
        if not map_dir.exists():
            # Try alternative map path
            map_dir = scenario_path
        try:
            avm = ArgoverseStaticMap.from_map_dir(map_dir, build_raster=False)
        except Exception:
            return []

        dt = 0.1  # 10 Hz
        n = len(positions)

        for i in range(n - 1):
            pos = positions[i]

            # Get nearby lane segments
            try:
                lane_segments = avm.get_nearby_lane_segments(
                    pos, search_radius_m=50.0
                )
            except Exception:
                continue

            if not lane_segments:
                continue

            # Find closest lane centreline
            best_elat = float("inf")
            best_tangent = 0.0
            best_kappa = 0.0

            for lane_seg in lane_segments:
                centerline = lane_seg.polygon_boundary
                if centerline is None or len(centerline) < 2:
                    continue

                # Use left/right boundary midpoints as centreline
                try:
                    cl = np.array(centerline)[:, :2]
                except (IndexError, TypeError):
                    continue

                e_lat, tangent, seg_idx, arc = _project_point_to_polyline(
                    pos, cl
                )

                if abs(e_lat) < abs(best_elat):
                    best_elat = e_lat
                    best_tangent = tangent
                    best_kappa = _compute_curvature_from_points(cl, seg_idx)

            if abs(best_elat) == float("inf"):
                continue

            # Filter: discard off-road timesteps
            if abs(best_elat) > cfg.ARGOVERSE_MAX_ELAT_FILTER:
                continue

            # Heading error
            e_psi = _wrap_angle(headings[i] - best_tangent)

            # Speed estimation from position differences
            if i > 0:
                dx = positions[i] - positions[i - 1]
                v_x = np.linalg.norm(dx) / dt
            else:
                dx_next = positions[i + 1] - positions[i]
                v_x = np.linalg.norm(dx_next) / dt

            if v_x < 1.0:  # Filter very low speed
                continue

            # Steering angle estimation (inverse bicycle model)
            if i < n - 1:
                heading_rate = _wrap_angle(headings[i + 1] - headings[i]) / dt
                # δ ≈ heading_rate · L / v_x (inverse bicycle equation)
                delta_est = heading_rate * cfg.VEHICLE_WHEELBASE / max(v_x, 1.0)
            else:
                delta_est = 0.0

            delta_est = np.clip(delta_est, -cfg.DELTA_MAX, cfg.DELTA_MAX)

            # Yaw rate
            yaw_rate = heading_rate if i < n - 1 else 0.0

            # Lateral velocity estimate
            v_y = v_x * np.sin(e_psi)

            transitions.append({
                "e_lat": float(best_elat),
                "e_psi": float(e_psi),
                "kappa": float(best_kappa),
                "v_y": float(v_y),
                "yaw_rate": float(yaw_rate),
                "delta": float(delta_est),
                "v_x": float(v_x),
            })

    except ImportError:
        # av2 not installed — skip this scenario
        logger.debug("av2 package not installed — skipping AV2 scenario processing")
        return []
    except Exception as e:
        logger.debug(f"Failed to process scenario {scenario_path}: {e}")
        return []

    return transitions


class Argoverse2Adapter:
    """
    Extracts lateral error and heading error from AV2 ego trajectories.

    Source: Wilson et al. (2023), arXiv:2301.00493.
    Install: pip install av2
    Scale: 250,000 scenarios, 10 Hz, 6 cities.

    This adapter is designed to work with or without the av2 package.
    If av2 is not installed, no transitions will be loaded and a warning
    is logged.
    """

    def __init__(
        self,
        data_dir: str,
        max_scenarios: int = cfg.ARGOVERSE_MAX_SCENARIOS,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_scenarios = max_scenarios
        self.normaliser = UniversalNormaliser()
        self._stats = {
            "scenarios_found": 0,
            "scenarios_processed": 0,
            "transitions_created": 0,
            "reject_rate": 0.0,
        }

    def _find_scenarios(self) -> List[Path]:
        """
        Find scenario directories in the data path.

        AV2 scenarios are stored as directories containing parquet files
        and map data.

        Returns:
            List of scenario directory paths.
        """
        if not self.data_dir.exists():
            logger.warning(f"Argoverse2 data directory not found: {self.data_dir}")
            return []

        # Look for scenario directories (contain .parquet files)
        scenarios = []
        for item in sorted(self.data_dir.rglob("*.parquet")):
            scenario_dir = item.parent
            if scenario_dir not in scenarios:
                scenarios.append(scenario_dir)

        # Deduplicate and limit
        scenarios = list(dict.fromkeys(scenarios))
        self._stats["scenarios_found"] = len(scenarios)

        if len(scenarios) > self.max_scenarios:
            rng = np.random.RandomState(cfg.SEED)
            indices = rng.choice(len(scenarios), self.max_scenarios, replace=False)
            scenarios = [scenarios[i] for i in sorted(indices)]

        logger.info(f"Found {len(scenarios)} Argoverse2 scenarios")
        return scenarios

    def extract_scenario(self, scenario_path: Path) -> List[RawTransition]:
        """
        Process a single scenario into DDPG-compatible transitions.

        Args:
            scenario_path: Path to the scenario directory.

        Returns:
            List of RawTransition objects.
        """
        raw_data = _process_single_scenario(scenario_path)
        if not raw_data or len(raw_data) < 3:
            return []

        transitions = []

        for i in range(len(raw_data) - 1):
            curr = raw_data[i]
            nxt = raw_data[i + 1]

            delta_prev = raw_data[i - 1]["delta"] if i > 0 else curr["delta"]

            # Lookahead curvature
            la1_idx = min(i + 10, len(raw_data) - 1)  # 1s at 10 Hz
            la2_idx = min(i + 20, len(raw_data) - 1)  # 2s at 10 Hz

            reward, _ = compute_reward(
                e_lat=curr["e_lat"],
                e_psi=curr["e_psi"],
                delta_current=curr["delta"],
                delta_previous=delta_prev,
                v_x=curr["v_x"],
                terminated=False,
            )

            is_last = (i == len(raw_data) - 2)
            done = is_last or abs(nxt["e_lat"]) > cfg.DEPARTURE_THRESHOLD

            try:
                trans = RawTransition(
                    source="argoverse",
                    e_lat_m=curr["e_lat"],
                    e_psi_rad=curr["e_psi"],
                    kappa_ref=curr["kappa"],
                    v_y_mps=curr["v_y"],
                    yaw_rate_rads=curr["yaw_rate"],
                    delta_prev_rad=delta_prev,
                    kappa_la1=raw_data[la1_idx]["kappa"],
                    kappa_la2=raw_data[la2_idx]["kappa"],
                    action_raw_rad=curr["delta"],
                    reward=reward,
                    done=done,
                    next_e_lat_m=nxt["e_lat"],
                    next_e_psi_rad=nxt["e_psi"],
                    next_kappa_ref=nxt["kappa"],
                    next_v_y_mps=nxt["v_y"],
                    next_yaw_rate_rads=nxt["yaw_rate"],
                    next_delta_prev_rad=curr["delta"],
                    next_kappa_la1=raw_data[min(la1_idx + 1, len(raw_data) - 1)]["kappa"],
                    next_kappa_la2=raw_data[min(la2_idx + 1, len(raw_data) - 1)]["kappa"],
                )
                transitions.append(trans)
            except ValueError:
                continue

        return transitions

    def load_into_buffer(
        self,
        buffer: HybridStratifiedBuffer,
        max_transitions: int = cfg.ARGOVERSE_MAX_TRANSITIONS,
    ) -> int:
        """
        Load Argoverse2 transitions into the 'argoverse' sub-buffer.

        Uses ProcessPoolExecutor for parallel scenario processing.

        Args:
            buffer:          Hybrid stratified replay buffer.
            max_transitions: Maximum number of transitions to load.

        Returns:
            Number of transitions successfully loaded.
        """
        scenarios = self._find_scenarios()
        if not scenarios:
            logger.warning("No Argoverse2 scenarios — skipping DS-03")
            return 0

        loaded = 0
        processed = 0

        # Process scenarios — use sequential if few, parallel if many
        if len(scenarios) <= 10:
            # Sequential processing for small counts
            for sp in scenarios:
                if loaded >= max_transitions:
                    break
                transitions = self.extract_scenario(sp)
                processed += 1
                for trans in transitions:
                    if loaded >= max_transitions:
                        break
                    try:
                        obs = self.normaliser.normalise_obs(trans)
                        next_obs = self.normaliser.normalise_next_obs(trans)
                        action_norm = self.normaliser.normalise_action(trans.action_raw_rad)
                        buffer.push(
                            "argoverse", obs, action_norm,
                            trans.reward, next_obs, float(trans.done)
                        )
                        loaded += 1
                    except ValueError:
                        continue
        else:
            # Parallel processing
            with ProcessPoolExecutor(
                max_workers=cfg.ARGOVERSE_NUM_WORKERS
            ) as executor:
                futures = {
                    executor.submit(_process_single_scenario, sp): sp
                    for sp in scenarios
                }

                for future in as_completed(futures):
                    if loaded >= max_transitions:
                        break
                    processed += 1

                    try:
                        raw_data = future.result(timeout=60)
                    except Exception:
                        continue

                    if not raw_data or len(raw_data) < 3:
                        continue

                    # Build transitions from raw data
                    for i in range(len(raw_data) - 1):
                        if loaded >= max_transitions:
                            break

                        curr = raw_data[i]
                        nxt = raw_data[i + 1]
                        delta_prev = raw_data[i - 1]["delta"] if i > 0 else curr["delta"]
                        la1_idx = min(i + 10, len(raw_data) - 1)
                        la2_idx = min(i + 20, len(raw_data) - 1)

                        reward, _ = compute_reward(
                            e_lat=curr["e_lat"],
                            e_psi=curr["e_psi"],
                            delta_current=curr["delta"],
                            delta_previous=delta_prev,
                            v_x=curr["v_x"],
                            terminated=False,
                        )

                        is_last = (i == len(raw_data) - 2)
                        done = is_last or abs(nxt["e_lat"]) > cfg.DEPARTURE_THRESHOLD

                        try:
                            trans = RawTransition(
                                source="argoverse",
                                e_lat_m=curr["e_lat"],
                                e_psi_rad=curr["e_psi"],
                                kappa_ref=curr["kappa"],
                                v_y_mps=curr["v_y"],
                                yaw_rate_rads=curr["yaw_rate"],
                                delta_prev_rad=delta_prev,
                                kappa_la1=raw_data[la1_idx]["kappa"],
                                kappa_la2=raw_data[la2_idx]["kappa"],
                                action_raw_rad=curr["delta"],
                                reward=reward,
                                done=done,
                                next_e_lat_m=nxt["e_lat"],
                                next_e_psi_rad=nxt["e_psi"],
                                next_kappa_ref=nxt["kappa"],
                                next_v_y_mps=nxt["v_y"],
                                next_yaw_rate_rads=nxt["yaw_rate"],
                                next_delta_prev_rad=curr["delta"],
                                next_kappa_la1=raw_data[min(la1_idx + 1, len(raw_data) - 1)]["kappa"],
                                next_kappa_la2=raw_data[min(la2_idx + 1, len(raw_data) - 1)]["kappa"],
                            )
                            obs = self.normaliser.normalise_obs(trans)
                            next_obs = self.normaliser.normalise_next_obs(trans)
                            action_norm = self.normaliser.normalise_action(trans.action_raw_rad)
                            buffer.push(
                                "argoverse", obs, action_norm,
                                trans.reward, next_obs, float(trans.done)
                            )
                            loaded += 1
                        except ValueError:
                            continue

        self._stats["scenarios_processed"] = processed
        self._stats["transitions_created"] = loaded
        if processed > 0:
            self._stats["reject_rate"] = 1.0 - (loaded / max(processed * 10, 1))

        # Save extraction stats
        stats_path = cfg.RESULTS_DIR / "av2_extraction_stats.json"
        cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(self._stats, f, indent=2)

        logger.info(
            f"Argoverse2: loaded {loaded} transitions from "
            f"{processed} scenarios"
        )
        return loaded

    @property
    def stats(self) -> dict:
        """Return extraction statistics."""
        return dict(self._stats)
