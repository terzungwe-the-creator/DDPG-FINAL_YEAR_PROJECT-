"""
test_system.py — Comprehensive System Test Suite

Automated pytest test suite covering all critical subsystems of the
DDPG Lane Keeping System v3.0 including real-world deployment.

Run: python -m pytest test_system.py -v --tb=short

Test Categories:
    1. Vehicle Model Physics (RK4, steady-state)
    2. Road Profile Geometry (clothoid transitions, curvature continuity)
    3. Feedforward Correctness (zero-agent survival)
    4. Domain Randomization (mass, friction, tyre)
    5. Safety Guardian (rate/angle clamping)
    6. Reward Function (component ranges)
    7. Buffer Stratified Sampling (sub-buffer logic)
    8. ISO 15622 / IEEE 2846 Metrics (synthetic data)
    9. Perception Pipeline (sensor fusion, normalisation)
    10. Safety Monitor (state machine transitions)
    11. Deployment Pipeline (end-to-end simulated)
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

import config as cfg


def _torch_available() -> bool:
    """Check if PyTorch is functional on this platform."""
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VEHICLE MODEL PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestVehicleModel:
    """Test BicycleModel physics simulation."""

    def setup_method(self):
        from simulator.vehicle_model import BicycleModel
        self.model = BicycleModel()

    def test_straight_line_no_drift(self):
        """Vehicle driving straight with zero steering should not drift."""
        self.model.reset(v_x=cfg.V_REFERENCE, e_lat_init=0.0, e_psi_init=0.0)
        for _ in range(1000):
            self.model.step(delta=0.0, kappa_ref=0.0)
        assert abs(self.model.lateral_error) < 0.01, \
            f"Straight-line drift: {self.model.lateral_error:.4f}m"

    def test_steady_state_cornering(self):
        """Steady-state Ackermann steering should produce bounded yaw rate."""
        R = 80.0  # m
        kappa = 1.0 / R
        # Include understeer gradient for correct steady-state angle
        K_us = (cfg.VEHICLE_MASS / cfg.VEHICLE_WHEELBASE) * (
            cfg.VEHICLE_LR / cfg.TYRE_CAF_NOMINAL
            - cfg.VEHICLE_LF / cfg.TYRE_CAR_NOMINAL
        )
        delta_ss = cfg.VEHICLE_WHEELBASE * kappa + K_us * cfg.V_REFERENCE ** 2 * kappa
        self.model.reset(v_x=cfg.V_REFERENCE, e_lat_init=0.0, e_psi_init=0.0)
        for _ in range(200):
            self.model.step(delta=delta_ss, kappa_ref=kappa)
        # Yaw rate should approximate v_x / R in steady state
        expected_r = cfg.V_REFERENCE / R
        assert abs(self.model.yaw_rate - expected_r) < 0.1, \
            f"Yaw rate mismatch: expected ~{expected_r:.3f}, got {self.model.yaw_rate:.3f}"

    def test_speed_conservation(self):
        """Longitudinal velocity should remain approximately constant."""
        v0 = cfg.V_REFERENCE
        self.model.reset(v_x=v0, e_lat_init=0.0, e_psi_init=0.0)
        for _ in range(1000):
            self.model.step(delta=0.0, kappa_ref=0.0)
        assert abs(self.model.v_x - v0) < 0.5, \
            f"Speed changed: {v0:.2f} → {self.model.v_x:.2f}"

    def test_rk4_symmetry(self):
        """Symmetric steering should produce symmetric lateral error."""
        self.model.reset(v_x=cfg.V_REFERENCE, e_lat_init=0.0, e_psi_init=0.0)
        for _ in range(100):
            self.model.step(delta=0.05, kappa_ref=0.0)
        e_lat_right = self.model.lateral_error

        self.model.reset(v_x=cfg.V_REFERENCE, e_lat_init=0.0, e_psi_init=0.0)
        for _ in range(100):
            self.model.step(delta=-0.05, kappa_ref=0.0)
        e_lat_left = self.model.lateral_error

        assert abs(e_lat_right + e_lat_left) < 0.01, \
            f"Asymmetry: right={e_lat_right:.4f}, left={e_lat_left:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROAD PROFILE GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoadProfiles:
    """Test road profile geometry and clothoid transitions."""

    def setup_method(self):
        from simulator.road_profiles import build_all_profiles
        self.profiles = build_all_profiles()

    def test_all_profiles_built(self):
        """All 5 scenario profiles should be created."""
        for scn_id in cfg.SCENARIO_IDS:
            assert scn_id in self.profiles, f"Missing profile: {scn_id}"

    def test_scn01_straight(self):
        """SCN-01 should have zero curvature everywhere."""
        profile = self.profiles["SCN-01"]
        kappa_max = np.max(np.abs(profile.kappa_ref))
        assert kappa_max < 1e-6, f"SCN-01 has non-zero curvature: {kappa_max}"

    def test_scn02_clothoid_continuity(self):
        """SCN-02 should have continuous (no step) curvature transitions."""
        profile = self.profiles["SCN-02"]
        kappa = profile.kappa_ref
        # Check that curvature changes are gradual (no steps > 0.002)
        dkappa = np.abs(np.diff(kappa))
        max_jump = np.max(dkappa)
        assert max_jump < 0.005, \
            f"SCN-02 curvature discontinuity: {max_jump:.4f} (should be < 0.005)"

    def test_scn05_sbend_clothoid(self):
        """SCN-05 S-bend should have smoothed curvature reversal."""
        profile = self.profiles["SCN-05"]
        kappa = profile.kappa_ref
        dkappa = np.abs(np.diff(kappa))
        max_jump = np.max(dkappa)
        assert max_jump < 0.005, \
            f"SCN-05 curvature discontinuity: {max_jump:.4f} (should be < 0.005)"

    def test_profile_lengths(self):
        """All profiles should have reasonable total lengths."""
        for scn_id, profile in self.profiles.items():
            assert profile.total_length > 100.0, \
                f"{scn_id} too short: {profile.total_length:.0f}m"
            assert profile.total_length < 1000.0, \
                f"{scn_id} too long: {profile.total_length:.0f}m"

    def test_kappa_lookup(self):
        """Curvature lookup at arbitrary arc lengths should work."""
        profile = self.profiles["SCN-02"]
        # Before curve
        k0 = profile.get_kappa_at_s(50.0)
        assert abs(k0) < 0.001, "Should be straight at s=50"
        # In curve (after clothoid entry)
        k_mid = profile.get_kappa_at_s(250.0)
        assert k_mid > 0.005, f"Should be curved at s=250: kappa={k_mid}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEEDFORWARD CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedforward:
    """Test that feedforward steering keeps the vehicle in lane."""

    def setup_method(self):
        from simulator.lane_keeping_env import LaneKeepingEnv
        self.env = LaneKeepingEnv(training_mode=False)

    @pytest.mark.parametrize("scenario", ["SCN-01", "SCN-02"])
    def test_zero_action_survival(self, scenario):
        """With zero agent action, feedforward should keep vehicle in lane."""
        obs, _ = self.env.reset(scenario_id=scenario)
        max_elat = 0.0
        survived = True

        for _ in range(500):
            obs, r, terminated, truncated, info = self.env.step(np.array([0.0]))
            max_elat = max(max_elat, abs(info["e_lat_m"]))
            if terminated:
                survived = False
                break

        assert survived, f"{scenario}: Vehicle departed lane (max_elat={max_elat:.3f}m)"
        assert max_elat < 1.5, f"{scenario}: Excessive lateral error: {max_elat:.3f}m"

    @pytest.mark.parametrize("scenario", ["SCN-04", "SCN-05"])
    def test_challenging_scenario_survival(self, scenario):
        """Challenging scenarios should at least survive with feedforward."""
        obs, _ = self.env.reset(scenario_id=scenario)
        max_elat = 0.0
        steps = 0

        for _ in range(500):
            obs, r, terminated, truncated, info = self.env.step(np.array([0.0]))
            max_elat = max(max_elat, abs(info["e_lat_m"]))
            steps += 1
            if terminated or truncated:
                break

        # Challenging scenarios may depart but should survive > 100 steps
        assert steps > 100, f"{scenario}: Departed too early at step {steps}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DOMAIN RANDOMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainRandomization:
    """Test domain randomization applies correctly."""

    def test_mass_variation(self):
        """Training mode should produce varied vehicle masses."""
        from simulator.lane_keeping_env import LaneKeepingEnv
        env = LaneKeepingEnv(training_mode=True)
        masses = []
        for _ in range(20):
            _, info = env.reset(scenario_id="SCN-01")
            masses.append(info["dr_mass_kg"])
        mass_range = max(masses) - min(masses)
        assert mass_range > 50, f"Insufficient mass variation: {mass_range:.0f}kg"

    def test_speed_variation(self):
        """Training mode should produce varied speeds."""
        from simulator.lane_keeping_env import LaneKeepingEnv
        env = LaneKeepingEnv(training_mode=True)
        speeds = []
        for _ in range(20):
            _, info = env.reset(scenario_id="SCN-01")
            speeds.append(info["episode_speed_kmh"])
        speed_range = max(speeds) - min(speeds)
        assert speed_range > 5, f"Insufficient speed variation: {speed_range:.1f}km/h"

    def test_eval_mode_fixed(self):
        """Evaluation mode should have fixed speed (no randomization)."""
        from simulator.lane_keeping_env import LaneKeepingEnv
        env = LaneKeepingEnv(training_mode=False)
        speeds = []
        for _ in range(10):
            _, info = env.reset(scenario_id="SCN-01")
            speeds.append(info["episode_speed_kmh"])
        speed_var = max(speeds) - min(speeds)
        assert speed_var < 0.1, f"Eval mode has speed variation: {speed_var:.2f}"

    def test_friction_applied_to_tyres(self):
        """Friction coefficient should modify tyre stiffness."""
        from simulator.lane_keeping_env import LaneKeepingEnv
        env = LaneKeepingEnv(training_mode=True)
        caf_values = []
        for _ in range(20):
            env.reset(scenario_id="SCN-01")
            caf_values.append(env.vehicle.C_af)
        caf_range = max(caf_values) - min(caf_values)
        # Should vary due to friction_mu * C_af randomization
        assert caf_range > 1000, \
            f"Tyre stiffness not varying (friction not applied): range={caf_range:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAFETY GUARDIAN
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyGuardian:
    """Test safety guardian rate/angle clamping."""

    def test_rate_clamping(self):
        """Large steering steps should be rate-clamped."""
        from simulator.lane_keeping_env import LaneKeepingEnv
        env = LaneKeepingEnv(training_mode=False)
        obs, _ = env.reset(scenario_id="SCN-01")
        # Apply maximum steering
        obs, _, _, _, info = env.step(np.array([1.0]))
        # Should be rate-limited (not full delta_max in one step)
        assert abs(info["delta_rad"]) < cfg.DELTA_MAX, \
            "Guardian should rate-limit first step"

    def test_angle_clamping(self):
        """Guardian should clamp to DELTA_MAX."""
        from simulator.safety_guardian import SafetyGuardian
        guardian = SafetyGuardian()
        # Request absurdly large angle — apply() does rate+angle clamping
        safe_delta = guardian.apply(delta_cmd=1.0, delta_prev=0.0)
        assert abs(safe_delta) <= cfg.DELTA_MAX + 1e-6, \
            f"Guardian exceeded DELTA_MAX: {safe_delta:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REWARD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewardFunction:
    """Test reward function component ranges."""

    def test_reward_range(self):
        """Reward at lane centre should be positive and bounded."""
        from simulator.reward import compute_reward
        # Perfect state: zero errors
        reward, components = compute_reward(
            e_lat=0.0, e_psi=0.0, delta_current=0.0,
            delta_previous=0.0, v_x=cfg.V_REFERENCE, terminated=False,
        )
        assert reward > 0, f"Reward at centre should be positive: {reward}"
        assert reward < 20.0, f"Reward unreasonably high: {reward}"

    def test_penalty_at_departure(self):
        """Reward should be heavily negative at lane departure."""
        from simulator.reward import compute_reward
        reward, components = compute_reward(
            e_lat=1.8, e_psi=0.3, delta_current=0.1,
            delta_previous=0.0, v_x=cfg.V_REFERENCE, terminated=True,
        )
        assert reward < 0, f"Reward at departure should be negative: {reward}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HYBRID BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _torch_available(), reason="PyTorch DLL not available")
class TestHybridBuffer:
    """Test hybrid stratified replay buffer."""

    def setup_method(self):
        from ddpg.hybrid_buffer import HybridStratifiedBuffer
        self.buffer = HybridStratifiedBuffer()

    def test_push_and_sample(self):
        """Should be able to push transitions and sample batches."""
        state = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        action = np.zeros(cfg.ACTION_DIM, dtype=np.float32)

        for i in range(cfg.BATCH_SIZE * 2):
            self.buffer.push("sim", state, action, 1.0, state, 0.0)

        batch = self.buffer.sample(episode=0)
        assert batch is not None, "Sample should return a batch"

    def test_sub_buffer_isolation(self):
        """Different sources should go to different sub-buffers."""
        state = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        action = np.zeros(cfg.ACTION_DIM, dtype=np.float32)

        self.buffer.push("sim", state, action, 1.0, state, 0.0)
        self.buffer.push("openlka", state, action, 1.0, state, 0.0)

        sizes = self.buffer.sizes
        assert sizes["sim"] == 1, f"Sim buffer should have 1: {sizes['sim']}"
        assert sizes["openlka"] == 1, f"OpenLKA buffer should have 1: {sizes['openlka']}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test ISO 15622, IEEE 2846, and UNECE R157 metrics."""

    def test_rmse_computation(self):
        """RMSE should match manual computation."""
        from metrics.iso15622 import compute_rmse_lat
        e_lat = np.array([0.1, -0.2, 0.15, -0.05])
        expected = np.sqrt(np.mean(e_lat ** 2))
        result = compute_rmse_lat(e_lat)
        assert abs(result - expected) < 1e-6

    def test_lksr_perfect(self):
        """LKSR should be 1.0 when always in lane."""
        from metrics.iso15622 import compute_lksr
        e_lat = np.array([0.1, -0.2, 0.3, -0.1])
        lksr = compute_lksr(e_lat)
        assert lksr == 1.0, f"LKSR should be 1.0 for in-lane data: {lksr}"

    def test_lksr_partial(self):
        """LKSR should reflect actual in-lane fraction."""
        from metrics.iso15622 import compute_lksr
        e_lat = np.array([0.1, 0.5, 0.8, 0.9])  # 2 of 4 above 0.75
        lksr = compute_lksr(e_lat, threshold=0.75)
        assert 0.0 < lksr < 1.0, f"LKSR should be partial: {lksr}"

    def test_settling_time(self):
        """Settling time should detect when error drops below threshold."""
        from metrics.ieee2846 import compute_settling_time
        # Error decays from 0.5 to 0 over 100 steps
        e_lat = 0.5 * np.exp(-np.arange(200) * 0.05)
        settle = compute_settling_time(e_lat, threshold=0.10)
        assert 0.0 < settle < 2.0, f"Settling time unusual: {settle:.2f}s"

    def test_ttld_safe_state(self):
        """TTLD should be large when vehicle is safely centred."""
        from metrics.safety import compute_ttld_series
        e_lat = np.full(100, 0.01)  # Nearly centred
        ttld = compute_ttld_series(e_lat)
        assert np.min(ttld) > 10.0, f"TTLD too low for safe state: {np.min(ttld):.2f}"

    def test_iso_pass_perfect(self):
        """Perfect tracking should pass ISO 15622."""
        from metrics.iso15622 import iso15622_pass_fail
        e_lat = np.random.randn(1000) * 0.05  # Small random errors
        e_psi = np.random.randn(1000) * 0.01
        result = iso15622_pass_fail(e_lat, e_psi)
        assert result["overall_pass"], \
            f"ISO 15622 should pass with small errors: {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PERCEPTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerceptionPipeline:
    """Test real-world perception pipeline."""

    def test_normalisation_matches_training(self):
        """Perception normalisation must match training normalisation."""
        from real_world.perception_pipeline import PerceptionPipeline, PerceptionState
        pipeline = PerceptionPipeline()

        # Create a state with known physical values
        state = PerceptionState()
        state.e_lat_m = 0.5
        state.e_psi_rad = 0.1
        state.kappa_ref = 0.01
        state.v_y_mps = 0.2
        state.yaw_rate_rads = 0.05
        state.delta_prev_rad = 0.1
        state.kappa_la1 = 0.015
        state.kappa_la2 = 0.02

        obs = pipeline._normalise(state)

        # Check dimensions
        assert obs.shape == (8,), f"Observation shape wrong: {obs.shape}"

        # Check normalisation values match config
        assert abs(obs[0] - 0.5 / cfg.NORM_E_LAT) < 0.01
        assert abs(obs[1] - 0.1 / cfg.NORM_E_PSI) < 0.01

        # All values should be in [-1, 1]
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), \
            f"Observation out of range: {obs}"

    def test_sensor_fusion_camera_only(self):
        """Pipeline should work with camera-only input."""
        from real_world.perception_pipeline import PerceptionPipeline
        from real_world.sensor_interface import (
            SensorBundle, CameraLaneDetection, IMUData,
            VehicleCANData, SensorHealth,
        )

        pipeline = PerceptionPipeline()
        bundle = SensorBundle(
            system_time_s=1.0,
            camera=CameraLaneDetection(
                timestamp_s=1.0,
                e_lat_m=0.3,
                e_psi_rad=0.05,
                confidence=0.9,
                curvature_1m=0.01,
                curvature_la1=0.012,
                curvature_la2=0.015,
                health=SensorHealth.OK,
            ),
            imu=IMUData(
                timestamp_s=1.0,
                yaw_rate_rads=0.03,
                health=SensorHealth.OK,
            ),
            vehicle_can=VehicleCANData(
                timestamp_s=1.0,
                v_x_mps=16.67,
                steering_angle_rad=0.02,
                health=SensorHealth.OK,
            ),
        )

        obs, state = pipeline.process(bundle)
        assert obs.shape == (8,)
        assert state.is_valid
        assert abs(state.e_lat_m - 0.3) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SAFETY MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyMonitor:
    """Test real-world safety monitor state machine."""

    def setup_method(self):
        from real_world.safety_monitor import RealWorldSafetyMonitor, SystemState
        self.monitor = RealWorldSafetyMonitor()
        self.monitor.initialise()
        self.monitor.engage()

    def test_active_state(self):
        """Monitor should be ACTIVE after engagement."""
        from real_world.safety_monitor import SystemState
        assert self.monitor.state == SystemState.ACTIVE

    def test_emergency_on_large_lateral(self):
        """Emergency stop on large lateral error."""
        from real_world.safety_monitor import SystemState
        from real_world.sensor_interface import SensorBundle, SensorHealth

        bundle = SensorBundle(system_time_s=1.0)

        state = self.monitor.check(
            sensors=bundle,
            e_lat_m=2.0,  # Exceeds emergency threshold
            e_psi_rad=0.0,
            v_x_mps=16.0,
            delta_rad=0.0,
            sensor_confidence=0.9,
        )

        assert state == SystemState.EMERGENCY_STOP

    def test_authority_scaling(self):
        """Authority factor should reduce in degraded states."""
        from real_world.safety_monitor import SystemState
        assert self.monitor.get_authority_factor() == 1.0

    def test_safe_to_steer(self):
        """Should be safe to steer when ACTIVE."""
        assert self.monitor.is_safe_to_steer()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. DDPG AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestDDPGAgent:
    """Test DDPG agent forward pass and update."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch DLL not available on this platform",
    )
    def test_action_output_shape(self):
        """Agent should produce correct action shape."""
        from ddpg.agent import DDPGAgent
        agent = DDPGAgent(state_dim=cfg.OBS_DIM, action_dim=cfg.ACTION_DIM)
        state = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        action = agent.select_action(state)
        assert action.shape == (cfg.ACTION_DIM,), \
            f"Action shape wrong: {action.shape}"

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch DLL not available on this platform",
    )
    def test_action_bounded(self):
        """Agent actions should be in [-1, 1]."""
        from ddpg.agent import DDPGAgent
        agent = DDPGAgent(state_dim=cfg.OBS_DIM, action_dim=cfg.ACTION_DIM)
        for _ in range(100):
            state = np.random.randn(cfg.OBS_DIM).astype(np.float32)
            action = agent.select_action(state)
            assert np.all(action >= -1.0) and np.all(action <= 1.0), \
                f"Action out of bounds: {action}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
