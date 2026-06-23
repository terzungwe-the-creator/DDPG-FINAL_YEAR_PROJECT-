"""Audit physical limits of the system to determine if all 5 scenarios CAN pass."""
import sys
sys.path.insert(0, ".")
import numpy as np
import config as cfg
from simulator.lane_keeping_env import LaneKeepingEnv

# We will use a Stanley controller to see if perfect feedback can pass the scenarios
# If Stanley fails, RL will likely fail too (due to physical/guardian limits).

class StanleyController:
    def __init__(self, k_e=2.5, k_v=0.0):
        self.k_e = k_e
        self.k_v = k_v

    def compute(self, e_lat, e_psi, v_x):
        # Stanley control law: delta = -e_psi + arctan(-k_e * e_lat / (v_x + k_v))
        # e_psi < 0 means heading right of path, need to steer left (delta > 0)
        delta = -e_psi + np.arctan2(-self.k_e * e_lat, v_x + self.k_v)
        return delta

print("=== Physical Audit: Stanley Controller Test ===")
env = LaneKeepingEnv(training_mode=False)
stanley = StanleyController(k_e=3.0)

for scn in ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]:
    obs, info = env.reset(scenario_id=scn)
    
    steps = 0
    max_elat = 0.0
    clips = 0
    
    # We will bypass the environment's step function partially to inject Stanley as the "perfect" RL agent
    # Actually, we can just use the environment's step, but our action is the correction.
    # We need to reverse-engineer the action.
    # delta_cmd = delta_nominal + action * 0.5 * DELTA_MAX
    # action = (delta_cmd - delta_nominal) / (0.5 * DELTA_MAX)
    
    for i in range(1000):
        v_x = env.vehicle.v_x
        e_lat = env.vehicle.lateral_error
        e_psi = env.vehicle.heading_error
        
        # Desired total steering from Stanley
        delta_desired = stanley.compute(e_lat, e_psi, v_x)
        
        # What is the nominal?
        profile = env.profiles[scn]
        preview_time = 0.4
        s_preview = env.arc_length_s + v_x * preview_time
        kappa_preview = profile.get_kappa_at_s(s_preview)
        
        K_us = (cfg.VEHICLE_MASS / cfg.VEHICLE_WHEELBASE) * (
            cfg.VEHICLE_LR / cfg.TYRE_CAF_NOMINAL - cfg.VEHICLE_LF / cfg.TYRE_CAR_NOMINAL
        )
        delta_nominal = (
            cfg.VEHICLE_WHEELBASE * kappa_preview
            + K_us * (v_x ** 2) * kappa_preview
        )
        
        # Calculate needed correction
        delta_correction = delta_desired - delta_nominal
        
        # Convert to RL action space [-1, 1]
        action_scalar_raw = delta_correction / (0.3 * cfg.DELTA_MAX)
        action_scalar = np.clip(action_scalar_raw, -1.0, 1.0)
        
        if abs(action_scalar_raw) > 1.0:
            clips += 1
            
        obs, r, t, tr, info = env.step(np.array([action_scalar]))
        
        if scn == "SCN-02" and steps % 50 == 0:
            print(f"[{steps:3d}] e_lat: {e_lat:6.3f}, e_psi: {e_psi:6.3f}, d_nom: {delta_nominal:6.3f}, d_des: {delta_desired:6.3f}, d_cor: {delta_correction:6.3f}, act: {action_scalar:6.3f}")
            
        steps += 1
        max_elat = max(max_elat, abs(info["e_lat_m"]))
        if t or tr:
            break
            
    status = "DEPARTED" if t else "OK"
    result = "PASS" if not t else "FAIL"
    print(f"  {scn}: steps={steps:4d}, max_elat={max_elat:.3f}m, {status} [{result}]")
    print(f"     -> rate clamps: {env.guardian.stats.rate_clamp_count}")
    print(f"     -> action clips: {clips}")

print("\nAudit Complete.")
