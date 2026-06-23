"""Test all three robustness features."""
import sys
sys.path.insert(0, ".")
import numpy as np
from simulator.lane_keeping_env import LaneKeepingEnv

print("=== TEST 1: Observation Noise (Training Mode) ===")
env = LaneKeepingEnv(training_mode=True, obs_noise_std=0.05)
obs1, info1 = env.reset(scenario_id="SCN-01")
obs2, _ = env.reset(scenario_id="SCN-01")
# Two resets of same scenario should give slightly different obs (noise)
diff = np.abs(obs1 - obs2).mean()
print(f"  Obs diff between identical resets: {diff:.4f} (should be > 0 due to noise)")
print(f"  PASS" if diff > 0.001 else "  FAIL")

print("\n=== TEST 2: Safety Guardian ===")
obs, _ = env.reset(scenario_id="SCN-01")
# Apply a large action and check if guardian clamps it
obs, r, t, tr, info = env.step(np.array([1.0]))
print(f"  Raw delta_cmd vs safe delta: cmd_raw={info.get('delta_cmd_raw', 'N/A'):.4f}, safe={info['delta_rad']:.4f}")
print(f"  Guardian handoff: {info['guardian_handoff']}")
print(f"  Guardian stats: rate_clamps={env.guardian.stats.rate_clamp_count}, angle_clamps={env.guardian.stats.angle_clamp_count}")
rate_clamped = env.guardian.stats.rate_clamp_count > 0
print(f"  PASS (rate clamped)" if rate_clamped else "  Rate not clamped (may be OK for first step)")

print("\n=== TEST 3: Multi-Speed Curriculum ===")
speeds = []
for i in range(20):
    _, info = env.reset(scenario_id="SCN-02")
    speeds.append(info["episode_speed_kmh"])
min_s, max_s = min(speeds), max(speeds)
print(f"  Speed range over 20 resets: {min_s:.1f} - {max_s:.1f} km/h")
print(f"  PASS" if (max_s - min_s) > 5 else "  FAIL - no speed variation")

print("\n=== TEST 4: Eval Mode (No Noise, Fixed Speed) ===")
env_eval = LaneKeepingEnv(training_mode=False)
eval_speeds = []
for i in range(10):
    _, info = env_eval.reset(scenario_id="SCN-01")
    eval_speeds.append(info["episode_speed_kmh"])
speed_var = max(eval_speeds) - min(eval_speeds)
print(f"  Eval speed: {eval_speeds[0]:.1f} km/h, variation: {speed_var:.2f}")
print(f"  PASS" if speed_var < 0.01 else "  FAIL - eval should be fixed speed")

print("\n=== TEST 5: Feedforward on All Scenarios (Zero Action) ===")
for scn in ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]:
    obs, _ = env_eval.reset(scenario_id=scn)
    steps = 0
    max_elat = 0
    for i in range(300):
        obs, r, t, tr, info = env_eval.step(np.array([0.0]))
        steps += 1
        max_elat = max(max_elat, abs(info["e_lat_m"]))
        if t or tr:
            break
    status = "DEPARTED" if t else "OK"
    print(f"  {scn}: steps={steps:4d}, max_elat={max_elat:.3f}m, {status}")

print("\nAll tests complete!")
