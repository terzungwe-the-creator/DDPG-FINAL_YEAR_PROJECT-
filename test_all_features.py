"""Test all features including SCN-03 fix."""
import sys
sys.path.insert(0, ".")
import numpy as np
from simulator.lane_keeping_env import LaneKeepingEnv

print("=== SCN-03 Feedforward Fix Test (Zero Agent Action) ===")
env_eval = LaneKeepingEnv(training_mode=False)
for scn in ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]:
    obs, _ = env_eval.reset(scenario_id=scn)
    steps = 0
    max_elat = 0
    for i in range(500):
        obs, r, t, tr, info = env_eval.step(np.array([0.0]))
        steps += 1
        max_elat = max(max_elat, abs(info["e_lat_m"]))
        if t or tr:
            break
    status = "DEPARTED" if t else "OK"
    result = "PASS" if not t else "FAIL"
    print(f"  {scn}: steps={steps:4d}, max_elat={max_elat:.3f}m, {status} [{result}]")

print("\n=== Domain Randomization Test ===")
env_train = LaneKeepingEnv(training_mode=True)
params = []
for i in range(10):
    _, info = env_train.reset(scenario_id="SCN-01")
    params.append((info["episode_speed_kmh"], info["dr_mass_kg"]))
speeds = [p[0] for p in params]
masses = [p[1] for p in params]
print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h")
print(f"  Mass range:  {min(masses):.0f} - {max(masses):.0f} kg")
print(f"  Variation OK: {max(speeds)-min(speeds) > 5 and max(masses)-min(masses) > 10}")

print("\n=== Guardian + Wind + Latency Test ===")
obs, _ = env_train.reset(scenario_id="SCN-02")
for i in range(50):
    obs, r, t, tr, info = env_train.step(np.array([0.0]))
print(f"  Guardian rate clamps: {env_train.guardian.stats.rate_clamp_count}")
print(f"  DR active: {env_train.domain_randomizer.enabled}")
print(f"  Wind force: {env_train.domain_randomizer.params.wind_force_n:.1f} N")
print(f"  Obs latency: {env_train.domain_randomizer.params.obs_latency_steps} steps")

print("\nAll tests complete!")
