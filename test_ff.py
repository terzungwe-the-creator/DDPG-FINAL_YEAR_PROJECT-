"""Quick test: verify feedforward compensation works on SCN-03 (the hardest scenario)."""
import sys
sys.path.insert(0, ".")
import numpy as np
from simulator.lane_keeping_env import LaneKeepingEnv

env = LaneKeepingEnv()

# Test: with zero agent action, feedforward alone should keep the vehicle roughly centered
for scn in ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]:
    obs, _ = env.reset(scenario_id=scn)
    r_sum = 0
    steps = 0
    max_elat = 0
    for i in range(500):
        obs, r, terminated, truncated, info = env.step(np.array([0.0]))
        r_sum += r
        steps += 1
        max_elat = max(max_elat, abs(info["e_lat_m"]))
        if terminated or truncated:
            break
    status = "DEPARTED" if terminated else "OK"
    print(f"{scn}: steps={steps:4d}, reward={r_sum:7.1f}, max_elat={max_elat:.3f}m, status={status}")

print("\nIf max_elat < 1.75m on curves, feedforward is working!")
