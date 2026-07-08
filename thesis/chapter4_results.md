# Chapter 4: Results and Discussion

## 4.1 Introduction

This chapter presents the comprehensive evaluation of the Deep Deterministic Policy Gradient (DDPG) lane keeping system developed in this work. The trained agent was subjected to a rigorous evaluation protocol following the guidelines of ISO 15622:2018, IEEE 2846-2022, and UNECE WP.29 R157. 

The evaluation encompasses 100 deterministic episodes (20 episodes across 5 distinct road scenarios) to assess the system's lateral tracking accuracy, steering smoothness, and safety margins. The results are discussed in the context of the regulatory thresholds and the fundamental trade-offs between tracking performance and control effort.

The training was conducted for 600 episodes, taking a total wall-clock time of 3,325 seconds (approximately 55 minutes) on a consumer-grade CPU backend using the custom analytical DDPG implementation.

---

## 4.2 Training Progression and Convergence

### 4.2.1 Learning Dynamics

The agent's learning progression through the four curriculum phases provides insight into the DDPG algorithm's sample efficiency and stability.

- **Phase 1 (Episodes 0–49, Straight Road)**: The agent rapidly learned to stabilise the vehicle. Initial random exploration (warmup) resulted in immediate lane departures (terminal reward of -10). However, by episode 30, the agent began tracking the centreline, and the episodic reward increased substantially, though lateral RMSE remained high (~0.9 m) due to early exploration noise.
  
- **Phase 2 (Episodes 50–149, Constant Curves)**: The introduction of constant-radius curves initially destabilised the policy, causing a temporary dip in rewards and LKSR (Lane Keeping Success Rate). The agent had to learn to utilise the feedforward curvature term (κ_la1, κ_la2) to anticipate curves. By episode 120, the agent converged to a stable policy for both straights and curves.
  
- **Phase 3 (Episodes 150–299, Winding Roads)**: The addition of sinusoidal curvature tested the agent's dynamic response. The RMSE initially spiked but steadily declined as the critic network mapped the continuous varying state space.
  
- **Phase 4 (Episodes 300–600, Full Scenarios)**: All scenarios, including the complex SCN-04 (Double Lane Change) and SCN-05 (Urban Profile), were introduced. The learning rate decay at episodes 450 and 540 successfully stabilised the policy, settling the rolling RMSE.

### 4.2.2 Final Training State

By the end of training (Episode 600), the rolling average RMSE across the final 50 episodes reached a minimum of **0.7394 m**, with a best-rolling-average of **0.7526 m**. While the agent learned to navigate the track without catastrophic failure, the final tracking error remained higher than the ISO 15622 target of 0.40 m.

![Training Convergence and Metrics](c:/Users/DELL/Desktop/preacher/results/figures/fig1_training_convergence.png)

---

## 4.3 Evaluation Results: ISO 15622:2018 Performance

The trained agent was evaluated using a deterministic policy (zero noise) across 100 episodes. The aggregated results across the five scenarios are summarised in Table 4.1.

**Table 4.1: ISO 15622:2018 Metrics Summary**

| Metric | Target | SCN-01 (Straight) | SCN-02 (Curve) | SCN-03 (Winding) | SCN-04 (Lane Change) | SCN-05 (Urban) |
|--------|--------|-------------------|----------------|------------------|----------------------|----------------|
| Mean \|e_lat\| (m) | < 0.30 | 0.568 | 0.553 | 0.556 | 0.550 | 0.557 |
| RMSE e_lat (m) | < 0.40 | 0.768 | 0.764 | 0.767 | 0.758 | 0.769 |
| Max \|e_lat\| (m) | — | 1.828 | 1.830 | 1.832 | 1.823 | 1.829 |
| RMSE e_ψ (rad) | < 0.087 | 0.264 | 0.265 | 0.258 | 0.267 | 0.267 |
| LKSR (%) | ≥ 95.0 | 68.6% | 68.9% | 68.5% | 69.3% | 68.8% |
| **Pass/Fail** | **Pass** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

![ISO 15622 Performance Metrics](c:/Users/DELL/Desktop/preacher/results/figures/fig4_iso_metrics.png)

### 4.3.1 Lateral Error Analysis (M-01, M-02, M-03)

The system failed to meet the strict ISO 15622:2018 requirements for lateral tracking. The RMSE across all scenarios averaged roughly **0.765 m**, which is significantly above the 0.40 m limit. Similarly, the mean absolute lateral error hovered around **0.55 m**, missing the 0.30 m target.

The maximum lateral deviation (e_lat_max) exceeded 1.82 m across all scenarios. Given the lane half-width of 1.75 m, these maximum deviations indicate brief lane departure events. The consistency of these results across drastically different geometries (from straight roads to double lane changes) suggests a systemic under-actuation or insufficient corrective gain in the policy, rather than an inability to handle specific curve types.

![Lateral Error Time Series](c:/Users/DELL/Desktop/preacher/results/figures/fig5_lateral_error_series.png)

### 4.3.2 Heading Error Analysis (M-04)

The heading error RMSE (M-04) averaged **0.26 rad (14.9°)**, failing to meet the target of 0.087 rad (5°). This significant heading misalignment correlates strongly with the high lateral tracking errors; the vehicle is likely oscillating or "crabbing" within the lane due to phase lag in the control response.

### 4.3.3 Lane Keeping Success Rate (M-06)

The Lane Keeping Success Rate (LKSR), defined as the percentage of time the vehicle remains within the 0.75 m departure threshold, averaged **~68.8%** across scenarios. This falls well short of the 95% requirement. Consequently, the Lane Departure Rate (LDR) averaged ~31.2%, meaning the vehicle spent nearly a third of its operation time outside the safe boundary zone.

---

## 4.4 Evaluation Results: IEEE 2846-2022 Control Quality

Table 4.2 details the control quality metrics aimed at assessing smoothness, energy consumption, and transient response.

**Table 4.2: IEEE 2846-2022 Control Quality Metrics**

| Metric | Target | SCN-01 | SCN-02 | SCN-03 | SCN-04 | SCN-05 |
|--------|--------|--------|--------|--------|--------|--------|
| Steering Rate RMS (rad/s) | < 0.20 | 4.62 | 4.61 | 3.44 | 4.60 | 4.60 |
| Control Effort (rad²·s) | Minimise | 1.03 | 1.03 | 0.70 | 1.04 | 1.04 |
| Settling Time (s) | Minimise | N/A | N/A | N/A | N/A | N/A |
| Overshoot (%) | Minimise | 104.4% | 104.6% | 104.7% | 104.2% | 104.5% |

### 4.4.1 Steering Smoothness (M-07, M-08)

The steering rate RMS significantly violated the IEEE 2846-2022 comfort limit of 0.20 rad/s, recording values between **3.44 and 4.62 rad/s**. This indicates highly aggressive, high-frequency steering actuation. The policy has learned a "bang-bang" style control strategy, rapidly oscillating the steering wheel to stay within the lane rather than finding a smooth equilibrium point.

Interestingly, SCN-03 (the sinusoidal winding road) exhibited a noticeably lower steering rate RMS (3.44 rad/s) and lower control effort (0.70 rad²·s) compared to the straight road (4.62 rad/s). This counterintuitive result suggests that the feedforward curvature term dominates the steering action on winding roads, effectively overriding the agent's noisy feedback corrections and producing a smoother overall trajectory.

### 4.4.2 Transient Response (M-09, M-10)

The system failed to achieve a settling time in any scenario (recorded as 0.0 s, meaning it never maintained an error below 0.10 m for 0.5 s continuously). The overshoot was consistently ~104%, meaning the peak error was slightly larger than the initial random perturbation.

![Trajectory Tracking Gallery](c:/Users/DELL/Desktop/preacher/results/figures/fig3_trajectory_gallery.png)

---

## 4.5 Evaluation Results: UNECE WP.29 R157 Safety Margins

Table 4.3 presents the safety-critical metrics required by UNECE WP.29 R157.

**Table 4.3: UNECE WP.29 R157 Safety Metrics**

| Metric | SCN-01 | SCN-02 | SCN-03 | SCN-04 | SCN-05 |
|--------|--------|--------|--------|--------|--------|
| 5th Percentile TTLD (s) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Safety Boundary Violation Rate (%) | 46.0% | 46.6% | 46.8% | 47.8% | 47.1% |

### 4.5.1 Time-to-Lane-Departure (M-11)

The 5th percentile Time-to-Lane-Departure (TTLD_p5) was exactly **0.00 seconds** across all scenarios. This is a direct consequence of the max(|e_lat|) exceeding 1.75 m. Since the vehicle crossed the physical lane boundary, the TTLD dropped to zero. A safe system should maintain a TTLD_p5 > 0.5 seconds to allow for driver intervention.

![Time-to-Lane-Departure (TTLD) Safety Margins](c:/Users/DELL/Desktop/preacher/results/figures/fig7_ttld_safety.png)

### 4.5.2 Safety Boundary Violations (M-12)

The Safety Boundary Violation Rate (SBVR) exceeded **46%** in all scenarios. This indicates that for nearly half of the evaluation duration, the vehicle was operating outside the designated safe zone (e_lat > 0.75 m), confirming the high LDR reported in Section 4.3.

---

## 4.6 Discussion and Failure Analysis

The evaluation results indicate a failure to meet the required ISO and IEEE standards. The uniformity of the metrics across different road geometries provides clear evidence of the underlying issues:

### 4.6.1 High-Frequency Policy Oscillation
The extremely high steering rate RMS (4.6 rad/s) combined with poor tracking accuracy (0.76 m RMSE) is the hallmark of a high-frequency limit cycle. The agent's neural network policy has learned to aggressively counter errors only when they become large, resulting in "ping-ponging" between lane boundaries. 

This behaviour typically arises from:
1. **Insufficient Action Smoothing**: The exponential moving average filter (α = 0.7) applied to the actions was likely insufficient to damp the neural network's high-frequency noise.
2. **Reward Function Tuning**: The smoothness penalty (w_smooth = 1.0) was overpowered by the lateral error penalty (w_lat = 5.0) and heading penalty (w_head = 2.0). The agent accepted the high steering rate penalty because smooth steering resulted in even higher tracking penalties due to phase lag.

### 4.6.2 Sim-Only Training Limitations
The system was trained purely in simulation for 600 episodes. DDPG is notoriously sample-inefficient. While 600 episodes (~60,000 steps) allowed the agent to learn to avoid catastrophic terminal states (staying roughly on the road), it is insufficient for fine-tuning the continuous control policy to millimeter accuracy. 

The original design of this system includes a **Data Fusion Pipeline** (incorporating OpenLKA, comma.ai, and Argoverse datasets) specifically to address this limitation. Pre-loading expert human driving demonstrations into the replay buffer provides the critic network with examples of smooth, high-accuracy tracking. The absence of these datasets in this training run forced the agent to rely entirely on online exploration, which failed to discover the optimal smooth policy within the allowed compute budget.

### 4.6.3 The Effectiveness of Feedforward Control
A notable positive finding is the performance on SCN-03 (Sinusoidal Winding Road). Despite the complex geometry, SCN-03 yielded the lowest control effort and the lowest steering rate RMS. This validates the architectural decision to include an analytical feedforward curvature term. The feedforward controller successfully handled the gross steering requirements for the winding road, leaving the RL agent to handle only minor corrections.

---

## 4.7 Conclusion

The simulated training run of the DDPG lane keeping agent produced a functional but unrefined controller. While the agent successfully avoided catastrophic departures that would terminate the episode early, it operated in a high-frequency oscillatory limit cycle, resulting in an RMSE of ~0.76 m and a steering rate RMS > 3.4 rad/s. Consequently, it failed the strict ISO 15622:2018 and IEEE 2846-2022 standards.

The analysis heavily implicates the lack of expert demonstration data and the short training duration as the primary causes of this sub-optimal convergence. Future work must leverage the implemented Hybrid Stratified Replay Buffer to inject expert human driving data (DS-01, DS-02, DS-03) prior to online training. By bootstrapping the critic with optimal trajectories, the agent can bypass the noisy exploration phase and converge to a smooth, standards-compliant policy.
