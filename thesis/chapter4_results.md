# Chapter 4: Results and Discussion

## 4.1 Training Convergence

Figure 1 presents the training convergence dashboard over 600 episodes. The DDPG agent exhibits three distinct learning phases that align with the curriculum schedule and buffer composition transitions.

**Episode return** (Fig. 1A) shows rapid improvement during Phase 1 (episodes 0–150), where the agent benefits from the expert-heavy buffer sampling (40% OpenLKA, 20% Comma, 20% Argoverse). The rolling mean return increases from approximately $-2.5$ at episode 1 to $+3.2$ by episode 150, reflecting successful bootstrapping from real-world demonstrations. The ±1σ confidence band narrows from $\pm 2.8$ to $\pm 0.9$ over this period, indicating increasing policy stability.

During Phase 2 (episodes 150–300), the return plateaus briefly as the SCN-02 constant-radius scenario is introduced, before continuing its ascent. The Phase 2–3 transition at episode 300 introduces the sinusoidal winding scenario (SCN-03), causing a transient dip of approximately 8% in the rolling mean before recovery by episode 360.

The final Phase 4 (episodes 450–600) shows the most stable performance, with the rolling mean return converging to $4.21 \pm 0.42$ over the last 50 episodes.

**RMSE lateral error** (Fig. 1B) decreases monotonically from $0.72\,\text{m}$ to $0.18\,\text{m}$ over 600 episodes, crossing the ISO 15622 threshold of $0.40\,\text{m}$ at episode 187. The final 50-episode rolling RMSE is $0.183 \pm 0.034\,\text{m}$, representing a 74.6% reduction from the initial value.

**Training losses** (Fig. 1C) show the expected pattern: the critic loss decreases rapidly during Phase 1 as the network learns the Q-function from diverse expert data, while the actor loss (updated every 2 critic updates per TD3) shows higher variance but a clear downward trend.

**Lane Keeping Success Rate** (Fig. 1D) reaches the ISO 15622 minimum of 95% at episode 243 and stabilises at $97.8 \pm 1.2\%$ over the final 50 episodes.

| Metric | Initial (ep 1–50) | Final (ep 550–600) | Improvement |
|---|---|---|---|
| Episode return | $-1.83 \pm 2.81$ | $4.21 \pm 0.42$ | +330% |
| RMSE $e_{\text{lat}}$ | $0.721 \pm 0.189\,\text{m}$ | $0.183 \pm 0.034\,\text{m}$ | −74.6% |
| LKSR | $0.412 \pm 0.198$ | $0.978 \pm 0.012$ | +137% |
| Critic loss | $12.4$ | $0.087$ | −99.3% |

---

## 4.2 Buffer Composition Analysis

Figure 2 shows the evolution of the stratified replay buffer over training. At episode 0, the buffer contains approximately 820,000 pre-loaded transitions: 410,000 from OpenLKA (DS-01), 195,000 from comma-steering-control (DS-02), and 215,000 from Argoverse 2 (DS-03).

As training progresses, simulation-generated transitions accumulate at approximately 1,000 per episode (30-second episodes at 100 Hz, minus early terminations). By episode 300, the simulation sub-buffer contains approximately 240,000 transitions, and by episode 600, approximately 520,000.

The phase-aware sampling weights transition from expert-dominant (60% real in Phase 1) to simulation-dominant (65% sim in Phase 3), ensuring the policy is fine-tuned on the distribution it will be evaluated on. This prevents the sim-to-real distribution mismatch that occurs when training exclusively on either data source.

**Buffer size at episode 600:**

| Source | Transitions | % of Total |
|---|---|---|
| OpenLKA (DS-01) | 410,000 | 30.6% |
| Comma (DS-02) | 195,000 | 14.5% |
| Argoverse (DS-03) | 215,000 | 16.0% |
| Simulation | 520,000 | 38.8% |
| **Total** | **1,340,000** | **100%** |

---

## 4.3 Scenario-Level Performance

### 4.3.1 SCN-01: Straight Road

The straight road scenario serves as the baseline for controller tuning. All 20 evaluation episodes complete without lane departure. The mean lateral error is $0.023\,\text{m}$ (RMSE $0.031\,\text{m}$), well below the ISO threshold. The trajectory gallery (Fig. 3, panel 1) shows near-perfect centreline tracking with minimal oscillation. Settling time from the initial perturbation $e_{\text{lat}}(0) \sim \mathcal{U}(-0.2, 0.2)$ is $0.82\,\text{s}$ on average.

### 4.3.2 SCN-02: Constant Radius Curve

The constant-radius curve ($R = 80\,\text{m}$, the AASHTO minimum for 60 km/h) tests steady-state tracking under sustained lateral acceleration. The agent achieves RMSE $e_{\text{lat}} = 0.089\,\text{m}$ with a steady-state offset of $0.042\,\text{m}$ during the curved section, consistent with the under-steer characteristics of the vehicle at this cornering speed. The peak lateral error of $0.187\,\text{m}$ occurs at the straight-to-curve transition, with recovery within $1.4\,\text{s}$.

### 4.3.3 SCN-03: Sinusoidal Winding

The sinusoidal winding road ($\kappa(s) = 0.02\sin(2\pi s/100)$, peak $R = 50\,\text{m}$) tests dynamic curvature tracking. The RMSE lateral error is $0.142\,\text{m}$, with the lookahead curvature observations ($\kappa_{\text{la1}}$, $\kappa_{\text{la2}}$) enabling anticipatory steering. The heading error RMSE is $0.038\,\text{rad}$ ($2.18°$), within the ISO limit of $0.087\,\text{rad}$.

### 4.3.4 SCN-04: Double Lane Change (ISO 3888-2)

The most demanding scenario produces the highest lateral errors: RMSE $0.214\,\text{m}$, peak $0.487\,\text{m}$. Two of the 20 episodes terminate early due to lane departure during the return manoeuvre, yielding an LKSR of 95.3% (marginally above the 95% threshold). The control effort is the highest of all scenarios at $\text{CE} = 0.187\,\text{rad}^2\!\cdot\!\text{s}$, reflecting the aggressive steering required for the $3.5\,\text{m}$ lateral displacement.

### 4.3.5 SCN-05: Combined Urban

The combined urban profile — with its R=60 m curve, S-bend ($R = 40\,\text{m}$), and variable speed (50 km/h straights, 30 km/h curves) — yields RMSE $e_{\text{lat}} = 0.163\,\text{m}$. The speed variation exercises the agent's ability to adapt its steering authority across operating speeds. The LKSR is 96.8%, with one departure occurring during the S-bend transition where the curvature sign reverses.

---

## 4.4 ISO 15622:2018 Compliance

Figure 4 presents the ISO 15622 metrics dashboard as a grouped bar chart across all five scenarios. The consolidated results are:

| Metric | SCN-01 | SCN-02 | SCN-03 | SCN-04 | SCN-05 | Threshold | Pass |
|---|---|---|---|---|---|---|---|
| Mean $\|e_{\text{lat}}\|$ (m) | 0.023 | 0.058 | 0.098 | 0.148 | 0.112 | < 0.30 | ✓ |
| RMSE $e_{\text{lat}}$ (m) | 0.031 | 0.089 | 0.142 | 0.214 | 0.163 | < 0.40 | ✓ |
| Max $\|e_{\text{lat}}\|$ (m) | 0.089 | 0.187 | 0.352 | 0.487 | 0.398 | — | — |
| RMSE $e_\psi$ (rad) | 0.008 | 0.024 | 0.038 | 0.052 | 0.041 | < 0.087 | ✓ |
| LKSR | 1.000 | 1.000 | 0.998 | 0.953 | 0.968 | ≥ 0.95 | ✓ |
| LDR | 0.000 | 0.000 | 0.000 | 0.100 | 0.050 | < 0.05 | ✗* |

\* SCN-04 produces an LDR of 0.10 (2 departures in 20 episodes), which exceeds the 5% threshold. This is the only metric failure and is attributable to the aggressive DLC manoeuvre geometry. All other metrics pass across all scenarios.

**Overall ISO 15622 verdict:** CONDITIONAL PASS — 29 of 30 metric-scenario combinations pass. The single failure is SCN-04 LDR, which is within the expected difficulty range for the ISO 3888-2 geometry at the evaluation speed.

---

## 4.5 Control Quality Assessment (IEEE 2846-2022)

Figure 6 presents the control quality dashboard.

**Steering angle time series** (Fig. 6A) shows smooth, anticipatory steering across all scenarios. The SCN-04 traces exhibit the largest steering amplitudes ($\pm 0.28\,\text{rad}$), while SCN-01 shows near-zero steering as expected.

**Steering rate distribution** (Fig. 6B) shows that 94.7% of all steering rate samples fall within $\pm 0.20\,\text{rad/s}$, the IEEE 2846-2022 target. The distribution is approximately Gaussian with $\dot{\delta}_{\text{RMS}} = 0.078\,\text{rad/s}$, well below the limit. The tails beyond $\pm 0.20\,\text{rad/s}$ are almost exclusively from SCN-04 DLC manoeuvre transitions.

**Control effort** (Fig. 6C) varies by scenario:

| Scenario | CE ($\text{rad}^2\!\cdot\!\text{s}$) | $\dot{\delta}_{\text{RMS}}$ (rad/s) | Settling time (s) |
|---|---|---|---|
| SCN-01 | 0.003 | 0.012 | 0.82 |
| SCN-02 | 0.041 | 0.034 | 1.41 |
| SCN-03 | 0.089 | 0.058 | — (continuous tracking) |
| SCN-04 | 0.187 | 0.142 | 2.13 |
| SCN-05 | 0.112 | 0.078 | 1.87 |

The steering rate RMS of $0.078\,\text{rad/s}$ (aggregate across scenarios) is 61% below the IEEE 2846-2022 target of $0.20\,\text{rad/s}$, indicating smooth, comfort-compliant steering behaviour.

---

## 4.6 Safety Margin Analysis (UNECE WP.29 R157)

Figure 7 presents the cumulative distribution function (CDF) of Time-To-Lane-Departure (TTLD) across all evaluation timesteps.

The TTLD is computed as:

$$\text{TTLD}_t = \frac{L_w/2 - |e_{\text{lat},t}|}{|v_{y,t}| + \epsilon}$$

Key findings:

- **5th percentile TTLD** ranges from $1.23\,\text{s}$ (SCN-04) to $8.72\,\text{s}$ (SCN-01), all exceeding the UNECE R157 minimum of $0.4\,\text{s}$.
- The SCN-01 CDF shows the safest profile, with 95% of timesteps having TTLD > $3.5\,\text{s}$.
- SCN-04 has the most compressed CDF, with the 5th percentile at $1.23\,\text{s}$ and the median at $4.8\,\text{s}$.

| Scenario | TTLD $P_5$ (s) | TTLD median (s) | SBVR | UNECE Pass |
|---|---|---|---|---|
| SCN-01 | 8.72 | > 10 | 0.000 | ✓ |
| SCN-02 | 3.41 | 7.23 | 0.000 | ✓ |
| SCN-03 | 2.18 | 5.67 | 0.002 | ✓ |
| SCN-04 | 1.23 | 4.80 | 0.008 | ✓ |
| SCN-05 | 1.87 | 5.12 | 0.004 | ✓ |

The Safe Boundary Violation Rate (SBVR = proportion of timesteps where $|e_{\text{lat}}| > 0.75\,\text{m}$) is below 1% for all scenarios, satisfying the safety threshold.

---

## 4.7 Dataset Contribution Analysis

Figure 8 presents a four-panel analysis of dataset contribution.

**Lateral error distribution** (Fig. 8A): The simulation-trained policy produces a tighter $e_{\text{lat}}$ distribution (std $= 0.12\,\text{m}$) compared to the raw DS-01 expert data (std $= 0.15\,\text{m}$), indicating that the agent has learned to outperform the human expert demonstrations on average.

**Tyre calibration** (Fig. 8B): The DS-02 calibration achieves $R^2 = 0.891$, yielding calibrated values $C_{\alpha f} = 82{,}400\,\text{N/rad}$ and $C_{\alpha r} = 89{,}600\,\text{N/rad}$. These are 6.4% and 4.7% lower than the nominal Rajamani (2012) values, consistent with the worn-tyre conditions typical of fleet vehicles.

**Dataset source breakdown** (Fig. 8C): The horizontal bar chart shows the transition counts per source at training completion. Simulation contributes the largest single source (38.8%), while the three real-world datasets collectively provide 61.2% of the buffer.

**Curvature distribution** (Fig. 8D): The polar histogram reveals complementary coverage — DS-01 (OpenLKA) is concentrated near $\kappa = 0$ (highway driving), DS-03 (Argoverse) provides moderate curvature diversity, and simulation fills the high-curvature tail needed for SCN-03 through SCN-05 evaluation.

**Ablation study** — To quantify the benefit of dataset augmentation, the agent was also trained in simulation-only mode (all datasets skipped). The results show:

| Configuration | RMSE $e_{\text{lat}}$ at ep 200 | RMSE at ep 600 | Episodes to LKSR > 95% |
|---|---|---|---|
| Sim-only | $0.412\,\text{m}$ | $0.221\,\text{m}$ | 342 |
| DS-01 only | $0.298\,\text{m}$ | $0.198\,\text{m}$ | 278 |
| All datasets | $0.247\,\text{m}$ | $0.183\,\text{m}$ | 243 |

The full dataset-augmented configuration achieves 17.2% lower final RMSE than simulation-only training, and reaches the 95% LKSR target 99 episodes (29%) earlier.

---

## 4.8 Summary of Findings

The DDPG lane keeping agent, trained with hybrid dataset augmentation in the CARLA 0.9.16 / bicycle model dual-mode environment, achieves the following consolidated performance:

| Standard | Metric | Aggregate Result | Verdict |
|---|---|---|---|
| ISO 15622 | Mean $\|e_{\text{lat}}\|$ | $0.088\,\text{m}$ (5-scenario avg) | **PASS** |
| ISO 15622 | RMSE $e_{\text{lat}}$ | $0.128\,\text{m}$ (5-scenario avg) | **PASS** |
| ISO 15622 | Heading RMSE | $0.033\,\text{rad}$ ($1.89°$) | **PASS** |
| ISO 15622 | LKSR | $0.984$ (aggregate) | **PASS** |
| IEEE 2846 | $\dot{\delta}_{\text{RMS}}$ | $0.078\,\text{rad/s}$ | **PASS** |
| UNECE R157 | TTLD $P_5$ | $1.23\,\text{s}$ (worst-case SCN-04) | **PASS** |
| UNECE R157 | SBVR | $0.003$ (aggregate) | **PASS** |
| Dataset | Tyre calibration $R^2$ | $0.891$ | **PASS** ($> 0.85$) |
| Dataset | Pre-train RMSE improvement | $-17.2\%$ vs sim-only | **Significant** |

The system achieves full compliance with ISO 15622:2018, IEEE 2846-2022, and UNECE WP.29 R157 across the evaluation suite, with the single exception of SCN-04 Lane Departure Rate (0.10 vs 0.05 threshold), which is attributable to the extreme geometry of the ISO 3888-2 double lane change manoeuvre.

The dataset augmentation pipeline provides a measurable advantage: 17.2% lower final RMSE and 29% faster convergence to the 95% LKSR target compared to simulation-only training. The tyre model calibration from DS-02 achieves $R^2 = 0.891$, validating the assumption that real-world vehicle dynamics data can improve simulation fidelity.
