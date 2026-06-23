# Chapter 3: Methodology

## 3.1 System Overview

This chapter presents the methodology for developing and evaluating a Deep Deterministic Policy Gradient (DDPG) agent for autonomous lane keeping, augmented with external driving dataset integration. The system operates in a dual-mode simulation environment: a high-fidelity CARLA 0.9.16 simulator with four-wheel PhysX vehicle dynamics for primary training and evaluation, and a nonlinear bicycle model for rapid prototyping and offline result generation. Both backends produce an identical eight-dimensional observation vector, enabling seamless policy transfer between simulation fidelity levels.

The complete pipeline comprises five stages:

1. **Dataset preloading** (Phase 0) — Three external driving datasets are parsed, normalised, and injected into a stratified replay buffer.
2. **Training** (Phases 1–4) — A 1200-episode curriculum progressively introduces harder road scenarios whilst transitioning the buffer sampling weights from expert-dominant to simulation-dominant.
3. **Deterministic evaluation** — 100 episodes (20 per scenario × 5 scenarios) with noise disabled.
4. **Metrics computation** — ISO 15622:2018, IEEE 2846-2022, and UNECE WP.29 R157 compliance assessment.
5. **Publication figure generation** — Eight IEEE-format figures at 300 DPI.

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (CLI)                         │
│   --all | --train | --eval | --plot | --backend          │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
    ┌──────────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
    │  Training Loop  │ │ Evaluator │ │  Plotting  │
    │   (1200 eps)    │ │ (100 eps) │ │ (8 figs)   │
    └──────┬──────────┘ └─────┬─────┘ └────────────┘
           │                  │
    ┌──────▼──────────────────▼──────────────────┐
    │        LaneKeepingEnv (Gymnasium)           │
    │  ┌─────────────┐    ┌───────────────────┐  │
    │  │  Bicycle     │    │  CARLA 0.9.16     │  │
    │  │  Model (RK4) │    │  (4-wheel PhysX)  │  │
    │  └─────────────┘    └───────────────────┘  │
    └────────────────────────────────────────────┘
```

---

## 3.2 Simulation Environment

### 3.2.1 CARLA Simulator Configuration

The primary simulation environment employs CARLA version 0.9.16 (September 2025), an open-source autonomous driving simulator built on Unreal Engine 4.26 with PhysX-based vehicle dynamics. CARLA provides a physically accurate rendering pipeline and a comprehensive Python API for programmatic control of vehicles, sensors, and world state.

The simulator operates in **synchronous mode** with a fixed timestep of $\Delta t = 0.01\,\text{s}$ (100 Hz), ensuring deterministic physics integration. This matches the control frequency specified in ISO 15622:2018 §8.1 for lane keeping assistance system evaluation. The synchronous mode guarantees that no physics steps are skipped between agent decisions, which is critical for reproducible reinforcement learning experiments.

**Server configuration:**

| Parameter | Value | Justification |
|---|---|---|
| CARLA version | 0.9.16 | Latest stable (September 2025) |
| Rendering engine | Unreal Engine 4.26 | PhysX stability |
| Physics timestep | $\Delta t = 0.01\,\text{s}$ | ISO 15622 §8.1 control rate |
| Synchronous mode | Enabled | Deterministic integration |
| Rendering mode | Enabled | Visual debugging capability |
| Traffic manager | Disabled | Ego-only evaluation |

**Map-scenario mapping:**

| Scenario | CARLA Map | Road Geometry |
|---|---|---|
| SCN-01 | Town04 | Long straight highway segments |
| SCN-02 | Town03 | Constant-radius curves ($R = 80\,\text{m}$) |
| SCN-03 | Town07 | Winding rural roads |
| SCN-04 | Town02 | Narrow lanes for double lane change |
| SCN-05 | Town01 | Mixed urban with curves and straights |

### 3.2.2 Four-Wheel Vehicle Dynamics

The ego vehicle employs the Tesla Model 3 blueprint (`vehicle.tesla.model3`) in CARLA, representing a Battery Electric Vehicle (BEV) midsize sedan. CARLA's PhysX engine simulates the vehicle with an independent four-wheel dynamics model that includes:

- **Independent tyre forces** for each wheel (front-left, front-right, rear-left, rear-right)
- **Ackermann steering geometry** with configurable maximum steering angle
- **Suspension spring-damper model** per wheel
- **Aerodynamic drag** and **rolling resistance**
- **Drivetrain model** with torque curve and automatic gearbox

The vehicle physics parameters are configured to match the representative BEV sedan class:

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Kerb mass | $m$ | $1650\,\text{kg}$ | BEV midsize sedan |
| Yaw moment of inertia | $I_z$ | $2315.3\,\text{kg}\cdot\text{m}^2$ | Rajamani (2012) Table 2.1 |
| CoM to front axle | $l_f$ | $1.105\,\text{m}$ | NHTSA NCAP |
| CoM to rear axle | $l_r$ | $1.738\,\text{m}$ | NHTSA NCAP |
| Wheelbase | $L$ | $2.843\,\text{m}$ | $l_f + l_r$ |
| Max front wheel angle | $\delta_{\max}$ | $0.35\,\text{rad}$ ($20°$) | Hardware limit at 60 km/h |
| Front cornering stiffness | $C_{\alpha f}$ | $88{,}000\,\text{N/rad}$ | Rajamani (2012) Table 3.2 |
| Rear cornering stiffness | $C_{\alpha r}$ | $94{,}000\,\text{N/rad}$ | Rajamani (2012) Table 3.2 |
| Reference speed | $V_{\text{ref}}$ | $16.67\,\text{m/s}$ ($60\,\text{km/h}$) | ISO 15622 test speed |

The tyre cornering stiffness values ($C_{\alpha f}$, $C_{\alpha r}$) are initially set to nominal values from Rajamani (2012) but may be overridden by the DS-02 calibration pipeline (§3.4.4) if the tyre model fit achieves $R^2 > 0.85$.

The four-wheel physics control is applied through CARLA's `VehiclePhysicsControl` API:

$$\mathbf{F}_{\text{wheel},i} = -C_{\alpha,i} \cdot \alpha_i \quad \text{for } i \in \{fl, fr, rl, rr\}$$

where $\alpha_i$ is the slip angle at each wheel, computed internally by PhysX from the vehicle state and tyre contact patch geometry.

### 3.2.3 Bicycle Model Fallback

For rapid prototyping and offline result generation without a running CARLA server, the system includes a nonlinear bicycle model with fourth-order Runge-Kutta (RK4) integration. This model is a two-degree-of-freedom approximation that captures the essential lateral dynamics of a four-wheel vehicle.

**State vector** ($\mathbf{x} \in \mathbb{R}^8$):

$$\mathbf{x} = [X, Y, \psi, v_x, v_y, r, e_{\text{lat}}, e_\psi]^T$$

where $X, Y$ are global position, $\psi$ is yaw angle, $v_x$ is longitudinal velocity (held constant), $v_y$ is lateral velocity, $r$ is yaw rate, $e_{\text{lat}}$ is lateral deviation from lane centre, and $e_\psi$ is heading error.

**Equations of motion** (Rajamani, 2012, Ch. 3, Eq. 3.6–3.11):

$$\dot{v}_y = \frac{F_{yf} + F_{yr}}{m} - v_x \cdot r$$

$$\dot{r} = \frac{l_f \cdot F_{yf} - l_r \cdot F_{yr}}{I_z}$$

$$\dot{e}_{\text{lat}} = v_x \sin(e_\psi) + v_y \cos(e_\psi)$$

$$\dot{e}_\psi = r - \kappa_{\text{ref}} \cdot v_x$$

**Tyre model** (linear cornering force):

$$\alpha_f = \delta - \arctan\left(\frac{v_y + l_f \cdot r}{v_x}\right)$$

$$\alpha_r = -\arctan\left(\frac{v_y - l_r \cdot r}{v_x}\right)$$

$$F_{yf} = -C_{\alpha f} \cdot \alpha_f, \quad F_{yr} = -C_{\alpha r} \cdot \alpha_r$$

**RK4 integration** (mandatory — Euler integration is insufficient for stiff tyre dynamics):

$$\mathbf{k}_1 = f(\mathbf{x}_t, u_t)$$
$$\mathbf{k}_2 = f(\mathbf{x}_t + \frac{\Delta t}{2}\mathbf{k}_1, u_t)$$
$$\mathbf{k}_3 = f(\mathbf{x}_t + \frac{\Delta t}{2}\mathbf{k}_2, u_t)$$
$$\mathbf{k}_4 = f(\mathbf{x}_t + \Delta t \cdot \mathbf{k}_3, u_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

### 3.2.4 Road Scenario Design

Five road scenarios are defined for progressive difficulty evaluation, conforming to ISO 15622:2018 and ISO 3888-2:2011 test specifications:

| ID | Name | Geometry | Length | Standard |
|---|---|---|---|---|
| SCN-01 | Straight Road | $\kappa = 0$ | 300 m | ISO 15622 §8.1 |
| SCN-02 | Constant Radius | $\kappa = 1/80\,\text{m}^{-1}$ | 500 m | ISO 15622 §8.2, AASHTO |
| SCN-03 | Sinusoidal Winding | $\kappa(s) = 0.02\sin(2\pi s/100)$ | 400 m | ISO 15622 §8.3 |
| SCN-04 | Double Lane Change | ISO 3888-2 exact geometry | 175 m | ISO 3888-2:2011 |
| SCN-05 | Combined Urban | Multi-segment with S-bend | 360 m | Euro NCAP AEB City |

The SCN-04 Double Lane Change follows the exact geometry of ISO 3888-2:2011: a 50 m approach straight, a sinusoidal lateral displacement of $\Delta y = 3.5\,\text{m}$ over 25 m longitudinal distance, a 25 m corridor at offset, a return manoeuvre, and a 50 m exit straight. The lateral displacement profile is:

$$y(s) = \frac{\Delta y}{2}\left(1 - \cos\left(\frac{\pi(s - s_0)}{L_c}\right)\right)$$

where $L_c = 25\,\text{m}$ is the lane change length.

---

## 3.3 DDPG Agent Architecture

### 3.3.1 Actor-Critic Networks

The agent employs the Deep Deterministic Policy Gradient (DDPG) algorithm (Lillicrap et al., 2016) with modifications from TD3 (Fujimoto et al., 2018). The actor-critic architecture operates on normalised observations and outputs continuous steering commands.

**Actor network** $\mu_\theta: \mathbb{R}^8 \to [-1, 1]$:

| Layer | Dimensions | Activation |
|---|---|---|
| Input | 8 | — |
| Hidden 1 | 256 | ReLU + LayerNorm |
| Hidden 2 | 128 | ReLU + LayerNorm |
| Output | 1 | tanh |

**Critic network** $Q_\phi: \mathbb{R}^8 \times \mathbb{R}^1 \to \mathbb{R}$:

| Layer | Dimensions | Activation |
|---|---|---|
| Input (state) | 8 | — |
| Hidden 1 | 256 + 1 (action injected) | ReLU + LayerNorm |
| Hidden 2 | 128 | ReLU + LayerNorm |
| Output | 1 | Linear |

Weight initialisation follows the fan-in scheme of Lillicrap et al. (2016): hidden layers are initialised uniformly in $[-1/\sqrt{f}, 1/\sqrt{f}]$ where $f$ is the fan-in, and the output layer in $[-3 \times 10^{-3}, 3 \times 10^{-3}]$.

### 3.3.2 Delayed Policy Updates

Following TD3 (Fujimoto et al., 2018), the actor is updated only every $d = 2$ critic updates. This reduces variance in the policy gradient by ensuring the critic is more accurately estimated before the actor adapts.

**Critic update** (every step):

$$y = r + \gamma(1 - d_{\text{done}}) \cdot Q_{\phi'}(s', \mu_{\theta'}(s'))$$

$$\mathcal{L}_{\text{critic}} = \frac{1}{N}\sum_{i=1}^{N}(Q_\phi(s_i, a_i) - y_i)^2$$

**Actor update** (every $d$ steps):

$$\nabla_\theta J \approx \frac{1}{N}\sum_{i=1}^{N}\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)$$

**Target network update** (Polyak averaging, every $d$ steps):

$$\theta' \leftarrow \tau\theta + (1 - \tau)\theta', \quad \phi' \leftarrow \tau\phi + (1 - \tau)\phi'$$

### 3.3.3 Ornstein-Uhlenbeck Noise with Annealing

Exploration employs an Ornstein-Uhlenbeck (OU) process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated noise, which produces smoother steering exploration than uncorrelated Gaussian noise:

$$dx_t = \theta(\mu - x_t)dt + \sigma\sqrt{dt} \cdot \mathcal{N}(0, 1)$$

where $\theta = 0.15$ is the mean reversion rate and $\mu = 0$ is the long-run mean. The volatility $\sigma$ is annealed linearly over the extended 1200-episode schedule:

$$\sigma(e) = \begin{cases} 0.15 & e < 100 \\ 0.15 + \frac{e - 100}{800}(0.03 - 0.15) & 100 \leq e < 900 \\ 0.03 & e \geq 900 \end{cases}$$

---

## 3.4 Hybrid Data Fusion Pipeline

### 3.4.1 External Dataset Integration

Three publicly available driving datasets are integrated to seed the replay buffer with real-world driving transitions, reducing the sample complexity of pure simulation training:

| ID | Dataset | Source | Content |
|---|---|---|---|
| DS-01 | OpenLKA | UVA/NHTSA | Expert LKA demonstrations with CAN signals |
| DS-02 | comma-steering-control | comma.ai | Steering angles, IMU, vehicle dynamics |
| DS-03 | Argoverse 2 | Argo AI | Trajectory data with HD map centrelines |

Each dataset adapter parses raw sensor data into a canonical `RawTransition` structure containing:

$$\mathbf{t} = (e_{\text{lat}},\; e_\psi,\; \kappa,\; v_y,\; r,\; \delta_{\text{prev}},\; \kappa_{\text{la1}},\; \kappa_{\text{la2}},\; a,\; r_{\text{reward}},\; \mathbf{s}',\; d_{\text{done}})$$

### 3.4.2 Universal Normaliser

All observations — from both real datasets and simulation — are normalised by physical limits rather than data-driven statistics. This guarantees zero distribution shift between sources:

$$\hat{o}_i = \frac{o_i}{c_i}, \quad \text{clipped to } [-1, 1]$$

| Dimension | Physical quantity | Normalisation constant $c_i$ |
|---|---|---|
| 0 | $e_{\text{lat}}$ | $L_w/2 = 1.75\,\text{m}$ |
| 1 | $e_\psi$ | $\pi/4 = 0.785\,\text{rad}$ |
| 2 | $\kappa$ | $0.05\,\text{m}^{-1}$ |
| 3 | $v_y$ | $2.0\,\text{m/s}$ |
| 4 | $r$ | $0.5\,\text{rad/s}$ |
| 5 | $\delta_{\text{prev}}$ | $\delta_{\max} = 0.35\,\text{rad}$ |
| 6 | $\kappa_{\text{la1}}$ | $0.05\,\text{m}^{-1}$ |
| 7 | $\kappa_{\text{la2}}$ | $0.05\,\text{m}^{-1}$ |

A 5× physical limit check rejects transitions where any raw value exceeds five times its normalisation constant, filtering parsing errors without discarding valid extreme manoeuvres.

### 3.4.3 Stratified Replay Buffer with Phase-Aware Sampling

The Hybrid Stratified Buffer maintains four independent sub-buffers — one per data source — and samples from them with configurable, episode-dependent weights:

| Phase | Episodes | OpenLKA | Comma | Argoverse | Simulation |
|---|---|---|---|---|---|
| Phase 1 | 0–150 | 0.40 | 0.20 | 0.20 | 0.20 |
| Phase 2 | 150–300 | 0.30 | 0.15 | 0.15 | 0.40 |
| Phase 3 | 300–600 | 0.15 | 0.10 | 0.10 | 0.65 |

This design follows Nair et al. (2018) for overcoming exploration with demonstrations. The phase transition ensures that:

1. **Phase 1** — The agent bootstraps from expert demonstrations, reducing initial random exploration.
2. **Phase 2** — Simulation data grows as the agent improves, balancing real and synthetic experience.
3. **Phase 3** — The policy is fine-tuned primarily on simulation rollouts matching the evaluation distribution.

Sub-buffers with zero entries are excluded from sampling and their weight is redistributed proportionally to non-empty sources. This allows the system to operate without any specific dataset by using the `--skip-ds0X` flags.

### 3.4.4 Tyre Model Calibration from Real-World Data

The DS-02 (comma-steering-control) adapter performs tyre model calibration by fitting effective cornering stiffness from real-world steering angle and lateral acceleration data:

$$a_y \approx \frac{C_{\text{eff}}}{m} \cdot \delta$$

where $C_{\text{eff}}$ is the effective cornering stiffness. A least-squares fit yields:

$$C_{\text{eff}} = \frac{\sum_{i} \delta_i \cdot a_{y,i}}{\sum_{i} \delta_i^2} \cdot m$$

The front-rear split is estimated assuming the ratio $C_{\alpha r}/C_{\alpha f} \approx l_f/l_r$:

$$C_{\alpha f} = C_{\text{eff}} \cdot \frac{l_r}{L}, \quad C_{\alpha r} = C_{\text{eff}} \cdot \frac{l_f}{L}$$

If the fit achieves $R^2 > 0.85$, the calibrated values replace the nominal tyre parameters in both the bicycle model and CARLA's `WheelPhysicsControl`.

---

## 3.5 Reward Function Design

The reward function comprises five components, each addressing a distinct control objective:

$$r_t = w_{\text{lat}} \cdot r_{\text{lat}} + w_{\text{head}} \cdot r_{\text{head}} + w_{\text{smooth}} \cdot r_{\text{smooth}} + w_{\text{prog}} \cdot r_{\text{prog}} + r_{\text{term}}$$

| Component | Weight | Formulation | Objective |
|---|---|---|---|
| Lateral | $w_{\text{lat}} = 2.5$ | $\exp\left(-5\left(\frac{e_{\text{lat}}}{L_w/2}\right)^2\right)$ | Lane keeping accuracy |
| Heading | $w_{\text{head}} = 2.0$ | $\exp\left(-3\left(\frac{e_\psi}{\pi/4}\right)^2\right)$ | Heading alignment (elevated for curved scenarios) |
| Smoothness | $w_{\text{smooth}} = 0.8$ | $\exp\left(-0.5\left(\frac{\dot{\delta}}{0.2}\right)^2\right)$ | ISO 15622 §9.2 comfort |
| Progress | $w_{\text{prog}} = 0.5$ | $\frac{v_x \cos(e_\psi)}{V_{\text{ref}}}$ | Prevent zero-speed collapse |
| Terminal | — | $-10.0$ if $|e_{\text{lat}}| \geq L_w/2$ | Lane departure penalty |

The lateral and heading components are co-dominant ($w = 2.5$ and $w = 2.0$ respectively), reflecting the dual requirement of both lane centreline tracking and heading alignment on curved roads. The heading weight was elevated from the initial $w = 1.0$ after preliminary experiments revealed that heading RMSE exceeded the ISO 15622 threshold of $0.087\,\text{rad}$ on SCN-02 through SCN-04. The smoothness component enforces the comfort requirement of ISO 15622:2018 §9.2 by penalising large steering rates, with the Gaussian width set to the IEEE 2846-2022 target of $\dot{\delta}_{\text{RMS}} < 0.2\,\text{rad/s}$.

The same reward function is applied to both real-world transitions (dataset adapters) and simulation rollouts, ensuring consistent reward signal across the fused replay buffer.

---

## 3.6 Training Protocol

### 3.6.1 Curriculum Learning Schedule

Training follows a four-phase curriculum that progressively introduces harder road scenarios:

| Phase | Episodes | Scenarios | Rationale |
|---|---|---|---|
| 1 | 0–100 | SCN-01 | Straight road baseline |
| 2 | 100–250 | SCN-01, SCN-02 | Add constant curvature |
| 3 | 250–450 | SCN-01, SCN-02, SCN-03 | Add dynamic curvature |
| 4 | 450–1200 | All (SCN-01–05) | Extended full scenario distribution (750 episodes) |

Within each phase, scenarios are sampled uniformly at each episode. Initial lateral perturbations are drawn from $\mathcal{U}(-0.2, 0.2)\,\text{m}$ for disturbance rejection assessment per ISO 15622 §8.4.

### 3.6.2 Hyperparameter Configuration

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Actor learning rate | $\eta_\mu$ | $1 \times 10^{-4}$ | Lillicrap et al. (2016) |
| Critic learning rate | $\eta_Q$ | $1 \times 10^{-3}$ | Lillicrap et al. (2016) |
| Discount factor | $\gamma$ | $0.99$ | ~100-step horizon at 100 Hz |
| Polyak coefficient | $\tau$ | $0.005$ | Standard DDPG |
| Batch size | $N$ | $256$ | Mini-batch |
| Policy update frequency | $d$ | $2$ | TD3-style delayed update |
| Critic gradient clip | — | $1.0$ | Prevents Q-value divergence |
| Warmup steps | — | $5{,}000$ | Reduced: buffer pre-seeded |
| Total episodes | — | $1{,}200$ | Extended for full convergence |
| Total buffer capacity | — | $2{,}000{,}000$ | ~1M real + ~600K sim |
| Random seed | — | $42$ | Reproducibility |

### 3.6.3 Convergence Monitoring

Convergence is monitored via a 50-episode rolling window of the RMSE lateral error. The agent is considered converged when the rolling RMSE consistently falls below the ISO 15622 threshold of $0.40\,\text{m}$ for at least 50 consecutive episodes.

Training logs are recorded per episode with 28 fields including: episode return, RMSE $e_{\text{lat}}$, LKSR, critic/actor losses, Q-value statistics, buffer composition, and noise parameters.

---

## 3.7 Evaluation Protocol

### 3.7.1 Deterministic Policy Assessment

Evaluation employs the trained actor network with noise disabled ($\sigma = 0$). For each of the five scenarios, 20 episodes are executed with random initial lateral perturbations drawn from $\mathcal{U}(-0.2, 0.2)\,\text{m}$, yielding 100 total evaluation episodes.

### 3.7.2 ISO 15622:2018 Metrics (M-01 to M-06)

| ID | Metric | Formula | Threshold |
|---|---|---|---|
| M-01 | Mean lateral error | $\bar{e}_{\text{lat}} = \frac{1}{T}\sum_{t=1}^{T} \|e_{\text{lat},t}\|$ | $< 0.30\,\text{m}$ |
| M-02 | RMSE lateral error | $e_{\text{lat,RMSE}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} e_{\text{lat},t}^2}$ | $< 0.40\,\text{m}$ |
| M-03 | Max lateral error | $e_{\text{lat,max}} = \max_t \|e_{\text{lat},t}\|$ | Reported |
| M-04 | Heading RMSE | $e_{\psi,\text{RMSE}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} e_{\psi,t}^2}$ | $< 0.087\,\text{rad}$ ($5°$) |
| M-05 | Lane Keeping Success Rate | $\text{LKSR} = \frac{\text{in-lane steps}}{\text{total steps}}$ | $\geq 0.95$ |
| M-06 | Lane Departure Rate | $\text{LDR} = \frac{\text{departure episodes}}{\text{total episodes}}$ | $< 0.05$ |

### 3.7.3 IEEE 2846-2022 Metrics (M-07 to M-10)

| ID | Metric | Formula | Target |
|---|---|---|---|
| M-07 | Settling time | Time to $\|e_{\text{lat}}\| < 0.10\,\text{m}$ sustained $0.5\,\text{s}$ | Reported |
| M-08 | Overshoot | $\max(e_{\text{lat}}) / e_{\text{lat}}(0)$ | Reported |
| M-09 | Control effort | $\text{CE} = \int_0^T \delta(t)^2\,dt$ | Reported |
| M-10 | Steering rate RMS | $\dot{\delta}_{\text{RMS}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} \dot{\delta}_t^2}$ | $< 0.20\,\text{rad/s}$ |

### 3.7.4 UNECE WP.29 R157 Safety Metrics (M-11 to M-13)

| ID | Metric | Formula | Threshold |
|---|---|---|---|
| M-11 | TTLD series | $\text{TTLD}_t = \frac{L_w/2 - \|e_{\text{lat},t}\|}{v_y}$ | — |
| M-12 | TTLD 5th percentile | $P_5(\text{TTLD})$ | $\geq 0.4\,\text{s}$ |
| M-13 | Safe Boundary Violation Rate | $\text{SBVR} = \frac{\text{steps with } \|e_{\text{lat}}\| > 0.75\text{m}}{\text{total steps}}$ | $< 0.01$ |

### 3.7.5 Dataset Quality Metrics (M-19 to M-22)

| ID | Metric | Description |
|---|---|---|
| M-19 | Real-to-sim ratio | Proportion of real vs. simulated transitions in buffer |
| M-20 | Tyre calibration $R^2$ | Goodness of fit from DS-02 calibration |
| M-21 | Curvature coverage | Distribution overlap of $\kappa$ across datasets |
| M-22 | Pre-train performance | RMSE $e_{\text{lat}}$ after pure pre-training (no sim) |

---

## 3.8 Summary

This chapter has presented a comprehensive methodology for DDPG-based lane keeping that integrates three external driving datasets with a dual-mode simulation environment. The CARLA 0.9.16 backend provides four-wheel PhysX dynamics for high-fidelity evaluation, while the bicycle model enables rapid offline training. The evaluation framework spans three international standards (ISO 15622, IEEE 2846, UNECE R157) with 22 quantitative metrics, ensuring industrial-grade assessment of the autonomous lane keeping controller.
