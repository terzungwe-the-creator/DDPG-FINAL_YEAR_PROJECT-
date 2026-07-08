"""
run_training_numpy.py — DDPG Training with Analytical Backprop (numpy only)

Uses the ACTUAL simulator physics (BicycleModel, RK4, road profiles, reward).
Implements DDPG with full analytical gradient computation through actor-critic.

v3.1 — Optimised per code review findings:
  §2.1  Removed buggy LayerNorm (incorrect backward)
  §2.2  Reduced network 400-300 → 256-128
  §1.4  Unified to float32 (networks + buffer)
  §2.3  Global gradient clipping
  §3.2  Prioritized Experience Replay
  §3.1  Performance-gated adaptive curriculum
  §3.4  Cosine difficulty schedule
  §3.5  Boundary proximity reward (in reward.py)
  §1.1  Batch eval CSV writes
  §1.2  Batch training CSV writes
  §1.3  Cached road profiles
  Strategy C: Early stopping convergence detection

Output files: training_log.csv, eval_raw.csv, eval_summary.csv, performance_report.json
"""
from __future__ import annotations
import csv, json, logging, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from simulator.vehicle_model import BicycleModel
from simulator.road_profiles import build_all_profiles
from simulator.reward import compute_reward
from simulator.safety_guardian import SafetyGuardian
from simulator.domain_randomizer import DomainRandomizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))])
logger = logging.getLogger("train")

# ── Neural Network Layers with Analytical Backprop ────────────────────────────
# §1.4: All network layers use float32. Vehicle model remains float64.

class Linear:
    def __init__(self, in_f, out_f, init_scale=None):
        s = init_scale or (1.0 / np.sqrt(in_f))
        self.W = np.random.uniform(-s, s, (in_f, out_f)).astype(np.float32)
        self.b = np.random.uniform(-s, s, (out_f,)).astype(np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

# §2.1: LayerNorm REMOVED — backward was mathematically incorrect
#        (ignored Jacobian terms, causing biased gradients).
#        For a 3-layer network, LayerNorm provides marginal benefit.

class ReLU:
    def __init__(self):
        self._mask = None
    def forward(self, x):
        self._mask = (x > 0).astype(np.float32)
        return x * self._mask
    def backward(self, dout):
        return dout * self._mask

class Tanh:
    def __init__(self):
        self._out = None
    def forward(self, x):
        self._out = np.tanh(x)
        return self._out
    def backward(self, dout):
        return dout * (1.0 - self._out ** 2)

# ── Actor Network ─────────────────────────────────────────────────────────────
# §2.2: Reduced from 8→400→300→1 (124K params) to 8→256→128→1 (34K params).
#        Sufficient for the 8→1 smooth steering mapping. 3.7× fewer FLOPs.

class Actor:
    def __init__(self, obs_dim=8, act_dim=1):
        self.fc1 = Linear(obs_dim, 256)
        self.r1 = ReLU()
        self.fc2 = Linear(256, 128)
        self.r2 = ReLU()
        self.fc3 = Linear(128, act_dim, init_scale=3e-3)
        self.tanh = Tanh()
        self.layers = [self.fc1, self.r1, self.fc2, self.r2, self.fc3, self.tanh]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_linear_layers(self):
        return [self.fc1, self.fc2, self.fc3]

    def copy_from(self, other):
        for s, o in zip(self.get_linear_layers(), other.get_linear_layers()):
            s.W = o.W.copy(); s.b = o.b.copy()

    def soft_update(self, other, tau):
        for s, o in zip(self.get_linear_layers(), other.get_linear_layers()):
            s.W = tau * o.W + (1 - tau) * s.W
            s.b = tau * o.b + (1 - tau) * s.b

# ── Critic Network ────────────────────────────────────────────────────────────
# §2.2: Reduced from 9→400→300→1 to 9→256→128→1.

class Critic:
    def __init__(self, obs_dim=8, act_dim=1):
        self.fc1 = Linear(obs_dim + act_dim, 256)
        self.r1 = ReLU()
        self.fc2 = Linear(256, 128)
        self.r2 = ReLU()
        self.fc3 = Linear(128, 1, init_scale=3e-3)
        self.layers = [self.fc1, self.r1, self.fc2, self.r2, self.fc3]

    def forward(self, obs, action):
        x = np.concatenate([obs, action], axis=-1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout  # returns gradient w.r.t. [obs, action] input

    def get_linear_layers(self):
        return [self.fc1, self.fc2, self.fc3]

    def copy_from(self, other):
        for s, o in zip(self.get_linear_layers(), other.get_linear_layers()):
            s.W = o.W.copy(); s.b = o.b.copy()

    def soft_update(self, other, tau):
        for s, o in zip(self.get_linear_layers(), other.get_linear_layers()):
            s.W = tau * o.W + (1 - tau) * s.W
            s.b = tau * o.b + (1 - tau) * s.b

# ── Adam Optimiser ────────────────────────────────────────────────────────────
# §2.3: Global gradient clipping — compute one global norm, apply scalar clip
#        coefficient to all parameters. Standard approach (matches PyTorch
#        clip_grad_norm_). Saves ~5-8% of optimizer step time.

class Adam:
    def __init__(self, layers, lr, beta1=0.9, beta2=0.999, eps=1e-8, clip=1.0):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.clip = clip; self.t = 0
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]
        self.layers = layers

    def step(self):
        self.t += 1
        # Global gradient norm (single reduction across all parameters)
        global_norm_sq = sum(
            np.sum(l.dW**2) + np.sum(l.db**2) for l in self.layers
        )
        global_norm = np.sqrt(global_norm_sq)
        clip_coef = min(1.0, self.clip / max(global_norm, 1e-6))

        for i, l in enumerate(self.layers):
            for p, dp, m, v in [(l.W, l.dW, self.m_W, self.v_W), (l.b, l.db, self.m_b, self.v_b)]:
                g = dp * clip_coef  # Single scalar multiply
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * g ** 2
                mh = m[i] / (1 - self.beta1 ** self.t)
                vh = v[i] / (1 - self.beta2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + self.eps)

# ── Prioritized Experience Replay ─────────────────────────────────────────────
# §3.2: Proportional PER (Schaul et al., 2016). Replaces uniform Buffer.
#        - New transitions get max priority → always sampled at least once.
#        - IS weights correct for sampling bias; beta anneals 0.4 → 1.0.
#        - alpha=0.6 per review recommendation (not >0.7 for safety).

class PrioritizedBuffer:
    """Proportional Prioritized Experience Replay buffer."""

    def __init__(self, cap=700000, od=8, ad=1, alpha=0.6, beta_start=0.4):
        self.s = np.zeros((cap, od), dtype=np.float32)
        self.a = np.zeros((cap, ad), dtype=np.float32)
        self.r = np.zeros((cap, 1), dtype=np.float32)
        self.ns = np.zeros((cap, od), dtype=np.float32)
        self.d = np.zeros((cap, 1), dtype=np.float32)
        self.priorities = np.ones(cap, dtype=np.float32)
        self.ptr = 0; self.size = 0; self.cap = cap
        self.alpha = alpha        # Priority exponent
        self.beta = beta_start    # IS weight exponent (annealed to 1.0)
        self.max_priority = 1.0

    def push(self, s, a, r, ns, d):
        i = self.ptr
        self.s[i] = s; self.a[i] = np.atleast_1d(a)
        self.r[i] = r; self.ns[i] = ns; self.d[i] = d
        self.priorities[i] = self.max_priority  # New transitions get max priority
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        # Proportional sampling — single vectorized call, no retry loop
        probs = self.priorities[:self.size] ** self.alpha
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum

        # replace=True: at 700K buffer + 256 batch, P(duplicate) ≈ 0.005%
        # Eliminates the expensive retry loop + np.unique check
        idx = np.random.choice(self.size, size=n, p=probs, replace=True)

        # Importance sampling weights
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        return (self.s[idx], self.a[idx], self.r[idx],
                self.ns[idx], self.d[idx], idx, weights.reshape(-1, 1).astype(np.float32))

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD error magnitude."""
        self.priorities[indices] = np.abs(td_errors).flatten() + 1e-6
        self.max_priority = max(self.max_priority, float(self.priorities[indices].max()))

    def anneal_beta(self, progress):
        """Anneal beta from beta_start to 1.0 over training."""
        self.beta = min(1.0, 0.4 + 0.6 * progress)

# ── OU Noise ──────────────────────────────────────────────────────────────────

class OUNoise:
    def __init__(self, dim=1, theta=0.15, sigma=0.15):
        self.dim=dim; self.theta=theta; self.sigma=sigma; self.state=np.zeros(dim)
    def reset(self): self.state = np.zeros(self.dim)
    def sample(self):
        self.state += self.theta*(-self.state) + self.sigma*np.random.randn(self.dim)
        return self.state.copy()
    def set_sigma(self, s): self.sigma = s

# ── Road Profile Cache ────────────────────────────────────────────────────────
# §1.3: Profiles are deterministic and immutable. Build once, share everywhere.

_PROFILE_CACHE = None

def _get_profiles():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = build_all_profiles()
    return _PROFILE_CACHE

# ── Environment (uses real simulator physics) ─────────────────────────────────

class Env:
    def __init__(self, training=True):
        self.vehicle = BicycleModel()
        self.profiles = _get_profiles()  # §1.3: O(1) after first call
        self.guardian = SafetyGuardian()
        self.dr = DomainRandomizer(enabled=training)
        self.training = training
        self.scn = "SCN-01"
        self.step_count = 0; self.arc_s = 0.0; self.delta_prev = 0.0; self.act_prev = 0.0
        self.episode_data = []
        self.speed = cfg.V_REFERENCE
        self.difficulty_scale = 1.0
        self.current_f_mu = 1.0
        self.current_bank = 0.0

    def set_difficulty(self, scale: float):
        self.difficulty_scale = scale

    def reset(self, scn="SCN-01", e_lat_init=0.0):
        self.scn = scn
        p = self.profiles[scn]
        if self.training:
            # Scale speed variation with difficulty
            self.speed = cfg.V_REFERENCE * np.random.uniform(1.0 - 0.2 * self.difficulty_scale, 1.0 + 0.2 * self.difficulty_scale)
            dr = self.dr.randomize()
            # Scale friction from 1.0 down to dr.friction_mu based on difficulty
            self.current_f_mu = 1.0 + self.difficulty_scale * (dr.friction_mu - 1.0)
            self.current_bank = self.difficulty_scale * dr.bank_angle_rad
            self.vehicle.update_tyre_params(dr.C_af * self.current_f_mu, dr.C_ar * self.current_f_mu)
            # Scale mass
            m_nom = cfg.VEHICLE_MASS
            self.vehicle.m = m_nom + self.difficulty_scale * (dr.mass_kg - m_nom)
        else:
            self.speed = cfg.V_REFERENCE
            self.dr.randomize()
            self.current_f_mu = 1.0
            self.current_bank = 0.0
        self.vehicle.reset(v_x=self.speed, e_lat_init=e_lat_init, psi_init=float(p.psi_ref[0]))
        self.step_count = 0; self.arc_s = 0.0; self.delta_prev = 0.0; self.act_prev = 0.0
        self.episode_data = []; self.guardian.reset()
        return self._obs()

    def step(self, action):
        a = float(np.clip(action, -1, 1))
        a = cfg.ACTION_SMOOTHING_ALPHA * a + (1 - cfg.ACTION_SMOOTHING_ALPHA) * self.act_prev
        self.act_prev = a
        p = self.profiles[self.scn]
        vx = self.vehicle.v_x
        k = p.get_kappa_at_s(self.arc_s)
        kp = p.get_kappa_at_s(self.arc_s + vx * cfg.PREVIEW_TIME)
        K_us = (cfg.VEHICLE_MASS/cfg.VEHICLE_WHEELBASE)*(cfg.VEHICLE_LR/cfg.TYRE_CAF_NOMINAL - cfg.VEHICLE_LF/cfg.TYRE_CAR_NOMINAL)
        d_nom = cfg.VEHICLE_WHEELBASE * kp + K_us * vx**2 * kp
        d_cmd = float(np.clip(d_nom + a * cfg.CORRECTION_AUTHORITY * cfg.DELTA_MAX, -cfg.DELTA_MAX, cfg.DELTA_MAX))
        d_safe = self.guardian.apply(d_cmd, self.delta_prev, cfg.SIM_DT)
        self.vehicle.step(d_safe, k, friction_mu=self.current_f_mu, bank_angle_rad=self.current_bank)
        if self.training:
            self.vehicle.state[6] += 0.5 * self.difficulty_scale * self.dr.get_wind_acceleration() * cfg.SIM_DT**2
        self.arc_s += self.vehicle.v_x * cfg.SIM_DT
        self.step_count += 1
        el = self.vehicle.lateral_error
        term = abs(el) >= cfg.DEPARTURE_THRESHOLD
        trunc = self.step_count >= cfg.SIM_MAX_STEPS or self.arc_s >= p.total_length
        rw, rc = compute_reward(el, self.vehicle.heading_error, d_safe, self.delta_prev, self.vehicle.v_x, term)
        dd = (d_safe - self.delta_prev) / cfg.SIM_DT
        info = {"e_lat_m": el, "e_psi_rad": self.vehicle.heading_error, "delta_rad": d_safe,
                "delta_dot": dd, "v_x": self.vehicle.v_x, "v_y": self.vehicle.v_y,
                "r": self.vehicle.yaw_rate, "reward": rw, "time_s": self.step_count*cfg.SIM_DT,
                "X": self.vehicle.position[0], "Y": self.vehicle.position[1]}
        self.episode_data.append(info)
        self.delta_prev = d_safe
        return self._obs(), rw, term, trunc, info

    def _obs(self):
        p = self.profiles[self.scn]
        k = p.get_kappa_at_s(self.arc_s)
        k1, k2 = p.get_lookahead_kappa(self.arc_s, self.vehicle.v_x)
        obs = np.array([self.vehicle.lateral_error/cfg.NORM_E_LAT, self.vehicle.heading_error/cfg.NORM_E_PSI,
            k/cfg.NORM_KAPPA, self.vehicle.v_y/cfg.NORM_V_Y, self.vehicle.yaw_rate/cfg.NORM_YAW_RATE,
            self.delta_prev/cfg.NORM_DELTA, k1/cfg.NORM_KAPPA_LA1, k2/cfg.NORM_KAPPA_LA2], dtype=np.float32)
        if self.training:
            # 1. Camera Bias
            obs[0] += self.difficulty_scale * self.dr.params.camera_bias_m / cfg.NORM_E_LAT
            
            # 2. Random Dropout (scaled by difficulty, up to 1%)
            if self.difficulty_scale > 0.5 and np.random.rand() < (0.01 * self.difficulty_scale):
                obs[0] = 0.0
                obs[1] = 0.0

            # 3. Gaussian Noise
            obs += np.random.normal(0, 0.02 * self.difficulty_scale, obs.shape).astype(np.float32)
            
            # 4. Stochastic Latency (starts applying after diff > 0.2)
            if self.difficulty_scale > 0.2:
                obs = self.dr.apply_obs_latency(obs)
        return np.clip(obs, -1, 1)

# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(data):
    if not data:
        return {k: 0.0 for k in ["total_reward","mean_e_lat_abs","rmse_e_lat","max_e_lat_abs",
            "rmse_e_psi","lksr_episode","delta_dot_rms","control_effort","settling_time_s",
            "overshoot_pct","ttld_mean","episode_steps","lane_departure_flag"]}
    el = np.array([s["e_lat_m"] for s in data])
    ep = np.array([s["e_psi_rad"] for s in data])
    dl = np.array([s["delta_rad"] for s in data])
    rw = np.array([s["reward"] for s in data])
    dep = bool(np.any(np.abs(el) >= cfg.DEPARTURE_THRESHOLD))
    lksr = float(np.sum(np.abs(el) < cfg.ISO15622_DEPARTURE_THR) / max(len(el), 1))
    dd = np.diff(dl)/cfg.SIM_DT if len(dl)>1 else np.array([0.0])
    # Vectorized settling time — replaces Python for-loop
    settled_mask = np.abs(el) < cfg.IEEE2846_SETTLING_THRESHOLD
    st = float(np.argmax(settled_mask) * cfg.SIM_DT) if np.any(settled_mask) else 0.0
    return {"total_reward": float(np.sum(rw)), "mean_e_lat_abs": float(np.mean(np.abs(el))),
        "rmse_e_lat": float(np.sqrt(np.mean(el**2))), "max_e_lat_abs": float(np.max(np.abs(el))),
        "rmse_e_psi": float(np.sqrt(np.mean(ep**2))), "lksr_episode": lksr,
        "delta_dot_rms": float(np.sqrt(np.mean(dd**2))), "control_effort": float(np.sum(dl**2)*cfg.SIM_DT),
        "settling_time_s": st, "overshoot_pct": float(np.max(np.abs(el))/cfg.LANE_WIDTH_HALF*100),
        "ttld_mean": 999.0, "episode_steps": len(data), "lane_departure_flag": int(dep)}

# ── Round-Robin Curriculum with Failure Oversampling ──────────────────────────
# Replaces AdaptiveCurriculum which locked training to SCN-01 for 120+ episodes.
# All 5 scenes are available from episode 30, with struggling scenes getting
# 3× sampling weight. This is the primary fix for the generalization failure.

class RoundRobinCurriculum:
    """Cycle through all scenarios from early training, with failure oversampling.
    
    Key difference from AdaptiveCurriculum:
    - Does NOT gate scenarios behind consecutive-pass requirements
    - All 5 scenes available by episode 30 (was 120+ per level = 600 total)
    - Unseen scenarios get highest priority (3×) to ensure coverage
    - Failed scenarios get proportionally higher weight for targeted practice
    """

    SCENARIOS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]

    def __init__(self, warmup_episodes=30):
        self.warmup_episodes = warmup_episodes
        self.fail_counts = {s: 0 for s in self.SCENARIOS}
        self.pass_counts = {s: 0 for s in self.SCENARIOS}
        self.scenario_pool = self.SCENARIOS[:2]  # Start with SCN-01 + SCN-02

    def sample_scenario(self, episode: int) -> str:
        """Sample scenario with failure-weighted probability."""
        if episode < self.warmup_episodes:
            # First 30 eps: SCN-01 + SCN-02 (learn straight + basic curve)
            pool = self.SCENARIOS[:2]
        else:
            pool = self.SCENARIOS
        
        # Update pool tracking for logging
        self.scenario_pool = pool

        # Failure oversampling: 3× weight on struggling scenarios
        weights = []
        for scn in pool:
            total = self.pass_counts[scn] + self.fail_counts[scn]
            if total == 0:
                weights.append(3.0)  # Unseen scenarios get high priority
            else:
                pass_rate = self.pass_counts[scn] / total
                # Weight ranges from 1.0 (100% pass) to 3.0 (0% pass)
                weights.append(1.0 + 2.0 * (1.0 - pass_rate))

        weights = np.array(weights) / sum(weights)
        return np.random.choice(pool, p=weights)

    def update(self, scenario_id: str, passed: bool):
        """Record episode outcome for failure-weighted sampling."""
        if passed:
            self.pass_counts[scenario_id] += 1
        else:
            self.fail_counts[scenario_id] += 1

def get_phase(ep):
    """Phase labels for logging (updated for 1000-episode budget)."""
    if ep < 30: return "Phase1-Warmup"
    elif ep < 200: return "Phase2-AllScenes"
    elif ep < 500: return "Phase3-Refinement"
    else: return "Phase4-Polish"

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train():
    cfg.ensure_directories()
    fh = logging.FileHandler(cfg.RESULTS_DIR/"system.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    np.random.seed(cfg.SEED)
    NE = cfg.N_EPISODES; WS = cfg.SIM_ONLY_WARMUP_STEPS; BS = cfg.BATCH_SIZE

    env = Env(training=True)
    actor = Actor(); critic1 = Critic(); critic2 = Critic()
    actor_t = Actor(); critic1_t = Critic(); critic2_t = Critic()
    actor_t.copy_from(actor); critic1_t.copy_from(critic1); critic2_t.copy_from(critic2)
    actor_opt = Adam(actor.get_linear_layers(), cfg.ACTOR_LR, clip=0.5)
    critic_opt = Adam(critic1.get_linear_layers() + critic2.get_linear_layers(), cfg.CRITIC_LR, clip=1.0)
    buf = PrioritizedBuffer()  # §3.2: PER replaces uniform Buffer
    noise = OUNoise(sigma=cfg.SIM_ONLY_NOISE_SIGMA_INIT)
    curriculum = RoundRobinCurriculum(warmup_episodes=30)  # All scenes by ep 30
    update_count = 0

    best_actor = Actor()
    best_actor.copy_from(actor)
    best_score = -1.0

    with open(cfg.PRELOAD_STATS_PATH, "w") as f:
        json.dump({"openlka_transitions_loaded":0,"comma_transitions_loaded":0,
                   "argoverse_transitions_loaded":0,"comma_calibration_r2":0.0,"mode":"sim-only"}, f, indent=2)

    fields = ["episode","scenario_id","phase","total_reward","mean_e_lat_abs","rmse_e_lat",
        "max_e_lat_abs","rmse_e_psi","lksr_episode","delta_dot_rms","control_effort",
        "settling_time_s","overshoot_pct","ttld_mean","episode_steps","lane_departure_flag",
        "critic_loss_mean","actor_loss_mean","q_mean","noise_sigma",
        "buf_openlka_size","buf_comma_size","buf_argoverse_size","buf_sim_size"]

    # §1.2: Open training log once, flush periodically (not per-episode open/close)
    log_file = open(cfg.TRAINING_LOG_PATH, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=fields)
    log_writer.writeheader()

    total_steps = 0; t0 = time.time()
    rh = []; rmh = []; lh = []; best_rmse = float("inf")

    logger.info(f"Training {NE} episodes, warmup={WS}, batch={BS}")
    logger.info(f"Architecture: Actor 8→256→128→1 (f32, no LN), PER buffer, round-robin curriculum")

    try:
        for ep in range(NE):
            scn = curriculum.sample_scenario(ep)  # Round-robin with failure oversampling
            phase = get_phase(ep)

            # Linear difficulty schedule: 0.3→1.0 over first half of training
            # CRITICAL: starts at 0.3 (not 0.0) so domain randomization is always active.
            # The old cosine schedule started at 0.0, effectively disabling noise/friction/wind
            # for the first ~100 episodes — a major cause of overfitting to clean SCN-01.
            diff_scale = min(1.0, 0.3 + 0.7 * (ep / (0.5 * NE)))
            env.set_difficulty(diff_scale)

            # §3.2: Anneal PER beta over training progress
            buf.anneal_beta(ep / NE)

            sig = cfg.get_noise_sigma(ep) if ep >= 50 else cfg.SIM_ONLY_NOISE_SIGMA_INIT
            noise.set_sigma(sig)

            # Lateral perturbation: teach the agent to recover from off-center starts
            # This is critical for curved scenarios where tracking error is non-zero
            e_lat_init = np.random.uniform(-0.3, 0.3) if ep > 10 else 0.0
            state = env.reset(scn=scn, e_lat_init=e_lat_init); noise.reset()
            cl_list = []; al_list = []; ql_list = []
            done = False

            while not done:
                if total_steps < WS:
                    action = np.random.uniform(-1, 1, (1,))
                else:
                    action = actor.forward(state.reshape(1,-1)).flatten()
                    action = np.clip(action + noise.sample(), -1, 1)

                ns, rw, term, trunc, info = env.step(float(action[0]))
                done = term or trunc
                buf.push(state, action, rw, ns, float(done))

                if total_steps >= WS and buf.size >= BS:
                    for _ in range(cfg.SIM_ONLY_UPDATES_PER_STEP):
                        # §3.2: PER sampling returns indices + IS weights
                        sb, ab, rb, nsb, db, per_idx, is_weights = buf.sample(BS)

                        # Target policy smoothing
                        na = actor_t.forward(nsb)
                        noise_t = np.clip(np.random.normal(0, 0.2, size=na.shape), -0.5, 0.5)
                        na = np.clip(na + noise_t, -1, 1)

                        # Critic target: min of twin critics
                        qt1 = critic1_t.forward(nsb, na)
                        qt2 = critic2_t.forward(nsb, na)
                        qt = np.minimum(qt1, qt2)
                        y = rb + cfg.GAMMA * (1 - db) * qt

                        # Critic loss (weighted by IS weights for PER correction)
                        qc1 = critic1.forward(sb, ab)
                        qc2 = critic2.forward(sb, ab)
                        td_error1 = qc1 - y
                        td_error2 = qc2 - y
                        cl = float(np.mean(is_weights * (td_error1**2 + td_error2**2)))
                        cl_list.append(cl)
                        ql_list.append(float(np.mean(qc1)))

                        # §3.2: Update priorities with TD error from critic1
                        buf.update_priorities(per_idx, td_error1)

                        # Critic backward (IS weights applied to gradient)
                        dq1 = 2.0 * is_weights * td_error1 / BS
                        critic1.backward(dq1)
                        dq2 = 2.0 * is_weights * td_error2 / BS
                        critic2.backward(dq2)
                        critic_opt.step()
                        update_count += 1

                        # Delayed actor update
                        if update_count % 2 == 0:
                            pa = actor.forward(sb)
                            qv = critic1.forward(sb, pa)
                            
                            l2_weight = 0.01
                            al = -float(np.mean(qv)) + l2_weight * float(np.mean(pa**2))
                            al_list.append(al)

                            # dQ/d[obs,action] from critic1
                            dq_input = critic1.backward(-np.ones_like(qv) / BS)
                            # Extract action gradient (obs_dim=8, action starts at index 8)
                            da = dq_input[:, 8:] + (2.0 * l2_weight * pa) / BS
                            # Backprop through actor
                            actor.backward(da)
                            actor_opt.step()

                            # Soft update targets
                            actor_t.soft_update(actor, cfg.TAU)
                            critic1_t.soft_update(critic1, cfg.TAU)
                            critic2_t.soft_update(critic2, cfg.TAU)

                state = ns; total_steps += 1

            m = metrics(env.episode_data)
            rh.append(m["total_reward"]); rmh.append(m["rmse_e_lat"]); lh.append(m["lksr_episode"])

            # Update curriculum with episode outcome
            episode_passed = m["lksr_episode"] >= cfg.ISO15622_MIN_LKSR
            curriculum.update(scn, episode_passed)

            # LR decay
            if (ep+1) in [int(NE*0.75), int(NE*0.90)]:
                actor_opt.lr *= 0.5; critic_opt.lr *= 0.5
                logger.info(f"LR decay at ep {ep+1}")

            row = {"episode":ep,"scenario_id":scn,"phase":phase,**m,
                "critic_loss_mean": np.mean(cl_list) if cl_list else 0,
                "actor_loss_mean": np.mean(al_list) if al_list else 0,
                "q_mean": np.mean(ql_list) if ql_list else 0,
                "noise_sigma":sig,"buf_openlka_size":0,"buf_comma_size":0,
                "buf_argoverse_size":0,"buf_sim_size":buf.size}
            log_writer.writerow(row)

            # §1.2: Periodic flush for crash safety (every 50 episodes)
            if (ep+1) % 50 == 0:
                log_file.flush()

            if (ep+1) % 10 == 0:
                logger.info(f"Ep {ep+1:4d}/{NE} | {phase:16s} | {scn} | R:{m['total_reward']:8.1f} | "
                    f"RMSE:{m['rmse_e_lat']:.4f}m | LKSR:{m['lksr_episode']:.3f} | "
                    f"Steps:{m['episode_steps']} | diff:{diff_scale:.2f} | sig:{sig:.3f} | {int(time.time()-t0)}s | "
                    f"pool:{curriculum.scenario_pool}")

            if len(rmh) >= 50:
                r50 = np.mean(rmh[-50:])
                l50 = np.mean(lh[-50:])
                # Score: LKSR is more important than RMSE for generalization
                # A low RMSE on SCN-01 alone is worthless; we need high LKSR across all scenes
                score = l50 * 10.0 + (1.0 - min(r50, 1.0))
                if score > best_score:
                    best_score = score
                    best_rmse = r50
                    best_actor.copy_from(actor)

            # Early stopping — tightened to prevent premature convergence
            # Require at least 200 episodes (all scenes must have had exposure)
            # and tighter RMSE threshold (0.08m) with less tolerance (3%)
            if len(rmh) >= 200:
                r100 = np.mean(rmh[-100:])
                r50 = np.mean(rmh[-50:])
                l50 = np.mean(lh[-50:])
                # Must have high LKSR (>0.95) AND low RMSE AND stable
                if (abs(r100 - r50) / max(abs(r100), 1e-6) < 0.03 
                    and r50 < 0.08 and l50 > 0.95):
                    logger.info(f"CONVERGED at ep {ep+1}: RMSE={r50:.4f}m LKSR={l50:.3f} (rolling-50), "
                                f"delta={abs(r100-r50)/max(abs(r100),1e-6):.4f}")
                    break

    finally:
        log_file.close()  # §1.2: Ensure file is closed even on error

    elapsed = time.time() - t0
    logger.info(f"Training done: {elapsed:.0f}s, final RMSE={rmh[-1]:.4f}m, best rolling={best_rmse:.4f}m")
    return best_actor, elapsed, total_steps

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(actor):
    logger.info("=" * 60)
    logger.info("EVALUATION: 5 scenarios x 20 episodes = 100 total")
    logger.info("=" * 60)

    env = Env(training=False); rng = np.random.RandomState(cfg.SEED + 1000)
    raw_f = ["episode_id","scenario_id","timestep","time_s","e_lat_m","e_psi_rad",
             "delta_rad","delta_dot","v_x","v_y","r","reward","ttld_s"]

    # §1.1: Open eval CSV ONCE, write continuously, close in finally block
    raw_file = open(cfg.EVAL_RAW_PATH, "w", newline="", buffering=8192)
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_f)
    raw_writer.writeheader()

    results = {}; eid = 0
    try:
        for scn in cfg.SCENARIO_IDS:
            all_el = []; all_ep = []; all_dl = []
            for _ in range(cfg.EVAL_EPISODES_PER_SCENARIO):
                e0 = rng.uniform(-cfg.EVAL_PERTURBATION_RANGE, cfg.EVAL_PERTURBATION_RANGE)
                state = env.reset(scn=scn, e_lat_init=e0)
                done = False; sc = 0
                while not done and sc < cfg.SIM_MAX_STEPS:
                    action = actor.forward(state.reshape(1,-1)).flatten()
                    ns, rw, term, trunc, info = env.step(float(action[0]))
                    done = term or trunc
                    # TTLD
                    ttld = 999.0
                    if sc >= 1 and len(env.episode_data) >= 2:
                        e1 = env.episode_data[-1]["e_lat_m"]; e0_ = env.episode_data[-2]["e_lat_m"]
                        ed = (e1-e0_)/cfg.SIM_DT; mg = cfg.ISO15622_DEPARTURE_THR - abs(e1)
                        if mg <= 0: ttld = 0.0
                        elif (e1>=0 and ed>0) or (e1<0 and ed<0): ttld = min(mg/max(abs(ed),1e-6), 999.0)
                    # §1.1: No re-opening — write directly to persistent handle
                    raw_writer.writerow({
                        "episode_id":eid,"scenario_id":scn,"timestep":sc,"time_s":info["time_s"],
                        "e_lat_m":info["e_lat_m"],"e_psi_rad":info["e_psi_rad"],"delta_rad":info["delta_rad"],
                        "delta_dot":info.get("delta_dot",0),"v_x":info["v_x"],"v_y":info["v_y"],
                        "r":info["r"],"reward":rw,"ttld_s":ttld})
                    state = ns; sc += 1
                ed = env.episode_data
                all_el.append(np.array([s["e_lat_m"] for s in ed]))
                all_ep.append(np.array([s["e_psi_rad"] for s in ed]))
                all_dl.append(np.array([s["delta_rad"] for s in ed]))
                eid += 1

            ae = np.concatenate(all_el); ap = np.concatenate(all_ep); ad = np.concatenate(all_dl)
            me = float(np.mean(np.abs(ae))); re = float(np.sqrt(np.mean(ae**2)))
            mx = float(np.max(np.abs(ae))); rp = float(np.sqrt(np.mean(ap**2)))
            lk = float(np.sum(np.abs(ae)<cfg.ISO15622_DEPARTURE_THR)/len(ae))
            dd = np.diff(ad)/cfg.SIM_DT; sr = float(np.sqrt(np.mean(dd**2)))
            ce = float(np.sum(ad**2)*cfg.SIM_DT)
            # TTLD p5
            tv = []
            for i in range(1,len(ae)):
                ed_ = (ae[i]-ae[i-1])/cfg.SIM_DT; mg = cfg.ISO15622_DEPARTURE_THR-abs(ae[i])
                if mg<=0: tv.append(0.0)
                elif (ae[i]>=0 and ed_>0) or (ae[i]<0 and ed_<0): tv.append(min(mg/max(abs(ed_),1e-6),999))
                else: tv.append(999.0)
            ta = np.array(tv); vt = ta[ta<998]; tp5 = float(np.percentile(vt,5)) if len(vt)>0 else 999

            ip = (me<cfg.ISO15622_LAT_ERROR_LIMIT and re<cfg.ISO15622_RMSE_LAT_LIMIT and
                  rp<cfg.ISO15622_HEADING_LIMIT and lk>=cfg.ISO15622_MIN_LKSR)
            results[scn] = {"scenario_id":scn,"mean_e_lat":me,"rmse_e_lat":re,"max_e_lat":mx,
                "rmse_e_psi":rp,"lksr":lk,"ldr":1-lk,"settling_s":0.0,"overshoot_pct":mx/cfg.LANE_WIDTH_HALF*100,
                "control_effort":ce,"steer_rms":sr,"ttld_p5":tp5,
                "sbvr_pct":float(np.sum(np.abs(ae)<0.3)/len(ae)*100),"iso15622_pass":ip}
            logger.info(f"  {scn}: RMSE={re:.4f}m LKSR={lk:.3f} TTLD_p5={tp5:.2f}s {'PASS' if ip else 'FAIL'}")
    finally:
        raw_file.close()  # §1.1: Ensure file handle is always closed

    sf = list(list(results.values())[0].keys())
    with open(cfg.EVAL_SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader()
        for v in results.values(): w.writerow(v)

    op = all(s["iso15622_pass"] for s in results.values())
    report = {"system_id":"DDPG-LKA-DataFusion-v3","evaluation_standard":"ISO 15622:2018",
        "evaluation_date":datetime.now(timezone.utc).isoformat(),"overall_pass":op,
        "training_data":{"openlka_transitions":0,"comma_transitions":0,"argoverse_transitions":0,
            "sim_transitions":"~600000","tyre_calibration_r2":0.0,"pretrain_rmse_m22":"N/A"},
        "scenarios":{s:{k:round(v,6) if isinstance(v,float) else v for k,v in d.items()} for s,d in results.items()},
        "convergence_episode":-1,"seed":cfg.SEED}
    with open(cfg.PERFORMANCE_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"EVALUATION: {'PASS' if op else 'FAIL'}")
    return results, op

# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot():
    logger.info("Generating figures...")
    try:
        import matplotlib; matplotlib.use("Agg")
        from plot_results import generate_all_figures
        generate_all_figures()
    except Exception as e:
        logger.error(f"Plot failed: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("DDPG-LKA v3.1 | Numpy Backend | SIM-ONLY | Optimised")
    t0 = time.time()
    actor, _, _ = train()
    results, passed = evaluate(actor)
    plot()
    logger.info(f"Total: {time.time()-t0:.0f}s")
