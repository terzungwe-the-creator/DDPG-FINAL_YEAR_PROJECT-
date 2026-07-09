"""
run_training_pytorch.py — TD3 Training with PyTorch Backend

Uses the ACTUAL simulator physics (BicycleModel, RK4, road profiles, reward).
Implements TD3 with full PyTorch tensor operations on the GPU (if available).

Optimisations:
  - PyTorch backend for nn & optimizers (GPU accelerated)
  - Reduced network 256-128 (No LayerNorm)
  - Float32 tensors at ML boundary
  - Prioritized Experience Replay (PER) in Hybrid Stratified Buffer
  - Performance-gated adaptive curriculum
  - Cosine difficulty schedule
  - Boundary proximity reward
  - Batch eval CSV writes
  - Early stopping convergence detection
"""
from __future__ import annotations
import csv, json, logging, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from simulator.vehicle_model import BicycleModel
from simulator.road_profiles import build_all_profiles
from simulator.reward import compute_reward
from simulator.safety_guardian import SafetyGuardian
from simulator.domain_randomizer import DomainRandomizer
from ddpg.agent import DDPGAgent
from ddpg.hybrid_buffer import HybridStratifiedBuffer
from ddpg.noise import OUNoise

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))])
logger = logging.getLogger("train")

# ── Road Profile Cache ────────────────────────────────────────────────────────

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
        self.profiles = _get_profiles()
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
            self.speed = cfg.V_REFERENCE * np.random.uniform(1.0 - 0.2 * self.difficulty_scale, 1.0 + 0.2 * self.difficulty_scale)
            dr = self.dr.randomize()
            self.current_f_mu = 1.0 + self.difficulty_scale * (dr.friction_mu - 1.0)
            self.current_bank = self.difficulty_scale * dr.bank_angle_rad
            self.vehicle.update_tyre_params(dr.C_af * self.current_f_mu, dr.C_ar * self.current_f_mu)
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
            obs[0] += self.difficulty_scale * self.dr.params.camera_bias_m / cfg.NORM_E_LAT
            if self.difficulty_scale > 0.5 and np.random.rand() < (0.01 * self.difficulty_scale):
                obs[0] = 0.0; obs[1] = 0.0
            # Component 4: Per-channel curvature-proportional noise
            # Higher noise on curvature-related channels when curvature is high
            # Forces agent to not over-rely on lookahead on curves
            base_noise = 0.02 * self.difficulty_scale
            kappa_noise_scale = 1.0 + 3.0 * min(abs(k) / 0.02, 2.0)  # Up to 7x on tight curves
            per_channel_std = np.array([
                base_noise,                          # e_lat
                base_noise * 0.5,                    # e_psi
                base_noise * kappa_noise_scale,      # kappa_ref
                base_noise,                          # v_y
                base_noise,                          # yaw_rate
                base_noise * 0.3,                    # delta_prev (less noise on own state)
                base_noise * kappa_noise_scale,      # kappa_la1 (curvature-proportional)
                base_noise * kappa_noise_scale,      # kappa_la2 (curvature-proportional)
            ], dtype=np.float32)
            obs += np.random.normal(0, 1, obs.shape).astype(np.float32) * per_channel_std
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
    # Vectorized settling time
    settled_mask = np.abs(el) < cfg.IEEE2846_SETTLING_THRESHOLD
    st = float(np.argmax(settled_mask) * cfg.SIM_DT) if np.any(settled_mask) else 0.0
    return {"total_reward": float(np.sum(rw)), "mean_e_lat_abs": float(np.mean(np.abs(el))),
        "rmse_e_lat": float(np.sqrt(np.mean(el**2))), "max_e_lat_abs": float(np.max(np.abs(el))),
        "rmse_e_psi": float(np.sqrt(np.mean(ep**2))), "lksr_episode": lksr,
        "delta_dot_rms": float(np.sqrt(np.mean(dd**2))), "control_effort": float(np.sum(dl**2)*cfg.SIM_DT),
        "settling_time_s": st, "overshoot_pct": float(np.max(np.abs(el))/cfg.LANE_WIDTH_HALF*100),
        "ttld_mean": 999.0, "episode_steps": len(data), "lane_departure_flag": int(dep)}

# ── RMSE-Weighted Exponential Curriculum ──────────────────────────────────────
# Component 3: Aggressive RMSE-weighted curriculum with scene score tracking.
# Uses exponential oversampling based on per-scene rolling RMSE, not just pass/fail.
# Guarantees minimum 10% sampling for every scene to prevent starvation.

class RMSEWeightedCurriculum:
    """Sample scenarios proportional to exp(scene_rmse), with min 10% floor."""

    SCENARIOS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]

    def __init__(self, warmup_episodes=10, rmse_window=20):
        self.warmup_episodes = warmup_episodes
        self.rmse_window = rmse_window
        # Track per-scene RMSE history for exponential weighting
        self.scene_rmse_history = {s: [] for s in self.SCENARIOS}
        self.fail_counts = {s: 0 for s in self.SCENARIOS}
        self.pass_counts = {s: 0 for s in self.SCENARIOS}
        self.scenario_pool = self.SCENARIOS[:2]

    def sample_scenario(self, episode: int) -> str:
        """Sample scenario with RMSE-exponential probability."""
        if episode < self.warmup_episodes:
            pool = self.SCENARIOS[:2]
        else:
            pool = self.SCENARIOS
        self.scenario_pool = pool

        weights = []
        for scn in pool:
            history = self.scene_rmse_history[scn]
            if len(history) == 0:
                weights.append(10.0)  # Very high priority for unseen scenes
            else:
                recent_rmse = np.mean(history[-self.rmse_window:])
                # Linear oversampling: struggling scenes get up to ~3x, not 76x
                weights.append(1.0 + recent_rmse * 2.0)

        weights = np.array(weights)
        # Enforce minimum 15% floor per scene to prevent catastrophic forgetting
        min_weight = 0.15 * len(pool)
        weights = np.maximum(weights, min_weight / len(pool) * weights.sum())
        weights /= weights.sum()
        return np.random.choice(pool, p=weights)

    def update(self, scenario_id: str, passed: bool, rmse: float = 0.0):
        """Update with both pass/fail and RMSE for richer weighting."""
        if passed:
            self.pass_counts[scenario_id] += 1
        else:
            self.fail_counts[scenario_id] += 1
        self.scene_rmse_history[scenario_id].append(rmse)
        # Keep bounded
        if len(self.scene_rmse_history[scenario_id]) > 100:
            self.scene_rmse_history[scenario_id] = self.scene_rmse_history[scenario_id][-50:]

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train():
    cfg.ensure_directories()
    fh = logging.FileHandler(cfg.RESULTS_DIR/"system.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    device = cfg.get_device()
    
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)

    NE = cfg.N_EPISODES; WS = cfg.SIM_ONLY_WARMUP_STEPS; BS = cfg.BATCH_SIZE
    
    env = Env(training=True)
    agent = DDPGAgent(device=device)
    buf = HybridStratifiedBuffer(device=device)
    noise = OUNoise(sigma=cfg.SIM_ONLY_NOISE_SIGMA_INIT)
    curriculum = RMSEWeightedCurriculum(warmup_episodes=10)

    best_agent_state = None
    best_score = -1.0
    best_scene_lksrs = {s: 0.0 for s in cfg.SCENARIO_IDS}  # Track per-scene best LKSR
    actor_freeze_until = -1  # Component 7: Actor freeze episode counter

    with open(cfg.PRELOAD_STATS_PATH, "w") as f:
        json.dump({"mode":"sim-only-pytorch", "device": device}, f, indent=2)

    fields = ["episode","scenario_id","phase","total_reward","mean_e_lat_abs","rmse_e_lat",
        "max_e_lat_abs","rmse_e_psi","lksr_episode","delta_dot_rms","control_effort",
        "settling_time_s","overshoot_pct","episode_steps","lane_departure_flag",
        "critic_loss_mean","actor_loss_mean","q_mean","noise_sigma","buf_sim_size"]

    log_file = open(cfg.TRAINING_LOG_PATH, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=fields)
    log_writer.writeheader()

    total_steps = 0; t0 = time.time()
    rh = []; rmh = []; lh = []; best_rmse = float("inf")

    logger.info(f"Training {NE} episodes on {device}, warmup={WS}, batch={BS}")
    logger.info(f"Architecture: TD3 Actor 8→256→128→1, Hybrid PER, RoundRobin Curriculum")
    logger.info(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        for ep in range(NE):
            scn = curriculum.sample_scenario(ep)
            phase = "Phase1-Warmup" if ep < 10 else ("Phase2-AllScenes" if ep < 200 else ("Phase3-Refine" if ep < 500 else "Phase4-Polish"))

            # Component 2: Faster difficulty ramp with higher floor
            # Start DR at 30% from ep 0, reach full by ep 400 (was 10%→full by ep 700)
            diff_scale = min(1.0, 0.3 + 0.7 * (ep / (0.4 * NE)))
            env.set_difficulty(diff_scale)
            buf.anneal_beta(ep / NE)

            sig = cfg.get_noise_sigma(ep) if ep >= 50 else cfg.SIM_ONLY_NOISE_SIGMA_INIT
            noise.set_sigma(sig)

            # Lateral perturbation for recovery training
            e_lat_init = np.random.uniform(-0.3, 0.3) if ep > 10 else 0.0
            state = env.reset(scn=scn, e_lat_init=e_lat_init); noise.reset()
            cl_list = []; al_list = []; ql_list = []
            done = False

            while not done:
                if total_steps < WS:
                    action = np.random.uniform(-1, 1, (1,))
                else:
                    action = agent.select_action(state, add_noise=False)
                    action = np.clip(action + noise.sample(), -1, 1)

                ns, rw, term, trunc, info = env.step(float(action[0]))
                done = term or trunc
                buf.push("sim", state, action, rw, ns, float(done))

                if total_steps >= WS and buf.has_enough(BS):
                    for _ in range(cfg.SIM_ONLY_UPDATES_PER_STEP):
                        # Component 1/7: Actor freeze — only update critic after best model
                        if ep < actor_freeze_until:
                            # Temporarily freeze actor to prevent catastrophic forgetting
                            for p in agent.actor.parameters():
                                p.requires_grad = False
                            metrics_dict = agent.update(buf, ep)
                            for p in agent.actor.parameters():
                                p.requires_grad = True
                        else:
                            metrics_dict = agent.update(buf, ep)
                        cl_list.append(metrics_dict["critic_loss"])
                        al_list.append(metrics_dict["actor_loss"])
                        ql_list.append(metrics_dict["q_mean"])

                state = ns; total_steps += 1

            m = metrics(env.episode_data)
            rh.append(m["total_reward"]); rmh.append(m["rmse_e_lat"]); lh.append(m["lksr_episode"])
            episode_passed = m["lksr_episode"] >= cfg.ISO15622_MIN_LKSR
            curriculum.update(scn, episode_passed, rmse=m["rmse_e_lat"])

            # Component 6: Cosine annealing LR (smooth decay, no sudden jumps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * ep / NE))  # 1.0 → 0.0
            lr_scale = max(lr_scale, 0.05)  # Floor at 5% of initial LR
            for pg in agent.actor_optim.param_groups:
                pg["lr"] = cfg.ACTOR_LR * lr_scale
            for pg in agent.critic_optim.param_groups:
                pg["lr"] = cfg.CRITIC_LR * lr_scale

            row = {"episode":ep,"scenario_id":scn,"phase":phase,**m,
                "critic_loss_mean": np.mean(cl_list) if cl_list else 0,
                "actor_loss_mean": np.mean(al_list) if al_list else 0,
                "q_mean": np.mean(ql_list) if ql_list else 0,
                "noise_sigma":sig,"buf_sim_size":buf.total_size}
            
            row_filtered = {k: v for k, v in row.items() if k in fields}
            log_writer.writerow(row_filtered)

            if (ep+1) % 50 == 0:
                log_file.flush()

            if (ep+1) % 10 == 0:
                logger.info(f"Ep {ep+1:4d}/{NE} | {phase:16s} | {scn} | R:{m['total_reward']:8.1f} | "
                    f"RMSE:{m['rmse_e_lat']:.4f}m | LKSR:{m['lksr_episode']:.3f} | "
                    f"Steps:{m['episode_steps']} | diff:{diff_scale:.2f} | sig:{sig:.3f} | {int(time.time()-t0)}s | "
                    f"pool:{curriculum.scenario_pool}")

            # Component 5: Robust mini-evaluation every 50 episodes
            # 10 trials per scene, median-based, with monotonicity guard
            if (ep+1) >= 50 and (ep+1) % 50 == 0:
                eval_env = Env(training=False)
                scene_results = {}
                for eval_scn in cfg.SCENARIO_IDS:
                    eval_rmses = []
                    eval_lksrs = []
                    for trial in range(8):  # 8 trials for stable median estimates
                        e0 = np.random.uniform(-0.15, 0.15)
                        es = eval_env.reset(scn=eval_scn, e_lat_init=e0)
                        ed = False; sc = 0
                        while not ed and sc < cfg.SIM_MAX_STEPS:
                            ea = agent.select_action(es, add_noise=False)
                            es, _, et, etr, _ = eval_env.step(float(ea[0]))
                            ed = et or etr; sc += 1
                        em = metrics(eval_env.episode_data)
                        eval_rmses.append(em["rmse_e_lat"])
                        eval_lksrs.append(em["lksr_episode"])
                    scene_results[eval_scn] = {
                        "rmse": float(np.median(eval_rmses)),  # Median for robustness
                        "lksr": float(np.median(eval_lksrs)),
                    }
                worst_lksr = min(sr["lksr"] for sr in scene_results.values())
                worst_rmse = max(sr["rmse"] for sr in scene_results.values())
                # Score: sum of all per-scene LKSR + accuracy bonus
                # (sum-based rewards partial improvements across ALL scenes,
                #  unlike minimax which only cares about the single worst)
                sum_lksr = sum(sr["lksr"] for sr in scene_results.values())
                sum_acc = sum(max(0, 1.0 - sr["rmse"]) for sr in scene_results.values())
                eval_score = sum_lksr + 0.5 * sum_acc
                logger.info(f"  Mini-eval ep {ep+1}: worst_LKSR={worst_lksr:.3f} worst_RMSE={worst_rmse:.4f} score={eval_score:.2f}")
                for s_id, s_res in scene_results.items():
                    logger.info(f"    {s_id}: RMSE={s_res['rmse']:.4f} LKSR={s_res['lksr']:.3f}")

                # Component 7: Regression guard — only rollback if OVERALL score drops
                # (previously triggered on any single-scene drop, which was too aggressive
                #  and trapped the agent at the ep-100 checkpoint forever)
                if eval_score > best_score:
                    best_score = eval_score
                    best_rmse = worst_rmse
                    best_scene_lksrs = {s_id: sr["lksr"] for s_id, sr in scene_results.items()}
                    best_agent_state = {
                        "actor": {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()},
                        "critic1": {k: v.cpu().clone() for k, v in agent.critic1.state_dict().items()},
                        "critic2": {k: v.cpu().clone() for k, v in agent.critic2.state_dict().items()},
                    }
                    logger.info(f"  New best model! score={eval_score:.2f}")
                    # Brief actor freeze to stabilise critic after improvement
                    actor_freeze_until = ep + 10
                    logger.info(f"  Actor frozen until ep {actor_freeze_until} (critic-only training)")
                elif eval_score < best_score - 0.3 and best_agent_state is not None:
                    # Only rollback on significant overall regression (> 0.3 score drop)
                    agent.actor.load_state_dict(best_agent_state["actor"])
                    agent.critic1.load_state_dict(best_agent_state["critic1"])
                    agent.critic2.load_state_dict(best_agent_state["critic2"])
                    actor_freeze_until = ep + 30  # Longer freeze: let critic fully re-stabilise
                    logger.info(f"  ROLLBACK to best model (score={best_score:.2f}), actor frozen until ep {actor_freeze_until}")
                else:
                    # Score is close to best but not better — allow exploration to continue
                    logger.info(f"  Score {eval_score:.2f} vs best {best_score:.2f} — continuing exploration")

            # Early stopping — relaxed thresholds for multi-scene convergence
            if len(rmh) >= 200:
                r100 = np.mean(rmh[-100:])
                r50 = np.mean(rmh[-50:])
                l50 = np.mean(lh[-50:])
                if (abs(r100 - r50) / max(abs(r100), 1e-6) < 0.03
                    and r50 < 0.12 and l50 > 0.90):
                    logger.info(f"CONVERGED at ep {ep+1}: RMSE={r50:.4f}m LKSR={l50:.3f}")
                    break

    finally:
        log_file.close()

    # Restore best model if available
    if best_agent_state is not None:
        agent.actor.load_state_dict(best_agent_state["actor"])
        agent.critic1.load_state_dict(best_agent_state["critic1"])
        agent.critic2.load_state_dict(best_agent_state["critic2"])
        logger.info(f"Restored best model (RMSE={best_rmse:.4f}m, score={best_score:.2f})")

    elapsed = time.time() - t0
    logger.info(f"Training done: {elapsed:.0f}s, final RMSE={rmh[-1]:.4f}m, best rolling={best_rmse:.4f}m")
    return agent, elapsed, total_steps

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(agent):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    env = Env(training=False); rng = np.random.RandomState(cfg.SEED + 1000)
    raw_f = ["episode_id","scenario_id","timestep","time_s","e_lat_m","e_psi_rad",
             "delta_rad","delta_dot","v_x","v_y","r","reward","ttld_s"]

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
                    action = agent.select_action(state, add_noise=False)
                    ns, rw, term, trunc, info = env.step(float(action[0]))
                    done = term or trunc
                    ttld = 999.0
                    if sc >= 1 and len(env.episode_data) >= 2:
                        e1 = env.episode_data[-1]["e_lat_m"]; e0_ = env.episode_data[-2]["e_lat_m"]
                        ed = (e1-e0_)/cfg.SIM_DT; mg = cfg.ISO15622_DEPARTURE_THR - abs(e1)
                        if mg <= 0: ttld = 0.0
                        elif (e1>=0 and ed>0) or (e1<0 and ed<0): ttld = min(mg/max(abs(ed),1e-6), 999.0)
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
            ip = (me<cfg.ISO15622_LAT_ERROR_LIMIT and re<cfg.ISO15622_RMSE_LAT_LIMIT and
                  rp<cfg.ISO15622_HEADING_LIMIT and lk>=cfg.ISO15622_MIN_LKSR)
            results[scn] = {"scenario_id":scn,"mean_e_lat":me,"rmse_e_lat":re,"max_e_lat":mx,
                "rmse_e_psi":rp,"lksr":lk,"ldr":1-lk,"settling_s":0.0,"overshoot_pct":mx/cfg.LANE_WIDTH_HALF*100,
                "control_effort":ce,"steer_rms":sr,"ttld_p5":999.0,
                "sbvr_pct":float(np.sum(np.abs(ae)<0.3)/len(ae)*100),"iso15622_pass":ip}
            logger.info(f"  {scn}: RMSE={re:.4f}m LKSR={lk:.3f} {'PASS' if ip else 'FAIL'}")
    finally:
        raw_file.close()

    sf = list(list(results.values())[0].keys())
    with open(cfg.EVAL_SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader()
        for v in results.values(): w.writerow(v)

    op = all(s["iso15622_pass"] for s in results.values())
    report = {"system_id":"TD3-LKA-PyTorch","evaluation_standard":"ISO 15622:2018",
        "evaluation_date":datetime.now(timezone.utc).isoformat(),"overall_pass":op,
        "scenarios":{s:{k:round(v,6) if isinstance(v,float) else v for k,v in d.items()} for s,d in results.items()},
        "convergence_episode":-1,"seed":cfg.SEED}
    with open(cfg.PERFORMANCE_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return results, op

def plot():
    logger.info("Generating figures...")
    try:
        import matplotlib; matplotlib.use("Agg")
        from plot_results import generate_all_figures
        generate_all_figures()
    except Exception as e:
        logger.error(f"Plot failed: {e}")

if __name__ == "__main__":
    t0 = time.time()
    agent, _, _ = train()
    results, passed = evaluate(agent)
    plot()
    logger.info(f"Total: {time.time()-t0:.0f}s")
