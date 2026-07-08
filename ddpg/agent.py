"""
agent.py — DDPG Agent with Delayed Policy Updates

Implements the Deep Deterministic Policy Gradient algorithm with:
    - Delayed policy updates (every 2 critic updates) — TD3-style.
      Reference: Fujimoto et al. (2018), "Addressing Function Approximation
      Error in Actor-Critic Methods", arXiv:1802.09477.
    - Polyak-averaged target networks (τ = 0.005).
      Reference: Lillicrap et al. (2016), arXiv:1509.02971, Appendix C.
    - Gradient clipping on critic (max norm = 1.0) to prevent Q-value
      divergence on large rewards.

The agent operates on normalised observations and actions.
The actor outputs actions in [-1, 1], which are scaled to physical steering
angles by the environment.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from ddpg.networks import Actor, Critic
from ddpg.hybrid_buffer import HybridStratifiedBuffer


class DDPGAgent:
    """
    DDPG agent with delayed policy updates and target networks.

    Attributes:
        actor:          Online actor network.
        critic:         Online critic network.
        actor_target:   Target actor network (Polyak-averaged).
        critic_target:  Target critic network (Polyak-averaged).
        actor_optim:    Actor optimiser (Adam).
        critic_optim:   Critic optimiser (Adam).
        device:         Compute device.
        update_count:   Total number of critic updates performed.
    """

    def __init__(
        self,
        obs_dim: int = cfg.OBS_DIM,
        act_dim: int = cfg.ACT_DIM,
        actor_lr: float = cfg.ACTOR_LR,
        critic_lr: float = cfg.CRITIC_LR,
        gamma: float = cfg.GAMMA,
        tau: float = cfg.TAU,
        policy_update_freq: int = cfg.POLICY_UPDATE_FREQ,
        critic_grad_clip: float = cfg.CRITIC_GRAD_CLIP,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.critic_grad_clip = critic_grad_clip
        self.device = torch.device(device)

        # Online networks
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim).to(self.device)

        # Target networks (deep copy, no grad)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        # Freeze target network parameters
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=critic_lr
        )

        # Update counter for delayed policy updates
        self.update_count: int = 0

    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Select action using the online actor network (deterministic policy).

        Args:
            state:     Observation array, shape (obs_dim,).
            add_noise: Unused (noise is added externally in training loop).

        Returns:
            Action array in [-1, 1], shape (act_dim,).
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train()

        return action.cpu().numpy().flatten()

    def update(
        self, buffer: HybridStratifiedBuffer, episode: int
    ) -> Dict[str, float]:
        """
        Perform one DDPG update step.

        Steps:
            1. Sample a mini-batch from the hybrid stratified buffer.
            2. Compute critic target: y = r + γ · (1 − d) · Q_target(s', μ_target(s'))
            3. Update critic by minimising MSE loss: L = (Q(s,a) − y)²
            4. Every `policy_update_freq` steps, update actor by maximising Q(s, μ(s))
            5. Soft-update target networks via Polyak averaging.

        Args:
            buffer:  Hybrid stratified replay buffer.
            episode: Current episode number (for phase-aware sampling).

        Returns:
            Dictionary with training metrics:
                critic_loss, actor_loss, q_mean, q_std.
        """
        if not buffer.has_enough(cfg.BATCH_SIZE):
            return {"critic_loss": 0.0, "actor_loss": 0.0, "q_mean": 0.0, "q_std": 0.0}

        # 1. Sample batch
        states, actions, rewards, next_states, dones, sources, indices, is_weights = buffer.sample(
            cfg.BATCH_SIZE, episode
        )

        # 2. Compute critic target
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            
            # Target policy smoothing
            noise = torch.randn_like(next_actions) * 0.2
            noise = noise.clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target_next = torch.min(q1_target, q2_target)
            y = rewards + self.gamma * (1.0 - dones) * q_target_next

        # 3. Update critic
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)
        
        td_error1 = q1_current - y
        td_error2 = q2_current - y
        
        critic_loss = (is_weights * (td_error1 ** 2 + td_error2 ** 2)).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()

        # Gradient clipping on critic
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.critic_grad_clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.critic_grad_clip)
        self.critic_optim.step()
        
        # Update buffer priorities using absolute TD error from critic1
        td_errors_np = td_error1.detach().cpu().numpy()
        buffer.update_priorities(sources, indices, td_errors_np)

        self.update_count += 1

        # Metrics
        q_vals = q1_current.detach()
        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": 0.0,
            "q_mean": float(q_vals.mean().item()),
            "q_std": float(q_vals.std().item()),
        }

        # 4. Delayed policy update
        if self.update_count % self.policy_update_freq == 0:
            # Maximise Q(s, μ(s))
            predicted_actions = self.actor(states)
            
            # L2 Action Regularisation
            l2_weight = 0.01
            l2_penalty = l2_weight * (predicted_actions ** 2).mean()
            
            actor_loss = -self.critic1(states, predicted_actions).mean() + l2_penalty

            self.actor_optim.zero_grad()
            actor_loss.backward()

            # Gradient clipping on actor (prevents policy collapse from noisy Q)
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), cfg.ACTOR_GRAD_CLIP
            )
            self.actor_optim.step()

            metrics["actor_loss"] = float(actor_loss.item())

            # 5. Soft-update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        Polyak-average update of target network parameters.

        θ_target ← τ · θ_source + (1 − τ) · θ_target

        Reference: Lillicrap et al. (2016), Eq. 5.

        Args:
            source: Online network.
            target: Target network.
        """
        for param_s, param_t in zip(source.parameters(), target.parameters()):
            param_t.data.copy_(
                self.tau * param_s.data + (1.0 - self.tau) * param_t.data
            )

    def save_checkpoint(self, episode: int, path: Optional[Path] = None, suffix: str = "") -> Path:
        if path is None:
            path = cfg.CHECKPOINTS_DIR
        path.mkdir(parents=True, exist_ok=True)

        filepath = path / f"ddpg_checkpoint_ep{episode:04d}{suffix}.pt"
        torch.save(
            {
                "episode": episode,
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic1_target_state_dict": self.critic1_target.state_dict(),
                "critic2_target_state_dict": self.critic2_target.state_dict(),
                "actor_optim_state_dict": self.actor_optim.state_dict(),
                "critic_optim_state_dict": self.critic_optim.state_dict(),
                "update_count": self.update_count,
            },
            filepath,
        )
        return filepath

    def load_checkpoint(self, filepath: Path) -> int:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
        self.update_count = checkpoint["update_count"]

        return checkpoint["episode"]
