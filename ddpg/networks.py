"""
networks.py — Actor and Critic Neural Networks for DDPG

Architecture:
    Actor:  Input(8) → Linear(400) → LayerNorm → ReLU → Linear(300) →
            LayerNorm → ReLU → Linear(1) → Tanh → scale by DELTA_MAX.

    Critic: [obs(8) cat action(1)] → Linear(9, 400) → LayerNorm → ReLU →
            Linear(300) → LayerNorm → ReLU → Linear(1).

Design decisions:
    - LayerNorm instead of BatchNorm: LayerNorm normalises per sample, not
      per batch. Required for online RL with non-i.i.d. mini-batches.
      Reference: Ba et al. (2016), "Layer Normalization", arXiv:1607.06450, §2.

    - Final actor layer initialised with uniform[-3e-3, 3e-3]: ensures
      near-zero initial policy, preventing large early steering commands
      that would deplete expert demonstrations from the buffer too quickly.
      Reference: Lillicrap et al. (2016), "Continuous control with deep
      reinforcement learning", arXiv:1509.02971, Appendix A.

    - Hidden layer weights initialised with fan-in uniform bound:
      1/sqrt(fan_in). Standard practice for DDPG.
      Reference: Lillicrap et al. (2016), Appendix C.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config as cfg


def _fan_in_init(layer: nn.Linear) -> None:
    """
    Initialise linear layer weights with uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)].

    Reference: Lillicrap et al. (2016), Appendix C — weight initialisation.

    Args:
        layer: PyTorch Linear layer to initialise.
    """
    fan_in = layer.weight.data.size(1)
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)


class Actor(nn.Module):
    """
    DDPG Actor network.

    Maps 8-dimensional observation to 1-dimensional steering action.
    Output is scaled by Tanh to [-1, 1] then by DELTA_MAX in the agent.

    Architecture:
        Input(8) → FC(400) → LN → ReLU → FC(300) → LN → ReLU → FC(1) → Tanh

    Reference: Lillicrap et al. (2016), arXiv:1509.02971, §7 — network architecture.
    """

    def __init__(self, obs_dim: int = cfg.OBS_DIM, act_dim: int = cfg.ACT_DIM) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Hidden layers
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output layer
        self.fc_out = nn.Linear(128, act_dim)

        # Weight initialisation
        _fan_in_init(self.fc1)
        _fan_in_init(self.fc2)

        # Final layer: near-zero init to prevent large initial steering
        # Reference: Lillicrap et al. (2016), Appendix A
        nn.init.uniform_(self.fc_out.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias.data, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → normalised steering action.

        Args:
            obs: Observation tensor, shape (batch, obs_dim).

        Returns:
            Action tensor in [-1, 1], shape (batch, act_dim).
        """
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc_out(x))
        return x


class Critic(nn.Module):
    """
    DDPG Critic (Q-value) network.

    Maps (observation, action) pair to scalar Q-value estimate.
    Observation and action are concatenated at the input.

    Architecture:
        [obs(8) | action(1)] → FC(400) → LN → ReLU → FC(300) → LN → ReLU → FC(1)

    Reference: Lillicrap et al. (2016), arXiv:1509.02971, §7.
    """

    def __init__(self, obs_dim: int = cfg.OBS_DIM, act_dim: int = cfg.ACT_DIM) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        input_dim = obs_dim + act_dim  # 8 + 1 = 9

        # Hidden layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Output layer
        self.fc_out = nn.Linear(128, 1)

        # Weight initialisation
        _fan_in_init(self.fc1)
        _fan_in_init(self.fc2)

        # Final layer: near-zero init
        nn.init.uniform_(self.fc_out.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias.data, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (observation, action) → Q-value.

        Args:
            obs:    Observation tensor, shape (batch, obs_dim).
            action: Action tensor, shape (batch, act_dim).

        Returns:
            Q-value tensor, shape (batch, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc_out(x)
        return q
