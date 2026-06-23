"""
noise.py — Ornstein-Uhlenbeck Noise Process with Sigma Annealing

Implements temporally correlated exploration noise for continuous action spaces.
The OU process is mean-reverting, producing smoother exploration trajectories
than Gaussian noise — critical for vehicle steering where jerky exploration
would cause unrealistic dynamics.

Reference: Uhlenbeck & Ornstein (1930). "On the Theory of Brownian Motion."
           Physical Review, 36(5), 823–841.
           Applied to DDPG: Lillicrap et al. (2016), arXiv:1509.02971, §3.

Process:
    dx_t = θ · (μ − x_t) · dt + σ · √dt · N(0, 1)

Where:
    θ (theta) = mean reversion rate (0.15)
    μ (mu)    = long-run mean (0.0 for zero-mean exploration)
    σ (sigma) = volatility, annealed over training

Annealing schedule (from config.py):
    Episodes 0–100:   σ = 0.15 (initial exploration)
    Episodes 100–450: σ linearly anneals from 0.15 → 0.03
    Episodes 450–600: σ = 0.03 (exploitation)
"""

from __future__ import annotations

import numpy as np

import config as cfg


class OUNoise:
    """
    Ornstein-Uhlenbeck noise process for DDPG exploration.

    Attributes:
        mu:     Long-run mean (default 0.0).
        theta:  Mean reversion rate (default 0.15).
        sigma:  Current volatility (annealed during training).
        dt:     Process timestep (default SIM_DT = 0.01 s).
        state:  Current noise state vector.
        dim:    Dimensionality of the noise (matches action space).
    """

    def __init__(
        self,
        dim: int = cfg.ACT_DIM,
        mu: float = 0.0,
        theta: float = cfg.NOISE_THETA,
        sigma: float = cfg.NOISE_SIGMA_INIT,
        dt: float = cfg.SIM_DT,
    ) -> None:
        self.dim = dim
        self.mu = mu * np.ones(dim, dtype=np.float64)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        self.state = np.zeros(dim, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset noise state to zero (start of new episode)."""
        self.state = np.zeros(self.dim, dtype=np.float64)

    def set_sigma(self, sigma: float) -> None:
        """
        Update the noise volatility (called at the start of each episode
        with the annealed sigma from config.get_noise_sigma).

        Args:
            sigma: New volatility value (must be > 0).
        """
        self.sigma = max(sigma, 1e-8)

    def sample(self) -> np.ndarray:
        """
        Generate one noise sample from the OU process.

        Update rule:
            dx = θ · (μ − x) · dt + σ · √dt · N(0, 1)
            x_{t+1} = x_t + dx

        Returns:
            Noise vector of shape (dim,).
        """
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        )
        self.state = self.state + dx
        return self.state.copy()

    @property
    def current_sigma(self) -> float:
        """Return current volatility."""
        return self.sigma
