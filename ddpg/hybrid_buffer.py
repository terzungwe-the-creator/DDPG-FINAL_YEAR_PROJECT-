"""
hybrid_buffer.py — Stratified Priority-Weighted Replay Buffer

Maintains separate sub-buffers for each data source and samples from them
with configurable, phase-aware weights. This prevents expert demonstrations
from being overwritten by RL rollouts in a naive FIFO buffer.

Design rationale:
    Naively mixing expert demonstrations with RL rollouts in a single FIFO
    buffer causes expert data to be overwritten too quickly.
    Reference: Nair et al. (2018), "Overcoming Exploration in Reinforcement
    Learning with Demonstrations", ICRA 2018.

Sub-buffers:
    'openlka'   — DS-01 transitions: expert LKA demonstrations
    'comma'     — DS-02 transitions: vehicle dynamics diversity
    'argoverse' — DS-03 transitions: trajectory geometry diversity
    'sim'       — Simulated RL rollouts: generated online during training

Sampling weights (phase-aware, from config.py):
    Phase 1 (ep   0–150): [0.40, 0.20, 0.20, 0.20] — heavy expert bias
    Phase 2 (ep 150–300): [0.30, 0.15, 0.15, 0.40] — transition to sim
    Phase 3 (ep 300–600): [0.15, 0.10, 0.10, 0.65] — sim-dominant
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

import config as cfg


class _SubBuffer:
    """
    Fixed-capacity ring buffer for a single data source.

    Stores transitions as contiguous numpy arrays for efficient sampling.
    Overwrites oldest entries when full (FIFO eviction).
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Pre-allocate arrays
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state:      Observation, shape (obs_dim,).
            action:     Action, shape (act_dim,) or scalar.
            reward:     Scalar reward.
            next_state: Next observation, shape (obs_dim,).
            done:       Done flag (0.0 or 1.0).
        """
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = np.atleast_1d(action).astype(np.float32)
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_indices(self, n: int) -> np.ndarray:
        """
        Sample n random indices from the current valid range.

        Args:
            n: Number of indices to sample.

        Returns:
            Array of indices, shape (n,).
        """
        return np.random.randint(0, self._size, size=n)

    def get_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Retrieve transitions at the given indices.

        Args:
            indices: Array of buffer indices.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    @property
    def size(self) -> int:
        """Current number of transitions stored."""
        return self._size


class HybridStratifiedBuffer:
    """
    Stratified replay buffer maintaining separate sub-buffers per data source.

    Sampling is phase-aware: the ratio of expert vs. simulation data changes
    over the course of training to transition from imitation to exploration.

    Attributes:
        sub_buffers: Dictionary mapping source name to _SubBuffer.
        source_keys: Ordered list of source names (for weight indexing).
        obs_dim:     Observation dimensionality.
        act_dim:     Action dimensionality.
        device:      PyTorch device for tensor conversion.
    """

    SOURCE_KEYS = ["openlka", "comma", "argoverse", "sim"]

    def __init__(
        self,
        capacities: Dict[str, int] | None = None,
        obs_dim: int = cfg.OBS_DIM,
        act_dim: int = cfg.ACT_DIM,
        device: str = "cpu",
    ) -> None:
        if capacities is None:
            capacities = cfg.BUFFER_CAPACITIES

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.source_keys = list(self.SOURCE_KEYS)

        self.sub_buffers: Dict[str, _SubBuffer] = {}
        for key in self.source_keys:
            cap = capacities.get(key, 100_000)
            self.sub_buffers[key] = _SubBuffer(cap, obs_dim, act_dim)

    def push(
        self,
        source: str,
        state: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        Push a transition to the specified source sub-buffer.

        Args:
            source:     Data source key ('openlka', 'comma', 'argoverse', 'sim').
            state:      Observation array, shape (obs_dim,).
            action:     Action array or scalar.
            reward:     Scalar reward.
            next_state: Next observation array, shape (obs_dim,).
            done:       Done flag (0.0 or 1.0).

        Raises:
            KeyError: If source is not a valid sub-buffer key.
        """
        if source not in self.sub_buffers:
            raise KeyError(
                f"Unknown source '{source}'. Valid: {self.source_keys}"
            )
        state_arr = np.asarray(state, dtype=np.float32).flatten()[:self.obs_dim]
        action_arr = np.atleast_1d(np.asarray(action, dtype=np.float32))
        next_state_arr = np.asarray(next_state, dtype=np.float32).flatten()[:self.obs_dim]

        self.sub_buffers[source].push(
            state_arr, action_arr, float(reward), next_state_arr, float(done)
        )

    def sample(
        self, batch_size: int, episode: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch using phase-appropriate source weights.

        The sampling weights determine how many transitions come from each
        source. Sources with zero entries are excluded and their weight is
        redistributed proportionally.

        Args:
            batch_size: Total number of transitions to sample.
            episode:    Current training episode (determines phase weights).

        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
            All on self.device.

        Raises:
            RuntimeError: If total buffer is empty.
        """
        # Determine phase weights
        phase = cfg.get_buffer_phase(episode)
        raw_weights = cfg.BUFFER_WEIGHTS[phase]

        # Compute effective weights (zero out empty sources, renormalise)
        effective_weights = np.zeros(len(self.source_keys))
        for i, key in enumerate(self.source_keys):
            if self.sub_buffers[key].size > 0:
                effective_weights[i] = raw_weights[key]

        total_weight = effective_weights.sum()
        if total_weight < 1e-12:
            raise RuntimeError("All sub-buffers are empty. Cannot sample.")

        effective_weights /= total_weight

        # Compute number of samples per source
        counts = np.zeros(len(self.source_keys), dtype=np.int64)
        for i in range(len(self.source_keys)):
            counts[i] = int(np.round(effective_weights[i] * batch_size))

        # Adjust to ensure exact batch_size
        diff = batch_size - counts.sum()
        if diff != 0:
            # Add/remove from the source with the largest allocation
            max_idx = np.argmax(counts)
            counts[max_idx] += diff

        # Sample from each source
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for i, key in enumerate(self.source_keys):
            n = int(counts[i])
            if n <= 0:
                continue
            buf = self.sub_buffers[key]
            if buf.size == 0:
                continue
            # Clamp n to available size (with replacement)
            indices = buf.sample_indices(n)
            s, a, r, ns, d = buf.get_batch(indices)
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_next_states.append(ns)
            all_dones.append(d)

        # Concatenate all sources
        states = torch.tensor(
            np.concatenate(all_states, axis=0), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.concatenate(all_actions, axis=0), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            np.concatenate(all_rewards, axis=0), dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.concatenate(all_next_states, axis=0), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            np.concatenate(all_dones, axis=0), dtype=torch.float32, device=self.device
        )

        # Shuffle to prevent source-ordering bias within the batch
        perm = torch.randperm(states.shape[0], device=self.device)
        return states[perm], actions[perm], rewards[perm], next_states[perm], dones[perm]

    @property
    def sizes(self) -> Dict[str, int]:
        """Returns {source: current_size} for all sub-buffers."""
        return {key: buf.size for key, buf in self.sub_buffers.items()}

    @property
    def total_size(self) -> int:
        """Total number of transitions across all sub-buffers."""
        return sum(buf.size for buf in self.sub_buffers.values())

    def has_enough(self, min_total: int = cfg.BATCH_SIZE) -> bool:
        """Check if the buffer has enough transitions for a batch."""
        return self.total_size >= min_total
