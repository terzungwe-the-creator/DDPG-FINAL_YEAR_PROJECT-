"""
hybrid_buffer.py — Stratified Priority-Weighted Replay Buffer

Maintains separate sub-buffers for each data source and samples from them
with configurable, phase-aware weights. This prevents expert demonstrations
from being overwritten by RL rollouts in a naive FIFO buffer.
Now includes Prioritized Experience Replay (PER).

Sub-buffers:
    'openlka'   — DS-01 transitions: expert LKA demonstrations
    'comma'     — DS-02 transitions: vehicle dynamics diversity
    'argoverse' — DS-03 transitions: trajectory geometry diversity
    'sim'       — Simulated RL rollouts: generated online during training
"""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import torch

import config as cfg


class _SubBuffer:
    """
    Fixed-capacity ring buffer for a single data source with Priority.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.alpha = alpha

        # Pre-allocate arrays
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.priorities = np.ones(capacity, dtype=np.float32)
        self.max_priority = 1.0

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
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = np.atleast_1d(action).astype(np.float32)
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.priorities[idx] = self.max_priority

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_with_weights(self, n: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        if self._size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            
        probs = self.priorities[:self._size] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self._size, size=n, p=probs, replace=False)
        
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return indices, weights.astype(np.float32)

    def get_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, ...]:
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors).flatten() + 1e-6
        self.max_priority = max(self.max_priority, float(self.priorities[indices].max()))

    @property
    def size(self) -> int:
        return self._size


class HybridStratifiedBuffer:
    SOURCE_KEYS = ["openlka", "comma", "argoverse", "sim"]

    def __init__(
        self,
        capacities: Dict[str, int] | None = None,
        obs_dim: int = cfg.OBS_DIM,
        act_dim: int = cfg.ACT_DIM,
        device: str = "cpu",
        alpha: float = 0.6,
        beta_start: float = 0.4
    ) -> None:
        if capacities is None:
            capacities = cfg.BUFFER_CAPACITIES

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.source_keys = list(self.SOURCE_KEYS)
        self.beta_start = beta_start
        self.beta = beta_start

        self.sub_buffers: Dict[str, _SubBuffer] = {}
        for key in self.source_keys:
            cap = capacities.get(key, 100_000)
            self.sub_buffers[key] = _SubBuffer(cap, obs_dim, act_dim, alpha=alpha)

    def push(
        self,
        source: str,
        state: np.ndarray,
        action: np.ndarray | float,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        if source not in self.sub_buffers:
            raise KeyError(f"Unknown source '{source}'. Valid: {self.source_keys}")
        state_arr = np.asarray(state, dtype=np.float32).flatten()[:self.obs_dim]
        action_arr = np.atleast_1d(np.asarray(action, dtype=np.float32))
        next_state_arr = np.asarray(next_state, dtype=np.float32).flatten()[:self.obs_dim]

        self.sub_buffers[source].push(
            state_arr, action_arr, float(reward), next_state_arr, float(done)
        )

    def anneal_beta(self, progress: float) -> None:
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * progress)

    def sample(
        self, batch_size: int, episode: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], np.ndarray, torch.Tensor]:
        
        phase = cfg.get_buffer_phase(episode)
        raw_weights = cfg.BUFFER_WEIGHTS[phase]

        effective_weights = np.zeros(len(self.source_keys))
        for i, key in enumerate(self.source_keys):
            if self.sub_buffers[key].size > 0:
                effective_weights[i] = raw_weights[key]

        total_weight = effective_weights.sum()
        if total_weight < 1e-12:
            raise RuntimeError("All sub-buffers are empty. Cannot sample.")

        effective_weights /= total_weight

        counts = np.zeros(len(self.source_keys), dtype=np.int64)
        for i in range(len(self.source_keys)):
            counts[i] = int(np.round(effective_weights[i] * batch_size))

        diff = batch_size - counts.sum()
        if diff != 0:
            max_idx = np.argmax(counts)
            counts[max_idx] += diff

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        all_sources = []
        all_indices = []
        all_weights = []

        for i, key in enumerate(self.source_keys):
            n = int(counts[i])
            if n <= 0:
                continue
            buf = self.sub_buffers[key]
            if buf.size == 0:
                continue
            
            indices, weights = buf.sample_with_weights(n, self.beta)
            s, a, r, ns, d = buf.get_batch(indices)
            
            all_states.append(s)
            all_actions.append(a)
            all_rewards.append(r)
            all_next_states.append(ns)
            all_dones.append(d)
            
            all_sources.extend([key] * n)
            all_indices.append(indices)
            all_weights.append(weights)

        states = torch.tensor(
            np.concatenate(all_states, axis=0), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.concatenate(all_actions, axis=0), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            np.concatenate(all_rewards, axis=0), dtype=torch.float32, device=self.device
        )  # Already shape (N, 1) from sub-buffer storage
        next_states = torch.tensor(
            np.concatenate(all_next_states, axis=0), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            np.concatenate(all_dones, axis=0), dtype=torch.float32, device=self.device
        )  # Already shape (N, 1) from sub-buffer storage
        is_weights = torch.tensor(
            np.concatenate(all_weights, axis=0), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        
        indices_arr = np.concatenate(all_indices, axis=0)

        # Shuffle
        perm = torch.randperm(states.shape[0], device=self.device)
        perm_cpu = perm.cpu().numpy()
        
        shuffled_sources = [all_sources[i] for i in perm_cpu]
        shuffled_indices = indices_arr[perm_cpu]

        return (states[perm], actions[perm], rewards[perm], next_states[perm], dones[perm], 
                shuffled_sources, shuffled_indices, is_weights[perm])

    def update_priorities(self, sources: List[str], indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for the sampled transitions."""
        # Group by source to update sub-buffers efficiently
        source_dict = {}
        for i, source in enumerate(sources):
            if source not in source_dict:
                source_dict[source] = ([], [])
            source_dict[source][0].append(indices[i])
            source_dict[source][1].append(td_errors[i])
            
        for source, (idxs, errors) in source_dict.items():
            self.sub_buffers[source].update_priorities(np.array(idxs), np.array(errors))

    @property
    def sizes(self) -> Dict[str, int]:
        return {key: buf.size for key, buf in self.sub_buffers.items()}

    @property
    def total_size(self) -> int:
        return sum(buf.size for buf in self.sub_buffers.values())

    def has_enough(self, min_total: int = cfg.BATCH_SIZE) -> bool:
        return self.total_size >= min_total
