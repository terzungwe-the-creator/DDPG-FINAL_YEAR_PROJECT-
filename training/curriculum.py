"""
curriculum.py — Round-Robin Scenario Curriculum Schedule

Implements failure-weighted round-robin sampling that progressively
introduces all road scenarios as training progresses.

Schedule:
    Phase 1 (ep   0– 30): SCN-01, SCN-02 (warmup)
    Phase 2 (ep  30+):    All five scenarios (failure-weighted)
"""

from __future__ import annotations

from typing import List

import numpy as np

import config as cfg


def get_curriculum_scenarios(episode: int) -> List[str]:
    """
    Return the list of available scenario IDs for a given episode.

    Args:
        episode: Current training episode number.

    Returns:
        List of scenario IDs available for this episode.
    """
    if episode < 30:
        return list(cfg.SCENARIO_IDS[:2])  # SCN-01, SCN-02 warmup
    return list(cfg.SCENARIO_IDS)


def get_curriculum_profile(episode: int) -> str:
    """
    Select a scenario ID for a given training episode.

    Randomly samples from the available scenarios for the current
    curriculum phase.

    Args:
        episode: Current training episode number.

    Returns:
        Scenario ID string (e.g. 'SCN-01').
    """
    scenarios = get_curriculum_scenarios(episode)
    return scenarios[np.random.randint(len(scenarios))]


def get_curriculum_phase_name(episode: int) -> str:
    """
    Return the curriculum phase name for a given episode.

    Args:
        episode: Current training episode number.

    Returns:
        Phase name string.
    """
    if episode < 30:
        return "phase1-warmup"
    elif episode < 200:
        return "phase2-allscenes"
    elif episode < 500:
        return "phase3-refinement"
    return "phase4-polish"
