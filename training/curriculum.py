"""
curriculum.py — Scenario Curriculum Schedule

Implements the phased curriculum that progressively introduces harder
road scenarios as training progresses.

Schedule:
    Phase 1 (ep   0–150): SCN-01 only (straight road)
    Phase 2 (ep 150–300): SCN-01, SCN-02 (add constant curve)
    Phase 3 (ep 300–450): SCN-01, SCN-02, SCN-03 (add sinusoidal)
    Phase 4 (ep 450–600): All five scenarios
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
    for phase_name, phase_info in cfg.CURRICULUM_PHASES.items():
        ep_start, ep_end = phase_info["episodes"]
        if ep_start <= episode < ep_end:
            return list(phase_info["scenarios"])

    # Default: all scenarios (beyond defined phases)
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
        Phase name string (e.g. 'phase1').
    """
    for phase_name, phase_info in cfg.CURRICULUM_PHASES.items():
        ep_start, ep_end = phase_info["episodes"]
        if ep_start <= episode < ep_end:
            return phase_name
    return "phase4"
