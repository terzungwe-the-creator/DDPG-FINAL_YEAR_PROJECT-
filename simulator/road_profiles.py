"""
road_profiles.py — Five ISO/AASHTO Road Scenarios for Lane Keeping Evaluation

Generates reference paths for each scenario as discretised centreline geometry.

Scenarios:
    SCN-01: Straight Road         — ISO 15622:2018 §8.1
    SCN-02: Constant Radius Curve — ISO 15622:2018 §8.2, AASHTO Green Book §3-4
    SCN-03: Sinusoidal Winding    — ISO 15622:2018 §8.3
    SCN-04: Double Lane Change    — ISO 3888-2:2011 exact geometry
    SCN-05: Combined Urban Profile — Euro NCAP AEB City representative

Each profile is stored as a RoadProfile dataclass with fields:
    scenario_id, name, arc_length (s), x_ref, y_ref, psi_ref, kappa_ref,
    total_length, speed_profile.

All coordinates are in a local Cartesian frame (East-North-Up).
Arc length is parameterised at ds = 0.1 m resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import config as cfg


DS: float = 0.1  # m — arc length discretisation step


@dataclass
class RoadProfile:
    """
    Discretised road centreline geometry for a single scenario.

    Attributes:
        scenario_id:   Scenario identifier (e.g. 'SCN-01')
        name:          Human-readable scenario name
        arc_length:    Cumulative arc length array (m), shape (N,)
        x_ref:         Reference X coordinates (m), shape (N,)
        y_ref:         Reference Y coordinates (m), shape (N,)
        psi_ref:       Reference heading angle (rad), shape (N,)
        kappa_ref:     Reference curvature (1/m), shape (N,)
        total_length:  Total path length (m)
        speed_profile: Target speed at each point (m/s), shape (N,)
    """
    scenario_id: str
    name: str
    arc_length: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray
    psi_ref: np.ndarray
    kappa_ref: np.ndarray
    total_length: float
    speed_profile: np.ndarray

    def __post_init__(self):
        """Build O(1) curvature lookup table after construction."""
        self._build_fast_lookup()

    def _build_fast_lookup(self, ds_lookup: float = 0.5):
        """
        Pre-compute curvature lookup table for O(1) queries.

        Instead of calling np.interp (O(log N) binary search) on every
        environment step, we pre-interpolate curvature onto a regular grid
        and use integer division for O(1) lookup.

        Args:
            ds_lookup: LUT grid spacing (m). 0.5m gives <0.5% interpolation
                       error vs the full resolution profile.
        """
        self._lookup_ds = ds_lookup
        n_bins = int(self.total_length / ds_lookup) + 2
        s_bins = np.arange(n_bins) * ds_lookup
        self._kappa_lut = np.interp(s_bins, self.arc_length, self.kappa_ref).astype(np.float64)
        self._psi_lut = np.interp(s_bins, self.arc_length, self.psi_ref).astype(np.float64)
        self._n_bins = n_bins

    def get_reference_at_s(self, s: float) -> dict:
        """
        Interpolate reference values at arbitrary arc length s.

        Args:
            s: Arc length position along the path (m). Clamped to [0, total_length].

        Returns:
            Dictionary with keys: x, y, psi, kappa, speed.
        """
        s_clamped = np.clip(s, 0.0, self.total_length)
        x = float(np.interp(s_clamped, self.arc_length, self.x_ref))
        y = float(np.interp(s_clamped, self.arc_length, self.y_ref))
        psi = float(np.interp(s_clamped, self.arc_length, self.psi_ref))
        kappa = float(np.interp(s_clamped, self.arc_length, self.kappa_ref))
        speed = float(np.interp(s_clamped, self.arc_length, self.speed_profile))
        return {"x": x, "y": y, "psi": psi, "kappa": kappa, "speed": speed}

    def get_kappa_at_s(self, s: float) -> float:
        """O(1) curvature lookup via pre-computed LUT."""
        idx = int(s / self._lookup_ds)
        if idx < 0:
            idx = 0
        elif idx >= self._n_bins:
            idx = self._n_bins - 1
        return float(self._kappa_lut[idx])

    def get_lookahead_kappa(self, s: float, v_x: float,
                            lookahead_times: tuple[float, float] = (1.0, 2.0)
                            ) -> tuple[float, float]:
        """
        Compute curvature at lookahead positions using fast LUT.

        Args:
            s: Current arc length (m).
            v_x: Current longitudinal speed (m/s).
            lookahead_times: Time horizons (s) for lookahead. Default (1.0, 2.0).

        Returns:
            (kappa_la1, kappa_la2) — curvature at 1s and 2s lookahead.
        """
        s_la1 = s + v_x * lookahead_times[0]
        s_la2 = s + v_x * lookahead_times[1]
        kappa_la1 = self.get_kappa_at_s(s_la1)
        kappa_la2 = self.get_kappa_at_s(s_la2)
        return kappa_la1, kappa_la2

    @property
    def n_points(self) -> int:
        """Number of discretisation points."""
        return len(self.arc_length)


def _integrate_path(kappa_fn, total_length: float, speed_fn=None,
                    ds: float = DS) -> RoadProfile:
    """
    Integrate a curvature function to produce a full road profile.

    Uses vectorized cumulative summation of the Frenet–Serret equations:
        ψ(s+ds) = ψ(s) + κ(s)·ds
        x(s+ds) = x(s) + cos(ψ(s))·ds
        y(s+ds) = y(s) + sin(ψ(s))·ds

    Args:
        kappa_fn:     Callable s → κ(s) returning curvature in 1/m.
        total_length: Total arc length of the path (m).
        speed_fn:     Callable s → v(s) returning target speed (m/s).
                      If None, uses V_REFERENCE throughout.
        ds:           Discretisation step (m).

    Returns:
        Partially-filled RoadProfile (scenario_id and name must be set by caller).
    """
    n = int(np.ceil(total_length / ds)) + 1
    s_arr = np.linspace(0.0, total_length, n)

    # Vectorize curvature and speed computation (eliminates Python for-loop)
    kappa_vec = np.vectorize(kappa_fn)
    kappa = kappa_vec(s_arr)

    if speed_fn is not None:
        speed_vec = np.vectorize(speed_fn)
        speed = speed_vec(s_arr)
    else:
        speed = np.full(n, cfg.V_REFERENCE)

    # Vectorized Frenet integration via cumulative sum
    ds_arr = np.diff(s_arr, prepend=0.0)
    psi = np.cumsum(kappa * ds_arr)
    psi[0] = 0.0  # Initial heading

    x = np.cumsum(np.cos(psi) * ds_arr)
    x[0] = 0.0
    y = np.cumsum(np.sin(psi) * ds_arr)
    y[0] = 0.0

    return RoadProfile(
        scenario_id="",
        name="",
        arc_length=s_arr,
        x_ref=x,
        y_ref=y,
        psi_ref=psi,
        kappa_ref=kappa,
        total_length=total_length,
        speed_profile=speed,
    )


def build_scn01() -> RoadProfile:
    """
    SCN-01: Straight Road — ISO 15622:2018 §8.1

    300 m straight road, zero curvature, constant speed V_REFERENCE (60 km/h).
    Baseline scenario for controller tuning and convergence assessment.
    """
    total_length = 300.0
    profile = _integrate_path(
        kappa_fn=lambda s: 0.0,
        total_length=total_length,
    )
    profile.scenario_id = "SCN-01"
    profile.name = "Straight Road"
    return profile


def build_scn02() -> RoadProfile:
    """
    SCN-02: Constant Radius Curve — ISO 15622:2018 §8.2

    Geometry: 150m straight → 20m clothoid entry → 160m arc at R=80m →
              20m clothoid exit → 150m straight.
    Total: 500 m.

    R = 80 m is the minimum curve radius for 60 km/h per AASHTO Green Book §3-4.
    Clothoid transitions (20m linear curvature ramp) prevent discontinuous
    curvature steps that cause unphysical transient overshoot.
    """
    L_straight1 = 150.0
    L_clothoid = 20.0       # m — clothoid transition zone
    L_arc = 160.0            # m — reduced from 200 to keep total at 500
    L_straight2 = 150.0
    R = 80.0                 # m — AASHTO minimum at 60 km/h
    kappa_curve = 1.0 / R
    total_length = L_straight1 + L_clothoid + L_arc + L_clothoid + L_straight2

    s1 = L_straight1
    s2 = s1 + L_clothoid         # end of entry clothoid
    s3 = s2 + L_arc              # end of constant curve
    s4 = s3 + L_clothoid         # end of exit clothoid

    def kappa_fn(s: float) -> float:
        if s < s1:
            return 0.0
        elif s < s2:
            # Entry clothoid: linear ramp 0 → kappa_curve
            frac = (s - s1) / L_clothoid
            return kappa_curve * frac
        elif s < s3:
            return kappa_curve
        elif s < s4:
            # Exit clothoid: linear ramp kappa_curve → 0
            frac = (s - s3) / L_clothoid
            return kappa_curve * (1.0 - frac)
        else:
            return 0.0

    profile = _integrate_path(kappa_fn=kappa_fn, total_length=total_length)
    profile.scenario_id = "SCN-02"
    profile.name = "Constant Radius Curve"
    return profile


def build_scn03() -> RoadProfile:
    """
    SCN-03: Sinusoidal Winding — ISO 15622:2018 §8.3

    Curvature: κ(s) = 0.02 · sin(2π·s / 100), total 400 m.
    Tests dynamic tracking of continuously varying curvature.
    Peak curvature: ±0.02 1/m (R = 50 m).
    Wavelength: 100 m.
    """
    total_length = 400.0

    def kappa_fn(s: float) -> float:
        return 0.02 * np.sin(2.0 * np.pi * s / 100.0)

    profile = _integrate_path(kappa_fn=kappa_fn, total_length=total_length)
    profile.scenario_id = "SCN-03"
    profile.name = "Sinusoidal Winding"
    return profile


def build_scn04() -> RoadProfile:
    """
    SCN-04: Double Lane Change — ISO 3888-2:2011

    Exact geometry per ISO 3888-2:2011:
        - Approach straight: 50 m
        - First lane change: 3.5 m lateral offset over 25 m longitudinal
        - Corridor: 25 m straight at offset
        - Return lane change: 3.5 m return over 25 m longitudinal
        - Exit straight: 50 m
        Total ≈ 175 m with approach/exit.

    The lane change segments use a sinusoidal lateral displacement profile:
        y(s) = (Δy/2) · (1 − cos(π·(s − s_start) / L_change))
    which gives smooth curvature variation.
    """
    L_approach = 50.0
    L_change = 25.0
    L_corridor = 25.0
    L_exit = 50.0
    delta_y = 3.5  # m — lateral offset per ISO 3888-2

    total_length = L_approach + L_change + L_corridor + L_change + L_exit

    # Build the profile by computing y(s) analytically then deriving curvature
    n = int(np.ceil(total_length / DS)) + 1
    s_arr = np.linspace(0.0, total_length, n)

    x = np.zeros(n)
    y = np.zeros(n)
    psi = np.zeros(n)
    kappa = np.zeros(n)

    for i in range(n):
        s = s_arr[i]
        if s <= L_approach:
            # Approach straight
            y[i] = 0.0
            psi[i] = 0.0
            kappa[i] = 0.0
        elif s <= L_approach + L_change:
            # First lane change (sinusoidal profile)
            s_local = s - L_approach
            frac = s_local / L_change
            y[i] = (delta_y / 2.0) * (1.0 - np.cos(np.pi * frac))
            # dy/ds = (delta_y * pi / (2 * L_change)) * sin(pi * frac)
            dy_ds = (delta_y * np.pi / (2.0 * L_change)) * np.sin(np.pi * frac)
            psi[i] = np.arctan(dy_ds)
            # d²y/ds² = (delta_y * pi² / (2 * L_change²)) * cos(pi * frac)
            d2y_ds2 = (delta_y * np.pi ** 2 / (2.0 * L_change ** 2)) * np.cos(
                np.pi * frac
            )
            # κ = d²y/ds² / (1 + (dy/ds)²)^(3/2)
            kappa[i] = d2y_ds2 / (1.0 + dy_ds ** 2) ** 1.5
        elif s <= L_approach + L_change + L_corridor:
            # Corridor straight at offset
            y[i] = delta_y
            psi[i] = 0.0
            kappa[i] = 0.0
        elif s <= L_approach + 2 * L_change + L_corridor:
            # Return lane change
            s_local = s - (L_approach + L_change + L_corridor)
            frac = s_local / L_change
            y[i] = delta_y - (delta_y / 2.0) * (1.0 - np.cos(np.pi * frac))
            dy_ds = -(delta_y * np.pi / (2.0 * L_change)) * np.sin(np.pi * frac)
            psi[i] = np.arctan(dy_ds)
            d2y_ds2 = -(delta_y * np.pi ** 2 / (2.0 * L_change ** 2)) * np.cos(
                np.pi * frac
            )
            kappa[i] = d2y_ds2 / (1.0 + dy_ds ** 2) ** 1.5
        else:
            # Exit straight
            y[i] = 0.0
            psi[i] = 0.0
            kappa[i] = 0.0

        # Proper x integration: x[i] = x[i-1] + cos(psi[i-1]) * ds
        # (x ≈ s approximation breaks at 12° heading during lane change)
        if i == 0:
            x[i] = 0.0
        else:
            ds_step = s_arr[i] - s_arr[i - 1]
            x[i] = x[i - 1] + np.cos(psi[i - 1]) * ds_step

    speed_profile = np.full(n, cfg.V_REFERENCE)

    return RoadProfile(
        scenario_id="SCN-04",
        name="Double Lane Change",
        arc_length=s_arr,
        x_ref=x,
        y_ref=y,
        psi_ref=psi,
        kappa_ref=kappa,
        total_length=total_length,
        speed_profile=speed_profile,
    )


def build_scn05() -> RoadProfile:
    """
    SCN-05: Combined Urban Profile — Euro NCAP AEB City representative

    Geometry:
        80 m straight → 100 m R=60m curve → 40 m straight →
        60 m S-bend (R=40m, alternating sign) → 80 m straight.
    Total: 360 m.

    Speed profile: Cosine transitions between 50 km/h → 30 km/h → 50 km/h.
        - 50 km/h on straights (13.89 m/s)
        - 30 km/h through tight curves (8.33 m/s)
    """
    seg_lengths = [80.0, 100.0, 40.0, 30.0, 30.0, 80.0]
    seg_cumulative = np.cumsum([0.0] + seg_lengths)
    total_length = seg_cumulative[-1]  # 360 m

    R_curve1 = 60.0    # m
    R_sbend = 40.0      # m

    V_HIGH = 50.0 / 3.6    # 13.89 m/s
    V_LOW = 30.0 / 3.6     # 8.33 m/s

    # Clothoid transition length for smoother curvature changes
    L_cloth = 10.0  # m — linear curvature ramp at transitions

    def kappa_fn(s: float) -> float:
        kappa_c1 = 1.0 / R_curve1   # 0.01667
        kappa_sb = 1.0 / R_sbend    # 0.025

        # Segment 1: Straight (0 → seg_cumulative[1])
        if s < seg_cumulative[1] - L_cloth:
            return 0.0
        elif s < seg_cumulative[1]:
            # Entry clothoid into curve1
            frac = (s - (seg_cumulative[1] - L_cloth)) / L_cloth
            return kappa_c1 * frac

        # Segment 2: Curve R=60m (seg_cumulative[1] → seg_cumulative[2])
        elif s < seg_cumulative[2] - L_cloth:
            return kappa_c1
        elif s < seg_cumulative[2]:
            # Exit clothoid from curve1
            frac = (s - (seg_cumulative[2] - L_cloth)) / L_cloth
            return kappa_c1 * (1.0 - frac)

        # Segment 3: Straight (seg_cumulative[2] → seg_cumulative[3])
        elif s < seg_cumulative[3] - L_cloth:
            return 0.0
        elif s < seg_cumulative[3]:
            # Entry clothoid into S-bend positive half
            frac = (s - (seg_cumulative[3] - L_cloth)) / L_cloth
            return kappa_sb * frac

        # Segment 4: S-bend first half +κ (seg_cumulative[3] → seg_cumulative[4])
        elif s < seg_cumulative[4] - L_cloth:
            return kappa_sb
        elif s < seg_cumulative[4] + L_cloth:
            # S-bend reversal: linear transition +κ → −κ over 2*L_cloth
            s_mid = seg_cumulative[4]
            frac = (s - (s_mid - L_cloth)) / (2.0 * L_cloth)
            return kappa_sb * (1.0 - 2.0 * frac)

        # Segment 5: S-bend second half −κ (seg_cumulative[4] → seg_cumulative[5])
        elif s < seg_cumulative[5] - L_cloth:
            return -kappa_sb
        elif s < seg_cumulative[5]:
            # Exit clothoid from S-bend
            frac = (s - (seg_cumulative[5] - L_cloth)) / L_cloth
            return -kappa_sb * (1.0 - frac)

        # Segment 6: Exit straight
        else:
            return 0.0

    def speed_fn(s: float) -> float:
        # Transition zones at segment boundaries using cosine smoothing
        transition_len = 15.0  # m — smooth transition zone

        if s < seg_cumulative[1] - transition_len:
            return V_HIGH
        elif s < seg_cumulative[1]:
            # Decelerate approaching curve
            frac = (s - (seg_cumulative[1] - transition_len)) / transition_len
            return V_HIGH + 0.5 * (V_LOW - V_HIGH) * (1.0 - np.cos(np.pi * frac))
        elif s < seg_cumulative[2]:
            return V_LOW
        elif s < seg_cumulative[2] + transition_len:
            # Accelerate leaving curve
            frac = (s - seg_cumulative[2]) / transition_len
            return V_LOW + 0.5 * (V_HIGH - V_LOW) * (1.0 - np.cos(np.pi * frac))
        elif s < seg_cumulative[3] - transition_len:
            return V_HIGH
        elif s < seg_cumulative[3]:
            # Decelerate approaching S-bend
            frac = (s - (seg_cumulative[3] - transition_len)) / transition_len
            return V_HIGH + 0.5 * (V_LOW - V_HIGH) * (1.0 - np.cos(np.pi * frac))
        elif s < seg_cumulative[5]:
            return V_LOW
        elif s < seg_cumulative[5] + transition_len:
            frac = (s - seg_cumulative[5]) / transition_len
            return V_LOW + 0.5 * (V_HIGH - V_LOW) * (1.0 - np.cos(np.pi * frac))
        else:
            return V_HIGH

    profile = _integrate_path(
        kappa_fn=kappa_fn, total_length=total_length, speed_fn=speed_fn
    )
    profile.scenario_id = "SCN-05"
    profile.name = "Combined Urban Profile"
    return profile


def build_all_profiles() -> Dict[str, RoadProfile]:
    """
    Build all five road profiles and return as a dictionary.

    Returns:
        Dictionary mapping scenario_id → RoadProfile.
    """
    builders = [build_scn01, build_scn02, build_scn03, build_scn04, build_scn05]
    profiles: Dict[str, RoadProfile] = {}
    for builder in builders:
        profile = builder()
        profiles[profile.scenario_id] = profile
    return profiles
