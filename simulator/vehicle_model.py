"""
vehicle_model.py — Nonlinear Bicycle Model with RK4 Integration

Implements the 8-state nonlinear bicycle model from:
    Rajamani, R. (2012). Vehicle Dynamics and Control, 2nd ed.
    Springer. Chapter 3: Lateral Vehicle Dynamics.

State vector: x = [X, Y, ψ, v_x, v_y, r, e_lat, e_psi]
    X       — global X position (m)
    Y       — global Y position (m)
    ψ       — yaw angle (rad)
    v_x     — longitudinal velocity (m/s) — held constant
    v_y     — lateral velocity (m/s)
    r       — yaw rate (rad/s)
    e_lat   — lateral deviation from lane centre (m)
    e_psi   — heading error relative to road tangent (rad)

Tyre model: Linear cornering force model
    α_f = δ − arctan((v_y + l_f·r) / v_x)   — front slip angle
    α_r =   − arctan((v_y − l_r·r) / v_x)   — rear slip angle
    F_yf = −C_af · α_f                       — front lateral force (N)
    F_yr = −C_ar · α_r                       — rear lateral force (N)

Integration: Runge-Kutta 4th order (RK4) at dt = 0.01 s (100 Hz).

ISO 26262 ASIL-B: All tyre parameter bounds are validated on assignment.
"""

from __future__ import annotations

import numpy as np

import config as cfg


class BicycleModel:
    """
    Nonlinear bicycle model with 8-dimensional state and RK4 integration.

    Reference: Rajamani (2012), Ch. 3, Eq. 3.6–3.11.

    Attributes:
        m:    Vehicle mass (kg)
        I_z:  Yaw moment of inertia (kg·m²)
        l_f:  Distance from CoM to front axle (m)
        l_r:  Distance from CoM to rear axle (m)
        C_af: Front axle cornering stiffness (N/rad)
        C_ar: Rear axle cornering stiffness (N/rad)
        dt:   Integration timestep (s)
    """

    # State indices for readability
    IDX_X = 0
    IDX_Y = 1
    IDX_PSI = 2
    IDX_VX = 3
    IDX_VY = 4
    IDX_R = 5
    IDX_ELAT = 6
    IDX_EPSI = 7

    N_STATES = 8

    def __init__(
        self,
        m: float = cfg.VEHICLE_MASS,
        I_z: float = cfg.VEHICLE_IZ,
        l_f: float = cfg.VEHICLE_LF,
        l_r: float = cfg.VEHICLE_LR,
        C_af: float = cfg.TYRE_CAF_NOMINAL,
        C_ar: float = cfg.TYRE_CAR_NOMINAL,
        dt: float = cfg.SIM_DT,
    ) -> None:
        self.m = m
        self.I_z = I_z
        self.l_f = l_f
        self.l_r = l_r
        self.C_af = C_af
        self.C_ar = C_ar
        self.dt = dt

        # Validate physical bounds
        assert self.m > 0.0, f"Vehicle mass must be positive, got {self.m}"
        assert self.I_z > 0.0, f"Yaw inertia must be positive, got {self.I_z}"
        assert self.l_f > 0.0, f"l_f must be positive, got {self.l_f}"
        assert self.l_r > 0.0, f"l_r must be positive, got {self.l_r}"
        assert self.C_af > 0.0, f"C_af must be positive, got {self.C_af}"
        assert self.C_ar > 0.0, f"C_ar must be positive, got {self.C_ar}"
        assert self.dt > 0.0, f"dt must be positive, got {self.dt}"

        # Pre-compute wheelbase
        self.L = self.l_f + self.l_r

        # Actuator lag state
        self.actual_delta = 0.0

        # Current state vector
        self.state = np.zeros(self.N_STATES, dtype=np.float64)

    def reset(
        self,
        v_x: float = cfg.V_REFERENCE,
        e_lat_init: float = 0.0,
        e_psi_init: float = 0.0,
        psi_init: float = 0.0,
    ) -> np.ndarray:
        """
        Reset the vehicle state.

        Args:
            v_x:        Initial longitudinal speed (m/s)
            e_lat_init: Initial lateral error (m)
            e_psi_init: Initial heading error (rad)
            psi_init:   Initial global yaw angle (rad)

        Returns:
            Copy of the initial state vector.
        """
        self.state = np.zeros(self.N_STATES, dtype=np.float64)
        self.state[self.IDX_VX] = v_x
        self.state[self.IDX_ELAT] = e_lat_init
        self.state[self.IDX_EPSI] = e_psi_init
        self.state[self.IDX_PSI] = psi_init
        self.actual_delta = 0.0
        return self.state.copy()

    def update_tyre_params(self, C_af: float, C_ar: float) -> None:
        """
        Update tyre cornering stiffness from DS-02 calibration.

        Args:
            C_af: Front axle cornering stiffness (N/rad), must be > 0
            C_ar: Rear axle cornering stiffness (N/rad), must be > 0

        Raises:
            ValueError: If calibrated values are non-physical.
        """
        if C_af <= 0.0 or C_ar <= 0.0:
            raise ValueError(
                f"Calibrated tyre stiffness must be positive: C_af={C_af}, C_ar={C_ar}"
            )
        if C_af > 500_000.0 or C_ar > 500_000.0:
            raise ValueError(
                f"Calibrated tyre stiffness implausibly large: C_af={C_af}, C_ar={C_ar}"
            )
        self.C_af = C_af
        self.C_ar = C_ar

    def _compute_tyre_forces(
        self, v_x: float, v_y: float, r: float, delta: float, friction_mu: float
    ) -> tuple[float, float]:
        """
        Compute front and rear lateral tyre forces.

        Pacejka Magic Formula (Simplified) to model saturation:
            F_y = D * sin(C * arctan(B * alpha))
            D = mu * F_z (peak force)
            B = C_alpha / (C * D) (stiffness factor)

        Args:
            v_x:   Longitudinal velocity (m/s)
            v_y:   Lateral velocity (m/s)
            r:     Yaw rate (rad/s)
            delta: Front wheel steering angle (rad)
            friction_mu: Road friction coefficient

        Returns:
            (F_yf, F_yr) — front and rear lateral forces in Newtons.
        """
        v_x_safe = max(abs(v_x), 0.5)

        alpha_f = np.arctan2(v_y + self.l_f * r, v_x_safe) - delta
        alpha_r = np.arctan2(v_y - self.l_r * r, v_x_safe)

        # Normal loads
        F_zf = self.m * 9.81 * self.l_r / self.L
        F_zr = self.m * 9.81 * self.l_f / self.L
        
        # Pacejka constants
        C = 1.3  # Shape factor
        D_f = friction_mu * F_zf
        D_r = friction_mu * F_zr
        
        B_f = self.C_af / (C * D_f) if D_f > 0 else 1.0
        B_r = self.C_ar / (C * D_r) if D_r > 0 else 1.0

        # Use -alpha since C_a applies a negative feedback
        F_yf = D_f * np.sin(C * np.arctan(B_f * -alpha_f))
        F_yr = D_r * np.sin(C * np.arctan(B_r * -alpha_r))

        return F_yf, F_yr

    def _dynamics(
        self, state: np.ndarray, delta: float, kappa_ref: float, friction_mu: float, bank_angle_rad: float
    ) -> np.ndarray:
        """
        Compute state derivatives ẋ for the nonlinear bicycle model.

        Equations — Rajamani (2012) Ch. 3:
            ẋ[0] = v_x·cos(ψ) − v_y·sin(ψ)        # Ẋ (global X)
            ẋ[1] = v_x·sin(ψ) + v_y·cos(ψ)        # Ẏ (global Y)
            ẋ[2] = r                                 # ψ̇ (yaw)
            ẋ[3] = 0.0                              # v̇_x (constant speed)
            ẋ[4] = (F_yf + F_yr)/m − v_x·r + g·sin(θ) # v̇_y (lateral accel + gravity)
            ẋ[5] = (l_f·F_yf − l_r·F_yr)/I_z        # ṙ (yaw accel)
            ẋ[6] = v_x·sin(e_psi) + v_y·cos(e_psi)  # ė_lat
            ẋ[7] = r − κ_ref·v_x                    # ė_psi

        Args:
            state:     8-dimensional state vector.
            delta:     Front wheel steering angle (rad).
            kappa_ref: Road reference curvature at current position (1/m).
            friction_mu: Road friction coefficient.
            bank_angle_rad: Road bank/camber angle (rad).

        Returns:
            8-dimensional state derivative vector.
        """
        X, Y, psi, v_x, v_y, r, e_lat, e_psi = state

        F_yf, F_yr = self._compute_tyre_forces(v_x, v_y, r, delta, friction_mu)

        dx = np.zeros(self.N_STATES, dtype=np.float64)
        dx[0] = v_x * np.cos(psi) - v_y * np.sin(psi)        # Ẋ
        dx[1] = v_x * np.sin(psi) + v_y * np.cos(psi)        # Ẏ
        dx[2] = r                                               # ψ̇
        dx[3] = 0.0                                             # v̇_x (constant)
        dx[4] = (F_yf + F_yr) / self.m - v_x * r + 9.81 * np.sin(bank_angle_rad) # v̇_y
        dx[5] = (self.l_f * F_yf - self.l_r * F_yr) / self.I_z  # ṙ
        dx[6] = v_x * np.sin(e_psi) + v_y * np.cos(e_psi)    # ė_lat
        dx[7] = r - kappa_ref * v_x                            # ė_psi

        return dx

    def step(
        self,
        delta_cmd: float = None,
        kappa_ref: float = 0.0,
        friction_mu: float = 1.0,
        bank_angle_rad: float = 0.0,
        delta: float = None,
    ) -> np.ndarray:
        if delta is not None:
            delta_cmd = delta
        if delta_cmd is None:
            delta_cmd = 0.0
        """
        Advance the vehicle state by one timestep using RK4 integration.

        Runge-Kutta 4th order (mandatory — no Euler):
            k1 = f(x, u)
            k2 = f(x + dt/2·k1, u)
            k3 = f(x + dt/2·k2, u)
            k4 = f(x + dt·k3, u)
            x_new = x + dt/6·(k1 + 2·k2 + 2·k3 + k4)

        Args:
            delta_cmd: Commanded front wheel steering angle (rad).
            kappa_ref: Road reference curvature at current position (1/m).
            friction_mu: Road friction coefficient.
            bank_angle_rad: Road bank/camber angle (rad).

        Returns:
            Updated state vector (copy).
        """
        # Clamp commanded steering input
        delta_cmd = np.clip(delta_cmd, -cfg.DELTA_MAX, cfg.DELTA_MAX)

        # Actuator lag (first order low-pass)
        tau = 0.1  # 100ms time constant
        alpha_filter = self.dt / (tau + self.dt)
        self.actual_delta = (1.0 - alpha_filter) * self.actual_delta + alpha_filter * delta_cmd
        
        delta = self.actual_delta

        # RK4 integration
        k1 = self._dynamics(self.state, delta, kappa_ref, friction_mu, bank_angle_rad)
        k2 = self._dynamics(self.state + 0.5 * self.dt * k1, delta, kappa_ref, friction_mu, bank_angle_rad)
        k3 = self._dynamics(self.state + 0.5 * self.dt * k2, delta, kappa_ref, friction_mu, bank_angle_rad)
        k4 = self._dynamics(self.state + self.dt * k3, delta, kappa_ref, friction_mu, bank_angle_rad)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Wrap yaw angle to [-π, π]
        self.state[self.IDX_PSI] = self._wrap_angle(self.state[self.IDX_PSI])
        # Wrap heading error to [-π, π]
        self.state[self.IDX_EPSI] = self._wrap_angle(self.state[self.IDX_EPSI])

        return self.state.copy()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @property
    def position(self) -> tuple[float, float]:
        """Global (X, Y) position in metres."""
        return float(self.state[self.IDX_X]), float(self.state[self.IDX_Y])

    @property
    def yaw(self) -> float:
        """Global yaw angle ψ in radians."""
        return float(self.state[self.IDX_PSI])

    @property
    def lateral_error(self) -> float:
        """Lateral deviation from lane centre e_lat in metres."""
        return float(self.state[self.IDX_ELAT])

    @property
    def heading_error(self) -> float:
        """Heading error e_psi in radians."""
        return float(self.state[self.IDX_EPSI])

    @property
    def v_x(self) -> float:
        """Longitudinal velocity in m/s."""
        return float(self.state[self.IDX_VX])

    @property
    def v_y(self) -> float:
        """Lateral velocity in m/s."""
        return float(self.state[self.IDX_VY])

    @property
    def yaw_rate(self) -> float:
        """Yaw rate r in rad/s."""
        return float(self.state[self.IDX_R])
