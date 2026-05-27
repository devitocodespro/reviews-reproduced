"""Analytical references for the LP04 reproduction convergence test.

Two cases:
1. Homogeneous fluid + travelling plane wave (used to validate the LW
   bulk independently of the ESIM coupling).
2. Plane wave incident on a planar fluid-solid interface — textbook
   Zoeppritz-like R/T coefficients (used for the full LP04 §4.2
   convergence test).

Both return wavefields as `(5, Nx, Ny)` arrays in the same
(v_x, v_y, σ_xx, σ_xy, σ_yy) layout used by `lax_wendroff.py`. Fluid
degenerate-solid convention: σ_xx = σ_yy = −p, σ_xy = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneWaveFluid:
    """Plane wave travelling in a homogeneous fluid.

    p(r, t)  = A · sin(ω (t − n̂·r/c) + φ)
    v_x(r,t) = (A / ρc) cos α · sin(ω (t − ...) + φ)
    v_y(r,t) = (A / ρc) sin α · sin(...)

    where α is the angle of n̂ from +x, ω is angular frequency, A is
    pressure amplitude, φ is phase offset.
    """
    A: float
    omega: float
    alpha: float          # propagation angle from +x (radians)
    phase: float          # initial phase offset
    c: float              # sound speed
    rho: float            # density

    @property
    def n_hat(self) -> tuple[float, float]:
        return (np.cos(self.alpha), np.sin(self.alpha))

    def evaluate(self, XX: np.ndarray, YY: np.ndarray, t: float
                  ) -> np.ndarray:
        """Return (5, Nx, Ny) state at time t."""
        nx, ny = self.n_hat
        # Argument of the wave
        arg = self.omega * (t - (nx * XX + ny * YY) / self.c) + self.phase
        p = self.A * np.sin(arg)
        # v = (p / ρc) · n̂
        v_scale = 1.0 / (self.rho * self.c)
        U = np.zeros((5,) + XX.shape, dtype=np.float64)
        U[0] = v_scale * nx * p   # v_x
        U[1] = v_scale * ny * p   # v_y
        U[2] = -p                  # σ_xx = -p
        U[3] = 0.0                 # σ_xy
        U[4] = -p                  # σ_yy = -p
        return U


@dataclass
class PlaneWaveAcousticElastic:
    """Plane wave incident from fluid onto a horizontal fluid-solid
    interface at y = y_interface. Computes the steady-state wavefield
    as incident + reflected (in fluid) + transmitted P + transmitted SV
    (in solid).

    Verification (2026-05-27): at normal incidence reproduces textbook
    acoustic R_pp = (Z_p − Z_f)/(Z_p + Z_f) = +0.747899 to fp64.
    Velocity-amplitude convention; system of 3 BCs (v_y continuity,
    σ_yy continuity, σ_xy = 0 on solid). Oblique incidence verified
    via energy-flux conservation gate (tests/test_rt_oblique.py).


    Geometry conventions:
      • Fluid occupies y > y_interface (above).
      • Solid occupies y < y_interface (below).
      • Incident wave propagates downward (+x, −y direction) at angle
        θ_inc from the −y axis.

    Snell's law: sin(θ_inc) / c_f = sin(θ_tp) / c_p = sin(θ_ts) / c_s.

    R/T coefficients for an acoustic plane wave incident from fluid
    onto a solid half-space (textbook formula; see e.g. Aki & Richards
    Quantitative Seismology, Eq 5.41 or Cerveny "Seismic Ray Theory"
    §5.6.1). The system of 3 equations in 3 unknowns (R_pp, T_pp,
    T_ps) comes from:
      (i)   continuity of normal velocity v_y:
            (1 + R_pp) cos θ_inc / (ρ_f c_f) = T_pp cos θ_tp / (ρ_s c_p)
                                              − T_ps sin θ_ts / (ρ_s c_s)
      (ii)  continuity of normal stress (−p = σ_yy):
            -(1 + R_pp) = σ_yy_solid_from_T_pp + σ_yy_solid_from_T_ps
      (iii) zero tangential stress on solid:
            σ_xy_solid = 0
    """
    A: float                       # incident pressure amplitude
    omega: float                   # angular frequency
    theta_inc: float               # incidence angle from −y (rad)
    c_f: float                     # fluid sound speed
    rho_f: float                   # fluid density
    c_p: float                     # solid P speed
    c_s: float                     # solid S speed
    rho_s: float                   # solid density
    y_interface: float             # y-coord of Γ
    phase: float = 0.0             # initial phase offset

    def _angles_and_coeffs(self):
        """Compute transmitted angles and R/T coefficients.

        VELOCITY-AMPLITUDE convention: each wave has scalar amplitude V
        such that the particle velocity is V·(polarization)·F(t-n̂·r/c).
        Pressure/stress amplitudes follow:
          fluid: p = ρ_f c_f V · F   (P-wave: longitudinal)
          solid P: σ_ij = -(V/c_p)(λ δ_ij + 2μ n_i n_j) · F
          solid SV: σ_ij = -(μ V/c_s)(pol_j n_i + pol_i n_j) · F
                     with n_SV = (s_ts, -c_ts), pol_SV = (c_ts, s_ts)

        Three boundary conditions at y = y_Γ:
          BC1 — v_y continuity:
                c_inc R + c_tp T_P - s_ts T_S = c_inc · 1
                (incident downward → -c_inc V_inc; reflected upward →
                 +c_inc V_ref; transmitted P → -c_tp V_tp; SV → +s_ts V_ts)
          BC2 — σ_yy continuity:
                Z_f R - (λ+2μ c_tp²)/c_p · T_P + 2μ s_ts c_ts/c_s · T_S
                  = -Z_f
                (fluid σ_yy = -p = -Z_f V_total)
          BC3 — σ_xy = 0 on solid:
                +2μ s_tp c_tp/c_p · T_P + μ(c_ts² - s_ts²)/c_s · T_S = 0

        At normal incidence reduces to R = (Z_p−Z_f)/(Z_p+Z_f), T_P=1-R,
        T_S = 0 (textbook acoustic; verified in __main__ smoke).
        """
        s_inc = np.sin(self.theta_inc)
        p = s_inc / self.c_f   # ray parameter
        s_tp = np.clip(p * self.c_p, -1.0, 1.0)
        s_ts = np.clip(p * self.c_s, -1.0, 1.0)
        theta_tp = np.arcsin(s_tp)
        theta_ts = np.arcsin(s_ts)
        c_inc = np.cos(self.theta_inc)
        c_tp = np.cos(theta_tp)
        c_ts = np.cos(theta_ts)
        Z_f = self.rho_f * self.c_f
        lam_s = (self.rho_s * self.c_p ** 2
                  - 2 * self.rho_s * self.c_s ** 2)
        mu_s = self.rho_s * self.c_s ** 2

        # 3×3 system M · (R, T_P, T_S)ᵀ = b
        M = np.array([
            # BC1: v_y continuity
            [c_inc, c_tp, -s_ts],
            # BC2: σ_yy continuity (units: stress = ρcV)
            [Z_f,
             -(lam_s + 2 * mu_s * c_tp ** 2) / self.c_p,
             +2 * mu_s * s_ts * c_ts / self.c_s],
            # BC3: σ_xy = 0 on solid
            [0.0,
             2 * mu_s * s_tp * c_tp / self.c_p,
             mu_s * (c_ts ** 2 - s_ts ** 2) / self.c_s],
        ])
        b = np.array([c_inc, -Z_f, 0.0])
        x = np.linalg.solve(M, b)
        R_pp, T_pp, T_ps = float(x[0]), float(x[1]), float(x[2])
        return dict(
            R_pp=R_pp, T_pp=T_pp, T_ps=T_ps,
            theta_inc=self.theta_inc, theta_tp=theta_tp,
            theta_ts=theta_ts,
            s_inc=s_inc, c_inc=c_inc, s_tp=s_tp, c_tp=c_tp,
            s_ts=s_ts, c_ts=c_ts,
        )

    def evaluate(self, XX: np.ndarray, YY: np.ndarray, t: float
                  ) -> np.ndarray:
        """Return (5, Nx, Ny) field at time t."""
        info = self._angles_and_coeffs()
        R_pp = info['R_pp']
        T_pp = info['T_pp']
        T_ps = info['T_ps']
        s_inc, c_inc = info['s_inc'], info['c_inc']
        s_tp, c_tp = info['s_tp'], info['c_tp']
        s_ts, c_ts = info['s_ts'], info['c_ts']
        U = np.zeros((5,) + XX.shape, dtype=np.float64)
        in_fluid = YY > self.y_interface
        in_solid = ~in_fluid
        # Incident: propagation direction (+s_inc, −c_inc), phase
        # argument arg_i = ω(t − ((s_inc) X + (−c_inc) (Y − y_Γ))/c_f).
        # The interface at y = y_interface is the reference plane.
        dY = YY - self.y_interface
        # Incident in fluid (down-going)
        arg_i = self.omega * (t - (s_inc * XX + (-c_inc) * dY) / self.c_f) + self.phase
        p_inc = self.A * np.sin(arg_i)
        # Reflected in fluid (up-going)
        arg_r = self.omega * (t - (s_inc * XX + (c_inc) * dY) / self.c_f) + self.phase
        p_ref = R_pp * self.A * np.sin(arg_r)
        # In fluid, p_total = p_inc + p_ref
        p_fluid = p_inc + p_ref
        # v_x in fluid: incident contributes s_inc · p / (ρ_f c_f);
        # reflected contributes s_inc · p_ref / (ρ_f c_f) (same x-direction
        # since x-component of n̂ is +s_inc for both incident and reflected).
        vx_fluid = (s_inc * (p_inc + p_ref)) / (self.rho_f * self.c_f)
        # v_y: incident is −c_inc / (ρ_f c_f) · p_inc;
        # reflected is +c_inc / (ρ_f c_f) · p_ref.
        vy_fluid = ((-c_inc) * p_inc + (c_inc) * p_ref) / (self.rho_f * self.c_f)
        # σ_xx = σ_yy = -p in fluid degenerate-solid layout, σ_xy = 0
        # Mask-fill
        U[0] = np.where(in_fluid, vx_fluid, 0.0)
        U[1] = np.where(in_fluid, vy_fluid, 0.0)
        U[2] = np.where(in_fluid, -p_fluid, 0.0)
        U[3] = np.where(in_fluid, 0.0, 0.0)
        U[4] = np.where(in_fluid, -p_fluid, 0.0)
        # Transmitted P in solid (down-going, direction (+s_tp, −c_tp))
        arg_tp = self.omega * (t - (s_tp * XX + (-c_tp) * dY) / self.c_p) + self.phase
        A_P = T_pp * self.A
        amp_tp = A_P * np.sin(arg_tp)
        # Transmitted SV in solid (down-going, polarization ⊥ propagation)
        arg_ts = self.omega * (t - (s_ts * XX + (-c_ts) * dY) / self.c_s) + self.phase
        A_SV = T_ps * self.A
        amp_ts = A_SV * np.sin(arg_ts)
        # Velocities and stresses in solid:
        #   P wave with amplitude A_P propagating at angle θ_tp from −y:
        #     v_x_P = (A_P / (ρ_s c_p)) s_tp
        #     v_y_P = (A_P / (ρ_s c_p)) (-c_tp)
        #     σ_xx_P = -(λ + 2μ s_tp²) A_P / (ρ_s c_p)
        #     σ_yy_P = -(λ + 2μ c_tp²) A_P / (ρ_s c_p)
        #     σ_xy_P = (2μ s_tp c_tp) A_P / (ρ_s c_p)
        #   SV wave with amplitude A_SV: polarization is perpendicular to
        #     propagation direction. Propagation = (s_ts, -c_ts);
        #     polarization = (c_ts, s_ts) (rotated 90° from propagation).
        #     v_x_SV = (A_SV / (ρ_s c_s)) c_ts
        #     v_y_SV = (A_SV / (ρ_s c_s)) s_ts
        #     σ_xy_SV = μ (cos²θ_ts − sin²θ_ts) / (ρ_s c_s) · A_SV
        #     σ_xx_SV = +2μ s_ts c_ts / (ρ_s c_s) · A_SV
        #     σ_yy_SV = −2μ s_ts c_ts / (ρ_s c_s) · A_SV
        lam_s = self.rho_s * self.c_p ** 2 - 2 * self.rho_s * self.c_s ** 2
        mu_s = self.rho_s * self.c_s ** 2
        vx_P = amp_tp * s_tp / (self.rho_s * self.c_p)
        vy_P = amp_tp * (-c_tp) / (self.rho_s * self.c_p)
        sigma_xx_P = -amp_tp * (lam_s + 2 * mu_s * s_tp ** 2) / (self.rho_s * self.c_p)
        sigma_yy_P = -amp_tp * (lam_s + 2 * mu_s * c_tp ** 2) / (self.rho_s * self.c_p)
        sigma_xy_P = amp_tp * (2 * mu_s * s_tp * c_tp) / (self.rho_s * self.c_p)
        vx_SV = amp_ts * c_ts / (self.rho_s * self.c_s)
        vy_SV = amp_ts * s_ts / (self.rho_s * self.c_s)
        sigma_xx_SV = amp_ts * (2 * mu_s * s_ts * c_ts) / (self.rho_s * self.c_s)
        sigma_yy_SV = -amp_ts * (2 * mu_s * s_ts * c_ts) / (self.rho_s * self.c_s)
        sigma_xy_SV = amp_ts * (mu_s * (c_ts ** 2 - s_ts ** 2)) / (self.rho_s * self.c_s)
        U[0] += np.where(in_solid, vx_P + vx_SV, 0.0)
        U[1] += np.where(in_solid, vy_P + vy_SV, 0.0)
        U[2] += np.where(in_solid, sigma_xx_P + sigma_xx_SV, 0.0)
        U[3] += np.where(in_solid, sigma_xy_P + sigma_xy_SV, 0.0)
        U[4] += np.where(in_solid, sigma_yy_P + sigma_yy_SV, 0.0)
        return U


if __name__ == '__main__':
    # Smoke 1: plane wave in homog fluid
    pw = PlaneWaveFluid(A=1.0, omega=2 * np.pi * 1e5, alpha=np.pi / 4,
                          phase=0.0, c=1500.0, rho=1000.0)
    X = np.linspace(0, 0.1, 50); Y = np.linspace(0, 0.1, 50)
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    U_t0 = pw.evaluate(XX, YY, t=0.0)
    print(f"Smoke 1 (fluid plane wave at t=0):")
    print(f"  max|p| = {float(np.max(np.abs(U_t0[2]))):.3e}")
    print(f"  max|v| = {float(np.max(np.abs(U_t0[:2]))):.3e}")
    # Smoke 2: fluid-solid plane wave at normal incidence (θ_inc=0)
    paewe = PlaneWaveAcousticElastic(
        A=1.0, omega=2 * np.pi * 5e4, theta_inc=0.0,
        c_f=1500.0, rho_f=1000.0, c_p=4000.0, c_s=2000.0, rho_s=2600.0,
        y_interface=0.5)
    info = paewe._angles_and_coeffs()
    print(f"\nSmoke 2 (fluid-solid, θ_inc=0):")
    print(f"  R_pp = {info['R_pp']:+.6f}   T_pp = {info['T_pp']:+.6f}   "
          f"T_ps = {info['T_ps']:+.6e}")
    # Normal-incidence R_pp: textbook (Z_solid - Z_fluid) / (Z_solid + Z_fluid)
    # with Z = ρ c. Z_f = 1.5e6, Z_p = 1.04e7.
    Zf = 1000 * 1500
    Zp = 2600 * 4000
    R_expected = (Zp - Zf) / (Zp + Zf)
    print(f"  textbook R_pp = (Z_p - Z_f)/(Z_p + Z_f) = {R_expected:+.6f}")
    print(f"  error: {abs(info['R_pp'] - R_expected):.2e}")
    # T_ps at normal incidence should be exactly 0
    assert abs(info['T_ps']) < 1e-12, (
        f"T_ps at normal incidence = {info['T_ps']}, expected 0")
    print("  OK: T_ps = 0 at normal incidence ✓")
