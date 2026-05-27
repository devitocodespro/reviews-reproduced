"""Lax-Wendroff 2nd-order bulk update for 2D acoustic-elastic media.

Implements the bulk scheme LP04 §3.1 uses as its workhorse: classical
Lax-Wendroff in two steps,

    U^{n+1} = U^n + dt L(U^n) + (dt^2 / 2) L²(U^n)

For the linear hyperbolic system ∂_t U + A ∂_x U + B ∂_y U = 0, this
becomes the Cauchy-Kowalevskaya formula

    U^{n+1} = U^n − dt (A ∂_x U + B ∂_y U)
              + (dt² / 2) (A² ∂²_x U + (AB+BA) ∂²_{xy} U + B² ∂²_y U)

We use centred 2nd-order FD for both ∂_x, ∂_y and ∂²_x, ∂²_y, ∂²_xy.

The state vector is the 5-component elastic layout
   U = (v_x, v_y, σ_xx, σ_xy, σ_yy)
on both sides of Γ. The fluid is treated as a degenerate elastic
medium with μ=0 (so c_s=0, λ=κ_fluid), which makes σ_xx=σ_yy=−p and
σ_xy=0 in equilibrium. This is the standard "fluid-as-degenerate-
solid" surrogate (parent repo uses it; bulk equations reduce
analytically to the fluid acoustic system). The ESIM projector
(esim_projector.py) is paper-faithful (3-fluid + 5-solid) so the
overall scheme retains LP04's interface coupling.

Flux Jacobians for 2D isotropic elastic (ρ, λ, μ):
  A_x = ∂F_x/∂U:
    F_x = (−σ_xx/ρ, −σ_xy/ρ, −(λ+2μ)v_x, −μ v_y, −λ v_x)
  A_y = ∂F_y/∂U:
    F_y = (−σ_xy/ρ, −σ_yy/ρ, −λ v_y, −μ v_x, −(λ+2μ) v_y)

Stability: CFL ≤ 1/(√2 · max(c_p)) for 2D LW; we use 0.5 for safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LWMaterial:
    """Per-cell material arrays. Shape (Nx, Ny) each."""
    rho: np.ndarray         # density
    lam: np.ndarray         # Lamé λ
    mu: np.ndarray          # Lamé μ (= 0 in fluid degenerate-solid surrogate)


def make_material_layered(Nx: int, Ny: int, dx: float,
                             interface_y: float,
                             tan_dip: float,
                             anchor_x: float,
                             rho_fluid: float, c_fluid: float,
                             rho_solid: float,
                             c_p_solid: float, c_s_solid: float
                             ) -> LWMaterial:
    """Material arrays for a dipping plane interface.

    z_Γ(x) = interface_y − tan_dip × (x − anchor_x)
    Fluid above (y > z_Γ); solid below (y < z_Γ).
    Fluid degenerate-solid: μ=0, λ=ρ_f c_f² (so c_p_fluid = c_f).
    """
    X = np.arange(Nx) * dx
    Y = np.arange(Ny) * dx
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    z_gamma = interface_y - tan_dip * (XX - anchor_x)
    mask_fluid = YY > z_gamma   # above Γ
    rho = np.where(mask_fluid, rho_fluid, rho_solid)
    mu_solid = rho_solid * c_s_solid ** 2
    mu = np.where(mask_fluid, 0.0, mu_solid)
    lam_fluid = rho_fluid * c_fluid ** 2
    lam_solid = rho_solid * c_p_solid ** 2 - 2 * mu_solid
    lam = np.where(mask_fluid, lam_fluid, lam_solid)
    return LWMaterial(rho=rho, lam=lam, mu=mu)


def _centered_dx(U: np.ndarray, dx: float, axis: int,
                   periodic: bool = False) -> np.ndarray:
    """2nd-order centred 1st derivative along `axis`. Zero-padded by
    default; pass ``periodic=True`` to use np.roll wrap-around."""
    if periodic:
        return (np.roll(U, -1, axis=axis) - np.roll(U, +1, axis=axis)) / (2 * dx)
    out = np.zeros_like(U)
    if axis == 1:
        out[:, 1:-1, :] = (U[:, 2:, :] - U[:, :-2, :]) / (2 * dx)
    elif axis == 2:
        out[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) / (2 * dx)
    return out


def _centered_d2(U: np.ndarray, dx: float, axis: int,
                   periodic: bool = False) -> np.ndarray:
    """2nd-order centred 2nd derivative along `axis`."""
    if periodic:
        return (np.roll(U, -1, axis=axis) - 2 * U
                 + np.roll(U, +1, axis=axis)) / (dx ** 2)
    out = np.zeros_like(U)
    if axis == 1:
        out[:, 1:-1, :] = (U[:, 2:, :] - 2 * U[:, 1:-1, :]
                             + U[:, :-2, :]) / (dx ** 2)
    elif axis == 2:
        out[:, :, 1:-1] = (U[:, :, 2:] - 2 * U[:, :, 1:-1]
                             + U[:, :, :-2]) / (dx ** 2)
    return out


def _centered_dxdy(U: np.ndarray, dx: float,
                     periodic: bool = False) -> np.ndarray:
    """Mixed 2nd derivative ∂²_xy via composition. 2nd-order centred."""
    if periodic:
        Up = np.roll(U, -1, axis=1); Um = np.roll(U, +1, axis=1)
        return (np.roll(Up, -1, axis=2) - np.roll(Up, +1, axis=2)
                 - np.roll(Um, -1, axis=2) + np.roll(Um, +1, axis=2)
                ) / (4 * dx * dx)
    out = np.zeros_like(U)
    out[:, 1:-1, 1:-1] = (U[:, 2:, 2:] - U[:, 2:, :-2]
                            - U[:, :-2, 2:] + U[:, :-2, :-2]
                            ) / (4 * dx * dx)
    return out


def _L_apply(U: np.ndarray, mat: LWMaterial, dx: float,
              periodic: bool = False) -> np.ndarray:
    """Apply L(U) = − A ∂_x U − B ∂_y U for the elastic system.

    Returns ∂_t U at every cell.
    """
    rho = mat.rho
    lam = mat.lam
    mu = mat.mu
    lam_p_2mu = lam + 2 * mu

    # Spatial derivatives of each component
    dU_dx = _centered_dx(U, dx, axis=1, periodic=periodic)
    dU_dy = _centered_dx(U, dx, axis=2, periodic=periodic)

    out = np.zeros_like(U)
    # ∂_t v_x = (1/ρ) (∂_x σ_xx + ∂_y σ_xy)
    out[0] = (dU_dx[2] + dU_dy[3]) / rho
    # ∂_t v_y = (1/ρ) (∂_x σ_xy + ∂_y σ_yy)
    out[1] = (dU_dx[3] + dU_dy[4]) / rho
    # ∂_t σ_xx = (λ+2μ) ∂_x v_x + λ ∂_y v_y
    out[2] = lam_p_2mu * dU_dx[0] + lam * dU_dy[1]
    # ∂_t σ_xy = μ (∂_y v_x + ∂_x v_y)
    out[3] = mu * (dU_dy[0] + dU_dx[1])
    # ∂_t σ_yy = λ ∂_x v_x + (λ+2μ) ∂_y v_y
    out[4] = lam * dU_dx[0] + lam_p_2mu * dU_dy[1]
    return out


def lw_step(U: np.ndarray, mat: LWMaterial, dx: float, dt: float,
             periodic: bool = False) -> np.ndarray:
    """One Lax-Wendroff step: U^{n+1} = U + dt L(U) + (dt²/2) L²(U)."""
    L_U = _L_apply(U, mat, dx, periodic=periodic)
    L2_U = _L_apply(L_U, mat, dx, periodic=periodic)
    return U + dt * L_U + 0.5 * dt * dt * L2_U


def cfl_dt(mat: LWMaterial, dx: float, cfl: float = 0.5) -> float:
    """LW CFL: dt ≤ cfl × dx / (√2 · c_p_max)."""
    lam_p_2mu = mat.lam + 2 * mat.mu
    c_p = np.sqrt(lam_p_2mu / mat.rho)
    c_max = float(np.max(c_p))
    return cfl * dx / (c_max * np.sqrt(2.0))


if __name__ == '__main__':
    # Quick smoke: propagate a Gaussian pulse on a small homog medium.
    Nx = Ny = 81
    dx = 0.01   # 10 mm cells
    rho = 1000.0
    c_f = 1500.0
    mat = make_material_layered(
        Nx, Ny, dx,
        interface_y=2.0,    # interface beyond domain → all-fluid
        tan_dip=0.0,
        anchor_x=0.5 * Nx * dx,
        rho_fluid=rho, c_fluid=c_f,
        rho_solid=rho, c_p_solid=c_f, c_s_solid=0.0,
    )
    dt = cfl_dt(mat, dx)
    print(f"Smoke: {Nx}×{Ny} fluid-only, dx={dx}, dt={dt:.3e}")
    print(f"  c_max = {np.sqrt(np.max((mat.lam + 2*mat.mu)/mat.rho)):.1f}")

    # Initial Gaussian pressure pulse at centre
    U = np.zeros((5, Nx, Ny))
    cx, cy = Nx // 2, Ny // 2
    X = np.arange(Nx) * dx
    Y = np.arange(Ny) * dx
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    sigma = 4 * dx
    pulse = np.exp(-((XX - X[cx]) ** 2 + (YY - Y[cy]) ** 2) / (2 * sigma ** 2))
    U[2] = -pulse  # σ_xx = -p
    U[4] = -pulse  # σ_yy = -p

    for n in range(50):
        U = lw_step(U, mat, dx, dt)

    max_v = float(np.max(np.abs(U[:2])))
    max_s = float(np.max(np.abs(U[2:])))
    finite = bool(np.all(np.isfinite(U)))
    print(f"  After 50 steps: finite={finite}, max|v|={max_v:.3e},"
          f" max|σ|={max_s:.3e}")
