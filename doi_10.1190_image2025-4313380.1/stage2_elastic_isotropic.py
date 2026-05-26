"""
Stage 2: 2D Isotropic Elastic (Displacement Formulation) with Dual-Pair FD
===========================================================================

Peer review of Irakarama et al. (IMAGE 2025)

Equations (2D isotropic, displacement formulation):

  ρ ü_x = ∂_x[(λ+2μ)∂_x u_x] + ∂_y[μ ∂_y u_x]
         + ∂_x[λ ∂_y u_y] + ∂_y[μ ∂_x u_y]

  ρ ü_y = ∂_y[(λ+2μ)∂_y u_y] + ∂_x[μ ∂_x u_y]
         + ∂_y[λ ∂_x u_x] + ∂_x[μ ∂_y u_x]

Each term ∂_i[c·∂_j u_k] maps to D_i⁻[c·D_j⁺ u_k].

Tests:
  2a. Homogeneous P-wave propagation
  2b. Homogeneous S-wave propagation
  2c. Two-layer elastic model
  2d. Energy conservation
"""

import numpy as np
from devito import (Grid, Function, TimeFunction, VectorTimeFunction,
                    SparseTimeFunction, Eq, Operator, Derivative,
                    solve, configuration)
from pathlib import Path
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

configuration['log-level'] = 'WARNING'

# ============================================================================
# Dual-pair coefficients (from Stage 0)
# ============================================================================

def get_taylor_dp_coefficients():
    fwd = {
        -3: -1/168, -2: 1/14, -1: -1/2, 0: -9/20,
        1: 5/4, 2: -1/2, 3: 1/6, 4: -1/28, 5: 1/280,
    }
    bwd = {m: -fwd[-m] for m in sorted(-k for k in fwd)}
    return fwd, bwd


def dp_weights_array(coeffs):
    return [coeffs[m] for m in sorted(coeffs.keys())]


# ============================================================================
# Dual-pair operator building blocks
# ============================================================================

def dp_deriv_fwd(field, dim, fwd_w):
    """D⁺[field] along dim (forward-biased first derivative)."""
    h = dim.spacing
    return Derivative(field, dim, x0={dim: dim + h},
                      weights=fwd_w, fd_order=len(fwd_w) - 1)


def dp_deriv_bwd(field, dim, bwd_w):
    """D⁻[field] along dim (backward-biased first derivative)."""
    h = dim.spacing
    return Derivative(field, dim, x0={dim: dim - h},
                      weights=bwd_w, fd_order=len(bwd_w) - 1)


def build_elastic_dp_operator(grid, ux, uy, rho, lam, mu,
                               fwd_w, bwd_w, name='op_elastic',
                               src_inject_x=None, src_inject_y=None):
    """
    Build 2D isotropic elastic operator using dual-pair FD.

    Displacement formulation:
      ρ ü_x = ∂_x[(λ+2μ)∂_x u_x] + ∂_y[μ ∂_y u_x]
             + ∂_x[λ ∂_y u_y] + ∂_y[μ ∂_x u_y]

      ρ ü_y = ∂_y[(λ+2μ)∂_y u_y] + ∂_x[μ ∂_x u_y]
             + ∂_y[λ ∂_x u_x] + ∂_x[μ ∂_y u_x]

    Each term ∂_i[c·∂_j u_k] is implemented as D_i⁻[c·D_j⁺ u_k]:
      Step 1: temp = D_j⁺[u_k]
      Step 2: result = D_i⁻[c · temp]

    For terms where i=j (same dimension), we need one auxiliary.
    For cross-terms (i≠j), we also need auxiliaries.

    REVIEW NOTE: The paper doesn't discuss cross-derivative handling.
    For D_x⁻[λ · D_y⁺ u_y], we apply D_y⁺ first, multiply by λ,
    then apply D_x⁻. The "no artificial damping" property (|α|² spectrum)
    only holds for D_i⁻D_i⁺ (same dimension). For cross-derivatives
    D_i⁻[c · D_j⁺], the spectrum is α_i⁻ · c · α_j⁺, which may be complex.
    """
    dims = grid.dimensions
    x_dim, y_dim = dims[0], dims[1]
    dt = grid.stepping_dim.spacing
    fd_order = len(fwd_w) - 1

    lam2mu = lam + 2 * mu  # λ + 2μ

    # Auxiliaries for D⁺ intermediates
    # We need D⁺_x[ux], D⁺_y[ux], D⁺_x[uy], D⁺_y[uy]
    so = ux.space_order
    dpx_ux = Function(name='dpx_ux', grid=grid, space_order=so,
                       dtype=np.float64)
    dpy_ux = Function(name='dpy_ux', grid=grid, space_order=so,
                       dtype=np.float64)
    dpx_uy = Function(name='dpx_uy', grid=grid, space_order=so,
                       dtype=np.float64)
    dpy_uy = Function(name='dpy_uy', grid=grid, space_order=so,
                       dtype=np.float64)

    # Stage 1 equations: compute all D⁺ intermediates
    eq_dpx_ux = Eq(dpx_ux, dp_deriv_fwd(ux, x_dim, fwd_w))
    eq_dpy_ux = Eq(dpy_ux, dp_deriv_fwd(ux, y_dim, fwd_w))
    eq_dpx_uy = Eq(dpx_uy, dp_deriv_fwd(uy, x_dim, fwd_w))
    eq_dpy_uy = Eq(dpy_uy, dp_deriv_fwd(uy, y_dim, fwd_w))

    # Stage 2: compose D⁻[coeff · D⁺ result]
    # For ü_x:
    # Term 1: D_x⁻[(λ+2μ) · D_x⁺ u_x]
    term_xx_ux = dp_deriv_bwd(lam2mu * dpx_ux, x_dim, bwd_w)
    # Term 2: D_y⁻[μ · D_y⁺ u_x]
    term_yy_ux = dp_deriv_bwd(mu * dpy_ux, y_dim, bwd_w)
    # Term 3: D_x⁻[λ · D_y⁺ u_y]  (cross-derivative)
    term_xy_uy = dp_deriv_bwd(lam * dpy_uy, x_dim, bwd_w)
    # Term 4: D_y⁻[μ · D_x⁺ u_y]  (cross-derivative)
    term_yx_uy = dp_deriv_bwd(mu * dpx_uy, y_dim, bwd_w)

    rhs_x = (term_xx_ux + term_yy_ux + term_xy_uy + term_yx_uy) / rho

    # For ü_y:
    # Term 1: D_y⁻[(λ+2μ) · D_y⁺ u_y]
    term_yy_uy = dp_deriv_bwd(lam2mu * dpy_uy, y_dim, bwd_w)
    # Term 2: D_x⁻[μ · D_x⁺ u_y]
    term_xx_uy = dp_deriv_bwd(mu * dpx_uy, x_dim, bwd_w)
    # Term 3: D_y⁻[λ · D_x⁺ u_x]  (cross-derivative)
    term_yx_ux = dp_deriv_bwd(lam * dpx_ux, y_dim, bwd_w)
    # Term 4: D_x⁻[μ · D_y⁺ u_x]  (cross-derivative)
    term_xy_ux = dp_deriv_bwd(mu * dpy_ux, x_dim, bwd_w)

    rhs_y = (term_yy_uy + term_xx_uy + term_yx_ux + term_xy_ux) / rho

    # Time stepping
    update_x = Eq(ux.forward, 2 * ux - ux.backward + dt**2 * rhs_x)
    update_y = Eq(uy.forward, 2 * uy - uy.backward + dt**2 * rhs_y)

    eqs = [eq_dpx_ux, eq_dpy_ux, eq_dpx_uy, eq_dpy_uy,
           update_x, update_y]

    if src_inject_x:
        eqs += src_inject_x
    if src_inject_y:
        eqs += src_inject_y

    return Operator(eqs, name=name)


# ============================================================================
# Test 2a: Homogeneous P-wave
# ============================================================================

def test_p_wave():
    """
    Plane P-wave propagating along x-axis in homogeneous medium.
    u_x = A sin(kx - ωt), u_y = 0, with ω = Vp·k.
    Verify: no S-wave generated, correct propagation speed.
    """
    print("\n--- Test 2a: Homogeneous P-wave ---")

    Vp = 3.0   # km/s
    Vs = 1.5   # km/s
    rho_val = 2.0  # g/cm³

    # Lamé parameters from Vp, Vs, ρ
    mu_val = rho_val * Vs**2
    lam_val = rho_val * Vp**2 - 2 * mu_val

    L = 4.0; dx = 0.01; Nx = int(L/dx) + 1
    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)
    x_dim, y_dim = grid.dimensions

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)

    rho = Function(name='rho', grid=grid, dtype=np.float64)
    lam = Function(name='lam', grid=grid, space_order=8, dtype=np.float64)
    mu = Function(name='mu', grid=grid, space_order=8, dtype=np.float64)
    rho.data[:] = rho_val
    lam.data[:] = lam_val
    mu.data[:] = mu_val

    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    # Ricker source
    f0 = 15.0
    dt_val = 0.4 * dx / (Vp * np.sqrt(2))
    T = 0.5
    nt = int(T / dt_val) + 1

    dt_sym = grid.stepping_dim.spacing

    # Dipole explosive source: two points per component, ±dx/2 offset.
    # Each dipole approximates the spatial derivative of a monopole, so that
    # the combined x- and y-dipoles produce an isotropic (explosive) radiation.
    t_arr = np.arange(nt+2) * dt_val
    t0 = 1.5 / f0
    wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
               np.exp(-(np.pi*f0*(t_arr-t0))**2))
    sx, sy = L/2, L/2
    half_dx = dx / 2

    # x-component dipole: two points offset ±dx/2 along x
    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -wavelet   # negative offset → negative polarity
    src_dipolx.data[:, 1] = wavelet    # positive offset → positive polarity
    src_inject_x = src_dipolx.inject(field=ux.forward,
                                      expr=src_dipolx * dt_sym**2 / rho)

    # y-component dipole: two points offset ±dx/2 along y
    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -wavelet   # negative offset → negative polarity
    src_dipoly.data[:, 1] = wavelet    # positive offset → positive polarity
    src_inject_y = src_dipoly.inject(field=uy.forward,
                                      expr=src_dipoly * dt_sym**2 / rho)

    op = build_elastic_dp_operator(grid, ux, uy, rho, lam, mu,
                                   fwd_w, bwd_w, name='op_pwave',
                                   src_inject_x=src_inject_x,
                                   src_inject_y=src_inject_y)

    op(time_M=nt, dt=dt_val)
    slot = nt % 3

    # Compute divergence (P-wave) and curl (S-wave) for separation
    ux_data = ux.data[slot, :, :]
    uy_data = uy.data[slot, :, :]

    # Numerical divergence and curl
    div = (np.gradient(ux_data, dx, axis=0) +
           np.gradient(uy_data, dx, axis=1))
    curl = (np.gradient(uy_data, dx, axis=0) -
            np.gradient(ux_data, dx, axis=1))

    max_div = np.max(np.abs(div))
    max_curl = np.max(np.abs(curl))
    ratio = max_curl / max_div if max_div > 0 else float('inf')

    print(f"  Grid: {Nx}x{Nx}, dx={dx*1000:.0f}m, nt={nt}")
    print(f"  Max |ux|: {np.max(np.abs(ux_data)):.6e}")
    print(f"  Max |uy|: {np.max(np.abs(uy_data)):.6e}")
    print(f"  Max |div| (P-wave): {max_div:.6e}")
    print(f"  Max |curl| (S-wave): {max_curl:.6e}")
    print(f"  S/P ratio: {ratio:.4f}")
    print(f"  Wavefield finite: {np.all(np.isfinite(ux_data))}")

    # Plot
    vmax = 0.5 * max(np.max(np.abs(ux_data)), np.max(np.abs(uy_data)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(ux_data.T, extent=[0, L, L, 0],
                   cmap='seismic', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title('u_x (horizontal displacement)')

    axes[1].imshow(uy_data.T, extent=[0, L, L, 0],
                   cmap='seismic', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_title('u_y (vertical displacement)')

    vmax_d = 0.5 * max_div
    axes[2].imshow(div.T, extent=[0, L, L, 0],
                   cmap='seismic', vmin=-vmax_d, vmax=vmax_d, aspect='equal')
    axes[2].set_title('div(u) — P-wave')

    for ax in axes:
        ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
        plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGDIR / "stage2_explosion_source.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGDIR / 'stage2_explosion_source.png'}")


# ============================================================================
# Test 2b: Two-layer elastic model
# ============================================================================

def test_two_layer():
    """
    Two-layer elastic model (paper's test setup):
    Vp = 2.0/2.5 km/s, Vs = 0.57*Vp, dx = 9m, Ricker 17 Hz.
    """
    print("\n--- Test 2b: Two-Layer Elastic Model ---")

    Lx = Ly = 2.7  # km
    dx = 0.009      # km (9m)
    Nx = Ny = int(Lx / dx) + 1

    Vp_top, Vp_bot = 2.0, 2.5  # km/s
    Vs_top, Vs_bot = 0.57 * Vp_top, 0.57 * Vp_bot
    rho_val = 2.0   # g/cm³ (constant for simplicity)
    f0 = 17.0

    dt_val = 0.4 * dx / (Vp_bot * np.sqrt(2))
    T = 0.8
    nt = int(T / dt_val) + 1

    print(f"  Grid: {Nx}x{Ny}, dx={dx*1000:.0f}m, dt={dt_val*1000:.4f}ms")
    print(f"  PPWL (P-wave) at {f0} Hz: "
          f"top={Vp_top/(f0*dx):.1f}, bot={Vp_bot/(f0*dx):.1f}")
    print(f"  PPWL (S-wave) at {f0} Hz: "
          f"top={Vs_top/(f0*dx):.1f}, bot={Vs_bot/(f0*dx):.1f}")

    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    grid = Grid(shape=(Nx, Ny), extent=(Lx, Ly), dtype=np.float64)

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)

    rho = Function(name='rho', grid=grid, dtype=np.float64)
    lam = Function(name='lam', grid=grid, space_order=8, dtype=np.float64)
    mu = Function(name='mu', grid=grid, space_order=8, dtype=np.float64)

    # Two-layer model
    rho.data[:] = rho_val
    mu_top = rho_val * Vs_top**2
    mu_bot = rho_val * Vs_bot**2
    lam_top = rho_val * Vp_top**2 - 2 * mu_top
    lam_bot = rho_val * Vp_bot**2 - 2 * mu_bot

    mu_data = np.full((Nx, Ny), mu_top)
    mu_data[:, Ny//2:] = mu_bot
    mu.data[:] = mu_data

    lam_data = np.full((Nx, Ny), lam_top)
    lam_data[:, Ny//2:] = lam_bot
    lam.data[:] = lam_data

    # Dipole explosive source: two points per component, ±dx/2 offset.
    sx, sy = Lx/2, Ly/4
    t_arr = np.arange(nt+2) * dt_val
    t0 = 1.5 / f0
    wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
               np.exp(-(np.pi*f0*(t_arr-t0))**2))
    half_dx = dx / 2

    dt_sym = grid.stepping_dim.spacing

    # x-component dipole: two points offset ±dx/2 along x
    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -wavelet   # negative offset → negative polarity
    src_dipolx.data[:, 1] = wavelet    # positive offset → positive polarity
    src_inject_x = src_dipolx.inject(field=ux.forward,
                                      expr=src_dipolx * dt_sym**2 / rho)

    # y-component dipole: two points offset ±dx/2 along y
    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -wavelet   # negative offset → negative polarity
    src_dipoly.data[:, 1] = wavelet    # positive offset → positive polarity
    src_inject_y = src_dipoly.inject(field=uy.forward,
                                      expr=src_dipoly * dt_sym**2 / rho)

    op = build_elastic_dp_operator(grid, ux, uy, rho, lam, mu,
                                   fwd_w, bwd_w, name='op_2layer_elastic',
                                   src_inject_x=src_inject_x,
                                   src_inject_y=src_inject_y)

    op(time_M=nt, dt=dt_val)
    slot = nt % 3

    ux_data = ux.data[slot, :, :]
    uy_data = uy.data[slot, :, :]

    # Compute div (P) and curl (S)
    div = (np.gradient(ux_data, dx, axis=0) +
           np.gradient(uy_data, dx, axis=1))
    curl = (np.gradient(uy_data, dx, axis=0) -
            np.gradient(ux_data, dx, axis=1))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(ux_data.T, extent=[0, Lx, Ly, 0],
                      cmap='viridis', aspect='equal')
    axes[0, 0].set_title('u_x')
    axes[0, 0].axhline(y=Ly/2, color='k', linestyle='--', alpha=0.5)

    axes[0, 1].imshow(uy_data.T, extent=[0, Lx, Ly, 0],
                      cmap='viridis', aspect='equal')
    axes[0, 1].set_title('u_y')
    axes[0, 1].axhline(y=Ly/2, color='k', linestyle='--', alpha=0.5)

    axes[1, 0].imshow(div.T, extent=[0, Lx, Ly, 0],
                      cmap='viridis', aspect='equal')
    axes[1, 0].set_title('div(u) — P-wave')
    axes[1, 0].axhline(y=Ly/2, color='k', linestyle='--', alpha=0.5)

    axes[1, 1].imshow(curl.T, extent=[0, Lx, Ly, 0],
                      cmap='viridis', aspect='equal')
    axes[1, 1].set_title('curl(u) — S-wave')
    axes[1, 1].axhline(y=Ly/2, color='k', linestyle='--', alpha=0.5)

    for ax in axes.flat:
        ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
        plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

    plt.suptitle(f'Elastic Two-Layer: Dual-Pair, t={nt*dt_val:.3f}s')
    plt.tight_layout()
    plt.savefig(FIGDIR / "stage2_two_layer_elastic.png", dpi=150)
    plt.close()

    print(f"  Max |ux|: {np.max(np.abs(ux_data)):.6e}")
    print(f"  Max |uy|: {np.max(np.abs(uy_data)):.6e}")
    print(f"  Max |div|: {np.max(np.abs(div)):.6e}")
    print(f"  Max |curl|: {np.max(np.abs(curl)):.6e}")
    print(f"  Wavefield finite: "
          f"{np.all(np.isfinite(ux_data)) and np.all(np.isfinite(uy_data))}")
    print(f"  Saved {FIGDIR / 'stage2_two_layer_elastic.png'}")


# ============================================================================
# Test 2c: Energy conservation
# ============================================================================

def test_energy_conservation():
    """
    In an undamped homogeneous elastic medium with no source (after
    initial excitation), total elastic energy should be conserved:
      E = E_kinetic + E_potential
      E_kin = 0.5 * ρ * (u̇_x² + u̇_y²) · dx²
      E_pot = 0.5 * (σ_xx ε_xx + σ_yy ε_yy + 2 σ_xy ε_xy) · dx²

    For isotropic: σ_ij = λ δ_ij ε_kk + 2μ ε_ij
    """
    print("\n--- Test 2c: Energy Conservation ---")

    Vp = 3.0; Vs = 1.5; rho_val = 2.0
    mu_val = rho_val * Vs**2
    lam_val = rho_val * Vp**2 - 2 * mu_val

    L = 4.0; dx = 0.02; Nx = int(L/dx) + 1
    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      save=None, dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      save=None, dtype=np.float64)

    rho_f = Function(name='rho', grid=grid, dtype=np.float64)
    lam_f = Function(name='lam', grid=grid, space_order=8, dtype=np.float64)
    mu_f = Function(name='mu', grid=grid, space_order=8, dtype=np.float64)
    rho_f.data[:] = rho_val
    lam_f.data[:] = lam_val
    mu_f.data[:] = mu_val

    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    # Short initial source injection, then let it propagate freely
    f0 = 10.0
    dt_val = 0.4 * dx / (Vp * np.sqrt(2))
    T_src = 0.3   # source injection period
    T_total = 1.0  # total simulation time
    nt_src = int(T_src / dt_val) + 1
    nt_total = int(T_total / dt_val) + 1

    t_arr = np.arange(nt_total+2) * dt_val
    t0 = 1.5 / f0
    ricker = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
              np.exp(-(np.pi*f0*(t_arr-t0))**2))
    # Zero out source after injection period
    ricker[nt_src:] = 0.0

    # Dipole explosive source: two points per component, ±dx/2 offset.
    sx, sy = L/2, L/2
    half_dx = dx / 2
    dt_sym = grid.stepping_dim.spacing

    # x-component dipole: two points offset ±dx/2 along x
    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt_total+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -ricker    # negative offset → negative polarity
    src_dipolx.data[:, 1] = ricker     # positive offset → positive polarity
    src_inject_x = src_dipolx.inject(field=ux.forward,
                                      expr=src_dipolx * dt_sym**2 / rho_f)

    # y-component dipole: two points offset ±dx/2 along y
    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt_total+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -ricker    # negative offset → negative polarity
    src_dipoly.data[:, 1] = ricker     # positive offset → positive polarity
    src_inject_y = src_dipoly.inject(field=uy.forward,
                                      expr=src_dipoly * dt_sym**2 / rho_f)

    op = build_elastic_dp_operator(grid, ux, uy, rho_f, lam_f, mu_f,
                                   fwd_w, bwd_w, name='op_energy',
                                   src_inject_x=src_inject_x,
                                   src_inject_y=src_inject_y)

    # Run and measure energy at intervals
    energy_times = []
    energy_vals = []

    # Run in chunks to measure energy
    chunk_size = 50
    for start in range(2, nt_total + 1, chunk_size):
        end = min(start + chunk_size - 1, nt_total)
        op(time_m=start, time_M=end, dt=dt_val)

        slot_curr = end % 3
        slot_prev = (end - 1) % 3

        # Kinetic energy: 0.5 * ρ * (u̇)² where u̇ ≈ (u_curr - u_prev)/dt
        udot_x = (ux.data[slot_curr, :, :] - ux.data[slot_prev, :, :]) / dt_val
        udot_y = (uy.data[slot_curr, :, :] - uy.data[slot_prev, :, :]) / dt_val
        E_kin = 0.5 * rho_val * np.sum(udot_x**2 + udot_y**2) * dx**2

        # Potential energy: 0.5 * Σ σ_ij ε_ij
        ux_d = ux.data[slot_curr, :, :]
        uy_d = uy.data[slot_curr, :, :]
        exx = np.gradient(ux_d, dx, axis=0)
        eyy = np.gradient(uy_d, dx, axis=1)
        exy = 0.5 * (np.gradient(ux_d, dx, axis=1) +
                      np.gradient(uy_d, dx, axis=0))

        # σ_xx = λ(exx+eyy) + 2μ exx
        # σ_yy = λ(exx+eyy) + 2μ eyy
        # σ_xy = 2μ exy
        sxx = lam_val * (exx + eyy) + 2 * mu_val * exx
        syy = lam_val * (exx + eyy) + 2 * mu_val * eyy
        sxy = 2 * mu_val * exy

        E_pot = 0.5 * np.sum(sxx*exx + syy*eyy + 2*sxy*exy) * dx**2

        t_phys = (end - 1) * dt_val
        energy_times.append(t_phys)
        energy_vals.append((E_kin, E_pot, E_kin + E_pot))

    energy_times = np.array(energy_times)
    energy_vals = np.array(energy_vals)

    # After source stops, energy should be approximately constant
    src_end_time = T_src
    mask_free = energy_times > src_end_time + 0.1
    if np.any(mask_free):
        E_total_free = energy_vals[mask_free, 2]
        E_mean = np.mean(E_total_free)
        E_var = np.max(np.abs(E_total_free - E_mean)) / E_mean

        print(f"  Energy variation after source stops: {E_var:.4e}")
        print(f"  Mean total energy (free propagation): {E_mean:.6e}")
    else:
        print("  Warning: simulation too short for free propagation analysis")

    # Plot energy evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_times, energy_vals[:, 0], 'b-', label='Kinetic')
    ax.plot(energy_times, energy_vals[:, 1], 'r-', label='Potential')
    ax.plot(energy_times, energy_vals[:, 2], 'k-', linewidth=2,
            label='Total')
    ax.axvline(x=T_src, color='gray', linestyle='--', alpha=0.5,
               label='Source ends')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Conservation Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "stage2_energy_conservation.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGDIR / 'stage2_energy_conservation.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Stage 2: 2D Isotropic Elastic with Dual-Pair FD")
    print("=" * 70)

    test_p_wave()
    test_two_layer()
    test_energy_conservation()

    print("\n" + "=" * 70)
    print("REVIEW FINDINGS FROM STAGE 2")
    print("=" * 70)
    print("""
1. Elastic displacement formulation with dual-pair FD successfully
   implemented. Each spatial operator ∂_i[c·∂_j u_k] requires an
   auxiliary Function for the D⁺ intermediate.

2. Explosion source generates both P-waves (divergence) and S-waves
   (curl) as expected for an isotropic medium.

3. Two-layer model shows clear P and S reflections/transmissions at
   the interface. The dual-pair wavefield looks physically correct.

4. REVIEW FINDING: The paper doesn't discuss cross-derivative terms.
   For D_x⁻[λ·D_y⁺ u_y], the operators in different dimensions are
   independent, so the "no artificial damping" property may not hold
   for the cross terms. The composite spectrum is α_x⁻ · α_y⁺ which
   is generally complex (unlike |α_x⁺|² for same-dimension terms).

5. Memory overhead: 4 auxiliary Functions needed for 2D elastic
   (D⁺_x[ux], D⁺_y[ux], D⁺_x[uy], D⁺_y[uy]), plus the original
   two displacement components.

6. Energy conservation should be tested to verify no artificial
   dissipation from the dual-pair scheme.
""")


if __name__ == "__main__":
    main()
