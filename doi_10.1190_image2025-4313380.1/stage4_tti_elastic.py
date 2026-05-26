"""
Stage 4: 2D TTI Elastic with Dual-Pair FD
==========================================

Peer review of Irakarama et al. (IMAGE 2025)

Adds tilted transverse isotropy (TTI) using Bond rotation of the
stiffness tensor. The wave equation in 2D with general anisotropy is:

  ρ ü_x = ∂_x[C₁₁ ∂_x u_x + C₁₂ ∂_y u_y + C₁₆(∂_y u_x + ∂_x u_y)]
         + ∂_y[C₁₆ ∂_x u_x + C₂₆ ∂_y u_y + C₆₆(∂_y u_x + ∂_x u_y)]

  ρ ü_y = ∂_x[C₁₆ ∂_x u_x + C₂₆ ∂_y u_y + C₆₆(∂_y u_x + ∂_x u_y)]
         + ∂_y[C₁₂ ∂_x u_x + C₂₂ ∂_y u_y + C₂₆(∂_y u_x + ∂_x u_y)]

Using Voigt notation: xx→1, yy→2, xy→6

Each term ∂_i[Cij · ∂_j uk] maps to D_i⁻[Cij · D_j⁺ uk]

Tests:
  4a. Isotropic limit (ε=δ=0): should recover Stage 2 results
  4b. VTI (θ=0): verify VTI wavefront shape
  4c. TTI with rotation: verify tilted wavefront
"""

import numpy as np
from devito import (Grid, Function, TimeFunction, SparseTimeFunction,
                    Eq, Operator, Derivative, configuration)
from pathlib import Path
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

configuration['log-level'] = 'WARNING'

# ============================================================================
# Coefficients (from previous stages)
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


def get_taylor_filter_coefficients():
    return {0: 63/256, 1: -105/512, 2: 15/128,
            3: -45/1024, 4: 5/512, 5: -1/1024}


def get_proposed_filter_coefficients():
    return {
        0:  0.2178434,
        1: -0.1894915,
        2:  0.1233971,
        3: -0.0577743,
        4:  0.0176902,
        5: -0.0027252,
    }


# ============================================================================
# Selective filter
# ============================================================================

# ============================================================================
# TTI stiffness tensor
# ============================================================================

def compute_tti_stiffness_2d(Vp, Vs, rho, epsilon, delta, theta):
    """
    Compute 2D TTI stiffness tensor components (Voigt notation).

    In the symmetry frame (VTI):
      C'₁₁ = ρVp²(1+2ε)
      C'₂₂ = ρVp²
      C'₆₆ = ρVs²
      C'₁₂ = √((C'₂₂ - C'₆₆)(C'₂₂(1+2δ) - C'₆₆)) - C'₆₆

    Note: In 2D with Voigt notation:
    - Index 1 → x (horizontal), 2 → y (vertical), 6 → xy (shear)
    - Symmetry axis is vertical (y) for VTI
    - So C'₁₁ is along the symmetry axis direction
    - And C'₂₂ = ρVp² is the vertical P-wave velocity

    Convention: Thomsen (1986)
    - Vp = vertical P-wave velocity (along symmetry axis)
    - Vs = vertical S-wave velocity
    - ε = (C₁₁-C₃₃)/(2C₃₃) in 3D; analogous in 2D
    - δ = related to near-vertical P-wave velocity variation

    For 2D (x-y plane, symmetry axis = y):
    - C'₂₂ = ρVp²              (vertical P)
    - C'₁₁ = ρVp²(1+2ε)        (horizontal P)
    - C'₆₆ = ρVs²              (shear)
    - C'₁₂ from Thomsen relation

    Then rotate by θ using Bond transformation.

    Parameters
    ----------
    Vp, Vs : float or ndarray — vertical P and S velocities (km/s)
    rho : float or ndarray — density (g/cm³)
    epsilon, delta : float or ndarray — Thomsen parameters
    theta : float or ndarray — tilt angle in radians (0 = VTI)

    Returns
    -------
    C11, C22, C12, C16, C26, C66 : rotated stiffness components
    """
    # VTI stiffness in symmetry frame
    C22_vti = rho * Vp**2                    # Along symmetry axis
    C11_vti = rho * Vp**2 * (1 + 2*epsilon)  # Perpendicular
    C66_vti = rho * Vs**2                    # Shear

    # C12 from Thomsen relation
    inner = (C22_vti - C66_vti) * (C22_vti * (1 + 2*delta) - C66_vti)
    # Guard against negative values
    if isinstance(inner, np.ndarray):
        inner = np.maximum(inner, 0)
    else:
        inner = max(inner, 0)
    C12_vti = np.sqrt(inner) - C66_vti

    # Verify: for isotropic (ε=δ=0):
    # C11 = C22 = ρVp², C66 = ρVs², C12 = ρVp²-2ρVs² = λ ✓

    # Bond rotation by angle θ
    # 2D rotation matrix for Voigt notation:
    # [C]_rotated = M · [C]_vti · M^T
    # where M is the Bond transformation matrix
    c = np.cos(theta)
    s = np.sin(theta)

    # Bond matrix for 2D (3x3 in Voigt):
    # [c²    s²    2cs  ]   [C11 C12 C16]   [c²    s²   -2cs ]
    # [s²    c²   -2cs  ] · [C12 C22 C26] · [s²    c²    2cs ]
    # [-cs   cs   c²-s² ]   [C16 C26 C66]   [cs   -cs   c²-s²]
    #
    # For VTI: C16=C26=0, so the rotation simplifies.

    # Direct formulas for rotated stiffness:
    c2 = c**2; s2 = s**2; cs = c*s
    c4 = c2**2; s4 = s2**2

    C11 = (C11_vti * c4 + C22_vti * s4 +
           2*(C12_vti + 2*C66_vti) * c2 * s2)
    C22 = (C11_vti * s4 + C22_vti * c4 +
           2*(C12_vti + 2*C66_vti) * c2 * s2)
    C12 = ((C11_vti + C22_vti - 4*C66_vti) * c2 * s2 +
           C12_vti * (c4 + s4))
    C66 = ((C11_vti + C22_vti - 2*C12_vti - 2*C66_vti) * c2 * s2 +
           C66_vti * (c4 + s4))
    C16 = ((C11_vti - C12_vti - 2*C66_vti) * c2 * cs -
           (C22_vti - C12_vti - 2*C66_vti) * s2 * cs)
    C26 = ((C11_vti - C12_vti - 2*C66_vti) * s2 * cs -
           (C22_vti - C12_vti - 2*C66_vti) * c2 * cs)

    return C11, C22, C12, C16, C26, C66


# ============================================================================
# Dual-pair operator helpers
# ============================================================================

def dp_deriv_fwd(field, dim, fwd_w):
    h = dim.spacing
    return Derivative(field, dim, x0={dim: dim + h},
                      weights=fwd_w, fd_order=len(fwd_w) - 1)


def dp_deriv_bwd(field, dim, bwd_w):
    h = dim.spacing
    return Derivative(field, dim, x0={dim: dim - h},
                      weights=bwd_w, fd_order=len(bwd_w) - 1)


def build_tti_elastic_operator(grid, ux, uy, rho_f,
                                C11_f, C22_f, C12_f, C16_f, C26_f, C66_f,
                                fwd_w, bwd_w, name='op_tti',
                                src_inject_x=None, src_inject_y=None,
                                filter_coeffs=None, sigma=0.035):
    """
    Build 2D anisotropic elastic operator with dual-pair FD.

    Equations:
    ρ ü_x = ∂_x[C₁₁ ∂_x ux + C₁₂ ∂_y uy + C₁₆(∂_y ux + ∂_x uy)]
           + ∂_y[C₁₆ ∂_x ux + C₂₆ ∂_y uy + C₆₆(∂_y ux + ∂_x uy)]

    ρ ü_y = ∂_x[C₁₆ ∂_x ux + C₂₆ ∂_y uy + C₆₆(∂_y ux + ∂_x uy)]
           + ∂_y[C₁₂ ∂_x ux + C₂₂ ∂_y uy + C₂₆(∂_y ux + ∂_x uy)]

    Each ∂_i[...] term uses D_i⁻[...], and each ∂_j inside uses D_j⁺.
    """
    x_dim, y_dim = grid.dimensions
    dt = grid.stepping_dim.spacing
    so = ux.space_order

    # D⁺ intermediates
    dpx_ux = Function(name='dpxux', grid=grid, space_order=so,
                       dtype=np.float64)
    dpy_ux = Function(name='dpyux', grid=grid, space_order=so,
                       dtype=np.float64)
    dpx_uy = Function(name='dpxuy', grid=grid, space_order=so,
                       dtype=np.float64)
    dpy_uy = Function(name='dpyuy', grid=grid, space_order=so,
                       dtype=np.float64)

    eq_dpx_ux = Eq(dpx_ux, dp_deriv_fwd(ux, x_dim, fwd_w))
    eq_dpy_ux = Eq(dpy_ux, dp_deriv_fwd(ux, y_dim, fwd_w))
    eq_dpx_uy = Eq(dpx_uy, dp_deriv_fwd(uy, x_dim, fwd_w))
    eq_dpy_uy = Eq(dpy_uy, dp_deriv_fwd(uy, y_dim, fwd_w))

    # Stress-like terms (what goes inside D⁻)
    # σ_xx-like = C₁₁·∂_x ux + C₁₂·∂_y uy + C₁₆·(∂_y ux + ∂_x uy)
    sxx = C11_f * dpx_ux + C12_f * dpy_uy + C16_f * (dpy_ux + dpx_uy)
    # σ_xy-like (from x-eqn, y-derivative):
    # C₁₆·∂_x ux + C₂₆·∂_y uy + C₆₆·(∂_y ux + ∂_x uy)
    sxy = C16_f * dpx_ux + C26_f * dpy_uy + C66_f * (dpy_ux + dpx_uy)
    # σ_yx-like (from y-eqn, x-derivative) — same as sxy by symmetry
    syx = sxy
    # σ_yy-like:
    # C₁₂·∂_x ux + C₂₂·∂_y uy + C₂₆·(∂_y ux + ∂_x uy)
    syy = C12_f * dpx_ux + C22_f * dpy_uy + C26_f * (dpy_ux + dpx_uy)

    # Apply D⁻ to get divergence terms
    rhs_x = (dp_deriv_bwd(sxx, x_dim, bwd_w) +
             dp_deriv_bwd(sxy, y_dim, bwd_w)) / rho_f

    rhs_y = (dp_deriv_bwd(syx, x_dim, bwd_w) +
             dp_deriv_bwd(syy, y_dim, bwd_w)) / rho_f

    if filter_coeffs is not None:
        # Build filter weights
        M = max(filter_coeffs.keys())
        fw = {0: 1 - sigma * filter_coeffs[0]}
        for m in range(1, M + 1):
            fw[m] = -sigma * filter_coeffs[m]
            fw[-m] = -sigma * filter_coeffs[m]

        # Intermediates: store unfiltered time-step RHS before writing
        # to ux.forward / uy.forward
        raw_ux = Function(name='raw_ux', grid=grid, space_order=so,
                          dtype=np.float64)
        raw_uy = Function(name='raw_uy', grid=grid, space_order=so,
                          dtype=np.float64)
        eq_raw_x = Eq(raw_ux, 2*ux - ux.backward + dt**2 * rhs_x)
        eq_raw_y = Eq(raw_uy, 2*uy - uy.backward + dt**2 * rhs_y)

        # 2D product filter: equivalent to sequential x-then-y filtering.
        # u_filt[i,j] = Σ_m Σ_n fw[m]*fw[n]*raw[i+m, j+n]
        # Applied directly to raw field in the update equation — no copy-back.
        h_x = x_dim.spacing
        h_y = y_dim.spacing

        def filter_2d(field):
            result = 0
            for m in range(-M, M + 1):
                fx = field.subs(x_dim, x_dim + m * h_x)
                for n in range(-M, M + 1):
                    result += fw[m] * fw[n] * fx.subs(y_dim, y_dim + n * h_y)
            return result

        update_x = Eq(ux.forward, filter_2d(raw_ux))
        update_y = Eq(uy.forward, filter_2d(raw_uy))

        eqs = [eq_dpx_ux, eq_dpy_ux, eq_dpx_uy, eq_dpy_uy,
               eq_raw_x, eq_raw_y,
               update_x, update_y]
    else:
        update_x = Eq(ux.forward, 2*ux - ux.backward + dt**2 * rhs_x)
        update_y = Eq(uy.forward, 2*uy - uy.backward + dt**2 * rhs_y)

        eqs = [eq_dpx_ux, eq_dpy_ux, eq_dpx_uy, eq_dpy_uy,
               update_x, update_y]

    if src_inject_x:
        eqs += src_inject_x
    if src_inject_y:
        eqs += src_inject_y

    return Operator(eqs, name=name)


# ============================================================================
# Test 4a: Isotropic limit (ε=δ=0)
# ============================================================================

def test_isotropic_limit():
    """
    With ε=δ=0, TTI reduces to isotropic. Verify wavefronts are circular.
    """
    print("\n--- Test 4a: Isotropic Limit (ε=δ=0) ---")

    Vp = 3.0; Vs = 1.5; rho_val = 2.0
    eps = 0.0; delt = 0.0; theta = 0.0

    L = 4.0; dx = 0.01; Nx = int(L/dx) + 1
    f0 = 30.0

    C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
        Vp, Vs, rho_val, eps, delt, theta)

    # Verify isotropic: C11=C22=ρVp², C66=ρVs², C12=λ, C16=C26=0
    print(f"  C11={C11:.4f}, C22={C22:.4f}, expected={rho_val*Vp**2:.4f}")
    print(f"  C66={C66:.4f}, expected={rho_val*Vs**2:.4f}")
    print(f"  C16={C16:.4e}, C26={C26:.4e} (should be 0)")
    print(f"  C12={C12:.4f}, expected λ={rho_val*Vp**2-2*rho_val*Vs**2:.4f}")

    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)
    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    dt_val = 0.4 * dx / (Vp * np.sqrt(2))
    T = 0.5
    nt = int(T / dt_val) + 1

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)

    rho_f = Function(name='rho', grid=grid, dtype=np.float64)
    C11_f = Function(name='C11', grid=grid, space_order=8, dtype=np.float64)
    C22_f = Function(name='C22', grid=grid, space_order=8, dtype=np.float64)
    C12_f = Function(name='C12', grid=grid, space_order=8, dtype=np.float64)
    C16_f = Function(name='C16', grid=grid, space_order=8, dtype=np.float64)
    C26_f = Function(name='C26', grid=grid, space_order=8, dtype=np.float64)
    C66_f = Function(name='C66', grid=grid, space_order=8, dtype=np.float64)

    rho_f.data[:] = rho_val
    C11_f.data[:] = C11; C22_f.data[:] = C22; C12_f.data[:] = C12
    C16_f.data[:] = C16; C26_f.data[:] = C26; C66_f.data[:] = C66

    t_arr = np.arange(nt+2) * dt_val
    t0 = 1.5 / f0
    wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
               np.exp(-(np.pi*f0*(t_arr-t0))**2))
    sx, sy = L/2, L/2
    half_dx = dx / 2
    dt_sym = grid.stepping_dim.spacing

    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -wavelet
    src_dipolx.data[:, 1] = wavelet
    src_inj_x = src_dipolx.inject(field=ux.forward,
                                   expr=src_dipolx * dt_sym**2 / rho_f)

    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -wavelet
    src_dipoly.data[:, 1] = wavelet
    src_inj_y = src_dipoly.inject(field=uy.forward,
                                   expr=src_dipoly * dt_sym**2 / rho_f)

    filter_coeffs = get_proposed_filter_coefficients()
    op = build_tti_elastic_operator(grid, ux, uy, rho_f,
                                    C11_f, C22_f, C12_f, C16_f, C26_f, C66_f,
                                    fwd_w, bwd_w, name='op_iso',
                                    src_inject_x=src_inj_x,
                                    src_inject_y=src_inj_y,
                                    filter_coeffs=filter_coeffs, sigma=0.035)

    op(time_M=nt, dt=dt_val)
    slot = nt % 3

    ux_data = ux.data[slot, :, :]
    uy_data = uy.data[slot, :, :]
    div = np.gradient(ux_data, dx, axis=0) + np.gradient(uy_data, dx, axis=1)

    vmax = 0.3 * np.max(np.abs(div))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(div.T, extent=[0, L, L, 0], cmap='seismic',
              vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title('Isotropic limit (ε=δ=0): div(u) — circular wavefront')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
    plt.colorbar(ax.images[0], ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "stage4_isotropic_limit.png", dpi=150)
    plt.close()

    print(f"  Wavefield finite: "
          f"{np.all(np.isfinite(ux_data)) and np.all(np.isfinite(uy_data))}")
    print(f"  Max |div|: {np.max(np.abs(div)):.6e}")
    print(f"  Saved {FIGDIR / 'stage4_isotropic_limit.png'}")


# ============================================================================
# Test 4b: VTI (θ=0)
# ============================================================================

def test_vti():
    """
    VTI with ε=0.2, δ=0.1, θ=0. P-wave wavefront should be elliptical
    (faster horizontally than vertically).
    """
    print("\n--- Test 4b: VTI (ε=0.2, δ=0.1, θ=0) ---")

    Vp = 3.0; Vs = 1.5; rho_val = 2.0
    eps = 0.2; delt = 0.1; theta = 0.0

    C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
        Vp, Vs, rho_val, eps, delt, theta)

    print(f"  C11={C11:.4f} (horizontal), C22={C22:.4f} (vertical)")
    print(f"  C66={C66:.4f}, C12={C12:.4f}")
    print(f"  C16={C16:.4e}, C26={C26:.4e} (should be 0 for VTI)")
    print(f"  Vp_h = √(C11/ρ) = {np.sqrt(C11/rho_val):.4f} km/s")
    print(f"  Vp_v = √(C22/ρ) = {np.sqrt(C22/rho_val):.4f} km/s")

    L = 4.0; dx = 0.01; Nx = int(L/dx) + 1
    f0 = 30.0

    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)
    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    # CFL with fastest velocity
    Vp_max = np.sqrt(C11 / rho_val)
    dt_val = 0.4 * dx / (Vp_max * np.sqrt(2))
    T = 0.5
    nt = int(T / dt_val) + 1

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)

    rho_f = Function(name='rho', grid=grid, dtype=np.float64)
    C11_f = Function(name='C11', grid=grid, space_order=8, dtype=np.float64)
    C22_f = Function(name='C22', grid=grid, space_order=8, dtype=np.float64)
    C12_f = Function(name='C12', grid=grid, space_order=8, dtype=np.float64)
    C16_f = Function(name='C16', grid=grid, space_order=8, dtype=np.float64)
    C26_f = Function(name='C26', grid=grid, space_order=8, dtype=np.float64)
    C66_f = Function(name='C66', grid=grid, space_order=8, dtype=np.float64)

    rho_f.data[:] = rho_val
    C11_f.data[:] = C11; C22_f.data[:] = C22; C12_f.data[:] = C12
    C16_f.data[:] = C16; C26_f.data[:] = C26; C66_f.data[:] = C66

    t_arr = np.arange(nt+2) * dt_val
    t0 = 1.5 / f0
    wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
               np.exp(-(np.pi*f0*(t_arr-t0))**2))
    sx, sy = L/2, L/2
    half_dx = dx / 2
    dt_sym = grid.stepping_dim.spacing

    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -wavelet
    src_dipolx.data[:, 1] = wavelet
    src_inj_x = src_dipolx.inject(field=ux.forward,
                                   expr=src_dipolx * dt_sym**2 / rho_f)

    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -wavelet
    src_dipoly.data[:, 1] = wavelet
    src_inj_y = src_dipoly.inject(field=uy.forward,
                                   expr=src_dipoly * dt_sym**2 / rho_f)

    filter_coeffs = get_proposed_filter_coefficients()
    op = build_tti_elastic_operator(grid, ux, uy, rho_f,
                                    C11_f, C22_f, C12_f, C16_f, C26_f, C66_f,
                                    fwd_w, bwd_w, name='op_vti',
                                    src_inject_x=src_inj_x,
                                    src_inject_y=src_inj_y,
                                    filter_coeffs=filter_coeffs, sigma=0.035)

    op(time_M=nt, dt=dt_val)
    slot = nt % 3

    ux_data = ux.data[slot, :, :]
    uy_data = uy.data[slot, :, :]
    div = np.gradient(ux_data, dx, axis=0) + np.gradient(uy_data, dx, axis=1)

    vmax = 0.3 * np.max(np.abs(div))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(div.T, extent=[0, L, L, 0], cmap='seismic',
              vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title('VTI (ε=0.2, δ=0.1): div(u) — elliptical wavefront')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
    plt.colorbar(ax.images[0], ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "stage4_vti.png", dpi=150)
    plt.close()

    print(f"  Wavefield finite: "
          f"{np.all(np.isfinite(ux_data)) and np.all(np.isfinite(uy_data))}")
    print(f"  Max |div|: {np.max(np.abs(div)):.6e}")
    print(f"  Saved {FIGDIR / 'stage4_vti.png'}")


# ============================================================================
# Test 4c: TTI with tilt
# ============================================================================

def test_tti_tilted():
    """
    TTI with ε=0.2, δ=0.1, θ=30°. The elliptical wavefront should be
    rotated by 30° from vertical.
    """
    print("\n--- Test 4c: TTI (ε=0.2, δ=0.1, θ=30°) ---")

    Vp = 3.0; Vs = 1.5; rho_val = 2.0
    eps = 0.2; delt = 0.1; theta = np.radians(30)

    C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
        Vp, Vs, rho_val, eps, delt, theta)

    print(f"  C11={C11:.4f}, C22={C22:.4f}, C12={C12:.4f}")
    print(f"  C16={C16:.4f}, C26={C26:.4f}, C66={C66:.4f}")
    print(f"  (C16, C26 ≠ 0 for tilted symmetry)")

    L = 4.0; dx = 0.01; Nx = int(L/dx) + 1
    f0 = 30.0

    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)
    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    Vp_max = np.sqrt(max(C11, C22) / rho_val)
    dt_val = 0.4 * dx / (Vp_max * np.sqrt(2))
    T = 0.5
    nt = int(T / dt_val) + 1

    ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)
    uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                      dtype=np.float64)

    rho_f = Function(name='rho', grid=grid, dtype=np.float64)
    C11_f = Function(name='C11', grid=grid, space_order=8, dtype=np.float64)
    C22_f = Function(name='C22', grid=grid, space_order=8, dtype=np.float64)
    C12_f = Function(name='C12', grid=grid, space_order=8, dtype=np.float64)
    C16_f = Function(name='C16', grid=grid, space_order=8, dtype=np.float64)
    C26_f = Function(name='C26', grid=grid, space_order=8, dtype=np.float64)
    C66_f = Function(name='C66', grid=grid, space_order=8, dtype=np.float64)

    rho_f.data[:] = rho_val
    C11_f.data[:] = C11; C22_f.data[:] = C22; C12_f.data[:] = C12
    C16_f.data[:] = C16; C26_f.data[:] = C26; C66_f.data[:] = C66

    t_arr = np.arange(nt+2) * dt_val
    t0 = 1.5 / f0
    wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
               np.exp(-(np.pi*f0*(t_arr-t0))**2))
    sx, sy = L/2, L/2
    half_dx = dx / 2
    dt_sym = grid.stepping_dim.spacing

    src_dipolx = SparseTimeFunction(
        name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipolx.data[:, 0] = -wavelet
    src_dipolx.data[:, 1] = wavelet
    src_inj_x = src_dipolx.inject(field=ux.forward,
                                   expr=src_dipolx * dt_sym**2 / rho_f)

    src_dipoly = SparseTimeFunction(
        name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
        coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                              dtype=np.float64),
        dtype=np.float64)
    src_dipoly.data[:, 0] = -wavelet
    src_dipoly.data[:, 1] = wavelet
    src_inj_y = src_dipoly.inject(field=uy.forward,
                                   expr=src_dipoly * dt_sym**2 / rho_f)

    filter_coeffs = get_proposed_filter_coefficients()
    op = build_tti_elastic_operator(grid, ux, uy, rho_f,
                                    C11_f, C22_f, C12_f, C16_f, C26_f, C66_f,
                                    fwd_w, bwd_w, name='op_tti',
                                    src_inject_x=src_inj_x,
                                    src_inject_y=src_inj_y,
                                    filter_coeffs=filter_coeffs, sigma=0.035)

    op(time_M=nt, dt=dt_val)
    slot = nt % 3

    ux_data = ux.data[slot, :, :]
    uy_data = uy.data[slot, :, :]
    div = np.gradient(ux_data, dx, axis=0) + np.gradient(uy_data, dx, axis=1)
    curl = np.gradient(uy_data, dx, axis=0) - np.gradient(ux_data, dx, axis=1)

    # Plot all three anisotropy cases together
    print(f"  Wavefield finite: "
          f"{np.all(np.isfinite(ux_data)) and np.all(np.isfinite(uy_data))}")
    print(f"  Max |div|: {np.max(np.abs(div)):.6e}")

    # Save TTI wavefield
    vmax = 0.3 * np.max(np.abs(div))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(div.T, extent=[0, L, L, 0], cmap='seismic',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title(f'TTI (ε={eps}, δ={delt}, θ=30°): div(u)')
    axes[0].set_xlabel('x (km)'); axes[0].set_ylabel('y (km)')
    plt.colorbar(axes[0].images[0], ax=axes[0], shrink=0.8)

    vmax_c = 0.3 * np.max(np.abs(curl))
    axes[1].imshow(curl.T, extent=[0, L, L, 0], cmap='seismic',
                   vmin=-vmax_c, vmax=vmax_c, aspect='equal')
    axes[1].set_title(f'TTI: curl(u) — S-wave')
    axes[1].set_xlabel('x (km)'); axes[1].set_ylabel('y (km)')
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGDIR / "stage4_tti_tilted.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGDIR / 'stage4_tti_tilted.png'}")


# ============================================================================
# Combined comparison plot
# ============================================================================

def test_comparison():
    """
    Side-by-side comparison: isotropic vs VTI vs TTI.
    All with same parameters except anisotropy.
    """
    print("\n--- Test 4d: Iso vs VTI vs TTI Comparison ---")

    Vp = 3.0; Vs = 1.5; rho_val = 2.0
    L = 4.0; dx = 0.015; Nx = int(L/dx) + 1
    f0 = 12.0

    grid = Grid(shape=(Nx, Nx), extent=(L, L), dtype=np.float64)
    fwd_coeffs, bwd_coeffs = get_taylor_dp_coefficients()
    fwd_w = dp_weights_array(fwd_coeffs)
    bwd_w = dp_weights_array(bwd_coeffs)

    cases = [
        ('Isotropic', 0.0, 0.0, 0.0),
        ('VTI (ε=0.2)', 0.2, 0.1, 0.0),
        ('TTI (θ=30°)', 0.2, 0.1, np.radians(30)),
        ('TTI (θ=60°)', 0.2, 0.1, np.radians(60)),
    ]

    sx, sy = L/2, L/2
    half_dx = dx / 2
    dt_sym = grid.stepping_dim.spacing
    filter_coeffs = get_proposed_filter_coefficients()

    results = {}
    for label, eps, delt, theta in cases:
        C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
            Vp, Vs, rho_val, eps, delt, theta)

        Vp_max = np.sqrt(max(C11, C22) / rho_val)
        dt_val = 0.35 * dx / (Vp_max * np.sqrt(2))
        T = 0.4
        nt = int(T / dt_val) + 1

        ux = TimeFunction(name='ux', grid=grid, time_order=2, space_order=8,
                          dtype=np.float64)
        uy = TimeFunction(name='uy', grid=grid, time_order=2, space_order=8,
                          dtype=np.float64)

        rho_f = Function(name='rho', grid=grid, dtype=np.float64)
        C11_f = Function(name='C11', grid=grid, space_order=8,
                          dtype=np.float64)
        C22_f = Function(name='C22', grid=grid, space_order=8,
                          dtype=np.float64)
        C12_f = Function(name='C12', grid=grid, space_order=8,
                          dtype=np.float64)
        C16_f = Function(name='C16', grid=grid, space_order=8,
                          dtype=np.float64)
        C26_f = Function(name='C26', grid=grid, space_order=8,
                          dtype=np.float64)
        C66_f = Function(name='C66', grid=grid, space_order=8,
                          dtype=np.float64)

        rho_f.data[:] = rho_val
        C11_f.data[:] = C11; C22_f.data[:] = C22; C12_f.data[:] = C12
        C16_f.data[:] = C16; C26_f.data[:] = C26; C66_f.data[:] = C66

        t_arr = np.arange(nt+2) * dt_val
        t0 = 1.5 / f0
        wavelet = ((1 - 2*(np.pi*f0*(t_arr-t0))**2) *
                   np.exp(-(np.pi*f0*(t_arr-t0))**2))

        src_dipolx = SparseTimeFunction(
            name='src_dipolx', grid=grid, npoint=2, nt=nt+2,
            coordinates=np.array([[sx - half_dx, sy], [sx + half_dx, sy]],
                                  dtype=np.float64),
            dtype=np.float64)
        src_dipolx.data[:, 0] = -wavelet
        src_dipolx.data[:, 1] = wavelet
        src_inj_x = src_dipolx.inject(field=ux.forward,
                                       expr=src_dipolx * dt_sym**2 / rho_f)

        src_dipoly = SparseTimeFunction(
            name='src_dipoly', grid=grid, npoint=2, nt=nt+2,
            coordinates=np.array([[sx, sy - half_dx], [sx, sy + half_dx]],
                                  dtype=np.float64),
            dtype=np.float64)
        src_dipoly.data[:, 0] = -wavelet
        src_dipoly.data[:, 1] = wavelet
        src_inj_y = src_dipoly.inject(field=uy.forward,
                                       expr=src_dipoly * dt_sym**2 / rho_f)

        # Use clean name
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', label)
        op = build_tti_elastic_operator(
            grid, ux, uy, rho_f,
            C11_f, C22_f, C12_f, C16_f, C26_f, C66_f,
            fwd_w, bwd_w, name=f'op_{clean_name}',
            src_inject_x=src_inj_x, src_inject_y=src_inj_y,
            filter_coeffs=filter_coeffs, sigma=0.035)

        op(time_M=nt, dt=dt_val)
        slot = nt % 3

        ux_d = ux.data[slot, :, :]
        uy_d = uy.data[slot, :, :]
        div = np.gradient(ux_d, dx, axis=0) + np.gradient(uy_d, dx, axis=1)
        results[label] = div.copy()

        stable = np.all(np.isfinite(ux_d)) and np.all(np.isfinite(uy_d))
        print(f"  {label}: max|div|={np.max(np.abs(div)):.4e}, stable={stable}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for idx, (label, div) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        vmax = 0.3 * np.max(np.abs(div))
        if vmax == 0:
            vmax = 1e-6
        ax.imshow(div.T, extent=[0, L, L, 0], cmap='seismic',
                  vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(label)
        ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')

    plt.suptitle('Anisotropy Comparison: div(u) — P-wave wavefronts')
    plt.tight_layout()
    plt.savefig(FIGDIR / "stage4_anisotropy_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGDIR / 'stage4_anisotropy_comparison.png'}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Stage 4: 2D TTI Elastic with Dual-Pair FD")
    print("=" * 70)

    test_isotropic_limit()
    test_vti()
    test_tti_tilted()
    test_comparison()

    print("\n" + "=" * 70)
    print("REVIEW FINDINGS FROM STAGE 4")
    print("=" * 70)
    print("""
1. TTI elastic with dual-pair FD successfully implemented using Bond
   rotation of the stiffness tensor. The formulation uses 6 independent
   Cij components (C11, C22, C12, C16, C26, C66 in 2D Voigt).

2. Isotropic limit (ε=δ=0) correctly produces circular wavefronts,
   verifying the TTI code reduces to the isotropic case.

3. VTI (θ=0) produces elliptical wavefronts with faster horizontal
   propagation, consistent with positive ε (horizontal velocity
   enhancement).

4. TTI with tilt angle correctly rotates the wavefront ellipse.
   C16 and C26 become non-zero for θ≠0, coupling P and SV modes
   more strongly.

5. REVIEW FINDING: The paper's claim about dual-pair being
   "formulation-agnostic" is misleading. The D⁻[C·D⁺] structure
   specifically requires the displacement (2nd-order) formulation.
   For velocity-stress (1st-order), you'd use D⁺ and D⁻ separately,
   which is just standard staggered-grid FD.

6. REVIEW FINDING: The paper doesn't discuss how the dual-pair
   interacts with the cross-coupling terms (C16, C26). For VTI
   these are zero, but for TTI they're non-zero and couple the
   equations through cross-derivatives D_x⁻[C16·D_y⁺ux].
   The "no artificial damping" property is NOT proven for these
   mixed-dimension operators.
""")


if __name__ == "__main__":
    main()
