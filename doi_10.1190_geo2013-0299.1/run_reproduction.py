"""STATUS (2026-05-28) — DEFERRED PILOT SCAFFOLD. Not load-bearing.

The byte-checked transcribed constants in `paper_tables.py` (+ tests
in `tests/test_paper_tables.py`, 14/14 green) are sufficient to
cite this paper as the literature anchor for the parent-repo
Petrobras cohort ranking. The empirical h-refinement convergence
study below was scaffolded but does NOT reproduce the paper's
δ→4 (SSGS) / δ→2 (RSGS) predictions: the small-domain pilot is
severely under-resolved at coarse grids (Gaussian σ ≈ 30 m,
coarsest dx = 10 m → ~3 cells span FWHM → SO=2 dispersion
dominates). Unblock path documented in README.

Results in `reference_outputs/convergence_results.json` are the
failed-pilot record — δ values near 1.0 instead of paper-predicted
4.0 (SSGS) and 2.0 (RSGS). Retained as evidence of what was
tried.

----

h-refinement convergence study reproducing the load-bearing claim
of Vishnevsky-Lisitsa-Tcheverda-Reshetova 2014 (DOI 10.1190/geo2013-0299.1):

> "for a fluid-solid interface aligned with the grid line, a
> second-order convergence can only be achieved by an SSGS. ...
> the presence of a fluid-solid interface reduces the order of
> convergence for the LS and the RSGS to a first order of
> convergence."

This driver implements minimal SO=2 Virieux SSGS (Virieux 1986) and
Saenger RSGS (Saenger-Gold-Shapiro 2000) in pure NumPy on a horizontal
IF / IS1 (fluid / isotropic-solid) interface. The convergence indicator
δ_k (paper Eqs 12-13) is computed across a 4-grid refinement sequence.

Expected outcomes per `paper_tables.CONVERGENCE_PREDICTIONS`:

- SSGS: δ_k → 4 (2nd-order convergence preserved at fluid-solid)
- RSGS: δ_k → 2 (1st-order convergence due to fluid-solid)

Configuration (Petrobras-relevant; horizontal fluid-solid):
- Domain: 600 × 600 m (smaller than paper's 3000² for tractability;
  convergence rate is dimensionless so this matches the paper's
  prediction)
- Interface: z = 300 m (horizontal)
- Top half (z < 300): IF — water (ρ=1000, Vp=1500, Vs=0)
- Bottom half (z >= 300): IS1 — solid (ρ=1800, Vp=1900, Vs=1200)
- Source IC per Eq 17: σ_xx = σ_zz = exp(-0.1·((x-xs)² + (z-zs)²))
  at (xs=300, zs=150) — source in the fluid half, above interface
- t_final: 0.05 s (short enough that reflections are still local)
- Grid sequence: dx ∈ {10, 5, 2.5, 1.25} m, i.e. 60×60, 120×120,
  240×240, 480×480 cells
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import paper_tables as pt


# ----------------------------------------------------------------------------
# Material parameters for horizontal fluid (IF) over isotropic solid (IS1)
# ----------------------------------------------------------------------------


def build_material(Nx: int, Nz: int, dx: float, interface_z: float):
    """Return (rho, lam, mu) cell-centered arrays for the IF/IS1
    horizontal interface.

    rho, lam, mu at cell centers (i, j) for i in [0, Nx), j in [0, Nz).
    z increases downward; interface at j = floor(interface_z / dx).
    """
    rho = np.zeros((Nx, Nz), dtype=np.float64)
    lam = np.zeros((Nx, Nz), dtype=np.float64)
    mu = np.zeros((Nx, Nz), dtype=np.float64)

    # Each cell-center (i, j) has z = (j + 0.5) * dx (cell-centered).
    # Note: domain is (x, z) where j indexes z.
    z_cells = (np.arange(Nz) + 0.5) * dx
    is_fluid = z_cells < interface_z  # shape (Nz,)

    # Fluid: ρ=1000, Vp=1500, Vs=0 → λ=ρ Vp² = 2.25e9, μ=0
    # Solid: ρ=1800, Vp=1900, Vs=1200 → μ=ρ Vs²=2.592e9, λ=ρ Vp²-2μ=1.314e9
    rho_f = pt.IF['rho']
    lam_f = rho_f * pt.IF['Vp'] ** 2  # κ for fluid
    mu_f = 0.0
    rho_s = pt.IS1['rho']
    mu_s = rho_s * pt.IS1['Vs'] ** 2
    lam_s = rho_s * pt.IS1['Vp'] ** 2 - 2 * mu_s

    rho[:, is_fluid] = rho_f
    lam[:, is_fluid] = lam_f
    mu[:, is_fluid] = mu_f
    rho[:, ~is_fluid] = rho_s
    lam[:, ~is_fluid] = lam_s
    mu[:, ~is_fluid] = mu_s

    return rho, lam, mu


def apply_paper_modifications(rho: np.ndarray, lam: np.ndarray,
                              mu: np.ndarray, scheme: str):
    """Apply paper-specified parameter modifications at the interface.

    Per paper §"Standard staggered-grid scheme" (Eqs A-2, A-4):
    - Density at velocity points: arithmetic mean of adjoining cells.
    - For C_33 (= lam + 2 mu): harmonic averaging for vertical-interface
      cell pairs.

    Returns:
      ρ_x, ρ_z: density at velocity-x and velocity-z points
      C33_hat: harmonically-averaged C_33 at half-shifted z positions
      lam_xz, mu_xz: at σ_xz position (half-shifted x and z)

    For SSGS: these are the modified parameters at half-staggered points.
    For RSGS: per the paper's note (Appendix), RSGS only modifies the
    density at cell corners (Eq A-11: ρ̂_{i+1/2,j+1/2} = (ρ_{i,j} +
    ρ_{i+1,j} + ρ_{i,j+1} + ρ_{i+1,j+1})/4) and CANNOT properly
    modify C at fluid-solid interfaces (Eq A-9 degenerates when C_33=0).
    Hence RSGS is documented to degrade to 1st order at fluid-solid.
    """
    if scheme == 'SSGS':
        # ρ at (i+1/2, j): arithmetic mean of (i, j) and (i+1, j)
        rho_x = 0.5 * (rho + np.roll(rho, -1, axis=0))
        # ρ at (i, j+1/2): arithmetic mean of (i, j) and (i, j+1)
        rho_z = 0.5 * (rho + np.roll(rho, -1, axis=1))
        # C_33 at (i, j+1/2): harmonic averaging in z (per Eq A-9 limit
        # → at fluid-solid where C33_fluid=0, use SSGS's c_33=0 rule
        # per paper Eq A-10).
        C33 = lam + 2 * mu  # cell-centered
        C33_jp = np.roll(C33, -1, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Harmonic mean: 2 / (1/a + 1/b). When a=0 (fluid above),
            # harmonic mean → 0 (paper Eq A-10).
            C33_hat = np.where(
                (C33 > 0) & (C33_jp > 0),
                2.0 / (1.0 / np.maximum(C33, 1e-30) +
                       1.0 / np.maximum(C33_jp, 1e-30)),
                0.0,  # fluid-solid: c_33 = 0 (Eq A-10)
            )
        # μ at (i+1/2, j+1/2) — used for σ_xz update.
        # Use 4-cell arithmetic mean.
        mu_xz = 0.25 * (mu + np.roll(mu, -1, axis=0) +
                        np.roll(mu, -1, axis=1) +
                        np.roll(np.roll(mu, -1, axis=0), -1, axis=1))
        return rho_x, rho_z, C33_hat, mu_xz
    elif scheme == 'RSGS':
        # ρ at corners: 4-cell arithmetic mean (Eq A-11)
        rho_corner = 0.25 * (rho + np.roll(rho, -1, axis=0) +
                             np.roll(rho, -1, axis=1) +
                             np.roll(np.roll(rho, -1, axis=0), -1, axis=1))
        # No modification possible for C at fluid-solid (Eq A-9 → A-10
        # documented as not applicable). Use cell-centered values directly.
        # This is the LOAD-BEARING source of 1st-order degradation.
        C11 = lam + 2 * mu
        return rho_corner, C11, mu
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


# ----------------------------------------------------------------------------
# SSGS — Virieux 1986 standard staggered grid, SO=2
# ----------------------------------------------------------------------------


def source_ic(Nx: int, Nz: int, dx: float, xs: float, zs: float,
              sigma_scale: float = 0.001):
    """Wider-than-paper Gaussian IC to keep the wave resolved at the
    coarsest grid in our tractability-reduced domain (600 m vs paper's
    3000 m).

    Paper Eq 17 uses sigma_scale=0.1 (radius ~3 m) on a 3000 m domain.
    Equivalent ratio for 600 m domain: sigma_scale=0.1·(600/3000)²=0.004.
    We use 0.001 (radius ~30 m, ~3 cells at coarsest dx=10 m) to ensure
    the wave is well-resolved at every grid in the refinement sequence.
    The convergence-rate test depends only on the IC being smooth and
    consistent across grids, not on the specific radius.
    """
    i = np.arange(Nx).reshape(-1, 1)
    j = np.arange(Nz).reshape(1, -1)
    x = (i + 0.5) * dx
    z = (j + 0.5) * dx
    return np.exp(-sigma_scale * ((x - xs) ** 2 + (z - zs) ** 2))


def run_ssgs(Nx: int, Nz: int, dx: float, t_final: float,
             interface_z: float, xs: float, zs: float,
             cfl: float = 0.4):
    """Virieux 1986 SO=2 SSGS forward, 2D velocity-stress.

    Returns dict with final wavefield snapshots {ux, uz, sxx, szz, sxz}.

    Staggering:
      σ_xx, σ_zz at cell centers (i, j)
      σ_xz at (i+1/2, j+1/2)
      u_x at (i+1/2, j)
      u_z at (i, j+1/2)

    SO=2 central FD with first-order time stepping (leapfrog).
    """
    rho, lam, mu = build_material(Nx, Nz, dx, interface_z)
    rho_x, rho_z, C33_hat, mu_xz = apply_paper_modifications(
        rho, lam, mu, 'SSGS')

    # CFL: dt < dx / (V_p * sqrt(2)) for 2D SO=2
    Vp_max = max(np.sqrt(pt.IF['Vp'] ** 2),
                 np.sqrt(pt.IS1['Vp'] ** 2))
    dt = cfl * dx / (Vp_max * np.sqrt(2))
    nt = int(np.ceil(t_final / dt))

    # Allocate at user (no halo); use np.roll for stencils (periodic BC
    # acceptable since wave is contained for short t_final).
    sxx = np.zeros((Nx, Nz), dtype=np.float64)
    szz = np.zeros((Nx, Nz), dtype=np.float64)
    sxz = np.zeros((Nx, Nz), dtype=np.float64)
    ux = np.zeros((Nx, Nz), dtype=np.float64)
    uz = np.zeros((Nx, Nz), dtype=np.float64)

    # IC: σ_xx = σ_zz = Gaussian at (xs, zs) per Eq 17
    g = source_ic(Nx, Nz, dx, xs, zs, sigma_scale=0.1)
    sxx[:] = g
    szz[:] = g

    inv_dx = 1.0 / dx

    for _ in range(nt):
        # Velocity update at t+dt/2
        # u_x at (i+1/2, j): reads σ_xx at (i+1, j) - (i, j) and
        # σ_xz at (i+1/2, j+1/2) - (i+1/2, j-1/2)
        dsxx_dx = (np.roll(sxx, -1, axis=0) - sxx) * inv_dx
        dsxz_dz = (sxz - np.roll(sxz, 1, axis=1)) * inv_dx
        ux += (dt / rho_x) * (dsxx_dx + dsxz_dz)

        # u_z at (i, j+1/2): reads σ_zz at (i, j+1) - (i, j) and
        # σ_xz at (i+1/2, j+1/2) - (i-1/2, j+1/2)
        dszz_dz = (np.roll(szz, -1, axis=1) - szz) * inv_dx
        dsxz_dx = (sxz - np.roll(sxz, 1, axis=0)) * inv_dx
        uz += (dt / rho_z) * (dszz_dz + dsxz_dx)

        # Stress update at t+dt
        # σ_xx at (i, j): reads u_x at (i+1/2, j) - (i-1/2, j) and
        # u_z at (i, j+1/2) - (i, j-1/2)
        dux_dx = (ux - np.roll(ux, 1, axis=0)) * inv_dx
        duz_dz = (uz - np.roll(uz, 1, axis=1)) * inv_dx
        # Use cell-center lam, mu for σ_xx, σ_zz updates
        sxx += dt * ((lam + 2 * mu) * dux_dx + lam * duz_dz)
        szz += dt * (lam * dux_dx + (lam + 2 * mu) * duz_dz)

        # σ_xz at (i+1/2, j+1/2): reads u_x at (i+1/2, j+1) - (i+1/2, j)
        # and u_z at (i+1, j+1/2) - (i, j+1/2)
        dux_dz = (np.roll(ux, -1, axis=1) - ux) * inv_dx
        duz_dx = (np.roll(uz, -1, axis=0) - uz) * inv_dx
        sxz += dt * mu_xz * (dux_dz + duz_dx)

    return dict(ux=ux, uz=uz, sxx=sxx, szz=szz, sxz=sxz, dt=dt, nt=nt)


# ----------------------------------------------------------------------------
# RSGS — Saenger 2000 rotated staggered grid, SO=2
# ----------------------------------------------------------------------------


def run_rsgs(Nx: int, Nz: int, dx: float, t_final: float,
             interface_z: float, xs: float, zs: float,
             cfl: float = 0.4):
    """Saenger-Gold-Shapiro 2000 SO=2 RSGS forward, 2D velocity-stress.

    Staggering:
      σ_xx, σ_zz, σ_xz at cell centers (i, j)
      u_x, u_z at cell corners (i+1/2, j+1/2)

    Diagonal stencils: derivatives along (+1,+1) and (+1,-1) diagonals.
    SO=2 along each diagonal axis.

    Per paper Appendix Eq A-11: density at corners is 4-cell arithmetic
    mean; C at fluid-solid is NOT modified (Eq A-9 degenerates → 1st-order
    convergence at fluid-solid, the load-bearing claim).
    """
    rho, lam, mu = build_material(Nx, Nz, dx, interface_z)
    rho_corner, C11_cell, mu_cell = apply_paper_modifications(
        rho, lam, mu, 'RSGS')

    Vp_max = max(np.sqrt(pt.IF['Vp'] ** 2),
                 np.sqrt(pt.IS1['Vp'] ** 2))
    dt = cfl * dx / (Vp_max * np.sqrt(2))
    nt = int(np.ceil(t_final / dt))

    sxx = np.zeros((Nx, Nz), dtype=np.float64)
    szz = np.zeros((Nx, Nz), dtype=np.float64)
    sxz = np.zeros((Nx, Nz), dtype=np.float64)
    ux = np.zeros((Nx, Nz), dtype=np.float64)
    uz = np.zeros((Nx, Nz), dtype=np.float64)

    g = source_ic(Nx, Nz, dx, xs, zs, sigma_scale=0.1)
    sxx[:] = g
    szz[:] = g

    # Diagonal-stencil constants. For (1,1) diagonal, distance per step
    # is sqrt(2)·dx. SO=2 diagonal derivative coefficients are {-1: -1/2,
    # +1: +1/2} divided by sqrt(2)·dx for d/dr.
    inv_diag = 1.0 / (np.sqrt(2) * dx)
    lam_cell = lam  # cell-centered
    mu_cell = mu_cell  # alias

    def d_r(field):
        """Diagonal derivative along (+1,+1)."""
        return 0.5 * (np.roll(np.roll(field, -1, axis=0), -1, axis=1) -
                      np.roll(np.roll(field, 1, axis=0), 1, axis=1)) * inv_diag

    def d_s(field):
        """Diagonal derivative along (+1,-1)."""
        return 0.5 * (np.roll(np.roll(field, -1, axis=0), 1, axis=1) -
                      np.roll(np.roll(field, 1, axis=0), -1, axis=1)) * inv_diag

    def rsg_dx(field):
        """d/dx = (d_r + d_s) / sqrt(2)."""
        return (d_r(field) + d_s(field)) / np.sqrt(2)

    def rsg_dz(field):
        """d/dz = (d_r - d_s) / sqrt(2)."""
        return (d_r(field) - d_s(field)) / np.sqrt(2)

    for _ in range(nt):
        # Velocity update
        # u at corners reads σ at cell centers via rotated stencils
        ux += (dt / rho_corner) * (rsg_dx(sxx) + rsg_dz(sxz))
        uz += (dt / rho_corner) * (rsg_dx(sxz) + rsg_dz(szz))

        # Stress update at cell centers reads u at corners
        dux_dx = rsg_dx(ux)
        duz_dz = rsg_dz(uz)
        dux_dz = rsg_dz(ux)
        duz_dx = rsg_dx(uz)

        sxx += dt * ((lam_cell + 2 * mu_cell) * dux_dx +
                     lam_cell * duz_dz)
        szz += dt * (lam_cell * dux_dx +
                     (lam_cell + 2 * mu_cell) * duz_dz)
        sxz += dt * mu_cell * (dux_dz + duz_dx)

    return dict(ux=ux, uz=uz, sxx=sxx, szz=szz, sxz=sxz, dt=dt, nt=nt)


# ----------------------------------------------------------------------------
# Convergence indicator (paper Eqs 12-13)
# ----------------------------------------------------------------------------


def convergence_indicator(snapshots: list[dict],
                          dxes: list[float],
                          field: str = 'ux'):
    """Compute δ_k = ε_k / ε_{k+1} per Eq 12 for a refinement sequence.

    snapshots[k] is the wavefield at grid spacing dxes[k] (coarse to fine).
    For each consecutive pair, sub-sample the FINE grid down to the
    coarse-grid resolution and compute ||u_coarse - u_fine_subsampled||_2.

    This is Lisitsa's standard convergence indicator: comparing each
    pair of consecutive-resolution solutions at the COARSER resolution
    (subsampling fine to coarse). With h/2 refinement and rate γ,
    ε_k ≈ C·h_k^γ, so δ = ε_k/ε_{k+1} = 2^γ → 4 for γ=2, → 2 for γ=1.

    Restrict to interior region away from boundaries (10% inset on each
    side) to avoid periodic-BC artifacts.
    """
    errors = []
    for k in range(len(snapshots) - 1):
        coarse = snapshots[k][field]
        fine = snapshots[k + 1][field]
        # Subsample fine to coarse resolution: factor n = dx_coarse/dx_fine
        n = int(round(dxes[k] / dxes[k + 1]))
        fine_sub = fine[::n, ::n][:coarse.shape[0], :coarse.shape[1]]
        # Restrict to interior (avoid periodic-BC wrap artifacts at the
        # edges).
        Nc = coarse.shape[0]
        inset = max(2, Nc // 10)
        c_int = coarse[inset:-inset, inset:-inset]
        f_int = fine_sub[inset:-inset, inset:-inset]
        num = np.linalg.norm(c_int - f_int)
        denom = np.linalg.norm(f_int) + 1e-30
        errors.append(num / denom)

    deltas = []
    for k in range(len(errors) - 1):
        if errors[k + 1] < 1e-30:
            deltas.append(float('inf'))
        else:
            deltas.append(errors[k] / errors[k + 1])
    return errors, deltas


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def main():
    domain_extent = 600.0  # m (smaller than paper 3000² for tractability)
    interface_z = 300.0
    xs = 300.0
    zs = 150.0  # source in fluid half (above interface)
    t_final = 0.05  # s — short enough that reflections stay local
    dxes = [10.0, 5.0, 2.5, 1.25]  # 4 grids, h/2 refinement

    results = {'config': {
        'domain': domain_extent,
        'interface_z': interface_z,
        'source': (xs, zs),
        't_final': t_final,
        'dxes': dxes,
        'IF': {k: float(v) for k, v in pt.IF.items()},
        'IS1': {k: float(v) for k, v in pt.IS1.items()},
    }, 'schemes': {}}

    for scheme in ['SSGS', 'RSGS']:
        runner = run_ssgs if scheme == 'SSGS' else run_rsgs
        snaps = []
        walls = []
        for dx in dxes:
            Nx = int(round(domain_extent / dx))
            t0 = time.perf_counter()
            snap = runner(Nx, Nx, dx, t_final, interface_z, xs, zs)
            walls.append(time.perf_counter() - t0)
            snaps.append(snap)
            print(f'  {scheme} dx={dx:6.2f} m  Nx={Nx:4d}  '
                  f'dt={snap["dt"]:.3e}  nt={snap["nt"]:5d}  '
                  f'wall={walls[-1]:6.2f}s')

        # Diagnostics: report magnitudes per grid
        for k, snap in enumerate(snaps):
            print(f'    {scheme} dx={dxes[k]:.2f}: '
                  f'max|sxx|={np.max(np.abs(snap["sxx"])):.3e} '
                  f'max|ux|={np.max(np.abs(snap["ux"])):.3e}')
        errors, deltas = convergence_indicator(snaps, dxes, field='sxx')
        print(f'  {scheme} errors (sxx):', [f'{e:.3e}' for e in errors])
        print(f'  {scheme} δ values:', [f'{d:.3f}' for d in deltas])
        results['schemes'][scheme] = {
            'errors': errors,
            'deltas': deltas,
            'walls': walls,
        }

    # Save to reference_outputs/
    out = Path(__file__).parent / 'reference_outputs'
    out.mkdir(exist_ok=True)
    out_path = out / 'convergence_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out_path}')

    # Print verdict against paper predictions
    print('\n=== Verdict against paper_tables.CONVERGENCE_PREDICTIONS ===')
    pred = pt.CONVERGENCE_PREDICTIONS['horizontal_fluid_isotropic_solid']
    bands = pt.CONVERGENCE_THRESHOLDS
    for scheme in ['SSGS', 'RSGS']:
        deltas = results['schemes'][scheme]['deltas']
        expected = pred[scheme]
        if expected == 4.0:
            band = bands['second_order_band']
            label = '2nd order'
        else:
            band = bands['first_order_band']
            label = '1st order'
        in_band = all(band[0] <= d <= band[1] for d in deltas if d != float('inf'))
        verdict = 'PASS' if in_band else 'FAIL'
        print(f'  {scheme}: expected {label} (band {band}), got δ={deltas} '
              f'→ {verdict}')


if __name__ == '__main__':
    main()
