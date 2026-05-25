"""Devito reproduction of Virieux (1986) — P-SV velocity-stress
staggered-grid 2nd-order finite-difference scheme.

Paper
-----
Virieux, J. (1986). "P-SV wave propagation in heterogeneous media;
velocity-stress finite-difference method." *Geophysics* 51(4),
889-901. DOI 10.1190/1.1442147.

Scheme
------
2D in-plane (P-SV) elastic wave propagation written as a first-
order velocity-stress system on a staggered Yee-style grid:

    ∂t vx = (1/ρ) (∂x σxx + ∂y σxy)
    ∂t vy = (1/ρ) (∂x σxy + ∂y σyy)
    ∂t σxx = (λ + 2μ) ∂x vx + λ ∂y vy
    ∂t σyy = λ ∂x vx + (λ + 2μ) ∂y vy
    ∂t σxy = μ (∂y vx + ∂x vy)

Field placement (Yee staggering — Virieux Fig 1):

    vx     at (i+½, j)
    vy     at (i,   j+½)
    σxx    at (i,   j)
    σyy    at (i,   j)
    σxy    at (i+½, j+½)

This script parametrises the spatial order over
``space_order ∈ {2, 4, 8, 16}`` per the
`devitocodespro/devito-fd-survey` repo-wide convention. The
canonical Virieux 1986 claim is at ``space_order=2``; higher
orders are natural Taylor-truncation extensions of the same
scheme (formally specified by Levander 1988 at order 4 — see
the companion folder ``doi_10.1190_1.1442422/``).

Output
------
For each space_order value, a 2D snapshot of vx + vy + sxx + syy
+ sxy at a fixed end-time is saved to ``reference_outputs/
wavefield_so<N>.npz``. These are the pinned reference outputs
that the parent `devitocodespro/devito-fd-survey` Method 1 SSG
must reproduce within tolerance (per the reproduction-as-
prerequisite gate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from devito import (
    Eq, Function, Grid, Operator, SparseTimeFunction,
    TensorTimeFunction, VectorTimeFunction,
    div, grad, diag, solve,
)


# Canonical reproduction configuration.
# All quantities in km / km/s / g·cm⁻³ per Devito unit-scaling guidance
# (keeps coefficients O(1) for FP32 safety).
DOMAIN_EXTENT_KM = (4.0, 4.0)        # 4 km × 4 km
NX = NY = 401                        # 10 m grid spacing
SOURCE_FREQ_HZ = 20.0                # Ricker f0
T_FINAL_S = 0.4                      # waves at ~1.2 km at this t,
                                     # still interior of 2 km half-extent
T_SNAPSHOT_S = T_FINAL_S             # snapshot at end of simulation

# Homogeneous isotropic material — chosen to be representative of
# upper-crust crystalline rock per Virieux 1986 Fig 4.
VP_KMS = 3.0     # km/s
VS_KMS = 1.7     # km/s
RHO_GCC = 2.2    # g/cm³

# Lamé parameters in repo units (kbar = g/cm³ · (km/s)² = 0.1 GPa).
RHO = RHO_GCC
MU = RHO * VS_KMS**2                # 2.2 · 2.89 = 6.358
LAM = RHO * VP_KMS**2 - 2.0 * MU    # 2.2 · 9 - 2·6.358 = 19.8 - 12.716 = 7.084
LAM_PLUS_2MU = LAM + 2.0 * MU       # 19.8 — i.e. ρ Vp²

# CFL coefficients per spatial order, 2D staggered grid. Per-order
# bounds derived from the modified-wavenumber spectrum of centred-FD
# at 2N-order × 2nd-order leapfrog (Saenger-Bohlen 2004 Table 2 +
# extrapolation for SO=16). Each value is dt_max = COEFF × dx / Vp.
# A 90 % safety margin is applied on top.
CFL_BY_ORDER: dict[int, float] = {
    2:  0.606,
    4:  0.495,
    8:  0.380,
    16: 0.280,
}
CFL_SAFETY = 0.90


def ricker(t_axis: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """Ricker wavelet centred at ``t0`` with central frequency ``f0``."""
    arg = (np.pi * f0 * (t_axis - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def build_grid(space_order: int) -> Grid:
    """Build a 2D Devito Grid for the reproduction. ``space_order``
    is recorded for stencil construction but the Grid itself is
    independent of it."""
    return Grid(
        shape=(NX, NY),
        extent=DOMAIN_EXTENT_KM,
        dtype=np.float64,
    )


def cfl_dt(grid: Grid, space_order: int) -> float:
    """CFL-stable dt for the chosen ``space_order``. Uses the
    per-order CFL coefficient from Saenger-Bohlen 2004 Table 2."""
    dx = grid.extent[0] / (grid.shape[0] - 1)
    if space_order not in CFL_BY_ORDER:
        coeff = min(CFL_BY_ORDER.values())
    else:
        coeff = CFL_BY_ORDER[space_order]
    return CFL_SAFETY * coeff * dx / VP_KMS


def build_operator(grid: Grid, space_order: int):
    """Assemble the 2D P-SV velocity-stress Devito Operator at the
    given ``space_order``. Uses Devito's first-class
    ``VectorTimeFunction`` and ``TensorTimeFunction`` so staggering
    and cross-stagger derivatives are handled by the framework
    (matches the tested pattern in
    ``devito/examples/seismic/elastic/operators.py``).

    The hand-written Yee version (commit history of this folder)
    was unstable at SO ≥ 4; the `_eval_at` machinery in the
    tensor-typed path correctly handles the half-grid offsets at
    every supported order.
    """
    dt = grid.stepping_dim.spacing

    v = VectorTimeFunction(
        name='v', grid=grid, time_order=1, space_order=space_order,
    )
    tau = TensorTimeFunction(
        name='tau', grid=grid, time_order=1, space_order=space_order,
    )

    # Particle velocity: ρ ∂_t v = ∇·τ
    eq_v = v.dt - div(tau) / RHO
    # Stress: ∂_t τ = λ (∇·v) I + μ (∇v + (∇v)^T)
    e = grad(v.forward) + grad(v.forward).transpose(inner=False)
    eq_tau = tau.dt - LAM * diag(div(v.forward)) - MU * e

    u_v = Eq(v.forward, solve(eq_v, v.forward))
    u_t = Eq(tau.forward, solve(eq_tau, tau.forward))

    fields = {'v': v, 'tau': tau}
    return [u_v, u_t], fields


def build_source_term(grid: Grid, tau, dt_value: float, nt: int):
    """Ricker explosive (isotropic) source injected into the diagonal
    stress components — Virieux 1986 §III.A. Domain-centre location."""
    src_coords = np.array([[DOMAIN_EXTENT_KM[0] / 2.0,
                            DOMAIN_EXTENT_KM[1] / 2.0]], dtype=np.float64)
    src = SparseTimeFunction(
        name='src', grid=grid, npoint=1, nt=nt,
        coordinates=src_coords,
    )
    t_axis = np.arange(nt) * dt_value
    t0 = 1.5 / SOURCE_FREQ_HZ
    src.data[:, 0] = ricker(t_axis, SOURCE_FREQ_HZ, t0)
    # Inject into both diagonal stresses (isotropic explosion).
    s = grid.stepping_dim.spacing
    src_term = src.inject(field=tau.forward.diagonal(), expr=src * s)
    return src, [src_term]


def run_reproduction(space_order: int, save_npz: bool = True,
                     output_dir: Path | None = None) -> dict:
    """Run the Virieux 1986 reproduction at the given ``space_order``.

    Returns a dict of final-step wavefield snapshots."""
    grid = build_grid(space_order)
    dt_value = cfl_dt(grid, space_order)
    nt = int(np.ceil(T_FINAL_S / dt_value)) + 1

    eqs, fields = build_operator(grid, space_order)
    src, src_terms = build_source_term(grid, fields['tau'], dt_value, nt)

    op = Operator(eqs + src_terms, name=f'VirieuxSO{space_order}')
    op.apply(time_M=nt - 2, dt=dt_value)

    # Read the most recent step from each component of v and tau.
    last = (nt - 1) % 2
    v, tau = fields['v'], fields['tau']
    snapshot = {
        'vx':  v[0].data[last].copy(),
        'vy':  v[1].data[last].copy(),
        'sxx': tau[0, 0].data[last].copy(),
        'syy': tau[1, 1].data[last].copy(),
        'sxy': tau[0, 1].data[last].copy(),
    }
    snapshot.update({
        'dt': dt_value, 'dx': grid.extent[0] / (grid.shape[0] - 1),
        'nt': nt, 'space_order': space_order,
        'vp_kms': VP_KMS, 'vs_kms': VS_KMS, 'rho_gcc': RHO_GCC,
        'f0_hz': SOURCE_FREQ_HZ, 't_final_s': T_FINAL_S,
    })

    if save_npz:
        output_dir = output_dir or (Path(__file__).resolve().parent
                                     / 'reference_outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f'wavefield_so{space_order}.npz'
        np.savez_compressed(out_path, **snapshot)
        print(f'  wrote {out_path} '
              f'(vx range [{snapshot["vx"].min():.3e}, '
              f'{snapshot["vx"].max():.3e}])')

    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--space-order', '-s', type=int, nargs='+',
        default=[2, 4, 8, 16],
        help='Space order(s) to run (default: 2 4 8 16 per repo convention)',
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not write reference_outputs/*.npz',
    )
    args = parser.parse_args()

    print(f'Virieux 1986 reproduction — running '
          f'space_order ∈ {args.space_order}')
    for so in args.space_order:
        print(f'  space_order={so} '
              f'(canonical={so == 2}, dt={cfl_dt(build_grid(so), so):.4e} s)')
        run_reproduction(so, save_npz=not args.no_save)


if __name__ == '__main__':
    main()
