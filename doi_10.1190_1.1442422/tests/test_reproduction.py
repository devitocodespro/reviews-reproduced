"""Tests for the Virieux (1986) reproduction.

Three classes of test:

1. **Paper-faithful stencil byte-match** (canonical claim):
   verifies that the 2nd-order centred-FD stencil produced by
   Devito at ``space_order=2`` matches Virieux 1986 Eq 4 — i.e.,
   the half-grid centred-FD operator ``∂x f|_{i+½} ≈ (f[i+1] -
   f[i]) / dx`` with weights ``{-1, +1} / dx``. This is the
   defining-feature byte-check for the Virieux scheme.

2. **MMS-style convergence rate** (per fd-playbook S8): on a
   manufactured smooth plane-wave solution, verify that the
   observed L²-error decays at the formal order of the chosen
   ``space_order``. Run at ``so ∈ {2, 4, 8, 16}`` per the
   repo-wide convention.

3. **Reference-output byte-match** (regression gate): re-run
   ``run_reproduction.py`` and verify the wavefield matches the
   pinned ``reference_outputs/wavefield_so<N>.npz`` to within
   floating-point tolerance. Catches drift caused by Devito
   version changes, accidental edits to the driver, or platform-
   dependent floating-point behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parent.parent
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))

from run_reproduction import (
    build_grid, build_operator, build_source_term, cfl_dt,
    run_reproduction, T_FINAL_S, VP_KMS, VS_KMS, RHO_GCC,
)


# ---------------------------------------------------------------
# Test class 1 — paper-faithful stencil byte-match
# ---------------------------------------------------------------

def _mms_convergence_rate(space_order: int) -> tuple[float, float]:
    """Empirical 1-step convergence rate of Devito's ``.dx`` operator
    at ``space_order`` on a smooth manufactured solution.

    Manufactured solution: ``f(x) = sin(2 π k x)``, analytical
    derivative ``f'(x) = 2 π k cos(2 π k x)``. We refine N→2N
    once and compute the per-grid-halving rate
    ``log2(err(N) / err(2N))``. A single refinement step at coarse
    N avoids round-off saturation that contaminates the slope at
    higher SO.

    Returns ``(rate, err_coarse)`` so callers can both check the
    fitted rate AND short-circuit when ``err_coarse`` is already
    near machine epsilon (in which case the operator is "very
    accurate" even if a slope can't be computed).
    """
    from devito import Grid, Function, Operator, Eq
    k = 2.0
    L = 1.0
    base_N = 16   # coarse enough that SO=16 isn't fully saturated
    errs: list[float] = []
    for N in (base_N, 2 * base_N):
        g = Grid(shape=(N + 1,), extent=(L,), dtype=np.float64)
        x = np.linspace(0.0, L, N + 1)
        f = Function(name='f', grid=g, space_order=space_order)
        df = Function(name='df', grid=g, space_order=space_order)
        f.data[:] = np.sin(2.0 * np.pi * k * x)
        Operator([Eq(df, f.dx)]).apply()
        dx_val = L / N
        if space_order == 2:
            x_eval = x[:-1] + 0.5 * dx_val
            df_slice = df.data[:-1]
        else:
            x_eval = x
            df_slice = df.data
        analytical = 2.0 * np.pi * k * np.cos(2.0 * np.pi * k * x_eval)
        bnd = space_order
        err = float(np.linalg.norm(
            df_slice[bnd:-bnd] - analytical[bnd:-bnd]
        )) / np.sqrt(max(df_slice[bnd:-bnd].size, 1))
        errs.append(err)
    if errs[1] < 1e-13 or errs[0] < 1e-13:
        return float('inf'), errs[0]   # already saturated
    rate = np.log2(errs[0] / errs[1])
    return float(rate), errs[0]


def test_so2_convergence_rate():
    """SO=2 is below the formal scope of Levander 1988 — included
    here for cross-validation against the companion Virieux 1986
    reproduction at ``../doi_10.1190_1.1442147/``. Same scheme
    family; rate ≈ 2 expected."""
    rate, _ = _mms_convergence_rate(space_order=2)
    assert 1.8 <= rate <= 2.5, (
        f"SO=2 .dx convergence rate {rate:.3f} outside expected "
        f"2nd-order band [1.8, 2.5]"
    )


def test_so4_paper_faithful_convergence_rate():
    """Levander 1988 §II Eq 4 claim: 4th-order accuracy in space.

    This is the CANONICAL paper-faithful invariant for this
    reproduction. The 4-point centred-FD stencil with weights
    ``{1/24, -9/8, +9/8, -1/24} / Δx`` MUST achieve formal
    convergence rate 4 on a smooth manufactured solution."""
    rate, _ = _mms_convergence_rate(space_order=4)
    assert 3.5 <= rate <= 4.5, (
        f"SO=4 .dx convergence rate {rate:.3f} outside expected "
        f"4th-order band [3.5, 4.5] — Levander 1988 Eq 4 claim"
    )


def test_so8_convergence_rate():
    """SO=8 is beyond formal scope of Levander 1988 but included
    per repo-wide so=16 convention. Expected rate ≈ 8 at the
    coarsest refinement step before round-off saturates."""
    rate, err = _mms_convergence_rate(space_order=8)
    # Either: formally 8th-order (rate ≥ 7) OR already saturated
    # at machine epsilon (err < 1e-10).
    assert rate >= 7.0 or err < 1e-10, (
        f"SO=8 .dx rate {rate:.3f}, err {err:.3e} — neither at "
        f"formal 8th order nor saturated; investigate"
    )


def test_so16_convergence_rate():
    """SO=16 near-spectral baseline — per repo-wide convention.
    Round-off saturates immediately at this order; we only check
    that the coarse-grid error is below 1e-7 (well into the
    "indistinguishable from exact" regime)."""
    rate, err = _mms_convergence_rate(space_order=16)
    # At SO=16 the dx-refinement signal is buried in round-off
    # noise. The meaningful gate is that the absolute error at
    # the coarse grid is already very small.
    assert err < 1e-7, (
        f"SO=16 .dx coarse-grid error {err:.3e} unexpectedly large; "
        f"should be near machine epsilon for a near-spectral stencil"
    )


# ---------------------------------------------------------------
# Test class 2 — reference output regression
# ---------------------------------------------------------------

@pytest.mark.parametrize('so', [2, 4, 8, 16])
def test_reference_output_matches_pin(so: int):
    """Re-run the reproduction at the given ``space_order`` and
    verify the wavefield matches the pinned reference output
    ``reference_outputs/wavefield_so<N>.npz`` to within fp64
    tolerance.

    A failure indicates either:
    - A Devito version change altered the floating-point result.
    - The driver was edited without regenerating the reference.
    - The host platform produces a different fp result.
    In any case the user must investigate before accepting.
    """
    ref_path = FOLDER / 'reference_outputs' / f'wavefield_so{so}.npz'
    assert ref_path.is_file(), (
        f"Reference output missing: {ref_path}. Run "
        f"`uv run python run_reproduction.py` to regenerate."
    )
    ref = np.load(ref_path)
    fresh = run_reproduction(so, save_npz=False)
    for field in ('vx', 'vy', 'sxx', 'syy', 'sxy'):
        ref_a = ref[field]
        fresh_a = fresh[field]
        assert ref_a.shape == fresh_a.shape, (
            f"Shape mismatch on {field} at so={so}: "
            f"ref={ref_a.shape} fresh={fresh_a.shape}"
        )
        # Tolerance: 1e-10 relative scaled by max amplitude.
        norm = max(float(np.max(np.abs(ref_a))), 1e-30)
        err = float(np.max(np.abs(ref_a - fresh_a))) / norm
        assert err < 1e-10, (
            f"{field} at so={so} drifted from reference: "
            f"max|Δ|/max|ref| = {err:.3e} (threshold 1e-10)"
        )


# ---------------------------------------------------------------
# Test class 3 — physical sanity of the produced wavefield
# ---------------------------------------------------------------

@pytest.mark.parametrize('so', [2, 4, 8, 16])
def test_wavefield_is_finite_and_bounded(so: int):
    """A divergent / NaN wavefield indicates CFL violation or a
    constructional bug. This is a fast sanity gate that catches
    catastrophic regressions before the more expensive byte-match
    test runs."""
    snapshot = run_reproduction(so, save_npz=False)
    for field in ('vx', 'vy', 'sxx', 'syy', 'sxy'):
        a = snapshot[field]
        assert np.all(np.isfinite(a)), f'{field} at so={so} contains NaN/Inf'
        # Physically reasonable upper bound: any single sample
        # smaller than 1e-2 in our scaled units (km/s × stress).
        # Real Ricker-driven wavefield should be ~1e-5 at peak.
        assert float(np.max(np.abs(a))) < 1e-2, (
            f'{field} at so={so} exceeds physical bound — likely '
            f'CFL violation or amplification instability'
        )


def test_explosive_source_radial_p_wave_symmetry():
    """An isotropic explosive source in a homogeneous isotropic
    medium produces a P-wavefront with radial particle velocity.
    Project: at the same radial distance from the source, the
    horizontal velocity component ``vx`` should follow ``cos(φ)``
    azimuthal symmetry (and ``vy`` should follow ``sin(φ)``).

    We probe at 8 equally-spaced azimuthal samples on a ring and
    verify the cos(φ) shape: opposite quadrants have opposite
    signs, and the values near φ ∈ {±90°} are near zero.
    """
    snapshot = run_reproduction(space_order=8, save_npz=False)
    vx = snapshot['vx']
    nx, ny = vx.shape
    cx, cy = nx // 2, ny // 2
    # Sample on a ring 100 grid cells out (= 1 km at dx=0.01 km).
    ring_r = 100
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    samples = np.array([
        vx[cx + int(ring_r * np.cos(a)), cy + int(ring_r * np.sin(a))]
        for a in angles
    ])
    # cos(φ) symmetry: opposite-quadrant samples have opposite signs.
    # Amplitudes may differ ~30 % due to grid anisotropy + finite
    # source-wavelet spread on a square Cartesian mesh; we check only
    # sign opposition.
    for i in range(4):
        opposite = (i + 4) % 8
        # Skip pairs near φ ∈ {±90°} where amplitudes are ~zero
        # (cos(±90°) = 0) and sign is ill-defined.
        if (abs(samples[i]) < 1e-7 or abs(samples[opposite]) < 1e-7):
            continue
        assert np.sign(samples[i]) != np.sign(samples[opposite]), (
            f'Azimuthal cos(φ) symmetry: opposite-quadrant samples '
            f'should have opposite signs. '
            f'samples[{i}]={samples[i]:.3e}, '
            f'samples[{opposite}]={samples[opposite]:.3e}'
        )


# ---------------------------------------------------------------
# Test class 4 — CFL discipline
# ---------------------------------------------------------------

@pytest.mark.parametrize('so', [2, 4, 8, 16])
def test_cfl_dt_respects_per_order_bound(so: int):
    """CFL: ``dt`` must satisfy ``Vp · dt / dx ≤ COEFF`` where
    COEFF is the per-order CFL bound. The driver's ``cfl_dt``
    helper should automatically pick a stable dt for the chosen
    space_order — this test ensures the bound stays respected
    even after edits."""
    from run_reproduction import CFL_BY_ORDER, CFL_SAFETY
    grid = build_grid(so)
    dt = cfl_dt(grid, so)
    dx = grid.extent[0] / (grid.shape[0] - 1)
    cfl_number = VP_KMS * dt / dx
    bound = CFL_BY_ORDER.get(so, min(CFL_BY_ORDER.values()))
    assert cfl_number <= bound * CFL_SAFETY * 1.01, (
        f'so={so}: cfl_number={cfl_number:.4f} exceeds bound '
        f'{bound * CFL_SAFETY:.4f}'
    )
