"""Convergence regression test for the LP04 LW bulk implementation.

Locks in the 2.0 order of accuracy on a homogeneous-fluid plane-wave
problem with periodic boundaries. This is the foundational
validation gate for the Lax-Wendroff machinery; the full LP04 §4.2
plane-interface test (Table 2) builds on top of this once the
fluid-solid R/T analytical reference is completed (currently
KNOWN-INCOMPLETE per analytical_reference.py).

The fitted log-log slopes recorded here serve as the regression
baseline:  any future edit that breaks LW's 2nd-order property
on smooth bulk will fail this gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from analytical_reference import PlaneWaveFluid  # noqa: E402
from run_reproduction import run_one_resolution, fit_order  # noqa: E402


@pytest.fixture(scope='module')
def homog_lw_sweep():
    """Run the homog plane-wave convergence sweep once per module."""
    L_x = 0.1
    rho = 1000.0
    c = 1500.0
    m, n = 2, 1
    norm = np.sqrt(m * m + n * n)
    alpha = np.arctan2(n, m)
    k_mag = (2 * np.pi / L_x) * norm
    omega = c * k_mag
    pw = PlaneWaveFluid(A=1.0, omega=omega, alpha=alpha, phase=0.0,
                          c=c, rho=rho)
    T_end = 0.5 / (omega / (2 * np.pi))
    results = [run_one_resolution(Nx, T_end, pw, L_x=L_x)
               for Nx in (50, 100, 200)]
    return {
        'results': results,
        'order_inf': fit_order([r['dx'] for r in results],
                                 [r['L_inf'] for r in results]),
        'order_1': fit_order([r['dx'] for r in results],
                              [r['L_1'] for r in results]),
    }


def test_LW_homog_L_inf_order_above_1p8(homog_lw_sweep):
    """LW is formally 2nd-order in space + time; require fitted L^∞
    order ≥ 1.8 across the 3-grid sweep (50 → 100 → 200).
    """
    order = homog_lw_sweep['order_inf']
    assert order >= 1.8, (
        f"LW L^∞ convergence order {order:.3f} < 1.8; LW is formally"
        f" 2nd-order, so this is a regression.")


def test_LW_homog_L_1_order_above_1p8(homog_lw_sweep):
    """L^1 order similarly ≥ 1.8."""
    order = homog_lw_sweep['order_1']
    assert order >= 1.8, (
        f"LW L^1 convergence order {order:.3f} < 1.8.")


def test_LW_finest_grid_L_inf_bounded(homog_lw_sweep):
    """At Nx=200 (PPW ≈ 89 for our chosen wavelength), L^∞ error
    should be ≤ 5e-3. This is a slack bound (3× the empirical value
    of 1.5e-3 observed at session-establishment 2026-05-27) to tolerate
    small platform-dependent fp64 noise.
    """
    finest = homog_lw_sweep['results'][-1]
    assert finest['L_inf'] < 5e-3, (
        f"Finest-grid L^∞ {finest['L_inf']:.3e} exceeds slack bound 5e-3."
        f" Either LW degraded or the test config drifted.")


def test_LW_finest_grid_solution_finite(homog_lw_sweep):
    """Sanity: solution at finest grid is finite (no NaN/Inf)."""
    finest = homog_lw_sweep['results'][-1]
    assert np.isfinite(finest['L_inf']) and np.isfinite(finest['L_1'])


def test_LW_error_decreases_with_grid_refinement(homog_lw_sweep):
    """Strictly monotone error decrease across the sweep — protects
    against accidental coarsest-grid lucky alignment.
    """
    results = homog_lw_sweep['results']
    for i in range(len(results) - 1):
        coarse_err = results[i]['L_inf']
        fine_err = results[i + 1]['L_inf']
        ratio = coarse_err / fine_err
        # 2nd-order halving should give ~4× reduction; loose bound 2×
        assert ratio > 2.0, (
            f"L^∞ at Nx={results[i]['Nx']}→{results[i+1]['Nx']} only"
            f" decreased by {ratio:.2f}× (expect ≥2× for any convergent"
            f" scheme on smooth bulk).")
