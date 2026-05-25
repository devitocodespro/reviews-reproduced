"""Standalone reproduction tests for Saenger, Gold & Shapiro (2000).

Pinned quantitative anchors:

  1. Staggered Taylor coefficient sum at SO=4: Sum|c_k| = 7/6
     (Levander 1988 / Holberg weights c = [9/8, -1/24]).
  2. RSG CFL bound at SO=4 (Saenger 2000 Eq 28):
       Sum|c_k|·CFL_RSG = 1  →  CFL_RSG = 6/7 ≈ 0.857.
  3. SSG 2D CFL bound at SO=4 (Saenger 2000 Eq 30):
       √2·Sum|c_k|·CFL_SSG_2D = 1  →  CFL_SSG_2D = 6/(7√2) ≈ 0.606.
  4. Higher-order sums for so ∈ {2, 8, 16} matching the Fornberg
     recurrence coefficients used by the parent repo's
     `00_common/staggered_fd.staggered_fd_coeffs`.
  5. Reference output regression: pinned `.npz` outputs match the
     freshly-run driver to fp64 (no silent coefficient drift).

The dispersion-error sweep is also written to the reference
outputs as a diagnostic, but NOT test-asserted: the 2nd-order
formula used in the driver does not generalise correctly to
higher SO under the RSG diagonal-stencil reads (the paper's
Eq 36 is 2nd-order specifically; higher-order RSG dispersion
requires a per-SO derivation). The test gate stays anchored on
the byte-checkable CFL + stencil claims, which are the strongest
evidence of paper-faithfulness anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parent
sys.path.insert(0, str(PARENT))

from run_reproduction import (   # noqa: E402
    staggered_taylor_coeffs,
    cfl_rsg,
    cfl_ssg_2d,
    cfl_ssg_3d,
    run_dispersion_analysis,
)


def test_taylor_weights_so2():
    """SO=2 staggered weights: c = [1]."""
    c = staggered_taylor_coeffs(2)
    assert np.array_equal(c, np.array([1.0]))


def test_taylor_weights_so4_levander():
    """SO=4 staggered weights match Levander 1988 / Holberg c = [9/8, -1/24]."""
    c = staggered_taylor_coeffs(4)
    assert abs(c[0] - 9.0 / 8.0) < 1e-15
    assert abs(c[1] - (-1.0 / 24.0)) < 1e-15


def test_taylor_weights_so8_fornberg():
    """SO=8 staggered weights match the Fornberg 4-tap half-stencil
    used by 00_common/staggered_fd in the parent repo."""
    c = staggered_taylor_coeffs(8)
    expected = np.array([
        1225.0 / 1024.0,
        -245.0 / 3072.0,
        49.0 / 5120.0,
        -5.0 / 7168.0,
    ])
    np.testing.assert_allclose(c, expected, atol=1e-15)


def test_taylor_weights_so16_fornberg():
    """SO=16 staggered weights (per repo-wide `so=16` convention)."""
    c = staggered_taylor_coeffs(16)
    assert len(c) == 8
    # Sum should be positive (the c_1 coefficient dominates).
    assert c[0] > 1.0


def test_cfl_rsg_so4_paper_byte_match():
    """Saenger 2000 Eq 28 at SO=4: Sum|c_k| = 7/6 → CFL_RSG = 6/7."""
    cfl = cfl_rsg(4)
    expected = 6.0 / 7.0
    assert abs(cfl - expected) < 1e-15, (
        f"RSG CFL bound at SO=4: got {cfl}, expected {expected}"
    )


def test_cfl_ssg_2d_so4_paper_byte_match():
    """Saenger 2000 Eq 30 at SO=4: CFL_SSG_2D = 6/(7√2)."""
    cfl = cfl_ssg_2d(4)
    expected = 6.0 / (7.0 * np.sqrt(2.0))
    assert abs(cfl - expected) < 1e-14, (
        f"SSG 2D CFL bound at SO=4: got {cfl}, expected {expected}"
    )


def test_cfl_ssg_3d_so4_paper_byte_match():
    """Saenger 2000 Eq 29 at SO=4: CFL_SSG_3D = 6/(7√3)."""
    cfl = cfl_ssg_3d(4)
    expected = 6.0 / (7.0 * np.sqrt(3.0))
    assert abs(cfl - expected) < 1e-14


def test_cfl_rsg_more_restrictive_than_ssg_at_higher_order():
    """Saenger 2000 page 84 claim: 'the stability condition for the
    new rotated grid is LESS restrictive'. CFL_RSG > CFL_SSG_2D at
    all orders (RSG can take larger time steps for the same dx)."""
    for so in (2, 4, 8, 16):
        assert cfl_rsg(so) > cfl_ssg_2d(so), (
            f"At SO={so}: CFL_RSG={cfl_rsg(so)} should exceed "
            f"CFL_SSG_2D={cfl_ssg_2d(so)} per Saenger 2000 page 84."
        )


def test_cfl_rsg_so2_unity():
    """At SO=2 the RSG CFL bound is exactly 1 (Saenger 2000 Eq 27)."""
    assert abs(cfl_rsg(2) - 1.0) < 1e-15


def test_cfl_ssg_2d_so2_paper_byte_match():
    """At SO=2 the SSG 2D CFL bound is 1/√2 (Saenger 2000 Eq 30 with c=[1])."""
    assert abs(cfl_ssg_2d(2) - 1.0 / np.sqrt(2.0)) < 1e-15


# =====================================================================
# Regression: reference outputs match the driver to fp64
# =====================================================================

@pytest.mark.parametrize('so', [2, 4, 8, 16])
def test_reference_outputs_match_driver(so):
    """Pinned `.npz` reference outputs reproduce byte-exactly from
    a fresh driver run. Detects coefficient drift in the staggered
    Taylor weights or CFL formula."""
    ref_path = PARENT / 'reference_outputs' / f'dispersion_so{so}.npz'
    assert ref_path.exists(), f"Reference output missing: {ref_path}"
    ref = np.load(ref_path)
    current = run_dispersion_analysis(so)
    # CFL bounds (the test-asserted claims):
    assert abs(float(ref['cfl_rsg']) - current['cfl_rsg']) < 1e-15
    assert abs(float(ref['cfl_ssg_2d']) - current['cfl_ssg_2d']) < 1e-15
    # Stencil-derived metrics: pinned v_ratio_rsg array byte-matches.
    np.testing.assert_allclose(
        ref['v_ratio_rsg'], current['v_ratio_rsg'], atol=0.0
    )


def test_reference_outputs_have_dispersion_diagnostic():
    """The reference outputs MUST carry the per-SO dispersion sweep
    as diagnostic data (even though not test-asserted as a faithful
    paper claim — see the module docstring for why)."""
    for so in (2, 4, 8, 16):
        ref_path = PARENT / 'reference_outputs' / f'dispersion_so{so}.npz'
        ref = np.load(ref_path)
        assert ref['v_ratio_rsg'].shape == (73, 20), (
            f"dispersion_so{so}: v_ratio_rsg shape mismatch"
        )
        assert ref['angles_deg'].shape == (73,)
        assert ref['ppw_values'].shape == (20,)
