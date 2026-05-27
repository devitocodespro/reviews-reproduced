"""Paper-faithfulness regression tests for Lombard & Piraux 2004.

These tests run under the reproduction folder's own venv (uv sync,
then uv run pytest tests/ -v).  Each test asserts a property of the
transcribed paper-tables that follows from the paper's equations,
not from our own implementation. The transcription itself was
user-confirmed via side-by-side review PDFs (see
transcription_review/ subdirectory + paper_tables.py provenance log).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

import paper_tables as pt


# ─── Eq 10-11 — fluid-solid jump conditions ───────────────────────────


def test_C1_zero_shape_and_normal_velocity_row():
    """C1^0 is 2x3; row 0 = (-y', x', 0) -> normal velocity v.n."""
    C = pt.C1_zero()
    assert C.shape == (2, 3)
    assert C[0, 0] == -pt.y_prime
    assert C[0, 1] == pt.x_prime
    assert C[0, 2] == 0


def test_C2_zero_shape_and_normal_velocity_row():
    """C2^0 is 2x5; row 0 = (-y', x', 0, 0, 0) -> normal velocity."""
    C = pt.C2_zero()
    assert C.shape == (2, 5)
    assert C[0, 0] == -pt.y_prime
    assert C[0, 1] == pt.x_prime
    assert tuple(C[0, 2:]) == (0, 0, 0)


def test_C2_zero_normal_stress_row_matches_nTsigma_n():
    """C2^0 row 2 . U2 = n^T sigma n x |t|^2 (Eq 11)."""
    assert pt.verify_C2_normal_stress_identity()


def test_L1_zero_is_empty_row():
    """L1^0 is 1x3 zero (perfect fluid has no extra BC beyond jump)."""
    L = pt.L1_zero()
    assert L.shape == (1, 3)
    assert all(L[0, j] == 0 for j in range(3))


def test_L2_zero_shape_and_tangential_shear():
    """L2^0 is 1x5; L2^0 . U2 = +/- t^T sigma n (sign-free)."""
    L = pt.L2_zero()
    assert L.shape == (1, 5)
    assert pt.verify_L2_tangential_shear_identity()


# ─── Eq 22 — alpha_1, alpha_2 elastic-coefficient combinations ────────


def test_alpha_coefficients_canonical_solid():
    """At c_p=4, c_s=2 km/s: alpha_1 = 1/3, alpha_2 = -1/6 (Eq 22)."""
    a1, a2 = pt.alpha_coefficients(4.0, 2.0)
    assert a1 == pytest.approx(1.0 / 3.0)
    assert a2 == pytest.approx(-1.0 / 6.0)


def test_alpha_sum_identity():
    """alpha_1 + alpha_2 = c_s^2 / [2 (c_p^2 - c_s^2)] from Eq 22."""
    assert pt.verify_alpha_sum_identity()


def test_alpha_coefficients_degenerate_raises():
    """Degenerate c_p = c_s solid must raise ValueError (Eq 22 denom = 0)."""
    with pytest.raises(ValueError):
        pt.alpha_coefficients(2.0, 2.0)


# ─── Appendix A.1, A.2 — G_i^k matrix construction ────────────────────


@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_G_fluid_dimensions(k):
    """Fluid G_i^k: rows = 3 x #monomials; cols = (k+1)(k+3)."""
    G = pt.build_G_fluid(k)
    monomials = (k + 1) * (k + 2) // 2
    assert G.shape == (3 * monomials, (k + 1) * (k + 3))


@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_G_solid_dimensions(k):
    """Solid G_i^k: rows = 5 x #monomials; cols = 2 k^2 + 8 k + 5."""
    G = pt.build_G_solid(k)
    monomials = (k + 1) * (k + 2) // 2
    assert G.shape == (5 * monomials, 2 * k * k + 8 * k + 5)


def test_G_fluid_k0_is_identity():
    """At k=0, fluid V_1^0 = (v1, v2, p) -> G_1^0 is 3x3 identity."""
    G = pt.build_G_fluid(0)
    assert G == sp.eye(3)


def test_G_solid_k0_is_identity():
    """At k=0, solid V_2^0 = (v1, v2, s11, s12, s22) -> 5x5 identity."""
    G = pt.build_G_solid(0)
    assert G == sp.eye(5)


def test_G_solid_carries_alpha_entries_at_k2():
    """At k=2 solid G has alpha_1 and alpha_2 entries (sigma-symmetry rows)."""
    G = pt.build_G_solid(2)  # symbolic alpha_1, alpha_2
    entries = set()
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            v = G[i, j]
            if v != 0 and v != 1:
                entries.add(v)
    assert sp.Symbol('alpha_1') in entries
    assert sp.Symbol('alpha_2') in entries


def test_G_dimensions_helper():
    """The aggregated verify_G_dimensions() agrees with per-k tests."""
    assert pt.verify_G_dimensions()


# ─── Numeric alpha substitution ───────────────────────────────────────


def test_G_solid_numeric_alpha_substitution():
    """build_G_solid with numeric alpha_1=1/3, alpha_2=-1/6 carries those
    rationals at the sigma-symmetry-row positions."""
    G = pt.build_G_solid(2, alpha_1=sp.Rational(1, 3),
                            alpha_2=sp.Rational(-1, 6))
    entries = set()
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            v = G[i, j]
            if v != 0 and v != 1:
                entries.add(v)
    assert sp.Rational(1, 3) in entries
    assert sp.Rational(-1, 6) in entries
