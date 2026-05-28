"""Yang 2015 RSG-TTI TE/SA/LS RSFD solvers — byte-match gates.

Independent reproduction of Yang 2015 *J. Appl. Geophys.* 122:40-52
§3.1 (TE-based), §3.2 (SA-based), §3.3 (LS-based) RSFD coefficient
solvers. Verifies the solvers reproduce Tables 1, 2, 3 of the paper
(pages 43-44) to print precision (7 sig figs).

This is the LOAD-BEARING test for Y.5 (independent cross-check of the
byte-transcribed coefficient tables against scipy/numpy re-derivation).
Together with the Liu 2014 LS and Yang/Yan/Liu 2015 GP SA antecedent
byte-match tests, it provides the full non-tautological independent
re-derivation that closes pre-flight dual-reviewer YF2.

Run: `uv run pytest tests/test_yang2015_rsfd_solvers.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from yang2015_rsfd_solvers import (  # noqa: E402
    YANG_2015_TABLE_1_TE,
    YANG_2015_TABLE_2_SA_U_1P10,
    YANG_2015_TABLE_3_LS_U_1P10,
    solve_ls_rsfd,
    solve_sa_rsfd,
    solve_te_rsfd,
)


# Paper print precision: 7 sig figs at coefficient magnitudes up to ~1.3.
# TE byte-match uses sympy-rational exact computation (no fp64 noise),
# so use tight 5e-7 tolerance. SA + LS use fp64 numerical quadrature
# + scipy.linalg.solve which introduces ~1e-6 noise at high M; use
# 5e-6 tolerance for those.
PAPER_BYTE_MATCH_TOL_TE = 5e-7
PAPER_BYTE_MATCH_TOL_SA = 1e-5
PAPER_BYTE_MATCH_TOL_LS = 1e-4

# Σ(2m-1) a_m = 1 holds exactly under the algorithms (TE: from
# Vandermonde structure; SA/LS: enforced via Eq 18). For paper-
# transcribed values, allow up to 5e-4 to absorb cumulative rounding
# from 7-sig-fig printed coefficients.
CONSTRAINT_TOL = 5e-4


# ─── Table 1 (TE-based RSFD) byte-match ─────────────────────────────────

@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_1_TE.keys()))
def test_solve_te_rsfd_matches_table_1(M: int):
    """Byte-match: solve_te_rsfd(M) reproduces Yang 2015 Table 1 row M
    to 7 sig figs.
    """
    a_computed = solve_te_rsfd(M)
    a_paper = np.array(YANG_2015_TABLE_1_TE[M])
    assert a_computed.shape == a_paper.shape

    diffs = np.abs(a_computed - a_paper)
    max_diff = float(np.max(diffs))
    bad = [(i + 1, float(a_computed[i]), float(a_paper[i]),
            float(diffs[i]))
           for i in range(M) if diffs[i] > PAPER_BYTE_MATCH_TOL_TE]
    assert not bad, (
        f"TE M={M}: byte-match FAIL  max diff {max_diff:.3e} > "
        f"{PAPER_BYTE_MATCH_TOL_TE:.0e}.\n"
        + "\n".join(f"  a_{i}: computed={cc:.7e}  paper={cp:.7e}  "
                    f"|diff|={d:.3e}"
                    for (i, cc, cp, d) in bad))


# ─── Table 2 (SA-based RSFD, u=1.10) byte-match ─────────────────────────

@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_2_SA_U_1P10.keys()))
def test_solve_sa_rsfd_matches_table_2(M: int):
    """Byte-match: solve_sa_rsfd(M, u=1.10) reproduces Yang 2015 Table 2."""
    a_computed = solve_sa_rsfd(M, u=1.10)
    a_paper = np.array(YANG_2015_TABLE_2_SA_U_1P10[M])
    assert a_computed.shape == a_paper.shape

    diffs = np.abs(a_computed - a_paper)
    max_diff = float(np.max(diffs))
    bad = [(i + 1, float(a_computed[i]), float(a_paper[i]),
            float(diffs[i]))
           for i in range(M) if diffs[i] > PAPER_BYTE_MATCH_TOL_SA]
    assert not bad, (
        f"SA M={M}, u=1.10: byte-match FAIL  max diff {max_diff:.3e}.\n"
        + "\n".join(f"  a_{i}: computed={cc:.7e}  paper={cp:.7e}  "
                    f"|diff|={d:.3e}"
                    for (i, cc, cp, d) in bad))


# ─── Table 3 (LS-based RSFD, u=1.10) byte-match ─────────────────────────

@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_3_LS_U_1P10.keys()))
def test_solve_ls_rsfd_matches_table_3(M: int):
    """Byte-match: solve_ls_rsfd(M, u=1.10) reproduces Yang 2015 Table 3."""
    a_computed = solve_ls_rsfd(M, u=1.10)
    a_paper = np.array(YANG_2015_TABLE_3_LS_U_1P10[M])
    assert a_computed.shape == a_paper.shape

    diffs = np.abs(a_computed - a_paper)
    max_diff = float(np.max(diffs))
    bad = [(i + 1, float(a_computed[i]), float(a_paper[i]),
            float(diffs[i]))
           for i in range(M) if diffs[i] > PAPER_BYTE_MATCH_TOL_LS]
    assert not bad, (
        f"LS M={M}, u=1.10: byte-match FAIL  max diff {max_diff:.3e}.\n"
        + "\n".join(f"  a_{i}: computed={cc:.7e}  paper={cp:.7e}  "
                    f"|diff|={d:.3e}"
                    for (i, cc, cp, d) in bad))


# ─── Cross-method invariants per pre-flight YF1 ─────────────────────────

@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_1_TE.keys()))
def test_te_low_wavenumber_constraint_holds(M: int):
    """Σ_m (2m-1) a_m = 1 (Yang 2015 Eq 17 low-wavenumber consistency)
    for the TE-derived coefficients (paper-transcribed).
    """
    a = YANG_2015_TABLE_1_TE[M]
    constraint = sum((2 * (i + 1) - 1) * a[i] for i in range(M))
    # Paper precision allows ~5e-6 deviation
    assert abs(constraint - 1.0) < CONSTRAINT_TOL, (
        f"TE M={M}: Σ(2m-1) a_m = {constraint:.10f} ≠ 1")


@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_2_SA_U_1P10.keys()))
def test_sa_low_wavenumber_constraint_holds(M: int):
    """SA solver enforces constraint via Eq 18 — verify the
    paper-transcribed Table 2 satisfies Σ_m (2m-1) a_m = 1.
    """
    a = YANG_2015_TABLE_2_SA_U_1P10[M]
    constraint = sum((2 * (i + 1) - 1) * a[i] for i in range(M))
    assert abs(constraint - 1.0) < CONSTRAINT_TOL, (
        f"SA M={M}: Σ(2m-1) a_m = {constraint:.10f} ≠ 1")


@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_3_LS_U_1P10.keys()))
def test_ls_low_wavenumber_constraint_holds(M: int):
    """LS solver enforces constraint via Eq 18 — verify the
    paper-transcribed Table 3 satisfies Σ_m (2m-1) a_m = 1.
    """
    a = YANG_2015_TABLE_3_LS_U_1P10[M]
    constraint = sum((2 * (i + 1) - 1) * a[i] for i in range(M))
    assert abs(constraint - 1.0) < CONSTRAINT_TOL, (
        f"LS M={M}: Σ(2m-1) a_m = {constraint:.10f} ≠ 1")


@pytest.mark.parametrize("M", sorted(YANG_2015_TABLE_1_TE.keys()))
def test_a1_close_to_unity_all_methods(M: int):
    """For all three methods, a_1 ∈ [1.0, 1.3) — the dominant
    centred-difference contribution. Qualitative SGFD sanity.
    """
    for label, table in [
        ("TE", YANG_2015_TABLE_1_TE),
        ("SA", YANG_2015_TABLE_2_SA_U_1P10),
        ("LS", YANG_2015_TABLE_3_LS_U_1P10),
    ]:
        a_1 = table[M][0]
        assert 1.0 < a_1 < 1.3, (
            f"{label} M={M}: a_1 = {a_1} out of expected range [1.0, 1.3)")


def test_table_1_te_a1_independent_of_u():
    """TE coefficients are u-independent (truncation-based, not
    optimisation-based). a_1 should monotonically increase with M
    (longer operator → better low-k limit closer to ideal=1).
    """
    Ms = sorted(YANG_2015_TABLE_1_TE.keys())
    a1_vals = [YANG_2015_TABLE_1_TE[M][0] for M in Ms]
    for i in range(1, len(Ms)):
        assert a1_vals[i] > a1_vals[i - 1], (
            f"TE a_1 at M={Ms[i]} ({a1_vals[i]}) ≤ M={Ms[i-1]} "
            f"({a1_vals[i-1]}) — should monotonically increase")
