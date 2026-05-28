"""Liu 2014 GJI 197:1033 — LS algorithm paper-print-precision gate.

Independent reproduction of Liu 2014 §2.1 via the closed-form
(M-1)×(M-1) linear solve (Eq 10-12) + Eq 9 constraint. Verifies
that `liu_2014_ls.solve_liu_2014_ls(M, b)` reproduces Liu 2014
Table 3 to paper print precision (7 sig figs) for M ∈ {2..10}
at η = 10⁻⁴ (b from Table 2).

This is the **non-tautological independent re-derivation** that
closes pre-flight dual-reviewer YF2 (Codex+Gemini DISAGREE on
initial test set being insufficient). Yang 2015's LS-RSG-TTI
method (Y.5.3) cross-check piggybacks on this same closed-form
solve once the RSG-TTI dispersion functional is set up.

Run: `uv run pytest tests/test_liu_2014_ls.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from liu_2014_ls import (  # noqa: E402
    LIU_2014_TABLE_2_B_ETA_1E_MINUS_4,
    LIU_2014_TABLE_3,
    solve_liu_2014_ls,
)


# Paper precision: Table 3 is printed at 7 significant figures.
# Use 5e-7 absolute tolerance to allow for the last-digit rounding
# in the paper's print + numerical-quadrature noise.
PAPER_BYTE_MATCH_TOL = 5e-7


@pytest.mark.parametrize("M", sorted(LIU_2014_TABLE_3.keys()))
def test_solve_matches_table_3_at_eta_1e_minus_4(M: int):
    """Byte-match: solve_liu_2014_ls(M, b) reproduces Liu 2014
    Table 3 row M to 7 sig figs at η = 10⁻⁴.

    This is the LOAD-BEARING test — if it fails, either the
    closed-form solve has a bug OR the hand-transcribed Table
    is wrong. Either way, fix before claiming reproduction.
    """
    b = LIU_2014_TABLE_2_B_ETA_1E_MINUS_4[M]
    c_computed = solve_liu_2014_ls(M, b)
    c_paper = np.array(LIU_2014_TABLE_3[M])

    assert c_computed.shape == c_paper.shape, (
        f"M={M}: shape mismatch  computed={c_computed.shape}  "
        f"paper={c_paper.shape}")

    diffs = np.abs(c_computed - c_paper)
    max_diff = float(np.max(diffs))
    bad = [(i + 1, float(c_computed[i]), float(c_paper[i]),
            float(diffs[i]))
           for i in range(M) if diffs[i] > PAPER_BYTE_MATCH_TOL]

    assert not bad, (
        f"M={M}, b={b}: paper-precision match FAIL for c_{[t[0] for t in bad]} "
        f"(max diff {max_diff:.3e} > {PAPER_BYTE_MATCH_TOL:.0e}).\n"
        + "\n".join(f"  c_{i}: computed={cc:.7e}  paper={cp:.7e}  "
                    f"|diff|={d:.3e}"
                    for (i, cc, cp, d) in bad))


@pytest.mark.parametrize("M", sorted(LIU_2014_TABLE_3.keys()))
def test_low_wavenumber_constraint_holds(M: int):
    """Liu 2014 Eq 9: Σ_m (2m-1) c_m = 1 (low-wavenumber
    consistency). Solver MUST enforce this exactly (computed
    via Eq 9 not via the LS solve).
    """
    b = LIU_2014_TABLE_2_B_ETA_1E_MINUS_4[M]
    c = solve_liu_2014_ls(M, b)
    constraint = sum((2 * (i + 1) - 1) * c[i] for i in range(M))
    assert abs(constraint - 1.0) < 1e-12, (
        f"M={M}: low-wavenumber constraint violation  "
        f"Σ(2m-1) c_m = {constraint:.15f} ≠ 1")


@pytest.mark.parametrize("M", sorted(LIU_2014_TABLE_3.keys()))
def test_table_3_satisfies_low_wavenumber_constraint(M: int):
    """The PAPER-PRINTED Table 3 coefficients must ALSO satisfy
    Σ_m (2m-1) c_m ≈ 1 to print precision. Catches transcription
    errors that flip a digit and break the constraint.
    """
    c = LIU_2014_TABLE_3[M]
    constraint = sum((2 * (i + 1) - 1) * c[i] for i in range(M))
    # Paper precision is 7 sig figs → constraint should hold to
    # roughly 1e-6 (typical roundoff at sum-of-M-terms scale).
    assert abs(constraint - 1.0) < 5e-6, (
        f"M={M}: Liu 2014 Table 3 row violates low-wavenumber "
        f"constraint  Σ(2m-1) c_m = {constraint:.10f} ≠ 1  "
        f"(byte-transcription bug?)")


def test_first_coefficient_close_to_unity():
    """For all M, c_1 should be close to 1 (the dominant
    centred-difference coefficient). Sanity check on the
    qualitative behaviour of LS-derived RSG stencils.
    """
    for M in sorted(LIU_2014_TABLE_3.keys()):
        c_1 = LIU_2014_TABLE_3[M][0]
        assert 1.0 < c_1 < 1.5, (
            f"M={M}: c_1 = {c_1} out of expected range [1.0, 1.5)")


def test_higher_order_coefficients_decreasing_magnitude():
    """For M ≥ 4, |c_m| should generally decrease as m grows
    (consistent with a converging spatial-FD operator). Catches
    transcribed-coefficient sign-error patterns.
    """
    for M in sorted(LIU_2014_TABLE_3.keys()):
        if M < 4:
            continue
        c = [abs(v) for v in LIU_2014_TABLE_3[M]]
        # Allow a small uptick at the very last entry due to
        # the LS adjustment near the integration boundary, but
        # the main trend must decrease.
        for m_idx in range(1, M - 1):
            assert c[m_idx + 1] < c[m_idx] * 1.1, (
                f"M={M}: |c_{m_idx + 2}| = {c[m_idx + 1]} > "
                f"|c_{m_idx + 1}| × 1.1 = {c[m_idx] * 1.1}  "
                f"(unexpected magnitude growth)")
