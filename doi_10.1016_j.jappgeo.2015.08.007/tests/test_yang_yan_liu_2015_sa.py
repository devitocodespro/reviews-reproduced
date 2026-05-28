"""Yang/Yan/Liu 2015 *Geophys. Prospect.* — SA-method byte-match gate.

Independent reproduction of Yang/Yan/Liu 2015 GP §"Optimal implicit
SGFD coefficients based on SA method" Eq 1-8. Verifies that
`yang_yan_liu_2015_sa.solve_yang_yan_liu_2015_sa(M, u)` reproduces
Table 1 (SA-based ISFD coefficients at u=1.25) to paper print
precision (7 sig figs) for M ∈ {2..11}.

Together with `test_liu_2014_ls.py`, this provides the second
independent re-derivation closing pre-flight dual-reviewer YF2
(Codex+Gemini: invariant gate set insufficient).

Codex YF2 concern: SA method is "summarized but not fully detailed
in the TTI-focused Yang 2015". This module shows the SA method
IS fully detailed in the antecedent paper (Eq 7-8 closed-form
linear solve), giving us a non-tautological cross-check at the
antecedent level. Yang 2015 J.Appl.Geophys. SA-RSG-TTI will then
be the explicit-RSG analog of this implicit-SGFD recipe.

Run: `uv run pytest tests/test_yang_yan_liu_2015_sa.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from yang_yan_liu_2015_sa import (  # noqa: E402
    YYL_2015_GP_TABLE_1_U_1P25,
    solve_yang_yan_liu_2015_sa,
)


# Paper precision: Table 1 prints at 7 significant figures (xxx.xxxxxxE±NN).
# Use 5e-7 absolute tolerance for byte-match.
PAPER_BYTE_MATCH_TOL = 5e-7

U_CANONICAL = 1.25   # Per Table 1 caption


@pytest.mark.parametrize("M", sorted(YYL_2015_GP_TABLE_1_U_1P25.keys()))
def test_solve_matches_table_1_at_u_1p25(M: int):
    """Byte-match: solve_yang_yan_liu_2015_sa(M, u=1.25) reproduces
    Yang/Yan/Liu 2015 GP Table 1 row M to 7 sig figs.

    Load-bearing test — if it fails, either the closed-form Eq 8
    solver has a bug OR the byte-transcribed Table 1 is wrong.
    """
    a_computed, b_computed = solve_yang_yan_liu_2015_sa(M, U_CANONICAL)
    row = YYL_2015_GP_TABLE_1_U_1P25[M]
    a_paper = np.array(row["a"])
    b_paper = row["b"]

    assert a_computed.shape == a_paper.shape, (
        f"M={M}: a-vector shape mismatch  computed={a_computed.shape}  "
        f"paper={a_paper.shape}")

    # Check a coefficients
    a_diffs = np.abs(a_computed - a_paper)
    a_max_diff = float(np.max(a_diffs))
    bad_a = [(i + 1, float(a_computed[i]), float(a_paper[i]),
              float(a_diffs[i]))
             for i in range(M) if a_diffs[i] > PAPER_BYTE_MATCH_TOL]

    # Check b parameter
    b_diff = abs(b_computed - b_paper)

    fail = bool(bad_a) or b_diff > PAPER_BYTE_MATCH_TOL

    assert not fail, (
        f"M={M}, u={U_CANONICAL}: SA byte-match FAIL.\n"
        f"  b: computed={b_computed:.7e}  paper={b_paper:.7e}  "
        f"|diff|={b_diff:.3e}\n"
        + ("  a-coefficients differ at: "
           + ", ".join(f"a_{i}={cc:.7e} vs paper {cp:.7e} "
                       f"(|diff|={d:.3e})"
                       for (i, cc, cp, d) in bad_a)
           if bad_a else "")
        + f"\n  Max a-diff: {a_max_diff:.3e} (tol {PAPER_BYTE_MATCH_TOL:.0e})"
    )


@pytest.mark.parametrize("M", sorted(YYL_2015_GP_TABLE_1_U_1P25.keys()))
def test_paper_table_1_b_in_expected_range(M: int):
    """b values in Table 1 trend from ~0.16 (M=2) to ~0.24 (M=11)
    — qualitative sanity. Catches gross transcription errors.
    """
    b = YYL_2015_GP_TABLE_1_U_1P25[M]["b"]
    assert 0.1 < b < 0.3, (
        f"M={M}: Table 1 b = {b} out of qualitative range [0.1, 0.3)")


@pytest.mark.parametrize("M", sorted(YYL_2015_GP_TABLE_1_U_1P25.keys()))
def test_paper_table_1_a1_in_expected_range(M: int):
    """a_1 values in Table 1 are positive and decrease from ~0.62
    (M=2) to ~0.34 (M=11) — qualitative sanity.
    """
    a_1 = YYL_2015_GP_TABLE_1_U_1P25[M]["a"][0]
    assert 0.3 < a_1 < 0.7, (
        f"M={M}: Table 1 a_1 = {a_1} out of qualitative range [0.3, 0.7)")


def test_a1_decreases_with_M():
    """a_1 should monotonically decrease as M grows (longer operator
    distributes weight across more taps). Catches transcription
    sign-error patterns.
    """
    Ms = sorted(YYL_2015_GP_TABLE_1_U_1P25.keys())
    a1_vals = [YYL_2015_GP_TABLE_1_U_1P25[M]["a"][0] for M in Ms]
    for i in range(1, len(Ms)):
        assert a1_vals[i] < a1_vals[i - 1], (
            f"a_1 at M={Ms[i]} ({a1_vals[i]}) ≥ a_1 at M={Ms[i-1]} "
            f"({a1_vals[i-1]}) — unexpected non-monotone trend")


def test_b_increases_with_M():
    """b should monotonically increase as M grows (more taps allow
    the Padé denominator to use a larger correction). Catches
    transcription sign-error patterns.
    """
    Ms = sorted(YYL_2015_GP_TABLE_1_U_1P25.keys())
    b_vals = [YYL_2015_GP_TABLE_1_U_1P25[M]["b"] for M in Ms]
    for i in range(1, len(Ms)):
        assert b_vals[i] > b_vals[i - 1], (
            f"b at M={Ms[i]} ({b_vals[i]}) ≤ b at M={Ms[i-1]} "
            f"({b_vals[i-1]}) — unexpected non-monotone trend")
