"""Yang 2015 §4 dispersion-error gate — Table 4 paper-print-precision
match + qualitative ordering invariants.

Closes pre-flight dual-reviewer YF3 (Codex + Gemini both DISAGREE on
strict k_cross byte-equality — use **qualitative ordering +
envelope** instead).

Table 4 (paper page 45) gives u-values at which the RMS dispersion
error ε first hits {10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³} for each (M ∈ {2..11},
scheme ∈ {TE, SA, LS}). This module matches that table to
paper-print precision (2 decimal places ± `TABLE_4_TOL = 0.012`
tolerance — absorbs binary-search + integration noise on top of the
paper's 0.005 print-rounding bound). The matching is tolerance-
bounded, NOT byte-equal.

Qualitative invariants additionally tested:
  - u_TE < u_SA < u_LS at every (ε, M)  [paper claim: SA/LS widen the
    accuracy band over TE]
  - u(ε, M) monotonically increases with M at fixed ε [longer operator
    → wider band]
  - u(ε, M) monotonically increases with ε at fixed M [larger error
    tolerance → wider band]

Run: `uv run pytest tests/test_yang2015_dispersion.py -v`
Note: test suite ~30 s runtime (each cell solves a fixed-point u via
binary search; SA/LS re-solve their coefficient system at each
candidate u).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from yang2015_dispersion import (  # noqa: E402
    YANG_2015_TABLE_4,
    find_u_for_target_error,
)


# Paper print precision: 2 decimal places (paper-rounded to 0.01).
# Tolerance 0.012 allows for:
#   - paper's 2-decimal rounding (±0.005)
#   - binary-search xtol=1e-4 on u + integration noise (~0.001)
#   - small drift from how the paper's integration scheme may differ
#     from scipy.integrate.quad
# Tightened from the initial 0.03 after Codex G1 (2026-05-27) found
# two OCR transcription typos that the loose tolerance had masked
# (Table 4 ε=1e-6 M=11 LS, ε=1e-5 M=5 TE). Per pre-flight YF1 the
# paper-precision tolerance MUST surface real transcription errors —
# loose tolerance is a documented anti-pattern.
#
# Phase Y/1.5b 2026-05-28 — relabeled from "byte-match" to
# "paper-print-precision match" per Codex round-2 review:
# TABLE_4_TOL = 0.012 exceeds the strict 2-decimal print rounding
# bound (±0.005) by ~2.4× to absorb binary-search + integration
# noise. This is honestly a tolerance-bounded match, NOT a
# byte-match.
TABLE_4_TOL = 0.012


_EPS_TARGETS = sorted(YANG_2015_TABLE_4.keys())
_M_VALUES = sorted(YANG_2015_TABLE_4[1e-3].keys())


@pytest.mark.parametrize("eps_target", _EPS_TARGETS)
@pytest.mark.parametrize("M", _M_VALUES)
@pytest.mark.parametrize("scheme", ["TE", "SA", "LS"])
def test_table_4_paper_precision_match(
    eps_target: float, M: int, scheme: str
):
    """Paper-print-precision match for Yang 2015 Table 4 (u-values
    where ε first hits threshold).

    For each (ε, M, scheme): solve for u via binary search such that
    the scheme's RMS error equals ε_target. Match against paper Table 4
    to 2-decimal paper precision ± TABLE_4_TOL = 0.012.

    NOTE (Phase Y/1.5b): this is a tolerance-bounded
    "compute matches paper print" gate, NOT a byte-match
    assertion. The tolerance bound is documented above.
    """
    u_computed = find_u_for_target_error(M, eps_target, scheme)
    u_paper = YANG_2015_TABLE_4[eps_target][M][scheme]
    diff = abs(u_computed - u_paper)
    assert diff < TABLE_4_TOL, (
        f"Table 4 paper-precision match FAIL: ε={eps_target:.0e}, "
        f"M={M}, scheme={scheme}\n"
        f"  computed u = {u_computed:.4f}\n"
        f"  paper u    = {u_paper:.2f}\n"
        f"  |diff|     = {diff:.4f}  (tol {TABLE_4_TOL})")


# ─── Qualitative invariants per YF3 ────────────────────────────────────


@pytest.mark.parametrize("eps_target", _EPS_TARGETS)
@pytest.mark.parametrize("M", _M_VALUES)
def test_u_te_less_than_u_sa(eps_target: float, M: int):
    """SA widens the accuracy band over TE (paper §4 claim)."""
    row = YANG_2015_TABLE_4[eps_target][M]
    assert row["TE"] < row["SA"], (
        f"ε={eps_target:.0e}, M={M}: u_TE = {row['TE']} should be less "
        f"than u_SA = {row['SA']}")


@pytest.mark.parametrize("eps_target", _EPS_TARGETS)
@pytest.mark.parametrize("M", _M_VALUES)
def test_u_sa_less_than_or_equal_u_ls(eps_target: float, M: int):
    """LS widens the accuracy band over SA at large wavenumbers
    (paper §4 + abstract claim).

    Strict u_SA < u_LS holds everywhere in Table 4 except a few
    high-M rows in ε=10⁻⁵ band where rounding ties to 0.005.
    """
    row = YANG_2015_TABLE_4[eps_target][M]
    # At ε=10⁻⁵ + M=11 the paper rounds u_SA=1.21, u_LS=1.27 — strict
    # inequality holds. At other rows the inequality is also strict.
    assert row["SA"] <= row["LS"] + 1e-6, (
        f"ε={eps_target:.0e}, M={M}: u_SA = {row['SA']} should be ≤ "
        f"u_LS = {row['LS']}")


@pytest.mark.parametrize("scheme", ["TE", "SA", "LS"])
@pytest.mark.parametrize("eps_target", _EPS_TARGETS)
def test_u_monotonic_in_M(scheme: str, eps_target: float):
    """At fixed (ε, scheme), u increases monotonically with M (longer
    operator → wider accuracy band).
    """
    u_vals = [YANG_2015_TABLE_4[eps_target][M][scheme] for M in _M_VALUES]
    for i in range(1, len(u_vals)):
        assert u_vals[i] >= u_vals[i - 1] - 1e-6, (
            f"{scheme} ε={eps_target:.0e}: u not monotonic in M  "
            f"M={_M_VALUES[i]}: {u_vals[i]} < M={_M_VALUES[i-1]}: "
            f"{u_vals[i-1]}")


@pytest.mark.parametrize("scheme", ["TE", "SA", "LS"])
@pytest.mark.parametrize("M", _M_VALUES)
def test_u_monotonic_in_eps_target(scheme: str, M: int):
    """At fixed (M, scheme), u increases monotonically with ε_target
    (larger tolerated error → wider band).
    """
    u_vals = [YANG_2015_TABLE_4[eps_target][M][scheme]
              for eps_target in _EPS_TARGETS]
    for i in range(1, len(u_vals)):
        assert u_vals[i] >= u_vals[i - 1] - 1e-6, (
            f"{scheme} M={M}: u not monotonic in ε  "
            f"ε={_EPS_TARGETS[i]:.0e}: {u_vals[i]} < "
            f"ε={_EPS_TARGETS[i-1]:.0e}: {u_vals[i-1]}")
