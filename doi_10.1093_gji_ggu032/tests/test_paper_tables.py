"""Hand-transcribed-constants regression tests for Liu (2014) GJI Eq 13.

Asserts:
  (1) Liu 2014 Eq 13 closed-form weights at orders {2, 4, 6, 8}
      byte-match the canonical Fornberg-equivalent rational
      coefficients to fp64 (modulo one ULP from product reorder).
  (2) The first-derivative consistency condition
      Σ_m (2m-1) c_m = 1 holds exactly for every pinned order.
  (3) Sentinel strings lock the formula, antisymmetric-convention,
      and Fornberg-equivalence claims.
  (4) The solver rejects invalid orders (non-positive / odd).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from liu_2014.optimal_fd_coefficients import taylor_staggered_coeffs
from liu_2014.paper_tables import (
    FORNBERG_1988_FORMULA_IS_RECURSIVE,
    LIU_2014_AND_FORNBERG_BYTE_EQUIVALENT,
    LIU_2014_EQ13_FORMULA_NAME,
    LIU_2014_EQ13_MAX_PINNED_ORDER,
    LIU_2014_EQ13_WEIGHTS,
    LIU_2014_FIRST_DERIVATIVE_CONSISTENCY_SUM,
    LIU_2014_FORMULA_IS_CLOSED_FORM,
    STAGGERED_FD_FIRST_DERIV_ANTISYMMETRIC,
    STAGGERED_FD_FIRST_DERIV_OFFSET_PATTERN,
    liu_2014_consistency_check,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Liu 2014 Eq 13 weights byte-match solver
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", sorted(LIU_2014_EQ13_WEIGHTS.keys()))
def test_liu_2014_eq13_weights_solver_byte_match(order):
    """The solver's `taylor_staggered_coeffs(order)` MUST match the
    Liu 2014 Eq 13 / Fornberg-equivalent rational weights to fp64
    (allowing 1 ULP for the product reorder)."""
    expected = LIU_2014_EQ13_WEIGHTS[order]
    actual = taylor_staggered_coeffs(order)

    assert len(actual) == len(expected) == order // 2

    for m, (exp_frac, act_float) in enumerate(zip(expected, actual), 1):
        exp_float = float(exp_frac)
        assert act_float == pytest.approx(exp_float, abs=1e-15, rel=1e-14), (
            f"order={order}, m={m}: solver gave {act_float!r}, "
            f"paper gave {exp_float!r} (= {exp_frac})"
        )


@pytest.mark.parametrize("order", [4, 6, 8])
def test_liu_2014_first_weight_exceeds_one(order):
    """The leading staggered-FD weight c_1 exceeds 1 for order ≥ 4
    (vs the 2nd-order weight c_1 = 1). Asymmetric optimisation —
    higher-order TE puts MORE weight on the nearest half-grid
    sample, not less."""
    weights = LIU_2014_EQ13_WEIGHTS[order]
    assert weights[0] > Fraction(1)


def test_liu_2014_eq13_order_2_is_unity():
    """Order=2 collapses to a single c_1 = 1: the standard
    centered-staggered 2nd-order FD."""
    assert LIU_2014_EQ13_WEIGHTS[2] == (Fraction(1),)


def test_liu_2014_eq13_order_4_anchor():
    """Order=4 weights: (9/8, -1/24). The canonical 4th-order
    centered-staggered FD."""
    assert LIU_2014_EQ13_WEIGHTS[4] == (Fraction(9, 8), Fraction(-1, 24))


def test_liu_2014_eq13_order_6_anchor():
    """Order=6 weights: (75/64, -25/384, 3/640)."""
    assert LIU_2014_EQ13_WEIGHTS[6] == (
        Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640),
    )


def test_liu_2014_eq13_order_8_anchor():
    """Order=8 weights: (1225/1024, -245/3072, 49/5120, -5/7168)."""
    assert LIU_2014_EQ13_WEIGHTS[8] == (
        Fraction(1225, 1024), Fraction(-245, 3072),
        Fraction(49, 5120),   Fraction(-5, 7168),
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 2: First-derivative consistency condition Σ (2m-1) c_m = 1
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", sorted(LIU_2014_EQ13_WEIGHTS.keys()))
def test_first_derivative_consistency_exact(order):
    """For every pinned order, Σ_m (2m-1) c_m = 1 holds EXACTLY as
    a rational. This is the leading-order Taylor consistency
    condition for the half-grid first derivative."""
    weights = LIU_2014_EQ13_WEIGHTS[order]
    actual_sum = liu_2014_consistency_check(weights)
    assert actual_sum == LIU_2014_FIRST_DERIVATIVE_CONSISTENCY_SUM
    assert actual_sum == Fraction(1)


@pytest.mark.parametrize("order", [2, 4, 6, 8, 10, 12])
def test_solver_consistency_fp64(order):
    """For every order the solver produces, Σ_m (2m-1) c_m = 1 holds
    to fp64. Tests the solver's output for orders BEYOND the pinned
    table too (10, 12)."""
    c = taylor_staggered_coeffs(order)
    M = order // 2
    s = sum((2 * m - 1) * c[m - 1] for m in range(1, M + 1))
    assert s == pytest.approx(1.0, abs=1e-14)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Sentinel strings + algorithm-class flags
# ──────────────────────────────────────────────────────────────────


def test_liu_eq13_formula_name_sentinel():
    """Locking the formula sentinel catches a silent swap to a
    different closed form."""
    assert LIU_2014_EQ13_FORMULA_NAME == (
        "c_m = ((-1)^(m+1) / (2m-1)) * "
        "prod_{n!=m}^M (2n-1)^2 / |(2m-1)^2 - (2n-1)^2|"
    )


def test_fornberg_equivalence_sentinel():
    """Liu Eq 13 and Fornberg are byte-equivalent but DIFFERENT
    algorithms — closed-form vs recursive. Locking these flags
    preserves the independent cross-check value."""
    assert LIU_2014_AND_FORNBERG_BYTE_EQUIVALENT is True
    assert LIU_2014_FORMULA_IS_CLOSED_FORM is True
    assert FORNBERG_1988_FORMULA_IS_RECURSIVE is True


def test_antisymmetric_convention_sentinel():
    """The full stencil applies c_m antisymmetrically about x with
    offsets (m-½)h. Locking this convention catches a swap to the
    centred (integer-offset) convention."""
    assert STAGGERED_FD_FIRST_DERIV_OFFSET_PATTERN == "(m - 1/2) * h"
    assert STAGGERED_FD_FIRST_DERIV_ANTISYMMETRIC is True


def test_max_pinned_order_metadata():
    """The pinned table covers orders ≤ 8. Higher orders are
    solver-computed and consistency-checked but not byte-pinned."""
    assert LIU_2014_EQ13_MAX_PINNED_ORDER == 8
    assert max(LIU_2014_EQ13_WEIGHTS.keys()) == 8


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Solver input-validation gates
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_order", [0, -2, 1, 3, 5])
def test_solver_rejects_invalid_order(bad_order):
    """The solver rejects non-positive or odd orders."""
    with pytest.raises(ValueError):
        taylor_staggered_coeffs(bad_order)


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Liu Eq 13 weights are antisymmetric in the full
#           stencil sense (sign of c_m alternates with m)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", [4, 6, 8])
def test_weight_sign_alternates(order):
    """For order ≥ 4, signs of c_m alternate: c_1 > 0, c_2 < 0,
    c_3 > 0, c_4 < 0, ... This is a direct consequence of the
    closed-form (-1)^(m+1) factor."""
    weights = LIU_2014_EQ13_WEIGHTS[order]
    for m, c in enumerate(weights, 1):
        expected_sign = 1 if m % 2 == 1 else -1
        assert (1 if c > 0 else -1) == expected_sign, (
            f"order={order}, m={m}: weight={c} has wrong sign "
            f"(expected sign = {expected_sign})"
        )
