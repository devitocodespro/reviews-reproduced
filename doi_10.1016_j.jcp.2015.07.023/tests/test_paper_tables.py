"""Hand-transcribed-constants regression tests for Petersson-Sjögreen
2015 sw4 SBP D1/D2 coefficients.

Asserts byte-equality between the canonical sw4 values in
``petersson_sjogreen_2015.paper_tables`` and the SBP solver's
internal tabulation in ``petersson_sjogreen_2015.sbp_sat``.
Plus the symbolic SBP-property invariant
``H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)`` for the canonical
diagonal-norm boundary configuration.

These checks use exact rational arithmetic (``fractions.Fraction``)
so the byte-match is unambiguous.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from petersson_sjogreen_2015.paper_tables import (
    BOP_4_2,
    INTERIOR_4,
    NORM_WEIGHTS_4_2,
    SBP_BOUNDARY_ACCURACY,
    SBP_BOUNDARY_NORM_ORDER,
    SBP_D2_IS_PS2015_COMPATIBLE_NARROW_STENCIL,
    SBP_INTERIOR_ORDER,
    SBP_PROPERTY_NAME,
    sbp_diagonal_norm_weight,
)
from petersson_sjogreen_2015.sbp_sat import (
    _BOP_4_2,
    _INTERIOR_4,
    _NORM_WEIGHTS_4_2,
    build_sbp_d1_matrix,
    build_sbp_norm_diag,
    sbp_d1_coeffs,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: (4, 2) D1 boundary stencil byte-match against sw4
# ──────────────────────────────────────────────────────────────────


def test_bop_4_2_matches_sw4():
    """The (4, 2) boundary stencil constants in paper_tables MUST
    equal the sbp_sat solver's internal tabulation, which itself
    is meant to byte-match sw4's `src/boundaryOpc.C`."""
    assert BOP_4_2 == _BOP_4_2


@pytest.mark.parametrize("i", range(4))
def test_bop_4_2_row_i_anchor(i):
    """Row-by-row sanity check on the boundary stencil. Catches
    a single-row drift more clearly than the bulk equality above."""
    expected_paper = BOP_4_2[i]
    actual_solver = _BOP_4_2[i]
    assert expected_paper == actual_solver, (
        f"Row {i} of the (4, 2) boundary stencil differs between "
        f"paper_tables.BOP_4_2[{i}] and sbp_sat._BOP_4_2[{i}]"
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Diagonal norm weights byte-match
# ──────────────────────────────────────────────────────────────────


def test_norm_weights_byte_match_sw4():
    """The 4 boundary norm weights (17/48, 59/48, 43/48, 49/48)
    are the canonical (4, 2) diagonal-norm values."""
    assert NORM_WEIGHTS_4_2 == _NORM_WEIGHTS_4_2
    assert NORM_WEIGHTS_4_2 == (
        Fraction(17, 48),
        Fraction(59, 48),
        Fraction(43, 48),
        Fraction(49, 48),
    )


def test_diagonal_norm_first_weight_is_17_48():
    """The boundary-row-0 norm weight is 17/48 (the canonical
    (4, 2) diagonal-norm value cited throughout the literature)."""
    assert sbp_diagonal_norm_weight(order=4) == Fraction(17, 48)


def test_diagonal_norm_other_orders_raise():
    """Sentinel: only order=4 is implemented; other orders raise
    NotImplementedError rather than silently returning a wrong
    value."""
    for bad_order in [2, 6, 8]:
        with pytest.raises(NotImplementedError):
            sbp_diagonal_norm_weight(order=bad_order)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Interior 4th-order centred FD stencil byte-match
# ──────────────────────────────────────────────────────────────────


def test_interior_4_byte_match():
    """The 4th-order centred FD interior stencil is the standard
    (1/12, -2/3, 0, 2/3, -1/12)."""
    assert INTERIOR_4 == _INTERIOR_4
    assert INTERIOR_4 == (
        Fraction(1, 12),
        Fraction(-2, 3),
        Fraction(0),
        Fraction(2, 3),
        Fraction(-1, 12),
    )


# ──────────────────────────────────────────────────────────────────
# Anchor 4: SBP property invariant — H D_1 + D_1^T H = diag
# ──────────────────────────────────────────────────────────────────


def test_sbp_property_invariant():
    """Symbolic check of the canonical SBP property:
        H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)
    using exact Fraction arithmetic. This is the load-bearing
    invariant that proves the operator is SBP-compatible per
    Strand 1994 + Olsson 1995 energy method.
    """
    N = 16  # Large enough to clear the boundary modification
    h = Fraction(1)  # Symbolic — drops out of the final equality

    D1 = build_sbp_d1_matrix(N, order=4)
    H = build_sbp_norm_diag(N, order=4)

    # Compute (H D_1)[i][j] = H[i] * D_1[i][j] and then the
    # full M = H D_1 + D_1^T H symbolic matrix.
    HD = [[H[i] * D1[i][j] for j in range(N)] for i in range(N)]
    # D1^T at [i][j] is D1[j][i]
    DTH = [[D1[j][i] * H[j] for j in range(N)] for i in range(N)]
    M = [[HD[i][j] + DTH[i][j] for j in range(N)] for i in range(N)]

    # Expected: -1 at (0,0), +1 at (N-1,N-1), 0 elsewhere
    for i in range(N):
        for j in range(N):
            if i == 0 and j == 0:
                expected = Fraction(-1)
            elif i == N - 1 and j == N - 1:
                expected = Fraction(1)
            else:
                expected = Fraction(0)
            assert M[i][j] == expected, (
                f"SBP property violation at ({i}, {j}): "
                f"H D_1 + D_1^T H = {M[i][j]} ≠ {expected}"
            )


def test_sbp_property_name_anchor():
    """Sentinel: the SBP property name string is exactly the
    canonical diagonal form."""
    assert SBP_PROPERTY_NAME == "H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)"


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Order metadata
# ──────────────────────────────────────────────────────────────────


def test_order_metadata_anchors():
    """The (4, 2) naming convention: 4th-order interior +
    2nd-order at the boundary, 4 boundary rows."""
    assert SBP_INTERIOR_ORDER == 4
    assert SBP_BOUNDARY_ACCURACY == 2
    assert SBP_BOUNDARY_NORM_ORDER == 4


def test_sbp_d2_is_ps2015_compatible_sentinel():
    """The SBP D2 implementation must claim to be the
    PS-2015-compatible narrow-stencil variant. A False here
    indicates the implementation has reverted to centred-FD
    + boundary SAT corrections — which is a distinct surrogate
    per CLAUDE.md Rule 1 + Bader-2023 §40e antipattern."""
    assert SBP_D2_IS_PS2015_COMPATIBLE_NARROW_STENCIL is True


# ──────────────────────────────────────────────────────────────────
# Cross-check: solver's public API uses the byte-matched values
# ──────────────────────────────────────────────────────────────────


def test_sbp_d1_coeffs_public_api_returns_byte_matched_values():
    """`sbp_d1_coeffs(order=4)` must expose the byte-matched
    sw4 constants under its public API contract."""
    coeffs = sbp_d1_coeffs(order=4)
    assert tuple(tuple(row) for row in coeffs["boundary_rows"]) == BOP_4_2
    assert tuple(coeffs["interior"]) == INTERIOR_4
    assert tuple(coeffs["norm_weights"]) == NORM_WEIGHTS_4_2


def test_sbp_d1_coeffs_other_orders_raise():
    """Sentinel: only order=4 is implemented."""
    for bad_order in [2, 6, 8]:
        with pytest.raises(NotImplementedError):
            sbp_d1_coeffs(order=bad_order)
