"""Hand-transcribed paper anchors for Petersson & Sjögreen (2015).

Petersson, N. A., & Sjögreen, B. (2015). "Wave propagation in
anisotropic elastic materials and curvilinear coordinates using a
summation-by-parts finite difference method." *Journal of
Computational Physics* 299: 820-841. DOI: 10.1016/j.jcp.2015.07.023.

Antecedent (the original (4, 2) diagonal-norm SBP D1 tabulation):
Strand, B. (1994). "Summation by parts for finite difference
approximations for d/dx." *Journal of Computational Physics*
110: 47-67.

Reference implementation: sw4 (Seismic Waves 4th-order),
open-source at https://github.com/geodynamics/sw4. The (4, 2)
diagonal-norm SBP D1 boundary stencils are tabulated in
sw4's source file `src/boundaryOpc.C` and reproduced byte-for-byte
in `sbp_sat._BOP_4_2`, `_NORM_WEIGHTS_4_2`, `_INTERIOR_4`.

This module exposes the same constants as named anchors so the
test suite can regression-guard against silent tabulation drift.
"""
from __future__ import annotations

from fractions import Fraction


# ──────────────────────────────────────────────────────────────────
# Anchor 1: (4, 2) diagonal-norm SBP D1 boundary stencil
# ──────────────────────────────────────────────────────────────────


# The (4, 2) SBP D1 boundary stencil. Rows 0-3 form the 4-row
# diagonal-norm boundary modification; columns 0-5 are the column
# indices in the discrete grid `u`. Apply as:
#     (D_1 u)_i = (1/h) * Σ_j BOP_4_2[i][j] * u_j   for i ∈ {0,1,2,3}.
# Mirror at the right boundary (row N-1-i ↔ row i, with sign flip).
#
# Source: sw4 src/boundaryOpc.C (Strand 1994 tabulation; cited by
# Petersson-Sjögreen 2015 as their D1 boundary operator).
BOP_4_2: tuple[tuple[Fraction, ...], ...] = (
    # Row 0: one-sided 2nd-order accurate at the boundary.
    (Fraction(-24, 17), Fraction(59, 34), Fraction(-4, 17),
     Fraction(-3, 34),  Fraction(0),      Fraction(0)),
    # Row 1: centred 2nd-order (forced by the SBP constraint).
    (Fraction(-1, 2), Fraction(0), Fraction(1, 2),
     Fraction(0),     Fraction(0), Fraction(0)),
    # Row 2: 4-point antisymmetric, 2nd-order at the boundary.
    (Fraction(4, 43),  Fraction(-59, 86), Fraction(0),
     Fraction(59, 86), Fraction(-4, 43),  Fraction(0)),
    # Row 3: 5-point antisymmetric, 2nd-order at the boundary.
    (Fraction(3, 98), Fraction(0), Fraction(-59, 98),
     Fraction(0),     Fraction(32, 49), Fraction(-4, 49)),
)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Diagonal-norm matrix weights (boundary)
# ──────────────────────────────────────────────────────────────────


# Diagonal H norm matrix at the boundary (H_{ii}/h for i ∈ {0..3}).
# Interior rows have H_{ii}/h = 1; the upper boundary mirrors
# these in reverse order.
NORM_WEIGHTS_4_2: tuple[Fraction, ...] = (
    Fraction(17, 48),
    Fraction(59, 48),
    Fraction(43, 48),
    Fraction(49, 48),
)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Interior 4th-order centred FD stencil
# ──────────────────────────────────────────────────────────────────


# Standard 4th-order centred FD weights at offsets (-2, -1, 0, 1, 2).
# Used for rows i ∈ {4, ..., N-5}.
INTERIOR_4: tuple[Fraction, ...] = (
    Fraction(1, 12),
    Fraction(-2, 3),
    Fraction(0),
    Fraction(2, 3),
    Fraction(-1, 12),
)


# ──────────────────────────────────────────────────────────────────
# Anchor 4: SBP-property sentinel + supporting metadata
# ──────────────────────────────────────────────────────────────────


# The norm matrix H and boundary stencil D_1 together satisfy the
# SBP property:
#     H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)
# This is the canonical Strand 1994 / PS 2015 SBP-energy-method
# constraint. The Olsson (1995) energy method then proves
# stability of the time-discretised scheme.
SBP_PROPERTY_NAME = "H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)"
SBP_ENERGY_METHOD_ANCHOR = "Olsson 1995"
SBP_BOUNDARY_NORM_ORDER = 4    # 4 boundary rows
SBP_INTERIOR_ORDER = 4         # 4th-order interior accuracy
SBP_BOUNDARY_ACCURACY = 2      # 2nd-order at the boundary (the (4, 2) naming)


def sbp_diagonal_norm_weight(order: int = 4) -> Fraction:
    """Boundary norm weight ``H_{00} / h`` for the canonical
    diagonal-norm SBP D1.

    Returns ``17/48`` (the (4, 2) Strand 1994 / PS 2015 value).
    Raises ``NotImplementedError`` for other orders (the higher
    orders 6/3 and 8/4 are not currently tabulated).
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D1 (the (4, 2) diagonal-norm "
            f"variant per Petersson-Sjögreen 2015) is currently "
            f"implemented; got order={order}."
        )
    return NORM_WEIGHTS_4_2[0]


# ──────────────────────────────────────────────────────────────────
# Anchor 5: SBP D2 / second-derivative reference
# ──────────────────────────────────────────────────────────────────


# Petersson-Sjögreen 2015 §3 also derives the (4, 2) compatible
# narrow-stencil D2 operator for the second derivative
# `∂_x (b(x) ∂_x u)` of a variable-coefficient field. The boundary
# stencils (acof) + ghost-cell weights (ghcof) are tabulated in
# sw4 `src/boundaryOpc.C` and reproduced byte-for-byte in the
# `sbp_sat.acof_value` / `sbp_sat.sbp_d2_coeffs` functions.
# Sentinel: the implementation uses the SBP-compatible narrow-stencil
# D2 (NOT centred-FD with SAT corrections — the latter would be a
# distinct surrogate per CLAUDE.md Rule 1).
SBP_D2_IS_PS2015_COMPATIBLE_NARROW_STENCIL = True
