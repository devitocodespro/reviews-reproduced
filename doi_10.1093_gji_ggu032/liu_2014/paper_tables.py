"""Hand-transcribed paper anchors for Liu (2014) + Fornberg (1988).

Primary paper:
    Liu, Y. (2014). "Optimal staggered-grid finite-difference schemes
    based on least-squares for wave equation modelling." *Geophysical
    Journal International* 197(2): 1033-1047.
    DOI: 10.1093/gji/ggu032.

Liu Eq 13 gives the closed-form Taylor-expansion (TE) coefficients
for the half-grid staggered first derivative

    ∂f/∂x|_x ≈ (1/h) Σ_{m=1}^M c_m [f(x + (m-½)h) − f(x − (m-½)h)]

where M = order/2. The closed form is:

    c_m = ((-1)^{m+1} / (2m-1)) ∏_{n=1, n≠m}^{M} (2n-1)² / |(2m-1)² − (2n-1)²|

Fornberg (1988) provides an equivalent **recursive** algorithm for
arbitrary stencil offsets; specialised to the half-grid first-
derivative case it produces byte-identical weights. The two
algorithms are independent in code but byte-equivalent in output —
the load-bearing sentinel.

This module exposes the orders {2, 4, 6, 8} weights as exact
`fractions.Fraction` rationals so byte-equality testing is unambiguous.
Higher orders (10, 12, …) are computed by the closed form in the
solver and cross-checked symbolically; their values are not pinned
here as they're rarely cited in standalone tables.
"""
from __future__ import annotations

from fractions import Fraction


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Liu 2014 Eq 13 closed-form TE weights as exact rationals
# ──────────────────────────────────────────────────────────────────


# Half-stencil staggered-grid first-derivative weights c_m for
# m ∈ {1, ..., M = order/2}. These are the canonical Fornberg /
# Liu 2014 Eq 13 closed-form coefficients. Byte-equivalent to the
# Fornberg 1988 recursive algorithm evaluated at the half-grid
# point.
LIU_2014_EQ13_WEIGHTS: dict[int, tuple[Fraction, ...]] = {
    # 2nd order: 1-point stencil at offset ½h.
    2: (Fraction(1),),
    # 4th order: canonical 2-point stencil.
    4: (Fraction(9, 8), Fraction(-1, 24)),
    # 6th order: 3-point stencil.
    6: (Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640)),
    # 8th order: 4-point stencil.
    8: (Fraction(1225, 1024), Fraction(-245, 3072),
        Fraction(49, 5120),  Fraction(-5, 7168)),
}


# Maximum order pinned in this anchor table. Higher orders are
# computed by the closed form but not byte-anchored here (they
# rarely appear in standalone literature tables).
LIU_2014_EQ13_MAX_PINNED_ORDER = 8


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Liu 2014 Eq 13 formula sentinel string
# ──────────────────────────────────────────────────────────────────


# The closed-form expression (verbatim from Liu 2014 §2.2).
# Sentinel for the FORMULA — locking it catches a silent swap to
# a different closed form (e.g., the Taylor-symmetric form which
# differs at the half-grid).
LIU_2014_EQ13_FORMULA_NAME = (
    "c_m = ((-1)^(m+1) / (2m-1)) * "
    "prod_{n!=m}^M (2n-1)^2 / |(2m-1)^2 - (2n-1)^2|"
)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Fornberg-equivalence sentinel
# ──────────────────────────────────────────────────────────────────


# The closed-form Liu 2014 Eq 13 weights are mathematically
# equivalent to the Fornberg 1988 recursive algorithm evaluated at
# the half-grid point — but the two are DIFFERENT algorithms in code:
#
#   * Liu Eq 13: direct closed-form, O(M) per coefficient
#     (one product over M-1 terms).
#   * Fornberg 1988: recursive δ_{n,k,m}(x) updates, O(M²) total
#     for all M coefficients.
#
# Byte-equality between the two outputs is the load-bearing
# verification property; conflating them would lose a useful
# independent cross-check.
LIU_2014_AND_FORNBERG_BYTE_EQUIVALENT = True
LIU_2014_FORMULA_IS_CLOSED_FORM = True
FORNBERG_1988_FORMULA_IS_RECURSIVE = True


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Antisymmetric-stencil convention
# ──────────────────────────────────────────────────────────────────


# The full first-derivative staggered stencil applies the
# half-stencil coefficients antisymmetrically:
#
#     (∂f/∂x)|_x ≈ (1/h) Σ_{m=1}^M c_m * [f(x + (m-½)h) − f(x − (m-½)h)]
#
# Sentinel: the c_m values returned are for the POSITIVE-offset
# side; mirroring produces the negative-offset side; the full
# expression contains a single division by h.
STAGGERED_FD_FIRST_DERIV_OFFSET_PATTERN = "(m - 1/2) * h"
STAGGERED_FD_FIRST_DERIV_ANTISYMMETRIC = True


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Liu 2014 Eq 13 sanity properties
# ──────────────────────────────────────────────────────────────────


# Σ_m (2m-1) * c_m = 1 — this is the formal consistency condition
# for the half-grid first derivative (the leading-order Taylor term
# matches the analytical derivative). All entries in
# LIU_2014_EQ13_WEIGHTS satisfy this to fp64.
LIU_2014_FIRST_DERIVATIVE_CONSISTENCY_SUM = Fraction(1)


def liu_2014_consistency_check(weights: tuple[Fraction, ...]) -> Fraction:
    """Returns Σ_m (2m-1) * c_m which MUST equal 1 for a
    consistent first-derivative half-grid stencil."""
    return sum(Fraction(2 * m - 1) * c for m, c in enumerate(weights, 1))
