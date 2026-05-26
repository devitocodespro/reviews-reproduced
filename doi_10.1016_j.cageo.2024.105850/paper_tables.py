"""Byte-checkable transcription of paper constants and recipes.

Zhang, O. & Schmitt, D. R. (2025). "An optimized 2D/3D
finite-difference seismic wave propagator using rotated staggered
grid for complex elastic anisotropic structures."
*Computers & Geosciences* **196**: 105850.
DOI: 10.1016/j.cageo.2024.105850.

Per the project's `feedback_reproduction_quantitative_first`
convention, this module hand-transcribes the quantitative claims
in the paper / upstream code so that `tests/test_paper_tables.py`
can verify the upstream code computes the same values. If a
future Devito update or upstream patch changes the numerics
silently, the byte-match test catches it.

Anchors transcribed here:

1. **Fornberg staggered first-derivative weights** at
   ``space_order ∈ {2, 4, 6, 8}`` (Eq 4 / Fornberg 1988 recurrence
   for grid spacing ``s = arange(order) - order//2 + 0.5``).
   Computed by ``upstream/src/wavesolver.py:308-321:fdcoeff_1st``.

2. **8-point isotropic source-weighting recipe** (per stress
   field: 4 face-centred + 4 diagonal-corner offsets, all weight
   ``1/8``, total moment 1). Encoded in
   ``upstream/src/operators.py:113-149:getsrcterm_2d_around``
   under the ``moment="iso"`` branch.

3. **2D RSG diagonal-stencil application**: derivatives
   composed via ``∂/∂x = d1 + d2`` and ``∂/∂y = d1 - d2``,
   where ``d1`` is the (+x, +y) diagonal and ``d2`` is the
   (+x, -y) diagonal (Saenger 2000 Eq 14). Encoded in
   ``upstream/src/wavesolver.py:250-273:dv_2d`` /
   ``dtau_2d``.
"""
from __future__ import annotations

import numpy as np


# ─── (1) Fornberg first-derivative weights ─────────────────────────────
#
# The upstream `fdcoeff_1st(order)` builds the staggered first-derivative
# coefficients by solving the linear system that enforces consistency at
# `s = arange(order) - order//2 + 0.5` (offsets in half-grid units),
# requiring `Σ c_k = 0`, `Σ s_k c_k = 1`, `Σ s_k^j c_k = 0` for `j > 1`.
#
# For SO=2, this reduces to the standard 2nd-order staggered stencil
# `[-1, +1]` (after normalisation by spacing).
#
# For SO=4: Levander 1988 fourth-order staggered Taylor weights
# `[1/24, -9/8, 9/8, -1/24]` divided by spacing.
#
# These are NOT divided by spacing in the values transcribed below —
# the upstream returns coefficients in units of "per cell spacing".

FORNBERG_STAGGERED_1ST_DERIV = {
    2: np.array([-1.0, 1.0]),
    4: np.array([1.0/24.0, -9.0/8.0, 9.0/8.0, -1.0/24.0]),
    6: np.array([
        -3.0 / 640.0,
        25.0 / 384.0,
        -75.0 / 64.0,
        75.0 / 64.0,
        -25.0 / 384.0,
        3.0 / 640.0,
    ]),
    8: np.array([
        5.0 / 7168.0,
        -49.0 / 5120.0,
        245.0 / 3072.0,
        -1225.0 / 1024.0,
        1225.0 / 1024.0,
        -245.0 / 3072.0,
        49.0 / 5120.0,
        -5.0 / 7168.0,
    ]),
}


def staggered_offsets(order: int) -> np.ndarray:
    """Half-grid offset positions for the staggered FD stencil of
    given order, matching upstream's
    ``s = np.arange(order) - order//2 + 0.5``."""
    return np.arange(order) - order // 2 + 0.5


# ─── (2) 8-point isotropic source-weighting recipe ─────────────────────
#
# OumZhang/rsg `getsrcterm_2d_around(src, tau, grid, dt, moment="iso")`:
#
# For each of the two normal stresses {tau_xx, tau_yy}:
#   - 4 face-centred shifts: (±x), (±y) → weight 1/8 each
#   - 4 diagonal-corner shifts: (±x, ±y) all sign combinations
#     → weight 1/8 each
#   - NO injection at the source's grid cell (centre)
#
# Total weight per stress field: 8 × (1/8) = 1 (preserves the
# monopole moment magnitude). Total inject equations in 2D:
# 2 stress fields × 8 positions = 16.

OUMZHANG_8POINT_OFFSETS_FACE = [
    (+1,  0),  # +x
    (-1,  0),  # -x
    ( 0, +1),  # +y
    ( 0, -1),  # -y
]

OUMZHANG_8POINT_OFFSETS_CORNER = [
    (+1, +1),
    (-1, -1),
    (-1, +1),
    (+1, -1),
]

OUMZHANG_8POINT_OFFSETS = (
    OUMZHANG_8POINT_OFFSETS_FACE + OUMZHANG_8POINT_OFFSETS_CORNER
)

# Equal weight per offset:
OUMZHANG_8POINT_WEIGHT = 1.0 / 8.0


# ─── (3) 2D Saenger Eq 14 diagonal-stencil composition ─────────────────
#
# OumZhang's `dv_2d(v_comp, grid, dd, so)`:
#   d1 = sum over k of c[k] * field.shift(x, (k+0.5)*hx).shift(y, (k+0.5)*hy)
#   d2 = sum over k of c[k] * field.shift(x, (k+0.5)*hx).shift(y, -(k+0.5)*hy)
#
# Combined into physical derivatives:
#   ∂/∂x = d1 + d2    (positive contributions from both diagonals)
#   ∂/∂y = d1 - d2    (positive d1, negative d2)
#
# This is the Saenger 2000 Wave Motion 31:77, Eq 14 prescription.

SAENGER_DIAGONAL_WEIGHTS_2D = {
    "x": {"d1": +1, "d2": +1},
    "y": {"d1": +1, "d2": -1},
}
