"""
Paper Table 1 + Table 2 — published reference values for byte-match.

Irakarama, Luo, Etgen, Shen & O'Brien (2026) — Fifth International
Meeting for Applied Geoscience & Energy (IMAGE 2025) Expanded
Abstracts pp. 825-829, Society of Exploration Geophysicists.
DOI: 10.1190/image2025-4313380.1

Direct transcription from the paper PDF (page 4):

- Table 1: First derivative forward operator 9-point FD stencils
  used for dual-pair FD. Rows: Taylor series; Proposed
  (least-squares optimised, per Liu 2014 method per paper §THEORY).
- Table 2: Selective filter coefficients for a 9-point FD stencil.
  Filters are symmetric; only j..j+5 shown. Rows: Taylor series;
  Bogey-Bailly 2004; Proposed.

These constants anchor the byte-match faithfulness tests in
``tests/test_paper_tables.py`` (this folder) AND the parent-repo
Method 6 / Method 12 faithfulness callback at
``tests/_paper_faithfulness_callbacks.py::_test_dualpair_irakarama_2026``.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Table 1 — First derivative forward operator D⁺ at 9 points (m = -3 .. +5)
# ---------------------------------------------------------------------------

# Paper-published Taylor row (matches `sympy.finite_diff_weights` at order=8
# byte-identically — already in 06_dualpair/dp_operators.py).
TABLE_1_TAYLOR_9PT: dict[int, float] = {
    -3: -0.0059524,
    -2:  0.0714286,
    -1: -0.5,
     0: -0.45,
     1:  1.25,
     2: -0.5,
     3:  0.1666667,
     4: -0.0357143,
     5:  0.0035714,
}

# Paper-published Proposed row (least-squares optimised, paper §THEORY,
# Liu 2014 method) — the paper's defining low-PPWL contribution.
# Independent solver in stage0_operator_analysis.py:
# `optimized_forward_coefficients_9pt()` produces values close to but not
# byte-identical to these — the paper does not specify its optimisation
# range. For faithful reproduction, this transcribed table is the
# load-bearing reference.
TABLE_1_PROPOSED_9PT: dict[int, float] = {
    -3: -0.0139592,
    -2:  0.1121395,
    -1: -0.5906789,
     0: -0.3410611,
     1:  1.1866666,
     2: -0.5047765,
     3:  0.2006772,
     4: -0.057683,
     5:  0.0086754,
}

# ---------------------------------------------------------------------------
# Table 2 — Selective filter coefficients (symmetric, j..j+5 shown)
# ---------------------------------------------------------------------------

TABLE_2_TAYLOR_FILTER: dict[int, float] = {
    0:  0.2460937,
    1: -0.2050781,
    2:  0.1171875,
    3: -0.0439453,
    4:  0.0097656,
    5: -0.0009766,
}

TABLE_2_BOGEY_BAILLY_2004_FILTER: dict[int, float] = {
    0:  0.2150449,
    1: -0.1877729,
    2:  0.1237559,
    3: -0.0592276,
    4:  0.0187216,
    5: -0.0029995,
}

# Paper-published Proposed filter — minimum-variation damping profile
# (paper §"Damping responses ... see Figure 3"). Currently reproduced
# byte-identically in 06_dualpair/dp_operators.py:get_proposed_filter_coefficients.
TABLE_2_PROPOSED_FILTER: dict[int, float] = {
    0:  0.2178434,
    1: -0.1894915,
    2:  0.1233971,
    3: -0.0577743,
    4:  0.0176902,
    5: -0.0027252,
}


def backward_from_forward(fwd: dict[int, float]) -> dict[int, float]:
    """D⁻ = -(D⁺)ᵀ per Irakarama 2026 §THEORY.

    For an asymmetric forward stencil with offsets {-nl, ..., nr},
    the backward stencil has offsets {-nr, ..., nl} with weights
    bwd[m] = -fwd[-m].
    """
    return {m: -fwd[-m] for m in sorted(-k for k in fwd)}
