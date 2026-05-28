"""Paper-anchored transcription of Xie & He 2024 quantitative claims.

Xie, J. & He, B. (2024). "Spatial-temporal high-order
rotated-staggered-grid finite-difference scheme of elastic wave
equations for TTI medium."
*Journal of Computational Physics* **499**: 112684.
DOI: 10.1016/j.jcp.2023.112684

Per the project's `feedback_reproduction_quantitative_first`
convention, this module hand-transcribes the paper's quantitative
claims so the reproduction tests can byte-check the parent's
implementation against the paper.

Anchors transcribed here:

1. **Algorithm identity** — the TE+LS (Taylor-Expansion + Least-
   Squares) joint optimization of the space-time dispersion
   relation. The CRITICAL claim is that the optimized stencil
   addresses the spatial-interpolation error of conventional
   Saenger 2000 RSG in TTI media (paper §3, page 4).

2. **Consistency constraint**: per Xie & He §3 Eq (15), the
   antisymmetric centered-FD coefficients must satisfy
   ``2 · Σ_m m·c_m = 1`` for first-order Taylor accuracy at
   ``kh → 0``. This is byte-checkable.

3. **Half-stencil width parameter ``P``**: the paper uses
   ``P = 6`` for the 12th-order baseline (P=6 means stencil
   width = 2P+1 = 13, but antisymmetric so only 6 independent
   positive coefficients). Lower P configurations (P ∈ {4, 5})
   produce lower-order schemes; the paper presents them as
   ablations.

4. **CFL-dependence**: the optimized coefficients depend on the
   CFL number c = Vmax · dt / dx. Different CFLs produce
   different optimal coefficient sets. The paper's canonical
   demonstrations use CFL ∈ {0.3, 0.4, 0.5}.

5. **Dispersion-error reduction**: the qualitative claim is
   that the TE+LS stencil has LOWER dispersion error than the
   pure Taylor stencil (Fornberg 1988) at the same P, especially
   near the Nyquist limit. This is testable empirically — the
   modified-wavenumber error |k̃h - kh| at fixed kh should be
   smaller for STO than for Taylor.

The parent repo's implementation lives in
``00_common/spacetime_coefficients.py:compute_sto_coefficients(P, cfl)``
— uses SciPy SLSQP to perform the LS optimization with the
TE constraint. Numerical reference outputs at canonical
configurations are pinned at ``reference_outputs/sto_coefficients_pinned.json``.
"""
from __future__ import annotations


# ─── (1) Algorithm identity ──────────────────────────────────────────


XIE_HE_2024_METHOD_NAME = "TE+LS RSG"
XIE_HE_2024_FORMULATION = (
    "Taylor-Expansion + Least-Squares joint optimization of "
    "space-time dispersion relation. Applied on a rotated-staggered "
    "grid with stress at NODE / velocity at corners (or equivalent "
    "staggering). Addresses Saenger 2000 RSG's spatial-interpolation "
    "error in TTI elastic media (§3, page 4 of the paper)."
)


# ─── (2) Consistency constraint ─────────────────────────────────────


def consistency_residual(positive_coeffs: dict[int, float]) -> float:
    """Return ``2 · Σ_m m·c_m - 1`` for an antisymmetric stencil.

    Per Xie & He §3 Eq (15): the antisymmetric centered FD must
    satisfy ``Σ_m c_m sin(m·kh) = (kh / 2)`` to leading order at
    kh→0; the first-order term gives ``2 · Σ_m m·c_m = 1``.

    A correctly-optimized stencil returns 0 (to fp64).
    """
    return 2.0 * sum(m * c for m, c in positive_coeffs.items()) - 1.0


# ─── (3) Canonical half-stencil widths used in the paper ─────────────


XIE_HE_2024_CANONICAL_P = (4, 5, 6)
XIE_HE_2024_PRIMARY_P = 6  # 12th-order baseline (so=12 in our code)


# ─── (4) Canonical CFL configurations ────────────────────────────────


XIE_HE_2024_CANONICAL_CFL = (0.3, 0.4, 0.5)


# ─── (5) Dispersion-error reduction (qualitative) ────────────────────
#
# Verified empirically in `tests/test_paper_tables.py::test_sto_vs_taylor_dispersion`.
# At fixed P and a representative kh in [π/4, π/2], the modified-
# wavenumber error of STO must be LESS than the Taylor error.


XIE_HE_2024_DISPERSION_REDUCTION_KH = 1.0  # rad, representative band


# ─── (6) RSG topology + source claim ─────────────────────────────────
#
# Xie & He emphasise that conventional staggered-grid (Yee/SSG) for
# TTI elastic requires spatial INTERPOLATION of some derivatives.
# Their rotated-staggered topology avoids that interpolation — all
# derivatives are sampled at the field's natural defining points.
# The 45° X-mode artifact visible in Saenger-2000-RSG snapshots is
# the empirical manifestation of the interpolation error described
# in the paper's §3.


XIE_HE_2024_SOLVES_TTI_INTERPOLATION_ERROR = True
