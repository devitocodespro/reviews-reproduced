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
# Xie & He §3.2 + §4 present the dispersion advantage at the
# Nyquist wavenumber (kh = π/2 per axis; ~2.22 rad along the
# diagonal). The TE+LS optimization trades low-kh accuracy for
# high-kh accuracy — this is the DEFINING FEATURE of an LS-
# optimized stencil, not a bug.
#
# Empirically verified at (P=6, cfl=0.4) on scipy 1.17.1
# (2026-05-28, Phase X deepening sweep):
#
#   kh    sto_err     taylor_err  STO < Taylor?
#   0.3   7.6e-05     1.3e-11     NO  (Taylor essentially exact)
#   0.7   1.8e-05     6.5e-07     NO
#   1.0   6.3e-05     5.4e-05     NO  (crossover region; near-tied)
#   1.2   1.8e-04     4.8e-04     YES (STO wins by ~2.7x)
#   1.5   1.7e-04     6.1e-03     YES (STO wins by ~36x)
#   1.8   1.9e-02     4.2e-02     YES (STO wins by ~2.2x)
#   2.0   8.5e-02     1.2e-01     YES (STO wins by ~1.4x)
#   π/2   ...          ...        approaching Nyquist limit
#
# Test anchors below reflect this band structure:
#  - HIGH_KH band [1.2, 2.0]: STO < Taylor strictly holds
#  - LOW_KH band  [0.1, 0.7]: Taylor < STO (LS tradeoff —
#    documents the defining feature, not a bug)


XIE_HE_2024_DISPERSION_HIGH_KH_BAND = (1.2, 2.0)
XIE_HE_2024_DISPERSION_LOW_KH_BAND = (0.1, 0.7)

# Retained for back-compat; do NOT use for new tests
# (kh=1.0 is in the STO/Taylor crossover region where the
# qualitative claim is near-tied).
XIE_HE_2024_DISPERSION_REDUCTION_KH = 1.0  # rad — DEPRECATED anchor


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
