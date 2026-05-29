"""Hand-transcribed paper anchors for Moczo et al. (2002), Muir (1992),
and Koene-Wittsten-Robertsson (2022).

Primary paper:
    Moczo, P., Kristek, J., Vavryčuk, V., Archuleta, R. J., & Halada, L.
    (2002). "3D heterogeneous staggered-grid finite-difference modeling
    of seismic motion with volume harmonic and arithmetic averaging of
    elastic moduli and densities." *BSSA* 92(8): 3042-3066.
    DOI: 10.1785/0120010167.

Co-anchors:
    Muir et al. (1992) — Schoenberg-Muir calculus foundational paper.
    Koene, Wittsten & Robertsson (2022) — production rotate-partition-
        average-rotate-back form (Eqs 23-27); DOI 10.1093/gji/ggac164.
    Kristek et al. (2017) — axis-aligned orthorhombic limit;
        DOI 10.1093/gji/ggw456.

The constants exposed here are the **load-bearing form rules** of the
Moczo 2002 isotropic-effective averaging plus the Schoenberg-Muir
rotate-partition-average-rotate-back invariants. They are NOT
floating-point tabulated coefficients (the implementation operates on
arbitrary continuous (λ, μ, ρ, f1) inputs), but the algebraic-form
sentinels prevent silent swaps of arithmetic ↔ harmonic averages or
swaps of the rotation-frame convention.
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Moczo 2002 isotropic-effective averaging (axis-aligned)
# ──────────────────────────────────────────────────────────────────


# The Moczo 2002 prescription combines density and moduli with
# DIFFERENT averaging schemes:
#   ρ_eff = f1 * ρ1 + f2 * ρ2                          (arithmetic)
#   1/K_eff = f1 / K1 + f2 / K2                        (harmonic)
#   1/μ_eff = f1 / μ1 + f2 / μ2                        (harmonic)
# where K = λ + μ (the in-plane Voigt bulk surrogate) and f2 = 1-f1.
#
# Silent-swap signatures (DO NOT silently swap any of these):
#   arithmetic → harmonic for ρ      ⇒ wrong large-density limit
#   harmonic   → arithmetic for K, μ ⇒ Voigt-Reuss / upper-vs-lower
#                                       bound confusion
MOCZO_2002_DENSITY_AVERAGE = "arithmetic"
MOCZO_2002_K_AVERAGE = "harmonic"
MOCZO_2002_MU_AVERAGE = "harmonic"


def moczo_2002_rho_anchor(rho1: float, rho2: float, f1: float) -> float:
    """Arithmetic density average (Moczo 2002 BSSA, Eq for ρ_eff)."""
    return f1 * rho1 + (1.0 - f1) * rho2


def moczo_2002_K_anchor(K1: float, K2: float, f1: float) -> float:
    """Harmonic K = λ+μ average. Silent-swap to arithmetic would
    move K_eff to the Voigt upper bound; the harmonic form is the
    Reuss lower bound — for layered media with normal incidence
    the Reuss is exact."""
    return 1.0 / (f1 / K1 + (1.0 - f1) / K2)


def moczo_2002_mu_anchor(mu1: float, mu2: float, f1: float) -> float:
    """Harmonic μ average."""
    return 1.0 / (f1 / mu1 + (1.0 - f1) / mu2)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Fluid-limit sentinels
# ──────────────────────────────────────────────────────────────────


# When both layers are fluid (μ1 = μ2 = 0), the harmonic μ_eff is
# 1 / (∞ + ∞) = 0 (compliance dominates → zero stiffness). The
# production code implements this via a `both_fluid` mask and a
# `1e-30` regulariser for the non-both-fluid mixed case.
MOCZO_2002_BOTH_FLUID_MU_EFF = 0.0
MOCZO_2002_FLUID_LIMIT_C66_EFF = 0.0  # μ = 0 ⇒ no shear-stiffness
MOCZO_2002_REGULARISER_MU_EPS = 1e-30  # numerical guard for 1/μ at μ=0


# ──────────────────────────────────────────────────────────────────
# Anchor 3: In-plane Voigt convention for the 2D effective medium
# ──────────────────────────────────────────────────────────────────


# The output Cij dict uses the (xx, zz, xz) 2D in-plane Voigt
# convention (3×3 matrix). The isotropic-effective medium has
# C16 = C26 = 0 (no off-diagonal shear-coupling).
MOCZO_2002_OUTPUT_VOIGT_BASIS = "(xx, zz, xz)"
MOCZO_2002_ISOTROPIC_C16_C26 = 0.0  # off-diagonal couplings vanish


def moczo_2002_c11_anchor(K_eff: float, mu_eff: float) -> float:
    """C11 = K + μ (in-plane Voigt convention; K = λ + μ)."""
    return K_eff + mu_eff


def moczo_2002_c12_anchor(K_eff: float, mu_eff: float) -> float:
    """C12 = K − μ (in-plane Voigt convention; K = λ + μ).
    Reduces to λ at C12 = K_eff − μ_eff = λ_eff."""
    return K_eff - mu_eff


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Schoenberg-Muir orientation-aware extension (Muir 1992 /
#           Koene-Robertsson 2022 Eq 23-27)
# ──────────────────────────────────────────────────────────────────


# The Schoenberg-Muir calculus generalises the Moczo 2002 axis-aligned
# form to arbitrary interface orientation θ via:
#     C_eff = R(θ)^T · M^{-1}[ f1 · M(C1) + f2 · M(C2) ] · R(θ)
# where R(θ) is the 2D Bond rotation in the (xx, zz, xz) Voigt basis
# and M is the Schoenberg-Muir partitioning into in-plane and out-of-
# plane sub-blocks.
#
# Reduction limits (load-bearing — verify in tests):
#   (β = 0, axis-aligned interface) ⇒ recovers Moczo 2002
#                                      isotropic-effective form
#   (f1 = 0 or f1 = 1) ⇒ recovers the single-phase Cij
#   (C1 = C2) ⇒ recovers C1 (idempotent under identical inputs)
SCHOENBERG_MUIR_AXIS_ALIGNED_REDUCES_TO_MOCZO_2002 = True
SCHOENBERG_MUIR_SINGLE_PHASE_LIMITS = (
    "f1=0 ⇒ C2; f1=1 ⇒ C1; C1=C2 ⇒ C1"
)
SCHOENBERG_MUIR_ROTATION_BASIS = "(xx, zz, xz) Voigt"


# Bond rotation matrix R(θ) form (Carcione 2014 Eq 1.59):
#     R = [[ c²,  s²,  2cs],
#          [ s²,  c², -2cs],
#          [-cs,  cs,  c²-s²]]
# where c = cos θ, s = sin θ. Identity at θ = 0.
BOND_ROTATION_2D_FORM_NAME = "Carcione 2014 Eq 1.59"


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Asymptotic / consistency checks
# ──────────────────────────────────────────────────────────────────


# Identical-medium limit: C1 = C2 ⇒ C_eff = C1.
# Single-phase limit: f1 = 1 ⇒ C_eff = C1; f1 = 0 ⇒ C_eff = C2.
# These are sentinel-tested in test_paper_tables.py for both
# `moczo2002_average` and the Schoenberg-Muir scalar/grid forms.
IDENTICAL_MEDIUM_RECOVERS_INPUT = True
SINGLE_PHASE_RECOVERS_INPUT = True
