"""Lombard-Piraux 2004 — byte-transcribed paper constants.

DOI 10.1016/j.jcp.2003.09.024 · HAL hal-00004813

All entries below come from the paper PDF via the side-by-side
user-confirmed transcription protocol (see
`feedback_transcription_workflow_arxiv_then_side_by_side` memory
entry). HAL deposit has PDF only, no LaTeX source.

Transcription provenance log (oldest → newest):

| Date       | Entry          | Source           | Status                       |
|------------|----------------|------------------|------------------------------|
| 2026-05-27 | Eq 10, 11      | paper p. 8       | ✓ user-confirmed (side-by-side v3) |
| 2026-05-27 | Eq 20-24       | paper p. 11      | ✓ user-confirmed (side-by-side v2; α₁/α₂ bracket fix) |
| 2026-05-27 | Eq 26-31       | paper p. 12      | ✓ user-confirmed (side-by-side v1) |
| 2026-05-27 | Eq 32-37       | paper pp. 13-14  | ✓ user-confirmed (stitched side-by-side v1) |
| 2026-05-27 | Eq 38-42       | paper p. 15      | ✓ user-confirmed (side-by-side v1; master ESIM) |
| 2026-05-27 | Eq 43-44       | paper p. 16      | ✓ user-confirmed (side-by-side v1; time-stepping injection) |
| 2026-05-27 | Appendix A.1-2 | paper pp. 30-31  | ✓ user-confirmed (side-by-side v1; G_i^k pseudocode) |
"""
from __future__ import annotations

import sympy as sp


# ─── Eq 10 — fluid-solid jump conditions ─────────────────────────────
#
# [v·n] = 0,    -p n = σ n
#
# where:
#   v = velocity, p = fluid pressure, σ = solid stress tensor
#   t = (x', y'), n = (-y', x') along Γ(τ)
#
# These are continuity-of-normal-velocity and continuity-of-normal-
# traction conditions across the fluid-solid interface. Tangential
# stress on the solid side vanishes (perfect-fluid contact).


# ─── Eq 11 — order-0 jump-condition matrices ─────────────────────────

# Parametrisation symbols
x_prime, y_prime = sp.symbols("x' y'", real=True)


def C1_zero():
    """Fluid-side order-0 coupling matrix C₁⁰ (2 × 3).

    Couples the 3-component fluid state U₁ = (v₁, v₂, p)ᵀ via:
      Row 0: -y' v₁ + x' v₂        = v·n      (normal velocity)
      Row 1:        (x'² + y'²) p              (pressure, scaled)
    """
    return sp.Matrix([
        [-y_prime, x_prime, 0],
        [0,        0,       x_prime**2 + y_prime**2],
    ])


def C2_zero():
    """Solid-side order-0 coupling matrix C₂⁰ (2 × 5).

    Couples the 5-component solid state U₂ = (v₁, v₂, σ₁₁, σ₁₂, σ₂₂)ᵀ via:
      Row 0: -y' v₁ + x' v₂                                  = v·n
      Row 1: y'² σ₁₁ - 2 x' y' σ₁₂ + x'² σ₂₂                 = nᵀ σ n |t|²
    """
    return sp.Matrix([
        [-y_prime, x_prime, 0,            0,             0           ],
        [0,        0,       y_prime**2,  -2*x_prime*y_prime, x_prime**2],
    ])


def L1_zero():
    """Fluid-side boundary-condition row L₁⁰ (1 × 3, empty).

    The perfect fluid has no boundary condition beyond the
    jump condition (Eq 10); L₁⁰ is the zero row.
    """
    return sp.Matrix([[0, 0, 0]])


def L2_zero():
    """Solid-side boundary-condition row L₂⁰ (1 × 5).

    Vanishing tangential shear stress tᵀ σ n = 0:
      x' y' σ₁₁ + (y'² - x'²) σ₁₂ - x' y' σ₂₂ = 0
    """
    return sp.Matrix([[0, 0, x_prime*y_prime,
                       y_prime**2 - x_prime**2, -x_prime*y_prime]])


# ─── Eq 20-24 — compatibility conditions ─────────────────────────────
#
# Fluid side Ω_i (Eq 20-21): vorticity is null outside Γ, so
#     ∂v₁/∂y − ∂v₂/∂x = 0                                    (20)
# Differentiating (20) (k−1) times in x and y:
#     ∂^k v₁ / (∂x^(k-j-1) ∂y^(j+1))
#         − ∂^k v₂ / (∂x^(k-j) ∂y^j)  = 0,
#                                k ≥ 1, j = 0, …, k−1.       (21)
#
# Solid side Ω_i (Eq 22-24): a necessary and sufficient condition for
# the stress tensor σ symmetry uses two elastic-coefficient combinations
#     α₁ = c_p² / [4 (c_p² − c_s²)]                          (22a)
#     α₂ = (2 c_s² − c_p²) / [4 (c_p² − c_s²)]               (22b)
# (note: denominator factor 4 multiplies the difference; transcription
#  trap caught by user 2026-05-27.)
# yielding the zeroth-order condition
#     α₂ σ₁₁,xx + α₁ σ₂₂,xx − σ₁₂,xy + α₁ σ₁₁,yy + α₂ σ₂₂,yy = 0  (23)
# and its (k−2)-times differentiated form (24) over the same (k,j)
# index set as (21).
#
# Role: (21) and (24) hold at every point of Ω_i, hence at P^+ and P^−
# near Γ. Used in Eq 25 to reduce U_i^k to V_i^k via U_i^k = G_i^k V_i^k.
# Fluid V_i^k: (k+1)(k+3) components.  Solid V_i^k: 2k² + 8k + 5
# components. Full G_i^k matrices are tabulated in Appendix A.
#
# For the dipping-line Petrobras case Γ is a single straight line ⇒
# α₁, α₂ are global constants; G_i^k collapses to a one-shot
# construction (no per-point lookup).


def alpha_coefficients(c_p: float, c_s: float) -> tuple[float, float]:
    """Compute α₁, α₂ per Eq 22 for the solid stress-symmetry condition.

    Parameters
    ----------
    c_p : P-wave speed in Ω_solid.
    c_s : S-wave speed in Ω_solid.

    Returns
    -------
    (α₁, α₂) such that the σ-symmetry condition (23) holds.

    Raises
    ------
    ValueError if c_p² = c_s² (degenerate; would divide by zero).
    """
    denom = 4.0 * (c_p**2 - c_s**2)
    if denom == 0.0:
        raise ValueError(
            f"Degenerate solid: c_p² = c_s² (c_p={c_p}, c_s={c_s}); "
            "Eq 22 denominator vanishes."
        )
    alpha_1 = c_p**2 / denom
    alpha_2 = (2.0 * c_s**2 - c_p**2) / denom
    return alpha_1, alpha_2


# ─── Eq 26-31 — interface coupling matrices ───────────────────────────
#
# Setup: each side Ω_i carries state U_i^k. Compatibility (Eq 25)
# reduces to V_i^k via U_i^k = G_i^k V_i^k. Boundary conditions
# (Eq 26) reduce further:
#     L_i^k G_i^k V_i^k = 0  ⇒  V_i^k = K_i^k W_i^k    (i = 1, 2)
# K_i^k is built from the SVD of L_i^k G_i^k (kernel basis).
#
# Jump conditions in V form (insert 25 into 19):
#     C_1^k G_1^k V_1^k = C_2^k G_2^k V_2^k             (27)
# Jump conditions in W form (insert 26):
#     C_1^k G_1^k K_1^k W_1^k = C_2^k G_2^k K_2^k W_2^k (28)
#
# Compact coupling matrix:
#     S_i^k = C_i^k G_i^k K_i^k    (i = 1, 2)           (29)
# yielding
#     S_1^k W_1^k = S_2^k W_2^k                         (30)
#
# SVD resolution of the under-determined system (30):
#     W_2^k = S_2^k^{-1} [S_1^k | R_{S_2}^k] (W_1^k, Λ^k)^T   (31)
# where S_2^k^{-1} is the SVD pseudoinverse and R_{S_2}^k spans
# the null-space of S_2^k. Λ^k = free parameters.
#
# Role: K_i^k, S_i^k, R_{S_2}^k are pre-computed once per interface
# point from C_i^k, L_i^k, G_i^k. For the dipping-line Petrobras
# case (uniform n and t along Γ) these matrices are GLOBAL CONSTANTS
# — one SVD per order k, not per irregular point. Eq 31 feeds
# directly into Eq 37 in §3.5 (Taylor expansion on the solid side).


# ─── Eq 32-37 — ESIM Taylor setup ─────────────────────────────────────
#
# Setup (Fig 3 of the paper): let M(x_I, y_J) ∈ Ω_2 be an irregular
# grid point — one whose stencil reaches across Γ. Let P be the
# orthogonal projection of M onto Γ. Let B be the set of grid points
# enclosed in the disk of radius q centred on P, partitioned
# B = B_1 ∪ B_2 by membership in Ω_1 vs Ω_2.
#
# Eq 32 — k-th order Taylor row vector at (x_i, y_j) about P:
#   Π_{i,j}^k = (1, (x_i - x_P), (y_j - y_P), …, (y_j - y_P)^k / k!)
#
# Eq 33 — modified value at the irregular point:
#   U*(x_I, y_J, t^n) = Π_{I,J}^k U_1^k
#
# Eq 34 — Taylor expansion at (i, j) ∈ B_1 using Eq 25, 26:
#   U(x_i, y_j, t^n)
#       = Π_{i,j}^k G_1^k K_1^k W_1^k + O(Δx^(k+1))
#
# Eq 35 — express via the full (W_1^k, Λ^k) vector:
#   U(x_i, y_j, t^n)
#       = Π_{i,j}^k G_1^k K_1^k [I | 0] (W_1^k, Λ^k)^T
#       + O(Δx^(k+1))
#
# Eq 36 — Taylor expansion at (i, j) ∈ B_2:
#   U(x_i, y_j, t^n)
#       = Π_{i,j}^k G_2^k K_2^k W_2^k + O(Δx^(k+1))
#
# Eq 37 — substitute Eq 31 to eliminate W_2^k:
#   U(x_i, y_j, t^n)
#       = Π_{i,j}^k G_2^k K_2^k S_2^k^{-1} [S_1^k | R_{S_2}^k]
#         (W_1^k, Λ^k)^T  +  O(Δx^(k+1))
#
# Role: Eq 34-37 give a row of the over-determined least-squares
# system (Eq 38) that recovers W_1^k from the numerical values of
# the solution on the disk B. Each row is a row of M; the rows on
# B_1 use Eq 35, the rows on B_2 use Eq 37.


# ─── Eq 38-42 — master ESIM equation ──────────────────────────────────
#
# Eq 38: stack the per-row Taylor expansions Eq 35 (rows on B_1)
# and Eq 37 (rows on B_2) into one over-determined system,
#     (U^n)_B = M · (W_1^k, Λ^k)^T + O(Δx^(k+1))
# The disk radius q is chosen so that |B| > dim(W_1^k) + dim(Λ^k).
#
# Eq 39: solve via least-squares pseudoinverse (normal equations or
# SVD), drop Taylor remainders, identify exact ↔ numerical values:
#     (W_1^k, Λ^k)^T = M^{-1} (U^n)_B
#
# Eq 40: only W_1^k is of interest; let M̄^{-1} denote the restriction
# of M^{-1} to the W_1^k rows (dropping the Λ^k rows):
#     W_1^k = M̄^{-1} (U^n)_B
#
# Eq 41: modified value at irregular point (I, J) ∈ Ω_2 (Eq 25, 26, 33):
#     U*_{I,J} = Π_{I,J}^k U_1^k = Π_{I,J}^k G_1^k K_1^k W_1^k
#
# Eq 42 (★ MASTER ESIM EQUATION): insert Eq 40 into Eq 41,
#     U*_{I,J} = Π_{I,J}^k · G_1^k · K_1^k · M̄^{-1} · (U^n)_B
#
# For an irregular point on the Ω_1 side, swap G_1^k → G_2^k and
# K_1^k → K_2^k. For a fluid-solid interface (Ω_1 fluid, Ω_2 solid),
# the (U^n)_B vector mixes the 3-component fluid solution and the
# 5-component solid solution.
#
# IMPLEMENTATION NOTES for the dipping-line Petrobras case:
#  - All four matrices Π_{I,J}^k, G_i^k, K_i^k, M̄^{-1} depend ONLY
#    on relative positions (x_i - x_P, y_j - y_P) and on the
#    (constant along Γ) normal/tangent direction. PRE-COMPUTE
#    ONCE per irregular point during set-up.
#  - Per-timestep cost (Eq 42): one mat-vec product
#    P := M̄^{-1} (U^n)_B, then mat-vec G_1^k K_1^k · P, then a
#    dot-product Π_{I,J}^k · result. Per paper §3.9: ~1% overhead
#    at 400×400.
#  - For k=2 on a fluid-solid interface, the paper's canonical
#    radius is q ≈ 3.5 Δx (~40 neighbours so Eq 38 is comfortably
#    over-determined).


# ─── Eq 43-44 — time-stepping at irregular points ─────────────────────
#
# Once all modified values U*_{I, J} are computed at t^n via Eq 42,
# update each irregular point using the SAME discrete operator H as
# at regular points (Eq 12 in the paper) but with substituted neighbour
# values. Denote Ω(i, j) the medium to which (i, j) belongs.
#
# Eq 43 — per-stencil-leg substitution rule (for each (α, β) in stencil):
#     Ω(i+α, j+β) = Ω(i, j)   ⇒  Ũ_{i+α, j+β} = U^n_{i+α, j+β}
#     Ω(i+α, j+β) ≠ Ω(i, j)   ⇒  Ũ_{i+α, j+β} = U*_{i+α, j+β}
#
# Eq 44 — modified update for an irregular point (i, j):
#     U^(n+1)_{i, j} = H(Ũ_{i+α, j+β})
#
# Key paper-stated properties:
#   - Scheme-agnostic: coupling ESIM with a wide class of schemes
#     is automatic; no modification of H is required.
#   - Sub-cell resolution: interface conditions (Eq 19) use x', y'
#     and derivatives at P up to k-th order, and the Taylor
#     expansions (Eq 35, 37) introduce sub-cell resolution about
#     the position of P inside the meshing.
#
# IMPLEMENTATION PATTERN for Devito (dipping-line Petrobras):
#   1. Identify irregular points (one-shot SDF gate; already done
#      in scripts/it_e_tier1_geometry.py).
#   2. Precompute per-irregular-point row vector
#        N_{I, J} := Π_{I, J}^k · G_1^k · K_1^k · M̄^{-1}
#      Because n, t, α_1, α_2 are constant along the dipping line,
#      the matrices Π, G, K, M̄^{-1} depend only on the integer
#      cell offset of B relative to P.
#   3. Each timestep:
#      a. Gather (U^n)_B from the disk B around each P on the grid.
#      b. Compute U*_{I, J} = N_{I, J} · (U^n)_B.
#      c. Apply H at each irregular point using the Ũ substitution
#         rule (Eq 43).
#
# Note on sides: for an irregular point on the Ω_1 side, the master
# row vector uses G_2^k K_2^k instead of G_1^k K_1^k (Eq 42 swap).
# Both sides have their own irregular-point lists and own
# precomputed N row vectors.


# ─── Appendix A — G_i^k construction ──────────────────────────────────
#
# G_i^k encodes the compatibility reduction U_i^k = G_i^k V_i^k (Eq 25).
# Rows index the (5 × number of derivative monomials at order ≤ k)
# components of U_i^k. Cols index the smaller independent sub-vector
# V_i^k.
#
# Dimensions (paper §3.2):
#   Fluid: dim(V_i^k) = (k+1)(k+3); dim(U_i^k) = 3 × (k+1)(k+2)/2
#   Solid: dim(V_i^k) = 2k² + 8k + 5; dim(U_i^k) = 5 × (k+1)(k+2)/2


def build_G_fluid(k: int):
    """Build G_i^k for a fluid side (Appendix A.1).

    Returns a sympy.SparseMatrix of shape (dim(U_i^k), dim(V_i^k)) with
    a 1 entry per assignment in the paper's pseudocode.
    """
    rows_U = 3 * (k + 1) * (k + 2) // 2  # 3 fluid components × #monomials
    cols_V = (k + 1) * (k + 3)
    G = sp.zeros(rows_U, cols_V)
    alpha = 0
    beta = 0
    for gamma in range(0, k + 1):
        for _eps in range(1, 4):  # ε = 1..3 (3 fluid components)
            alpha += 1
            beta += 1
            G[alpha - 1, beta - 1] = 1
        for _delta in range(1, gamma + 1):
            alpha += 1
            beta -= 1
            G[alpha - 1, beta - 1] = 1
            alpha += 1
            beta += 2
            G[alpha - 1, beta - 1] = 1
            alpha += 1
            beta += 1
            G[alpha - 1, beta - 1] = 1
    assert alpha == rows_U, f"fluid α-count mismatch: {alpha} vs {rows_U} for k={k}"
    assert beta == cols_V, f"fluid β-count mismatch: {beta} vs {cols_V} for k={k}"
    return G


def build_G_solid(k: int, alpha_1=None, alpha_2=None):
    """Build G_i^k for a solid side (Appendix A.2).

    Returns a sympy.SparseMatrix of shape (dim(U_i^k), dim(V_i^k)).
    The α₁/α₂ entries from Eq 22 enter at the rows corresponding to
    the σ-symmetry compatibility (Eq 24). If `alpha_1` or `alpha_2`
    are None, symbolic ``sp.Symbol('alpha_1')`` and ``sp.Symbol('alpha_2')``
    are used.
    """
    if alpha_1 is None:
        alpha_1 = sp.Symbol('alpha_1')
    if alpha_2 is None:
        alpha_2 = sp.Symbol('alpha_2')
    rows_U = 5 * (k + 1) * (k + 2) // 2  # 5 solid components × #monomials
    cols_V = 2 * k * k + 8 * k + 5
    G = sp.zeros(rows_U, cols_V)
    alpha = 0
    beta = 0
    for gamma in range(0, k + 1):
        for delta in range(0, gamma + 1):
            if delta == 0:
                for _eps in range(1, 6):  # ε = 1..5 (5 solid components)
                    alpha += 1
                    beta += 1
                    G[alpha - 1, beta - 1] = 1
            if gamma != 0 and delta != 0 and gamma != delta:
                # Branch coefficients ν, η ∈ {0, 1}:
                if gamma == 2:
                    nu, eta = 0, 0
                elif delta == 1:
                    nu, eta = 0, 1
                elif delta == gamma - 1:
                    nu, eta = 1, 0
                else:
                    nu, eta = 1, 1
                # Three identity rows
                alpha += 1
                beta += 1
                G[alpha - 1, beta - 1] = 1
                alpha += 1
                beta += 1
                G[alpha - 1, beta - 1] = 1
                alpha += 1
                beta += 1
                G[alpha - 1, beta - 1] = 1
                # σ-symmetry row with four α₁/α₂ entries on the same α
                alpha += 1
                beta -= 5 - nu
                G[alpha - 1, beta - 1] = alpha_2
                beta += 2 - nu
                G[alpha - 1, beta - 1] = alpha_1
                beta += 7
                G[alpha - 1, beta - 1] = alpha_1
                beta += 2 - eta
                G[alpha - 1, beta - 1] = alpha_2
                # Closing identity row
                alpha += 1
                beta -= 5 - eta
                G[alpha - 1, beta - 1] = 1
            if gamma != 0 and gamma == delta:
                for _eps in range(1, 6):
                    alpha += 1
                    beta += 1
                    G[alpha - 1, beta - 1] = 1
    assert alpha == rows_U, f"solid α-count mismatch: {alpha} vs {rows_U} for k={k}"
    assert beta == cols_V, f"solid β-count mismatch: {beta} vs {cols_V} for k={k}"
    return G


# ─── Sanity-check helpers ─────────────────────────────────────────────

def verify_G_dimensions():
    """Confirm G_i^k dimensions match the paper's stated formulas
    (fluid: (k+1)(k+3) cols; solid: 2k²+8k+5 cols) for k = 0, 1, 2.
    """
    for k in (0, 1, 2):
        Gf = build_G_fluid(k)
        if Gf.shape != (3 * (k + 1) * (k + 2) // 2, (k + 1) * (k + 3)):
            return False
        Gs = build_G_solid(k)
        if Gs.shape != (5 * (k + 1) * (k + 2) // 2, 2 * k * k + 8 * k + 5):
            return False
    return True


def verify_alpha_sum_identity():
    """Confirm α₁ + α₂ = c_s² / [2(c_p² − c_s²)] (algebraic identity from Eq 22).

    Test point: c_p = 4 km/s, c_s = 2 km/s (typical solid).
    """
    c_p, c_s = 4.0, 2.0
    a1, a2 = alpha_coefficients(c_p, c_s)
    expected = c_s**2 / (2.0 * (c_p**2 - c_s**2))
    return abs((a1 + a2) - expected) < 1e-14


def verify_C2_normal_stress_identity():
    """Confirm C₂⁰ row 2 corresponds to nᵀ σ n × (x'² + y'²).

    The jump condition -p n = σ n implies n·σ·n = -p, which (after
    multiplying by t² = (x'² + y'²)) gives the C₂⁰ row 2 formula.
    """
    n = sp.Matrix([-y_prime, x_prime])
    sigma = sp.Matrix([[sp.Symbol('sigma_11'), sp.Symbol('sigma_12')],
                       [sp.Symbol('sigma_12'), sp.Symbol('sigma_22')]])
    nT_sigma_n = (n.T * sigma * n)[0]
    # Compare with C₂⁰ row 2 dot (0, 0, σ₁₁, σ₁₂, σ₂₂):
    U2 = sp.Matrix([0, 0, sp.Symbol('sigma_11'),
                    sp.Symbol('sigma_12'), sp.Symbol('sigma_22')])
    c2_row2_dot_U2 = (C2_zero().row(1) * U2)[0]
    diff = sp.simplify(c2_row2_dot_U2 - nT_sigma_n)
    return diff == 0


def verify_L2_tangential_shear_identity():
    """Confirm L₂⁰ U₂ = ±tᵀ σ n (vanishing tangential shear, sign-free).

    The paper writes L₂⁰ = (0, 0, x'y', y'² - x'², -x'y') with a sign
    convention such that L₂⁰·U₂ = -tᵀσn. Since L₂⁰·U₂ = 0 is the
    boundary-condition equation, the overall sign is physically
    irrelevant — both ±tᵀσn = 0 enforce the same vanishing-tangential-
    shear condition. The verification accepts either sign.
    """
    t_vec = sp.Matrix([x_prime, y_prime])
    n = sp.Matrix([-y_prime, x_prime])
    sigma = sp.Matrix([[sp.Symbol('sigma_11'), sp.Symbol('sigma_12')],
                       [sp.Symbol('sigma_12'), sp.Symbol('sigma_22')]])
    tT_sigma_n = (t_vec.T * sigma * n)[0]
    U2 = sp.Matrix([0, 0, sp.Symbol('sigma_11'),
                    sp.Symbol('sigma_12'), sp.Symbol('sigma_22')])
    l2_dot_U2 = (L2_zero() * U2)[0]
    diff_plus = sp.simplify(l2_dot_U2 - tT_sigma_n)
    diff_minus = sp.simplify(l2_dot_U2 + tT_sigma_n)
    return (diff_plus == 0) or (diff_minus == 0)


if __name__ == '__main__':
    print('Lombard-Piraux 2004 — Eq 10/11 transcribed matrices:')
    print('C₁⁰ =')
    sp.pprint(C1_zero())
    print('\nC₂⁰ =')
    sp.pprint(C2_zero())
    print('\nL₁⁰ =')
    sp.pprint(L1_zero())
    print('\nL₂⁰ =')
    sp.pprint(L2_zero())
    print()
    print('Self-consistency:')
    print(f'  C₂⁰ row 2 ↔ nᵀσn × |t|²:                    '
          f'{verify_C2_normal_stress_identity()}')
    print(f'  L₂⁰ ↔ tᵀσn (sign-free):                     '
          f'{verify_L2_tangential_shear_identity()}')
    print(f'  α₁ + α₂ = c_s² / [2(c_p²−c_s²)]:           '
          f'{verify_alpha_sum_identity()}')
    print(f'  G_i^k dimensions (fluid + solid, k=0,1,2):  '
          f'{verify_G_dimensions()}')
    print()
    print('Eq 22 (solid, c_p=4, c_s=2 km/s):')
    a1, a2 = alpha_coefficients(4.0, 2.0)
    print(f'  α₁ = {a1:.10f}')
    print(f'  α₂ = {a2:.10f}')
