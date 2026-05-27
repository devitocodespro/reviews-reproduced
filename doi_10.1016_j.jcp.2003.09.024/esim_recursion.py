"""Lombard-Piraux 2004 — paper-faithful recursive C_i^k / L_i^k builder.

DOI 10.1016/j.jcp.2003.09.024 · HAL hal-00004813

This module implements LP04 §3.1 Eq 13-18 recursion that derives
higher-order C_i^k, L_i^k jump-condition matrices by:
  (a) differentiating the order-0 condition `∂t (M_{k-1} U_{k-1}) = 0`
      and substituting the conservation law ∂t U = -A ∂x U - B ∂y U,
  (b) differentiating along the interface tangent
      `∂τ (M_{k-1} U_{k-1}) = 0`.

The two new row blocks at level k stack onto the existing M_{k-1}
condition to give the order-k jump-condition matrix M_k:

    M_k = [[ M_{k-1},   0,                  0                  ],
           [ 0,        -M_{k-1} A_lifted,  -M_{k-1} B_lifted   ],
           [ 0,         x' M_{k-1},         y' M_{k-1}         ]]

acting on `U_k = (U_{k-1}, ∂x U_{k-1}, ∂y U_{k-1})`.

For piecewise-flat Γ (constant x', y') the τ derivative of M_i^0 is
zero, but the PDE-substitution rows -M·A_lifted, -M·B_lifted are
NOT zero — they encode the conservation law and are load-bearing for
the O(dx^{k+1}) truncation claim in LP04 §3.5.

This is the paper-faithful replacement for `esim_projector.block_diag_C_k`
and `esim_projector.block_diag_L_k`, which were a non-equivalent
simplification per the prior dual-reviewer Codex DISAGREE verdict
(see `transcription_review/lp04_close_out_codex.txt`).

Provenance
----------
The recursion core (recurse_one_level, recurse_to_order,
_lift_jacobian, flux_jacobians_isotropic, collapse_matrix) is a
vendored byte-copy from the parent repository's
`scripts/tier3_esim_recursion.py` (verified working in the parent's
Tier 3 chain through 8+ commits since 2026-05-25, with static-
projection tests at k ∈ {0, 1, 2} hitting fp64 noise on polynomial
fields + plane-wave h-refinement showing O(h^{k+1}) convergence).

Two REPRODUCTION-SPECIFIC differences from the parent's module:

  (1) `minimal_multi_indices(k)` uses **paper/x-first** ordering
      [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...] to match the
      reproduction's `esim_projector._monomial_indices` and
      `paper_tables.build_G_*` ordering. The parent uses y-first
      ordering which would cause a silent semantic permutation in
      `C/L @ G` per the LP04-R plan-review Codex F1 finding.

  (2) NEW `flux_jacobians_fluid(rho, c)` for the acoustic
      3-state (v_x, v_y, p) conservation law in LP04's
      **extensional-positive** p convention (p_LP04 = -p_physical).
      Per the existing repo sign-fix at `esim_apply.py:159-180`,
      U_B for B_1 fluid cells passes `p_LP04 = σ_xx_physical =
      -p_physical`. The flux Jacobians must use the SAME convention
      or the PDE-substitution rows go in with the wrong sign (Codex
      F2 finding).

The standalone-isolation discipline of `reviews-reproduced/` allows
vendored copies; it forbids importing from the parent. The byte-
copy is documented and the only edits are the two listed above.
"""
from __future__ import annotations

import numpy as np
import sympy as sp


# ─── Material flux Jacobians ──────────────────────────────────────────


def flux_jacobians_isotropic(rho: float, vp: float, vs: float
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Build A, B for the 2D isotropic elastic conservation law
        ∂t U + A ∂x U + B ∂y U = 0,
    with U = (v_1, v_2, σ_11, σ_12, σ_22)^T.

    The PDE:
      ∂t v_1 = (∂x σ_11 + ∂y σ_12) / ρ
      ∂t v_2 = (∂x σ_12 + ∂y σ_22) / ρ
      ∂t σ_11 = (λ + 2μ) ∂x v_1 + λ ∂y v_2
      ∂t σ_12 = μ (∂y v_1 + ∂x v_2)
      ∂t σ_22 = λ ∂x v_1 + (λ + 2μ) ∂y v_2

    In conservation form ∂t U = -A ∂x U - B ∂y U, so A and B carry
    NEGATIVE of the coefficients above.

    Sign convention: standard σ < 0 in compression (the same
    convention paper_tables.C2_zero uses with -y' v_1 + x' v_2 row
    structure).

    Byte-copy from parent's `scripts/tier3_esim_recursion.py`.
    """
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    A = np.zeros((5, 5), dtype=np.float64)
    B = np.zeros((5, 5), dtype=np.float64)
    # Row 0 (v_1): ∂t v_1 = ∂x σ_11 / ρ + ∂y σ_12 / ρ
    A[0, 2] = -1.0 / rho
    B[0, 3] = -1.0 / rho
    # Row 1 (v_2): ∂t v_2 = ∂x σ_12 / ρ + ∂y σ_22 / ρ
    A[1, 3] = -1.0 / rho
    B[1, 4] = -1.0 / rho
    # Row 2 (σ_11): ∂t σ_11 = (λ + 2μ) ∂x v_1 + λ ∂y v_2
    A[2, 0] = -(lam + 2 * mu)
    B[2, 1] = -lam
    # Row 3 (σ_12): ∂t σ_12 = μ (∂y v_1 + ∂x v_2)
    A[3, 1] = -mu
    B[3, 0] = -mu
    # Row 4 (σ_22): ∂t σ_22 = λ ∂x v_1 + (λ + 2μ) ∂y v_2
    A[4, 0] = -lam
    B[4, 1] = -(lam + 2 * mu)
    return A, B


def flux_jacobians_fluid(rho: float, c: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Build A, B for the 2D acoustic conservation law in LP04's
    **extensional-positive p** convention.

        ∂t U + A ∂x U + B ∂y U = 0,    U = (v_1, v_2, p_LP04)^T

    Convention: p_LP04 = -p_physical (extensional positive), matching
    `paper_tables.C1_zero` row 1 entry `(x'² + y'²) p` and the existing
    repo sign-fix at `esim_apply.py:159-180`. In a fluid represented as
    a degenerate solid, p_LP04 = σ_xx_physical = σ_yy_physical.

    Derivation in LP04's convention:
      Physical: ∂t v_x = -(1/ρ) ∂x p_phys,  ∂t p_phys = -ρ c² div(v)
      Substitute p_phys = -p_LP04:
        ∂t v_x = +(1/ρ) ∂x p_LP04         → -A[0,2] = -1/ρ → A[0,2] = -1/ρ
        ∂t v_y = +(1/ρ) ∂y p_LP04         → B[1,2] = -1/ρ
        ∂t p_LP04 = -∂t p_phys
                  = +ρ c² (∂x v_x + ∂y v_y)
                                          → A[2,0] = -ρc², B[2,1] = -ρc²

    Eigenvalues of A: solve det(A - λI) = 0 →
      -λ (λ² - (1/ρ)(ρc²)) = -λ (λ² - c²) = 0 → λ ∈ {0, ±c}
    (The eigenvalue smoke does NOT distinguish convention; the
    PDE-residual test at LP04-R.0.3 is the right discriminator.)

    Parameters
    ----------
    rho : float
        Fluid density (g/cm³ or kg/m³).
    c : float
        Acoustic wave speed (consistent units with rho).

    Returns
    -------
    A, B : (3, 3) ndarray, dtype float64
        Flux Jacobians in LP04's extensional-positive p convention.
    """
    A = np.zeros((3, 3), dtype=np.float64)
    B = np.zeros((3, 3), dtype=np.float64)
    rho_c2 = rho * c * c
    # Row 0 (v_x): ∂t v_x = +(1/ρ) ∂x p_LP04 → A[0, 2] = -1/ρ
    A[0, 2] = -1.0 / rho
    # Row 1 (v_y): ∂t v_y = +(1/ρ) ∂y p_LP04 → B[1, 2] = -1/ρ
    B[1, 2] = -1.0 / rho
    # Row 2 (p_LP04): ∂t p_LP04 = +ρc² (∂x v_x + ∂y v_y)
    A[2, 0] = -rho_c2
    B[2, 1] = -rho_c2
    return A, B


def flux_jacobians_isotropic_sym(rho, vp, vs):
    """SymPy symbolic variant of flux_jacobians_isotropic — useful for
    symbolic-derivation cross-checks. Byte-copy from parent.
    """
    mu = rho * vs**2
    lam = rho * vp**2 - 2 * mu
    A = sp.zeros(5, 5)
    B = sp.zeros(5, 5)
    A[0, 2] = -1 / rho
    B[0, 3] = -1 / rho
    A[1, 3] = -1 / rho
    B[1, 4] = -1 / rho
    A[2, 0] = -(lam + 2 * mu)
    B[2, 1] = -lam
    A[3, 1] = -mu
    B[3, 0] = -mu
    A[4, 0] = -lam
    B[4, 1] = -(lam + 2 * mu)
    return A, B


# ─── Recursion step ────────────────────────────────────────────────────


def _lift_jacobian(A_base, current_state_dim: int, base_state_dim: int):
    """Block-diagonal lift of the base flux Jacobian to act on a
    higher-level U_k vector of size current_state_dim.

    At level k, U_k carries `current_state_dim / base_state_dim` =
    `3^(k-1)` recursive monomial copies of U_0. The conservation law
    applies to each sub-block, so the lifted Jacobian is
    block_diag(A_base, A_base, ...).

    Byte-copy from parent's `scripts/tier3_esim_recursion.py`.
    """
    n_blocks = current_state_dim // base_state_dim
    assert n_blocks * base_state_dim == current_state_dim, \
        f"non-integer block count: {current_state_dim}/{base_state_dim}"
    if isinstance(A_base, sp.MatrixBase):
        return sp.diag(*[A_base for _ in range(n_blocks)])
    return np.kron(np.eye(n_blocks), A_base)


def recurse_one_level(M_prev, A_base, B_base, x_prime, y_prime,
                       base_state_dim: int):
    """Apply LP04 Eq 17-18 to lift M_{k-1} → M_k, acting on
    U_k = (U_{k-1}, ∂x U_{k-1}, ∂y U_{k-1}).

    The three row blocks:
      row1 = M_{k-1} on the (U_{k-1}, 0, 0) sub-vector
      row2 = -M_{k-1} · A_lifted on ∂x sub-vector,
             -M_{k-1} · B_lifted on ∂y sub-vector
             (encodes ∂t(M_{k-1} U_{k-1}) = 0 + PDE substitution)
      row3 = x' M_{k-1} on ∂x sub-vector,
             y' M_{k-1} on ∂y sub-vector
             (encodes ∂τ(M_{k-1} U_{k-1}) = 0)

    Returns M_k of shape (3*m, 3*n) where M_prev is (m, n).

    Byte-copy from parent's `scripts/tier3_esim_recursion.py`. Note
    `base_state_dim` is the SOURCE state dim (3 for fluid, 5 for
    solid), passed explicitly because M_prev's column count is
    `3^k * base_state_dim`.
    """
    is_sym = isinstance(M_prev, sp.MatrixBase)
    m, n = M_prev.shape
    A_lifted = _lift_jacobian(A_base, n, base_state_dim)
    B_lifted = _lift_jacobian(B_base, n, base_state_dim)
    if is_sym:
        Z = sp.zeros(m, n)
        row1 = sp.Matrix.hstack(M_prev, Z, Z)
        row2 = sp.Matrix.hstack(Z, -M_prev * A_lifted, -M_prev * B_lifted)
        row3 = sp.Matrix.hstack(Z, x_prime * M_prev, y_prime * M_prev)
        return sp.Matrix.vstack(row1, row2, row3)
    Z = np.zeros_like(M_prev)
    row1 = np.hstack([M_prev, Z, Z])
    row2 = np.hstack([Z, -M_prev @ A_lifted, -M_prev @ B_lifted])
    row3 = np.hstack([Z, x_prime * M_prev, y_prime * M_prev])
    return np.vstack([row1, row2, row3])


def recurse_to_order(M_zero, A_base, B_base, x_prime, y_prime, k: int,
                      base_state_dim: int):
    """Iterate `recurse_one_level` from k=0 up to the target k.

    Byte-copy from parent's `scripts/tier3_esim_recursion.py`.
    """
    M = M_zero
    for _ in range(k):
        M = recurse_one_level(M, A_base, B_base, x_prime, y_prime,
                              base_state_dim=base_state_dim)
    return M


# ─── Minimal-basis collapse ────────────────────────────────────────────


def recursive_multi_indices(k: int) -> list[tuple[int, int]]:
    """Multi-index (a, b) at each position in the recursive U_k layout.

    Layout: U_k = (U_{k-1}, ∂x U_{k-1}, ∂y U_{k-1}). Each entry is
    a derivative ∂^a ∂^b U_0 with a + b ≤ k. The recursion produces
    3^k entries, with duplicates: e.g., ∂x∂y appears via both
    (∂x then ∂y) and (∂y then ∂x) recursion paths.

    Byte-copy from parent's `scripts/tier3_esim_recursion.py`.
    """
    if k == 0:
        return [(0, 0)]
    prev = recursive_multi_indices(k - 1)
    out = list(prev)
    out += [(a + 1, b) for (a, b) in prev]
    out += [(a, b + 1) for (a, b) in prev]
    return out


def minimal_multi_indices(k: int) -> list[tuple[int, int]]:
    """Unique (a, b) with a + b ≤ k in **paper/x-first ordering**.

    For k=2: [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)].

    **REPRODUCTION-SPECIFIC**: this ordering matches the
    reproduction's `esim_projector._monomial_indices` and
    `paper_tables.build_G_*` conventions. The parent's
    `scripts/tier3_esim_recursion.py` uses **y-first** ordering
    instead — using the parent's version here would cause a silent
    semantic permutation in `C/L @ G` per the LP04-R plan-review
    Codex F1 finding.

    Length (k+1)(k+2)/2.
    """
    out: list[tuple[int, int]] = []
    for gamma in range(k + 1):
        for j in range(gamma + 1):
            bx = gamma - j
            by = j
            out.append((bx, by))
    return out


def collapse_matrix(k: int, state_components: int) -> np.ndarray:
    """T such that U_recursive = T · U_minimal.

    Shape: (state_components · 3^k, state_components · (k+1)(k+2)/2).
    Used to RIGHT-multiply a recursive-layout matrix M_rec to get
    M_min = M_rec · T acting on the minimal U_min vector.

    Without this collapse, cond(M_min^T M_min) at k=2 hits ~1e21
    (numerical garbage from the ∂x∂y vs ∂y∂x duplicate columns);
    with it, ~1e5 (well-conditioned).

    **Uses the reproduction's paper/x-first minimal_multi_indices**
    so output columns align with `paper_tables.build_G_*` and
    `esim_projector._monomial_indices`.
    """
    rec = recursive_multi_indices(k)
    mini = minimal_multi_indices(k)
    mini_lookup = {ab: idx for idx, ab in enumerate(mini)}
    n_rec = len(rec)
    n_min = len(mini)
    T_scalar = np.zeros((n_rec, n_min))
    for i, ab in enumerate(rec):
        j = mini_lookup[ab]
        T_scalar[i, j] = 1.0
    return np.kron(T_scalar, np.eye(state_components))


def minimal_taylor_weights(dxo: float, dyo: float, k: int) -> np.ndarray:
    """Π Taylor weights in the MINIMAL basis (k+1)(k+2)/2 entries.

    Order matches `minimal_multi_indices(k)`:
        weight[(a,b)] = (dxo)^a (dyo)^b / (a! b!)

    Mirrors `esim_projector.taylor_row` to ensure consistency.
    """
    from math import factorial
    return np.array([
        (dxo ** a) * (dyo ** b) / (factorial(a) * factorial(b))
        for (a, b) in minimal_multi_indices(k)
    ])


# ─── Composite builder for the projector integration ──────────────────


def build_jump_matrix_k_minimal(M_zero: np.ndarray, A_base: np.ndarray,
                                  B_base: np.ndarray, x_prime: float,
                                  y_prime: float, k: int,
                                  state_components: int) -> np.ndarray:
    """Build the order-k jump-condition matrix M^k acting on the
    MINIMAL Taylor basis U^k_min = (U_0, ∂x U_0, ∂y U_0, ∂x² U_0, ...).

    Composes recurse_to_order + right-multiply by collapse_matrix(k).

    This is the user-facing function called by `build_projector` in
    place of the old `block_diag_C_k` / `block_diag_L_k`.

    Parameters
    ----------
    M_zero : (m, state_components) ndarray
        Order-0 jump-condition matrix (C_i^0 or L_i^0) acting on U_0.
    A_base, B_base : (state_components, state_components) ndarray
        Flux Jacobians of the SAME side (fluid → fluid Jacobians,
        solid → solid Jacobians).
    x_prime, y_prime : float
        Interface tangent components (unit vector for piecewise-flat
        Γ; constants if dipping straight line).
    k : int
        Target Taylor order.
    state_components : int
        3 for fluid (v_x, v_y, p_LP04); 5 for solid.

    Returns
    -------
    M_k_min : (m * (1 + 2*k_blocks), state_components * (k+1)(k+2)/2) ndarray
        Where the row count is determined by the recursion (M_0 rows
        × (3^k - related count); see recurse_one_level).

    Notes
    -----
    The recursion produces M_k of shape `(m * sum_{i=0}^{k} 3^i / 3,
    state_components * 3^k)` ... actually:
      shape M_0 = (m, n)
      shape M_1 = (3*m, 3*n)
      shape M_2 = (9*m, 9*n) before collapse, (9*m, 6*n) after collapse
    So output is `(3^k * m, state_components * (k+1)(k+2)/2)`.
    """
    M_k_rec = recurse_to_order(M_zero, A_base, B_base, x_prime, y_prime,
                                k, base_state_dim=state_components)
    if k == 0:
        return M_k_rec
    T = collapse_matrix(k, state_components)
    return M_k_rec @ T
