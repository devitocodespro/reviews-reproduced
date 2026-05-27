"""Lombard-Piraux 2004 — paper-faithful standalone ESIM projector.

Builds the modified-value row vector N_{I,J} per Eq 42:
    U*_{I,J} = Π_{I,J}^k · G_1^k · K_1^k · M̄^{-1} · (U^n)_B

This module is the standalone reproduction's projector. Per the
reviews-reproduced/CLAUDE.md isolation discipline it imports ONLY
the byte-transcribed paper-table primitives from `paper_tables.py`
in the same folder. It must NOT import from the parent repository's
`scripts/tier3_esim_projector.py` (which is the implementation under
test in the cross-validation at LP04-R.2).

Paper-faithful conventions (LP04 §3):
  - Fluid state U_1 has 3 components: (v_1, v_2, p)
  - Solid state U_2 has 5 components: (v_1, v_2, σ_11, σ_12, σ_22)
  - At a fluid-solid interface, the disk B is partitioned B = B_1 ∪
    B_2 by which side each grid cell lies on (SDF sign).
  - For a STRAIGHT interface (constant tangent t along Γ) the matrices
    Π, G, K, S, M̄^{-1} depend only on the integer cell offsets of B
    relative to P. They are precomputed once.

The implementation handles arbitrary k ∈ {0, 1, 2}. Higher k requires
recursive differentiation of the jump conditions per LP04 Eq 17-18;
deferred until k=2 reproduces the paper's Table 2 convergence rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.linalg import svd
from scipy.linalg import pinv as scipy_pinv

import paper_tables as pt
from esim_recursion import (
    build_jump_matrix_k_minimal,
    flux_jacobians_fluid,
    flux_jacobians_isotropic,
)


# ─── helpers ─────────────────────────────────────────────────────────────

def _monomial_indices(k: int) -> list[tuple[int, int]]:
    """Return all (β_x, β_y) with β_x + β_y ≤ k, ordered by total degree
    then by β_x descending.

    For k=2: [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)].
    Length = (k+1)(k+2)/2.
    """
    out: list[tuple[int, int]] = []
    for gamma in range(k + 1):
        for j in range(gamma + 1):
            bx = gamma - j
            by = j
            out.append((bx, by))
    return out


def taylor_row(dx: float, dy: float, k: int) -> np.ndarray:
    """LP04 Eq 32 — row vector Π^k of Taylor monomial weights at offset
    (dx, dy), normalised by β_x! β_y!.

    Π[m] = (dx)^β_x (dy)^β_y / (β_x! β_y!) for m-th monomial.
    Length (k+1)(k+2)/2.
    """
    mono = _monomial_indices(k)
    out = np.zeros(len(mono), dtype=np.float64)
    for m, (bx, by) in enumerate(mono):
        out[m] = (dx ** bx) * (dy ** by) / (factorial(bx) * factorial(by))
    return out


def pi_block_at_offset(dx: float, dy: float, k: int,
                        state_components: int) -> np.ndarray:
    """Π_{i,j}^k as a (state_components, state_components × (k+1)(k+2)/2)
    block matrix.

    For each Taylor monomial m, the corresponding state-component block
    is `taylor[m] * I_{state_components}`. So Π_block · U^k_vector yields
    the full state at offset (dx, dy) by Taylor expansion about P.

    LP04 indexes Π as a scalar row of monomial weights AND tensors with
    identity_{state_components} to act on a multi-component state. This
    function returns the explicit tensor product.
    """
    row = taylor_row(dx, dy, k)
    n_mono = len(row)
    M = np.zeros((state_components, state_components * n_mono),
                   dtype=np.float64)
    for m, w in enumerate(row):
        for c in range(state_components):
            M[c, m * state_components + c] = w
    return M


def block_diag_C_k(C0: np.ndarray, k: int) -> np.ndarray:
    """C_i^k = block_diag(C_i^0, C_i^0, ..., C_i^0) with (k+1)(k+2)/2
    blocks (one per Taylor monomial).

    For a STRAIGHT interface the tangent/normal are constant along Γ, so
    the order-k jump conditions are simply the order-0 conditions applied
    to each derivative monomial. This is the simplification LP04 §3.6
    notes for piecewise-flat Γ.

    Shape: (2 × n_mono, C0.shape[1] × n_mono).
    """
    n_mono = (k + 1) * (k + 2) // 2
    n_rows, n_cols = C0.shape
    out = np.zeros((n_rows * n_mono, n_cols * n_mono), dtype=np.float64)
    for m in range(n_mono):
        out[m * n_rows:(m + 1) * n_rows,
            m * n_cols:(m + 1) * n_cols] = C0
    return out


def block_diag_L_k(L0: np.ndarray, k: int) -> np.ndarray:
    """L_i^k = block_diag(L_i^0, ..., L_i^0) — same construction as C_k.

    For a straight interface, the BC L_i^0 V_i^0 = 0 holds for each
    Taylor derivative independently.

    Shape: (L0.shape[0] × n_mono, L0.shape[1] × n_mono).
    """
    return block_diag_C_k(L0, k)


def K_via_svd(L_k: np.ndarray, V_dim: int) -> np.ndarray:
    """Construct K_i^k as a basis of ker(L_i^k) inside V-space.

    LP04 Eq 26: V_i^k = K_i^k W_i^k. K_i^k spans the null-space of
    L_i^k. Built from the right singular vectors of L_i^k with
    singular value < cutoff.
    """
    if L_k.shape[0] == 0:
        return np.eye(V_dim, dtype=np.float64)
    U, s, Vt = svd(L_k, full_matrices=True)
    tol = max(L_k.shape) * np.finfo(np.float64).eps * (s.max() if s.size else 0.0)
    rank = int((s > tol).sum())
    # Null-space: rows of Vt with index >= rank, transposed to give columns.
    return Vt[rank:, :].T


# ─── main projector ──────────────────────────────────────────────────────

@dataclass
class ProjectorInfo:
    """Diagnostic metadata returned alongside the N row vector."""
    k: int
    side: str                  # 'fluid' or 'solid' — which side U* lives on
    n_state_self: int          # dim of U^0 on the side U* lives
    n_state_other: int         # dim of U^0 on the other side
    n_w1: int                  # dim of W_1^k
    n_lambda: int              # dim of Λ^k
    n_B1: int                  # |B_1|
    n_B2: int                  # |B_2|
    cond_M: float              # condition number of M
    rank_M: int                # numerical rank of M


def _alpha_for_side(side: str, c_p: float, c_s: float
                    ) -> tuple[float, float]:
    if side == 'solid':
        return pt.alpha_coefficients(c_p, c_s)
    # Fluid side: α₁, α₂ unused (G_fluid has no α entries); return zero.
    return 0.0, 0.0


def _build_G_for_side(side: str, k: int, alpha_1: float, alpha_2: float
                       ) -> np.ndarray:
    """Wrap paper_tables.build_G_{fluid,solid} returning a NumPy array.

    `paper_tables.build_G_*` return sympy.Matrix; convert here.
    """
    if side == 'fluid':
        G_sp = pt.build_G_fluid(k)
    else:
        G_sp = pt.build_G_solid(k, alpha_1=alpha_1, alpha_2=alpha_2)
    return np.array(G_sp.tolist(), dtype=np.float64)


def _C0_L0_for_side(side: str, tangent: tuple[float, float]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Substitute the parametrisation symbols (x', y') in the paper's
    C_i⁰, L_i⁰ with numeric values from the interface tangent.

    LP04 §3 parametrises Γ by arc-length τ with t = (x', y') the
    tangent and n = (-y', x') the normal. For our STRAIGHT dipping
    interface, t is the unit tangent vector.
    """
    x_pr, y_pr = float(tangent[0]), float(tangent[1])
    if side == 'fluid':
        C_sp = pt.C1_zero()
        L_sp = pt.L1_zero()
    else:
        C_sp = pt.C2_zero()
        L_sp = pt.L2_zero()
    C0 = np.array(C_sp.subs({pt.x_prime: x_pr, pt.y_prime: y_pr})
                   .tolist(), dtype=np.float64)
    L0 = np.array(L_sp.subs({pt.x_prime: x_pr, pt.y_prime: y_pr})
                   .tolist(), dtype=np.float64)
    return C0, L0


def build_projector(
    target_side: str,   # 'fluid' or 'solid' — side where U* lives
    target_xy: tuple[float, float],  # cell where we want U*
    P_xy: tuple[float, float],       # foot of perpendicular onto Γ
    tangent: tuple[float, float],    # unit tangent of Γ at P
    B1_coords: list[tuple[float, float]],  # disk neighbours in Ω_1 (fluid)
    B2_coords: list[tuple[float, float]],  # disk neighbours in Ω_2 (solid)
    k: int,
    c_p_solid: float,
    c_s_solid: float,
    rho_fluid: float = 1.0,
    c_fluid: float = 1.5,
    rho_solid: float = 2.6,
    ) -> tuple[np.ndarray, ProjectorInfo]:
    """Build the modified-value projector N row matrix per LP04 Eq 42.

    Parameters
    ----------
    target_side
        'fluid' if (I,J) lies in Ω_1 (irregular fluid cell — its
        stencil reaches across Γ into Ω_2); 'solid' if (I,J) lies in
        Ω_2. The projector swaps G_1·K_1 ↔ G_2·K_2 accordingly.
    target_xy
        Cartesian coords of the irregular cell where U* is computed.
    P_xy
        Foot of perpendicular onto Γ.
    tangent
        Unit tangent vector along Γ at P.
    B1_coords, B2_coords
        Physical coordinates (km, km) of the disk-B grid cells on the
        fluid side (B_1 ⊂ Ω_1) and solid side (B_2 ⊂ Ω_2).
    k
        Taylor order (1 or 2 supported; 0 trivial).
    c_p_solid, c_s_solid
        Solid wave speeds — used to compute α_1, α_2 (Eq 22) which
        enter the σ-symmetry compatibility for k ≥ 2.

    Returns
    -------
    N : np.ndarray of shape (state_components_target, len(U_B_vec))
        Row matrix such that  U*_{target} = N · (U^n)_B
        where (U^n)_B is the column vector stacking
        [U^n at B_1 cells (3 fluid components each),
         U^n at B_2 cells (5 solid components each)]
        in the order B_1 first then B_2.
    info : ProjectorInfo
    """
    n_state_fluid = 3
    n_state_solid = 5
    n_mono = (k + 1) * (k + 2) // 2

    alpha_1, alpha_2 = pt.alpha_coefficients(c_p_solid, c_s_solid)

    # --- Order-k C, L, G for both sides ---
    C0_f, L0_f = _C0_L0_for_side('fluid', tangent)
    C0_s, L0_s = _C0_L0_for_side('solid', tangent)

    # LP04 §3.1 Eq 13-18: build C_i^k and L_i^k via the recursive
    # construction (differentiate order-0 in time + along Γ; substitute
    # the PDE for time derivatives). The block-diagonal copy of
    # C_i^0 / L_i^0 that this code formerly used is NOT equivalent to
    # the LP04 recipe even for a piecewise-flat Γ — the PDE-substitution
    # rows -M_{k-1}·A and -M_{k-1}·B are non-zero and load-bearing for
    # the O(dx^{k+1}) truncation claim. Codex DISAGREE 2026-05-27 (see
    # transcription_review/lp04_close_out_codex.txt and lp04_R_plan_review.txt).
    x_pr, y_pr = float(tangent[0]), float(tangent[1])
    A_f, B_f = flux_jacobians_fluid(rho_fluid, c_fluid)
    A_s, B_s = flux_jacobians_isotropic(rho_solid, c_p_solid, c_s_solid)
    C_f_k = build_jump_matrix_k_minimal(C0_f, A_f, B_f, x_pr, y_pr,
                                          k, n_state_fluid)
    C_s_k = build_jump_matrix_k_minimal(C0_s, A_s, B_s, x_pr, y_pr,
                                          k, n_state_solid)
    L_f_k = build_jump_matrix_k_minimal(L0_f, A_f, B_f, x_pr, y_pr,
                                          k, n_state_fluid)
    L_s_k = build_jump_matrix_k_minimal(L0_s, A_s, B_s, x_pr, y_pr,
                                          k, n_state_solid)

    G_f_k = _build_G_for_side('fluid', k, alpha_1, alpha_2)
    G_s_k = _build_G_for_side('solid', k, alpha_1, alpha_2)

    # Dim sanity
    assert G_f_k.shape[0] == n_state_fluid * n_mono, (
        f"G_f shape rows={G_f_k.shape[0]} != "
        f"{n_state_fluid} × {n_mono} = {n_state_fluid * n_mono}")
    assert G_s_k.shape[0] == n_state_solid * n_mono, (
        f"G_s shape rows={G_s_k.shape[0]} != "
        f"{n_state_solid} × {n_mono} = {n_state_solid * n_mono}")
    assert C_f_k.shape[1] == G_f_k.shape[0], (
        f"C_f.cols={C_f_k.shape[1]} != G_f.rows={G_f_k.shape[0]}")
    assert C_s_k.shape[1] == G_s_k.shape[0], (
        f"C_s.cols={C_s_k.shape[1]} != G_s.rows={G_s_k.shape[0]}")

    # --- K via SVD of L·G ---
    LG_f = L_f_k @ G_f_k
    LG_s = L_s_k @ G_s_k
    V_f_dim = G_f_k.shape[1]
    V_s_dim = G_s_k.shape[1]
    K_f = K_via_svd(LG_f, V_f_dim)
    K_s = K_via_svd(LG_s, V_s_dim)
    n_w1 = K_f.shape[1]  # paper's W_1^k dim
    n_w2 = K_s.shape[1]

    # --- S_1 = C_1 G_1 K_1, S_2 = C_2 G_2 K_2 ---
    S_1 = C_f_k @ G_f_k @ K_f       # (2·n_mono, n_w1)
    S_2 = C_s_k @ G_s_k @ K_s       # (2·n_mono, n_w2)

    # SVD-resolve W_2 = S_2_pinv [S_1 | R_{S_2}] (W_1, Λ)
    U_s2, sig_s2, Vt_s2 = svd(S_2, full_matrices=True)
    tol_s2 = max(S_2.shape) * np.finfo(np.float64).eps * (
        sig_s2.max() if sig_s2.size else 0.0)
    rank_s2 = int((sig_s2 > tol_s2).sum())
    S_2_pinv = scipy_pinv(S_2, rtol=1e-10)
    R_S2 = Vt_s2[rank_s2:, :].T     # (n_w2, n_lambda)
    n_lambda = R_S2.shape[1]

    A_W1 = S_2_pinv @ S_1           # (n_w2, n_w1)
    A_lam = R_S2                    # (n_w2, n_lambda)

    # --- M matrix: rows from B_1 (Eq 35) and B_2 (Eq 37) ---
    # Eq 35: U(B_1 cell) = Π · G_1 · K_1 · [W_1] + O(Δx^(k+1))
    #         which in the (W_1, Λ) layout is Π · G_1 · K_1 · [I | 0].
    # Eq 37: U(B_2 cell) = Π · G_2 · K_2 · [S_2_pinv S_1 | R_{S_2}]
    M_rows: list[np.ndarray] = []
    for (x, y) in B1_coords:
        dx_off = x - P_xy[0]
        dy_off = y - P_xy[1]
        pi_block = pi_block_at_offset(dx_off, dy_off, k, n_state_fluid)
        # Π_f · G_f · K_f · [I | 0]
        first = pi_block @ G_f_k @ K_f
        row = np.hstack([first, np.zeros((n_state_fluid, n_lambda),
                                            dtype=np.float64)])
        M_rows.append(row)
    for (x, y) in B2_coords:
        dx_off = x - P_xy[0]
        dy_off = y - P_xy[1]
        pi_block = pi_block_at_offset(dx_off, dy_off, k, n_state_solid)
        # Π_s · G_s · K_s · [A_W1 | A_lam]
        GsKs = G_s_k @ K_s
        col1 = pi_block @ GsKs @ A_W1
        col2 = pi_block @ GsKs @ A_lam
        row = np.hstack([col1, col2])
        M_rows.append(row)

    if not M_rows:
        raise ValueError("Disk B is empty — increase q radius or check"
                          " SDF partitioning.")
    M = np.vstack(M_rows)

    # --- M̄^{-1}: pseudoinverse of M, restrict to W_1 rows ---
    # M maps (W_1, Λ) → (U^n)_B; M^{-1} maps (U^n)_B → (W_1, Λ).
    # We want only the W_1 component — take first n_w1 rows.
    M_pinv = scipy_pinv(M, rtol=1e-10)
    Mbar_inv = M_pinv[:n_w1, :]     # (n_w1, total_B_components)

    # --- LP04 Eq 41-42: per-target U* extension formula ---
    #
    # For target in Ω_2 (SOLID): U*_{I,J} = Π · G_1 · K_1 · W_1
    #   = Π · G_1 · K_1 · M̄^{-1} · (U^n)_B
    #   The target's modified value is the FLUID extension to that
    #   solid position — output is in fluid 3-component layout
    #   (v_x, v_y, p).
    #
    # For target in Ω_1 (FLUID): "swap" rule — use G_2 · K_2 · W_2,
    # where W_2 = A_W1 W_1 + A_lam Λ (Eq 31). The target's modified
    # value is the SOLID extension to that fluid position — output
    # is in solid 5-component layout (v_x, v_y, σ_xx, σ_xy, σ_yy).
    #
    # IMPORTANT: prior version of this code had the labels SWAPPED:
    # 'fluid' branch computed FLUID extension at fluid position
    # (trivial same-side) and 'solid' branch computed SOLID extension
    # at solid position (also trivial). The mistake was undetectable
    # under uniform-input tests because constant extensions are
    # identical regardless of which side the projector extends. The
    # error surfaced in the LP04 §4.2 plane-interface convergence
    # test where the jump wavefield has different fluid vs solid
    # amplitudes (2026-05-27).
    dx_t = target_xy[0] - P_xy[0]
    dy_t = target_xy[1] - P_xy[1]

    if target_side == 'solid':
        # Target in Ω_2 → fluid extension at solid position
        n_state_self = n_state_fluid
        pi_target = pi_block_at_offset(dx_t, dy_t, k, n_state_self)
        G_K = G_f_k @ K_f                # (3·n_mono, n_w1)
        N = pi_target @ G_K @ Mbar_inv   # (3, total_B_components)
    else:
        # Target in Ω_1 → solid extension at fluid position
        n_state_self = n_state_solid
        pi_target = pi_block_at_offset(dx_t, dy_t, k, n_state_self)
        G_K = G_s_k @ K_s                # (5·n_mono, n_w2)
        Lam_inv = M_pinv[n_w1:n_w1 + n_lambda, :]
        N = pi_target @ G_K @ (A_W1 @ Mbar_inv + A_lam @ Lam_inv)

    cond_M = float(np.linalg.cond(M))
    info = ProjectorInfo(
        k=k,
        side=target_side,
        n_state_self=n_state_self,
        n_state_other=(n_state_solid if target_side == 'fluid'
                        else n_state_fluid),
        n_w1=n_w1,
        n_lambda=n_lambda,
        n_B1=len(B1_coords),
        n_B2=len(B2_coords),
        cond_M=cond_M,
        rank_M=int(np.linalg.matrix_rank(M, tol=1e-10 * np.linalg.norm(M, 2))),
    )
    return N, info


def disk_neighbours_around_P(P_xy: tuple[float, float],
                               sdf_sign_at_P_positive_side: str,
                               dx: float, q_factor: float = 3.5,
                               ) -> tuple[list, list]:
    """Build B_1 (fluid, sdf > 0) and B_2 (solid, sdf < 0) lists at the
    Cartesian cells within disk radius q_factor·dx around P, for a
    HORIZONTAL interface (Γ at y = P_y).

    `sdf_sign_at_P_positive_side` is 'above' or 'below' depending on
    the convention. For LP04 'fluid above, solid below' use 'above'.
    """
    q = q_factor * dx
    cells_above: list[tuple[float, float]] = []
    cells_below: list[tuple[float, float]] = []
    # Search integer cell offsets within q radius
    n_search = int(np.ceil(q_factor))
    for i in range(-n_search, n_search + 1):
        for j in range(-n_search, n_search + 1):
            x = P_xy[0] + i * dx
            y = P_xy[1] + j * dx
            r2 = (x - P_xy[0]) ** 2 + (y - P_xy[1]) ** 2
            if r2 > q ** 2:
                continue
            if y > P_xy[1]:
                cells_above.append((x, y))
            elif y < P_xy[1]:
                cells_below.append((x, y))
            # cells exactly on Γ (y == P_y) are excluded
    if sdf_sign_at_P_positive_side == 'above':
        return cells_above, cells_below
    return cells_below, cells_above


if __name__ == '__main__':
    # Smoke: horizontal interface (dip=0°) at y=0; fluid above, solid below.
    dx_grid = 0.005   # 5 mm cells, LP04 §4.2 scale
    P = (0.0, 0.0)
    tangent = (1.0, 0.0)     # interface horizontal
    B1, B2 = disk_neighbours_around_P(P, 'above', dx_grid, q_factor=3.5)
    print(f"|B_1| (fluid above Γ) = {len(B1)}")
    print(f"|B_2| (solid below Γ) = {len(B2)}")
    target = (0.0, -dx_grid)  # one solid cell adjacent to Γ
    N, info = build_projector(
        target_side='solid', target_xy=target, P_xy=P, tangent=tangent,
        B1_coords=B1, B2_coords=B2, k=2,
        c_p_solid=4.0, c_s_solid=2.0,
    )
    print(f"N shape = {N.shape}")
    print(f"|N|_2   = {np.linalg.norm(N):.3e}")
    print(f"cond(M) = {info.cond_M:.3e}")
    print(f"rank(M) = {info.rank_M}")
    print(f"n_w1, n_λ = {info.n_w1}, {info.n_lambda}")
    comp_names = ['v_x', 'v_y', 'sxx', 'sxy', 'syy']
    n_b1_cols = 3 * info.n_B1
    for c in range(5):
        # Sum |N[c, B_1 cols]| vs |N[c, B_2 cols]| to see B_1 / B_2 coupling
        b1_norm = float(np.linalg.norm(N[c, :n_b1_cols]))
        b2_norm = float(np.linalg.norm(N[c, n_b1_cols:]))
        print(f"  N[{comp_names[c]:>3s}]:  |B1 cols|={b1_norm:+.3e}  "
              f"|B2 cols|={b2_norm:+.3e}")
