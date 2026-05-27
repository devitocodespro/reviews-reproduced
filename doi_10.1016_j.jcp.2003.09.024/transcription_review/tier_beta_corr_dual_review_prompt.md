Architectural sanity-check + library-choice review for an ESIM (Explicit Simplified Interface Method) implementation in a 2D rotated-staggered-grid (RSG) elastic finite-difference solver.

═══════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════

We're implementing Lombard-Piraux 2004 (DOI 10.1016/j.jcp.2003.09.024) ESIM for a 2D RSG elastic solver on a Petrobras dipping fluid-solid interface (dx=0.005 km, Lx=6.0 km, Ly=4.30 km, dip=30°, ~11,913 irregular cells along Γ, RSG SO=8).

The seabed Γ is a STATIONARY 1D manifold in 2D (a straight dipping line). So the LP04 per-cell projector tensors are TIME-INVARIANT. We precompute them once and apply each timestep.

We've already built and verified:
- Tier 3 scaffold (paper-faithful algebra through k=2, validated on Tier α plane-wave h-refinement to fp64).
- β.1 host-side precompute API (`scripts/tier3_esim_precompute.py` → `ESIMTables`).
- β.3 packed Devito-friendly storage (`scripts/tier3_esim_storage.py` → `PackedESIMTables`, ~98 MB at Petrobras scale).
- β.4.1 small-scale Devito prototype (validated indirect-indexing pattern in a compiled stencil; passes at 100 cells × 41 offsets, 0.64 ms apply).
- β.4.2 SCAFFOLD failed at full Petrobras: the offset alphabet at production is ~115 distinct relative offsets (because the disk B around the cell's interface projection has variable cell-to-projection offset). The unrolled Devito stencil has ~14k symbolic terms; gcc JIT compile stalls indefinitely.

A user reframe surfaced the right architecture: this is a **SPARSE LINEAR OPERATOR** problem, not a Devito-stencil problem. The LP04 substitution can be decomposed as bulk + correction, with the correction being a sparse matrix-vector product. Then we don't fight Devito at all — the bulk RSG operator runs unmodified, and a sparse correction step runs post-bulk each timestep.

═══════════════════════════════════════════════════════════════════════
QUESTION 1 — Correction-term decomposition (math)
═══════════════════════════════════════════════════════════════════════

LP04 Eq 43-44 says: at irregular cell M, the discrete operator H reads U^n at SAME-side stencil legs and U* at CROSS-Γ legs.

For a LINEAR discrete operator H (which our RSG is — it's a sum of stencil-weight × U value products with no nonlinear function applied to U):

    U^(n+1, LP04)_M = H_M(Ũ_neighbours)
    U^(n+1, bulk)_M  = H_M(U^n_neighbours)
    Δ_M := U^(n+1, LP04)_M - U^(n+1, bulk)_M
         = H_M(Ũ - U^n)
         = sum over CROSS-Γ legs (α,β) at M:
             stencil_weight[M, (α,β), c_out, c_in]
             × (U*_(M+α,β, c_in) - U^n_(M+α,β, c_in))

where Ũ has U^n at same-side legs (zero difference, no contribution) and U* at cross-Γ legs.

Each cell M has typically 5-15 CROSS-Γ legs (half the SO=8 diagonal stencil). Total non-zeros in the sparse correction matrix C:
  n_cells × n_state² × n_cross_leg_avg ≈ 12000 × 25 × 10 = ~3M

Per-timestep flow:
  1. Standard Devito op.apply(time_m=n, time_M=n, dt=dt) advances U → U^(n+1, bulk) everywhere.
  2. SpMV #1: U_star_per_cell = N @ flatten(U^n)  [precomputed N row vectors per β.3].
  3. SpMV #2: delta_per_cell = C @ (U_star - U^n_at_cross_neighbours).
  4. Apply: U.data[c, target_i[cell], target_j[cell]] += delta_per_cell[cell, c] at all irregular cells.

VERDICT QUESTIONS:

Q1a. Is the correction-term decomposition Δ_M = H_M(Ũ - U^n) MATHEMATICALLY CORRECT for the LP04 substitution under linearity of H? Verify via algebraic identity. Note: the RSG operator H has the form
  v^(n+1) = v^n + dt/ρ · (∂_x σ_components + ∂_y σ_components)
  σ^(n+1) = σ^n + dt · Cij · (∂_x v_components + ∂_y v_components)
which IS linear in the field U at each cell.

Q1b. Does running the BULK operator without LP04 substitution at irregular cells, then ADDING Δ post-hoc, give the same final state U^(n+1) as a hypothetical operator that did read-time LP04 substitution everywhere? In particular: at non-irregular cells whose stencil reads an irregular cell, do we need any correction too?
   - Hypothesis: at non-irregular cell N, the stencil reads neighbours including possibly irregular cells M. For LP04 correctness, N should read U^n_M at those positions (same-side from N's perspective if N and M are on the same side; OPPOSITE side otherwise). Bulk operator already reads U^n_M correctly. So no correction needed at N.
   - But wait: if N is on the OPPOSITE side from M, the bulk read of U^n_M gives N the "wrong-side" value. LP04 says N should read U*_M (extrapolation from N's side). This means N IS irregular too (its stencil crosses Γ), and the correction Δ_N handles this case.
   - So the correction is needed at ALL irregular cells (both sides of Γ), which is what we already have.

═══════════════════════════════════════════════════════════════════════
QUESTION 2 — Numerical library choice for ill-conditioned pseudoinverse
═══════════════════════════════════════════════════════════════════════

In the LP04 precompute (`scripts/tier3_esim_projector.py`), we compute the pseudoinverse of the over-determined Taylor-fit matrix M (LP04 Eq 38-40):
  Mbar^{-1} = pinv(M)  followed by row-restriction to the W_1 unknowns

M has shape (n_eq, n_unknown) where n_eq = state_components × |B| (≈ 5 × 40 = 200) and n_unknown = n_w1 + n_lambda (≈ 30).

At the BULK of cells, cond(M) ≈ 1.4e5 (median). At 4 EDGE cells out of 11,917 (interface exits the grid bottom), cond(M) blows up to 4.5e16 due to disk-B asymmetry (~3 cells on one side, ~15 on the other). We currently DROP those cells via a min-per-side filter (in-sponge so the drop is harmless for Petrobras; documented in plan §"Tier β robustness").

The user asks: which numerical library/algorithm is best for inverting/curve-fitting WITH ill-conditioned or degenerate data?

CANDIDATE LIBRARIES + ALGORITHMS:

(A) `numpy.linalg.pinv` — current default. Moore-Penrose pseudoinverse via SVD with `rcond` truncation. Stable but not optimised for ill-conditioned cases.

(B) `scipy.linalg.pinv` — wraps LAPACK gelsd. More explicit rcond control.

(C) `scipy.linalg.lstsq` with explicit rcond — LAPACK gelsd (SVD-based, divide-and-conquer), returns least-squares solution + rank. The rank info lets us detect degenerate cells programmatically.

(D) Tikhonov / ridge regularisation — adds λ·I to M^T M; trades bias for numerical stability. Per-cell adaptive λ.

(E) Truncated SVD (sklearn.decomposition.TruncatedSVD or manual via scipy.linalg.svd) — keep only top-k singular values. Effective rank reduction.

(F) Randomised SVD (Halko-Martinsson-Tropp) — for large matrices, not relevant here (M is tiny).

(G) Pivoted QR / Rank-revealing QR — alternative rank-detection mechanism.

For our case: M is small (200×30) so all candidates are fast. The discriminator is ROBUSTNESS to varying cond(M). At the worst 4 cells, what's the most principled handling?

VERDICT QUESTIONS:

Q2a. For the LP04 pseudoinverse precompute (per-cell M ~200×30, varying cond from 1e5 to 1e16), what's the best library + parameter recipe? Specifically:
  - scipy.linalg.lstsq with rcond=1e-10? Tighter or looser?
  - Tikhonov with λ = 1e-8 · σ_max(M)?
  - Detect rank-deficiency via rank-revealing QR + skip-flag (cleaner than drop-by-side-count)?

Q2b. For the per-timestep SpMV at production (3M-NNZ matrix C applied 8800 times):
  - scipy.sparse.csr_matrix.dot (~5-10 GFlops/s with OpenMP via scipy.sparse)
  - Intel MKL Sparse BLAS (scipy.sparse interface; ~2-5× faster on Intel CPUs)
  - cuSPARSE (10-100× faster on GPU)
  - Hand-tuned AVX512 + OpenMP custom SpMV
  Given Devito itself runs on CPU and the per-step overhead budget is < 10 ms (~1% of baseline RSG step time), what's the right choice?

═══════════════════════════════════════════════════════════════════════
QUESTION 4 — Existing IBM/cut-cell/correction-term packages (was QUESTION 4; user clarified scope to mean numerical libraries, NOT domain IBM packages — SKIPPED)
═══════════════════════════════════════════════════════════════════════

(Per user clarification, this question is REPLACED by Q2 above's library focus on SVD/pinv/SpMV libraries.)

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

Four numbered sections — Q1a, Q1b, Q2a, Q2b. Each section:
 - VERDICT: AGREE | DISAGREE | INCONCLUSIVE
 - 4-8 sentences of justification
 - Specific library / algorithm recommendation with rationale
 - DOI / package version citations where applicable

End with two lines:
SUMMARY: <one-sentence overall recommendation for the implementation>
VERDICT: <AGREE | DISAGREE | INCONCLUSIVE>
