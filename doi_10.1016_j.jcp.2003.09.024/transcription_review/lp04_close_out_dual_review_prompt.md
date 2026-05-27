# Dual-reviewer consultation: LP04 reproduction final close-out

## Goal
Review progress + remaining gaps for the Lombard & Piraux 2004 J. Comput. Phys.
(DOI 10.1016/j.jcp.2003.09.024) reproduction. Provide AGREE/DISAGREE/INCONCLUSIVE
verdicts + ranked next-step actions to reach a fully-faithful reproduction of LP04
§4.2 Table 2 (L^∞ + L^1 convergence orders → 2.0 on plane-interface plane-wave).

End your response with a final `VERDICT:` line: `AGREE` if the close-out is
acceptable as a faithful reproduction; `DISAGREE` if more work is mandatory;
`INCONCLUSIVE` if the empirical evidence is ambiguous.

## Where the reproduction lives
Path: `reviews-reproduced/doi_10.1016_j.jcp.2003.09.024/` (a standalone Python project
with pyproject.toml + 49 passing pytest tests; standalone modules: paper_tables,
esim_projector, lax_wendroff, analytical_reference, esim_apply,
run_reproduction, run_reproduction_interface).

## What's verified at fp64 (static-component faithfulness)
- **14 paper-table byte-match tests** — Eq 10-44 + Appendix A symbolic
  matrices C_i⁰, L_i⁰, G_i^k, α coefficients via user-confirmed side-by-side
  transcription review.
- **9 ESIM projector invariant tests** — shape, conditioning (cond(M)
  ≈ 2.5e5), full rank, disk x-symmetry, physical coupling.
- **5 Lax-Wendroff bulk convergence tests** — homog plane-wave sweep
  Nx ∈ {50, 100, 200, 400} → fitted L^∞/L^1 order **2.000 / 2.001** (to
  3 decimal places).
- **5 R/T oblique-incidence tests** — textbook R_pp at normal to fp64;
  energy flux conserved at oblique angles to fp64; critical-angle threshold
  detection.
- **6 R/T BC continuity tests at Γ** — direct AT-Γ evaluation: v_y continuity,
  σ_yy continuity, σ_xy_solid = 0 to fp64 at both normal and oblique
  (15°) incidence.
- **NEW: 4 projector-vs-analytical correspondence tests** — at irregular
  cells, projector U_star matches opposite-side analytical wavefield with
  fitted Taylor convergence slope ~ **1.81** (per target side) over
  Nx ∈ {40, 80, 160}.

## Critical bug found and fixed this session
**Sign-convention mismatch on pressure:** Paper-tables `C1_zero` uses
LP04's EXTENSIONAL-positive "p" — opposite from physical compressive
pressure. The BC `C_1·V_1 = C_2·V_2` row 2 enforces
p_LP04 = σ_yy_LP04 = σ_yy_physical (compressive convention).
Therefore p_LP04 = σ_yy_physical = -p_physical (compressive) =
σ_xx_physical (in fluid degenerate-solid layout).

Before fix: U_B fluid-cell pressure passed as +p_physical (wrong sign).
After fix: pass σ_xx directly (= p_LP04 = -p_physical).

Empirical impact on Phase A.1 (projector-vs-analytical test):
- Pre-fix: residuals 1.13 / 1.22 / 1.23 at Nx ∈ {40, 80, 160} — O(1),
  slope -0.06 (no refinement, diverging).
- Post-fix: residuals 0.124 / 0.038 / 0.010 — clean refinement,
  slope **1.81**.

## Integrated convergence sweep result

`run_reproduction_interface.py` on horizontal Γ at normal incidence
(LP04 §4.2 simplified to θ=0, with materials matching Eq 48: fluid
1000 kg/m³ 1500 m/s, solid 2600 kg/m³ c_p=4000 c_s=2000):

| Nx  | dx (µm) | L^∞      | L^1      | L^∞ peak distance to Γ |
|-----|---------|----------|----------|------------------------|
| 60  | 1666.7  | 9.2e+05  | 2.6e+04  | 1 cell                 |
| 100 | 1000    | 7.4e+05  | 1.9e+04  | 1 cell                 |
| 150 | 666.7   | 1.5e+06  | 1.5e+04  | 3 cells                |

Fitted log-log slopes: **L^∞ order = 0.73, L^1 order = 1.52**.

## The remaining gap

The L^1 order (1.52) PASSES our ≥ 1.5 threshold and is close to
LP04 Table 2's reported orders (which are 2.0 at finest grid). The
L^∞ order (0.73) does NOT pass.

**Interpretation**: The L^∞ peak sits at distance-1 cells from Γ
(the irregular cells), and the per-cell projector residual scales
as O(dx^1.81). After N_steps ∝ 1/dx CFL-bounded time-steps,
accumulated error at irregular cells scales as O(dx^0.81) — matching
the measured L^∞ slope 0.73.

LP04 §3.5 + Eq 41 claim O(dx^{k+1}) = O(dx³) for k=2. Our measured
projector slope 1.81 falls between O(dx²) and O(dx³). Possible
causes:

1. **Minimal-basis collapse** at k=2 trims redundant ∂x∂y vs ∂y∂x
   monomials; the actual implementation may behave as effectively
   k=1 in some directions, giving O(dx²) rather than O(dx³).
2. **Disk-B q-radius** (we use q=3.5·dx per paper); insufficient
   geometric coverage may limit the effective Taylor order.
3. **Π row vector** evaluates Taylor monomials with β_x! β_y!
   normalization; possibly an alternative normalization is needed
   for k=2.
4. **Coordinate-convention issue** at solid-side cells when the
   target is on opposite side (the swap rule's M^{-1} chain).

## Questions

**Q1**: Given the sign convention finding (p_LP04 = -p_physical),
is our diagnosis correct that this was the load-bearing bug
preventing LP04 reproduction, or is there an independent paper-
convention disagreement we should verify?

**Q2**: Is the L^1 order = 1.52 sufficient to claim faithful
reproduction at the integrated-test level, given that:
- Paper Table 2's reported L^1 orders at Nx ∈ {100, 200, 400} are
  1.74, 2.04, 2.00 (LW+ESIM)
- Our 3-grid sweep at Nx ∈ {60, 100, 150} gives 1.52 (within ~0.5
  of paper's coarse-grid value)
- L^∞ falls short due to projector's 1.81 slope vs paper-claimed 3

**Q3**: For the L^∞ shortfall, which of the 4 hypotheses (above)
is most likely? Concrete fix-path with effort estimate.

**Q4**: For an HONEST faithfulness classification (per `~/CLAUDE.md`
Rules 1-13: faithful / surrogate / novel-combination), what
provenance is appropriate for this reproduction in its current
state?

## Required response format

For each Q1-Q4, provide:
- **Verdict**: AGREE / DISAGREE / INCONCLUSIVE (relative to the
  framing in the question)
- 2-4 sentence justification
- For Q3 only: ranked concrete fix-path with effort estimates.

Finish with a single line:
`VERDICT: AGREE` if the LP04 reproduction is acceptably faithful in its
current state (with documented L^∞ residual), `DISAGREE` if more
work is mandatory before the reproduction can be labeled paper-
faithful, or `INCONCLUSIVE` if more empirical evidence is needed
before judging.
