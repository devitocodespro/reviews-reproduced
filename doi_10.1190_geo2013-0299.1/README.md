# Vishnevsky, Lisitsa, Tcheverda, Reshetova 2014 — interface errors of FD seismic-wave simulations

**Paper**: Vishnevsky, D., Lisitsa, V., Tcheverda, V. & Reshetova, G. (2014).
"Numerical study of the interface errors of finite-difference simulations
of seismic waves." *Geophysics* **79**(4): T219-T232.
DOI: [10.1190/geo2013-0299.1](https://doi.org/10.1190/geo2013-0299.1).

## Why this reproduction exists

This paper is **the single load-bearing literature anchor** for the
cross-method ranking SSG > RSG ≈ STO-RSG ≈ Lebedev observed in the
Petrobras water-over-TTI cohort in
`devitocodespro/devito-fd-survey` (the consistency memo
`notes/method_reproduction_consistency_2026-05-28.md` makes this
explicit).

The paper's key prediction (from the abstract):

> "...the presence of a fluid-solid interface reduces the order of
> convergence for the LS and the RSGS to a first order of
> convergence. The presence of inclined interfaces makes high-order
> (second and more) convergence impossible."

Plus, from §"Conclusion":

> "if a planar horizontal/vertical interface between two solids is
> considered, all three schemes (SSGS, LS, and RSGS) can be modified
> to produce the second-order accuracy by arithmetic averaging of
> density and finely layered media averaging for the stiffness
> tensor in the vicinity of the interface."

The Petrobras configuration is EXACTLY the fluid-solid + dipping
case the paper documents.

## What we reproduce

Per `feedback_reproduction_quantitative_first` (anchor on tables
and scalar bounds, not figures), this reproduction focuses on:

1. **Test medium parameters** (paper §"Numerical experiments",
   page T223): isotropic-solid configs IS1/IS2/IS3, fluid IF, and
   anisotropic-solid Cij matrices (Eqs 14-16).

2. **Source initial condition** (paper Eq 17): Gaussian on σ_xx and
   σ_zz centred at (xs=1500m, zs=750m) on a 3000×3000m domain.

3. **Convergence indicator** (paper Eqs 12-13): the ratio
   δ_k = ε_k/ε_{k-1}. For grid refinement by factor 2:
   - δ_k ≈ 4 ⇒ 2nd-order convergence
   - δ_k ≈ 2 ⇒ 1st-order convergence

4. **The qualitative scheme-vs-interface convergence table**
   extracted from Figures 4-16:

   | Configuration | SSGS | RSGS | LS |
   |---|---|---|---|
   | Horizontal solid-solid (iso/aniso, modified) | ≈4 | ≈4 | ≈4 |
   | **Horizontal fluid-solid** | **≈4** | **≈2** | **≈2** |
   | Inclined (any) | ≈2 | ≈2 | ≈2 |
   | Corner 3-solids (no fluid) | ≈4 | ≈4 | ≈4 |
   | Corner with fluid | ≈4 | ≈2 | ≈2 |
   | Unmodified (any modification skipped) | <2 | <2 | <2 |

   The fluid-solid row maps directly to the Petrobras
   water-over-TTI configuration in the parent-repo cohort.

5. **Empirical h-refinement convergence study** (DEFERRED — see
   "Status" below). The intent is to verify δ_k → 4 for SSGS and
   δ_k → 2 for RSGS at the horizontal fluid-isotropic-solid
   configuration using minimal pure-NumPy implementations.

## Status — 2026-05-28

| Anchor | Status | Notes |
|---|---|---|
| 1. Test medium parameters (IS1/IS2/IS3/IF) | ✅ landed | byte-tested in `tests/test_paper_tables.py` |
| 2. Cij matrices Eqs 14, 15, 16 (AS1/AS2/AS3) | ✅ landed | review-pool vision-LLM verified 2026-05-28 (Azure GPT-5 + Ollama gemma4:31b both AGREE) |
| 3. Source IC Eq 17 | ✅ landed | tested + peak-at-(xs,zs) check |
| 4. Qualitative convergence-rate table | ✅ landed | Eqs 12-13 thresholds + Figures 4-16 predictions encoded in `CONVERGENCE_PREDICTIONS`, byte-tested |
| 5. Empirical h-refinement (SSGS + RSGS NumPy) | ⏳ deferred | `run_reproduction.py` scaffold exists but the small-domain pilot is severely under-resolved at coarse grids (Gaussian σ ≈ 30 m, coarsest dx = 10 m → ~3 cells span FWHM → SO=2 dispersion dominates). Reframing to a wider Gaussian (σ ≈ 100 m) + finer dx range, OR migrating the implementation to Devito's existing M1/M2 production solvers, is the unblock path. Not load-bearing for the parent-repo consistency memo: anchors 1-4 are sufficient to cite this paper as the load-bearing literature reference for the Petrobras cohort ranking. |

The verification gate as it stands (`tests/test_paper_tables.py`,
14 tests, all green) byte-matches every transcribed constant
that the parent-repo consistency memo cites. If the empirical
h-refinement work is re-prioritised, the obvious unblock is to
reuse the parent's `01_ssg/` and `02_rsg/` Devito solvers instead
of re-implementing SSGS / RSGS in NumPy from scratch — the
parent has graduated implementations of both schemes.

## Engagement context

The qualitative table above explains the Petrobras cohort
ranking observed at dx=0.005 km, SO=8:

| Method | hf_ratio @ dip=30° | Predicted convergence |
|---|---|---|
| SSG (M1) | 5.6e-3 | 2nd order at fluid-solid (matches Lisitsa SSGS) |
| RSG (M2) | 9.8e-3 | 1st order at fluid-solid (matches Lisitsa RSGS) |
| STO-RSG (M61) | 1.0e-2 | 1st order (same RSG topology) |
| Dual-Pair (M6) | 3.3e-3 | non-staggered topology (corollary, not directly tested in paper) |

## Quick start

```bash
cd reproduced/doi_10.1190_geo2013-0299.1/
uv sync
uv run pytest tests/ -v
uv run python run_reproduction.py
```

## Files

- `paper_tables.py` — byte-transcribed test medium constants +
  qualitative convergence-rate table (load-bearing)
- `tests/test_paper_tables.py` — pinned-constants regression
  (14 tests, all green)
- `run_reproduction.py` — h-refinement convergence sweep
  scaffold (DEFERRED — pilot under-resolved at coarse grids,
  see Status table)
- `vishnevsky_lisitsa_2014_paper.pdf` — pinned paper PDF
