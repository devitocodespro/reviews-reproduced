# Saenger, Gold & Shapiro (2000) — Rotated-Staggered Grid

**Status**: dispersion-only reproduction (stencil + CFL bounds).
15/15 tests pass. Wave-propagation reproduction deferred to
T5.2 physical-staggering rewrite.

## Paper

> Saenger, E. H., Gold, N., & Shapiro, S. A. (2000). Modeling the
> propagation of elastic waves using a modified finite-difference
> grid. *Wave Motion*, 31(1), 77–92.
> DOI: [10.1016/S0165-2125(99)00023-2](https://doi.org/10.1016/S0165-2125(99)00023-2)

## Reproduction scope

This is a **stencil-and-stability reproduction** (no Devito wave
propagation). Saenger 2000 derives the RSG, then provides:

- §5.1 von Neumann stability analysis → Eq 27/28 (RSG bound),
  Eq 29/30 (SSG 3D / 2D bounds)
- §5.2 dispersion relations → Eq 36–39 (closed-form expressions
  for the RSG and the SSG)

The quantitative anchors we byte-check (per the
`feedback_reproduction_quantitative_first` repo discipline):

1. **Staggered Taylor coefficients** (Levander 1988 / Holberg
   tabulated weights) at so ∈ {2, 4, 8, 16}.
2. **RSG CFL bound** (Saenger 2000 Eq 28) at every order — at
   SO=4 this gives `CFL_RSG = 6/7 ≈ 0.857`.
3. **SSG 2D / 3D CFL bounds** (Eq 30 / Eq 29) — at SO=4:
   `CFL_SSG_2D = 6/(7√2) ≈ 0.606`, `CFL_SSG_3D = 6/(7√3) ≈ 0.495`.
4. **Saenger's qualitative claim** "the stability condition for
   the new rotated grid is LESS restrictive" (paper page 84):
   `CFL_RSG > CFL_SSG_2D` at every order tested.
5. **Reference-output regression**: pinned `.npz` outputs match
   the driver to fp64.

## What is NOT reproduced

- **Wave-propagation snapshots**: Saenger 2000 Fig 2/3/4 are
  wavefield comparisons (RSG vs SSG on heterogeneous test models).
  A faithful Devito reproduction needs physically-staggered fields
  (the parent repo's Method 2 uses co-located fields per the
  `staggered=` discussion in `02_rsg/README.md`), which is the
  T5.2 physical-staggering rewrite — deferred multi-day work.
- **Eq 36 dispersion-error sweep**: the driver writes a per-SO
  dispersion sweep to `reference_outputs/dispersion_so<N>.npz`
  as a diagnostic, but the test gate does NOT assert against the
  paper's Fig 6/7 bounds. The 2nd-order formula used in the
  driver does not generalise correctly to higher SO under the
  RSG diagonal-stencil reads (the paper's Eq 36 is 2nd-order
  specifically; higher-order RSG dispersion requires a per-SO
  derivation). The diagnostic is preserved for future review
  and is byte-stable across driver invocations.

## Reference implementations

- `github.com/OumZhang/rsg` — Devito-RSFD reproduction by Zhang et
  al. (DOI 10.1016/j.cageo.2024.105850); uses Devito's
  `method='RSFD'` symbolic-derivative path and `staggered=` field
  placement.
- `devito-recipes` elastic_tti solver: also exercises RSFD via
  `recipes/elastic_tti/solver.py`.

## Distinction from parent-repo Method 2

Parent repo Method 2's `02_rsg/rsg_elastic_tti.py` uses
**co-located** Devito TimeFunctions (no `staggered=` kwarg) with
diagonal-tap derivative stencils applied via explicit `.subs()`.
This is a "co-located variant" of Saenger 2000. Per Codex
Round 7/8 findings: the OumZhang/rsg reference uses Devito's
`staggered=` field placement; the parent repo's
`02_rsg/run_geo.py:3` docstring explicitly names this gap and
points at the T5.2 physical-staggering rewrite as the path to
faithful Saenger 2000 reproduction.

This reproduction folder therefore does NOT exercise the
parent-repo Method 2 operator. The byte-checkable claims
(coefficients + CFL bounds) are universal across both
co-located and physically-staggered RSG variants because they
depend only on the diagonal-stencil tap pattern, not on the
field placement.

## Running

```bash
cd reproduced/doi_10.1016_S0165-2125\(99\)00023-2
uv sync                            # one-time setup
uv run python run_reproduction.py --all-orders   # generate pinned outputs
uv run pytest tests/ -v            # 15/15 pass
```

## Future work

- T5.2 (deferred): physical-staggering Devito rewrite of Method 2
  → enables wave-propagation reproduction of Saenger 2000 Fig 2/3/4.
- Higher-order RSG dispersion derivation → enables test-asserting
  the dispersion diagnostic against the paper's Fig 6/7 bounds.
