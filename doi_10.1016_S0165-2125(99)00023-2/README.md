# Saenger, Gold & Shapiro (2000) — Rotated-Staggered Grid

**Status**: stub.

## Paper

> Saenger, E. H., Gold, N., & Shapiro, S. A. (2000). Modeling the
> propagation of elastic waves using a modified finite-difference
> grid. *Wave Motion*, 31(1), 77–92.
> DOI: [10.1016/S0165-2125(99)00023-2](https://doi.org/10.1016/S0165-2125(99)00023-2)

## Reference implementations

- `github.com/OumZhang/rsg` — Devito-RSFD reproduction by Zhang et
  al. (DOI 10.1016/j.cageo.2024.105850); uses Devito's
  `method='RSFD'` symbolic-derivative path and `staggered=` field
  placement.
- `devito-recipes` elastic_tti solver: also exercises RSFD via
  `recipes/elastic_tti/solver.py`.

## Key feature

Rotated diagonal stencils on a physically staggered grid (each
field has its own half-grid offset; stress and velocity components
sit at different grid positions). The "rotated" character comes
from sampling derivatives along diagonals (1, ±1) rather than the
Cartesian axes — this allows materials with sharp contrasts to be
sampled symmetrically by every stencil.

## Distinction from parent-repo Method 2

Parent repo Method 2's `02_rsg/rsg_elastic_tti.py` uses
**co-located** Devito TimeFunctions (no `staggered=` kwarg) with
diagonal-tap derivative stencils applied via explicit `.subs()`.
This is a "co-located variant" of Saenger 2000, not the
physically-staggered original. Per Codex Round 7 finding
(2026-05-24): the OumZhang/rsg reference uses Devito's `staggered=`
field placement, justifying the framing.

A **faithful** Saenger 2000 reproduction would need physical
staggering — that's the target for this folder.

## TODO

- [ ] `pyproject.toml`
- [ ] `run_reproduction.py` — physically-staggered RSG reproducing
      Saenger 2000 Fig 2-4 (homogeneous + heterogeneous test cases)
- [ ] `tests/test_reproduction.py` — defining-feature invariants
      (diagonal stencil byte-match; cross-stagger auto-shift
      verification)
- [ ] Graduation review
