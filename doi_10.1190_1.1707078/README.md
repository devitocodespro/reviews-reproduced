# Bohlen & Saenger (2004) — RSG Density Averaging for Rayleigh Waves

**Status**: minimal reproduction (corner-averaging recipe).
7/7 tests pass. The recipe verified here is the helper used by
parent-repo Method 2 (`02_rsg/rsg_elastic_tti.py:run_rsg(..., density_averaging='bohlen_saenger')`).

## Paper

> Bohlen, T., & Saenger, E. H. (2004). Accuracy of heterogeneous
> staggered-grid finite-difference modeling of Rayleigh waves.
> *Geophysics*, 69(2), 583–591.
> DOI: [10.1190/1.1707078](https://doi.org/10.1190/1.1707078)

## What's reproduced

The BS04 paper's prescribed recipe for averaging cell-centre ρ
values to the rotated-staggered-grid (RSG) corner velocity
positions:

```
rho_corner(i+½, j+½) = ¼ · (ρ[i,j] + ρ[i+1,j] + ρ[i,j+1] + ρ[i+1,j+1])
```

The 7 tests verify:
1. **Homogeneous invariance** — constant ρ is unchanged.
2. **Vertical-contrast arithmetic mean** — step ρ=1→3 averaged at
   interface = 2.0.
3. **Horizontal-contrast arithmetic mean** — same recipe, x-axis.
4. **Paper recipe byte-match** — every interior cell is the literal
   2×2 mean of its 4 surrounding ρ values to fp64.
5. **Boundary replicate-extend** — shape preserved (Nx, Ny in →
   Nx, Ny out).
6. **Reference output regression** — pinned `.npz` byte-stable
   across driver invocations.
7. **Cross-validation against parent repo helper** —
   `00_common/material_averaging.py:apply_bohlen_saenger_density_averaging`
   produces byte-identical output. This is the key gate proving
   the parent repo's helper implements the paper recipe.

## What's NOT reproduced

- **Rayleigh-wave amplitude error plots** (BS04 Fig 4-5): the
  paper quantifies how much the averaging recipe reduces Rayleigh-
  wave dispersion error at sharp ρ contrasts. A faithful
  Devito reproduction would run a layered model with and without
  the averaging, measure the Rayleigh-wave dispersion error at
  the surface, and confirm the averaging reduces it. Deferred —
  the test gate stays anchored on the byte-checkable corner-
  averaging recipe, which is the load-bearing claim for the
  parent repo's Method 2.
- **Free-surface treatment**: BS04 also touches on free-surface
  conditions for RSG. Out of scope here; covered separately in
  the parent repo's `00_common/free_surface.py` and
  `02_rsg/rsg_elastic_tti.py`.

## Reference implementation

`devitocodes/reviews/scripts/run_rsg_bohlen_saenger_v2.py:_averaged_arrays`
on the `legacy` branch — the §64 audit's empirical-tuning
recipe (a 1-D y-axis convolution). Per Codex M2 Round 8 (R8-2):
that 1-D recipe did NOT match the BS04 paper prescription;
this reproduction folder's tests pin the paper-faithful 2×2
recipe instead. The parent-repo helper at
`00_common/material_averaging.py` was reimplemented post-R8-2 to
match the BS04 paper, with `tests/test_bohlen_saenger_density_averaging.py`
locking the new behaviour. The cross-validation test in this
folder confirms standalone-and-parent agree to fp64.

## Running

```bash
cd reproduced/doi_10.1190_1.1707078
uv sync                            # one-time setup
uv run python run_reproduction.py  # generate reference outputs
uv run pytest tests/ -v            # 7/7 pass
```

## Future work

- Devito-based Rayleigh-wave Fig 4/5 reproduction (multi-day,
  deferred). Would compare RSG with and without averaging at a
  layered model with sharp ρ contrast and surface-source
  receiver.
