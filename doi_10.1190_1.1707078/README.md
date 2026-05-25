# Bohlen & Saenger (2004) — RSG Density Averaging for Rayleigh Waves

**Status**: stub (helper landed in parent repo as
`00_common/material_averaging.py:apply_bohlen_saenger_density_averaging`
under T1.2; full reproduction folder pending).

## Paper

> Bohlen, T., & Saenger, E. H. (2004). Accuracy of heterogeneous
> staggered-grid finite-difference modeling of Rayleigh waves.
> *Geophysics*, 69(2), 583–591.
> DOI: [10.1190/1.1707078](https://doi.org/10.1190/1.1707078)

## Key result

Arithmetic averaging of cell-centre density values at corners (2×2
kernel) on the rotated-staggered grid (Saenger 2000) preserves
momentum conservation at sharp density contrasts — without this,
Rayleigh-wave dispersion errors grow.

## Reference implementation

`devitocodes/reviews/scripts/run_rsg_bohlen_saenger_v2.py:_averaged_arrays`
on the `legacy` branch — implements the §64 audit's corrected
convolution kernel.

## TODO

- [ ] `pyproject.toml` with Devito pin
- [ ] `run_reproduction.py` reproducing Bohlen-Saenger Fig 4-5
      (Rayleigh-wave dispersion at sharp ρ contrast)
- [ ] `tests/test_reproduction.py` with byte-match against
      `apply_bohlen_saenger_density_averaging` (already locked in
      parent repo's `tests/test_bohlen_saenger_density_averaging.py`)
- [ ] Graduation review verdict
