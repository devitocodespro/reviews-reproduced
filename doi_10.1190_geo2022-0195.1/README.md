# Bader, Almquist & Dunham (2023) — SBP-SAT Acoustic-Elastic Coupling

**Status**: stub (reproduction work-in-progress in parent repo;
to be migrated here per refactor Phase 5).

## Paper

> Bader, M., Almquist, M., & Dunham, E. M. (2023). Modeling and
> inversion in acoustic-elastic coupled media using energy-stable
> summation-by-parts operators. *Geophysics*, 88(3), T137–T150.
> DOI: [10.1190/geo2022-0195.1](https://doi.org/10.1190/geo2022-0195.1)

## Reference implementation

[`github.com/nmbader/fwi2d`](https://github.com/nmbader/fwi2d) —
Bader's own C++ implementation. The companion SPECFEM2D input deck
(no precomputed reference seismograms) lives at
`github.com/nmbader/sbp_sat_geophysics_2022`.

**Critical bulk-operator reading**:
`fwi2d/src/include/spatial_operators.hpp:32-90` (`Dxx_var`,
`Dzz_var`) is a fully SBP-restricted variable-coefficient
narrow-stencil D2 operator with 6-row boundary closures and a 5-tap
interior stencil. Our parent-repo Method 5 currently uses centred-FD
bulk with SAT corrections only at boundary rows — this is a
deviation, not a reference match (Codex Round 7 misread, corrected
during refactor planning 2026-05-25).

`fwi2d/src/cpp/we_op.cpp:2487-2512` —
`nl_we_op_ae::propagate` shows the seafloor coupling is a **2×2
implicit Crank-Nicolson-like solve per interface column**, not the
explicit Inc-add SAT pattern that the paper text suggests.

## Reproduction targets

- Fig. 4 — canonical-geometry waveform comparison against SPECFEM2D
  (substitute: high-resolution Devito self-consistency reference
  per parent repo `tti/CLAUDE.md` self-consistency fallback rule).
- Energy-norm conservation property `H D + D^T H ≤ 0` at boundary
  rows.

## TODO

- [ ] `pyproject.toml` with `devito @ git+...@<commit>`
- [ ] `run_reproduction.py` ported from
      `devitocodes/reviews/scripts/run_dip_layered_benchmark.py`
      Method 5 path (Phase 5 milestone)
- [ ] `tests/test_reproduction.py` with operator-level SBP property
      callback + seafloor implicit-solve structural check
- [ ] `reference_outputs/` pinned reference traces
- [ ] Graduation review verdict recorded
