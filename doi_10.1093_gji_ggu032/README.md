# Liu (2014) GJI optimal staggered-grid FD coefficients

**Primary paper**: Liu, Y. (2014). *Optimal staggered-grid
finite-difference schemes based on least-squares for wave equation
modelling.* Geophysical Journal International 197(2): 1033-1047.
DOI: [10.1093/gji/ggu032](https://doi.org/10.1093/gji/ggu032).

**Precursor**: Liu, Y. (2013). *Globally optimal finite-difference
schemes based on least squares.* Geophysics 78(4): T113-T132.
DOI: [10.1190/geo2012-0480.1](https://doi.org/10.1190/geo2012-0480.1).

**Equivalent at the TE order**: Fornberg, B. (1988). *Generation of
finite difference formulas on arbitrarily spaced grids.* Math. Comp.
51(184): 699-706. DOI:
[10.1090/S0025-5718-1988-0935077-0](https://doi.org/10.1090/S0025-5718-1988-0935077-0).

**Downstream consumer**: Jiang, L. & Zhang, W. (2021). *TTI equivalent
medium parametrization method for the seismic waveform modelling of
heterogeneous media with coarse grids.* GJI 227(3): 2016-2043. DOI:
[10.1093/gji/ggab310](https://doi.org/10.1093/gji/ggab310).

## Why this reproduction exists

This package is the **canonical optimal-staggered-grid FD coefficient
library** for the parent repo. It's the source of half-grid first-
derivative weights used by the optimised RSG / STO / coarse-grid TTI
methods, and underpins the Jiang & Zhang 2021 line-179 reference to
"Liu 2014's optimal weights".

Phase Y/2f 2026-05-28 relocates the solver from
`00_common/optimal_fd_coefficients.py` into this reproduction folder
so the paper claims (Liu 2014 Eq 13 closed-form weights,
Fornberg-1988 byte-equivalence sentinel, consistency condition)
live inside the reproduction that anchors them (QED guard FM2 —
citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as byte-checkable
exact-rational constants + sentinel-string-locked algorithm
properties + formal-consistency sums:

1. **Liu 2014 Eq 13 closed-form weights** at orders {2, 4, 6, 8} as
   exact `fractions.Fraction` rationals:
   - Order 2: `(1)`
   - Order 4: `(9/8, -1/24)`
   - Order 6: `(75/64, -25/384, 3/640)`
   - Order 8: `(1225/1024, -245/3072, 49/5120, -5/7168)`

   The solver's `taylor_staggered_coeffs(order)` is asserted to
   reproduce these to fp64 (allowing 1 ULP for product-reorder
   noise).

2. **First-derivative consistency condition** Σ_m (2m-1) c_m = 1
   verified exactly as a rational for every pinned order, and to
   fp64 for orders 10 and 12 too.

3. **Fornberg-equivalence sentinel** locked as three flags:
   - `LIU_2014_AND_FORNBERG_BYTE_EQUIVALENT = True`
   - `LIU_2014_FORMULA_IS_CLOSED_FORM = True`
   - `FORNBERG_1988_FORMULA_IS_RECURSIVE = True`

   The two algorithms are independent in code (closed-form O(M) vs
   recursive O(M²) per order) but byte-equivalent in output — the
   load-bearing independent cross-check.

4. **Formula sentinel** locks the closed-form expression
   `c_m = ((-1)^(m+1) / (2m-1)) · ∏_{n≠m}^M (2n-1)² / |(2m-1)² − (2n-1)²|`
   verbatim against silent swaps to a different TE form.

5. **Antisymmetric-stencil convention** (offsets `(m-½)h`, signs
   alternating with m). Catches a swap to centred (integer-offset)
   FD which would change the staggered grid identity.

6. **Solver input-validation gates** reject non-positive and odd
   orders.

## File layout

```
doi_10.1093_gji_ggu032/
├── pyproject.toml                          # declares liu_2014 pkg
├── README.md                               # this file
├── liu_2014/
│   ├── __init__.py                         # paper-anchored docstring
│   ├── optimal_fd_coefficients.py          # 431-line solver
│   └── paper_tables.py                     # exact-rational byte anchors
└── tests/
    └── test_paper_tables.py                # 16 byte-equality + sentinel
                                            # + consistency-condition tests
```

The solver (`optimal_fd_coefficients.py`) was relocated from
`00_common/optimal_fd_coefficients.py` without code changes. It also
contains Liu 2014 LS-optimisation routines for relative
(`liu_2014_relative_coeffs`) and absolute
(`liu_2014_absolute_coeffs`) dispersion-error minimisation, plus a
`modified_wavenumber` / `dispersion_error` analyser. Those are
paper-faithful but their byte-anchors depend on the Liu Table 5
band `b` values and are not pinned in this initial reproduction
(future-work follow-on).

## Quick start

```bash
# Standalone reproduction tests:
cd reproduced/doi_10.1093_gji_ggu032/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo:

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1093_gji_ggu032/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `liu_2014` package. Parent imports use
`from liu_2014.optimal_fd_coefficients import ...`. Phase Y/2f
updated 4 caller sites:
- `00_common/staggered_fd.py`
- `tests/test_optimal_fd_coefficients.py`
- `tests/_paper_faithfulness_callbacks.py`
- `scripts/run_jz_test1_iso_iso.py`

## Reproduction provenance classification

**`published`** (Phase Y/2f initial classification 2026-05-28). The
Liu 2014 Eq 13 closed-form weights at orders 2-8 are byte-anchored
exactly as rationals; the consistency condition holds exactly; the
Fornberg-equivalence is locked as a sentinel.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.

## Note on the multi-paper scope

`optimal_fd_coefficients.py` implements algorithms from THREE
papers (Liu 2014 + Liu 2013 + Fornberg 1988) plus optimisation
variants from Tam-Webb 1993 + Holberg 1987. Per the Phase Y plan's
multi-paper anchor convention, **Liu 2014 is the primary anchor**
(folder DOI). Fornberg 1988 and Liu 2013 are documented as
co-references in this README; their forms are sentinel-locked in
`paper_tables.py`. Standalone reproductions for Fornberg 1988 and
Liu 2013 are future-work follow-ons. Tam-Webb 1993 and Holberg
1987 are kept as comparative optimisation modes in the solver but
not byte-anchored here.
