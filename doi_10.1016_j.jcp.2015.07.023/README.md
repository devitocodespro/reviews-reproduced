# Petersson & Sjögreen (2015) — SBP-SAT for anisotropic elastic

**Paper**: Petersson, N. A., & Sjögreen, B. (2015). *Wave propagation
in anisotropic elastic materials and curvilinear coordinates using
a summation-by-parts finite difference method.* Journal of
Computational Physics 299: 820-841.
DOI: [10.1016/j.jcp.2015.07.023](https://doi.org/10.1016/j.jcp.2015.07.023).

**Antecedent** (the original (4, 2) diagonal-norm SBP D1
tabulation):
Strand, B. (1994). *Summation by parts for finite difference
approximations for d/dx.* Journal of Computational Physics 110:
47-67. DOI: [10.1006/jcph.1994.1005](https://doi.org/10.1006/jcph.1994.1005).

**Reference implementation**: sw4 (Seismic Waves 4th-order),
open-source at [github.com/geodynamics/sw4](https://github.com/geodynamics/sw4).
The (4, 2) diagonal-norm SBP D1 boundary stencils are tabulated
in `sw4/src/boundaryOpc.C` and reproduced byte-for-byte by this
package's `sbp_sat` module.

## Why this reproduction exists

This package is the **canonical SBP-SAT operator library** for
the parent repo. Six method directories depend on its boundary
stencils + norm matrix + SAT-coupling helpers:

- `05_sbp/` — Method 5 elastic displacement SBP-SAT
- `09_viscoelastic_sbp/` — viscoelastic SBP extension
- `44_elastic_lw4_sbp/` — LW4 + SBP-SAT elastic
- `50_viscoelastic_lw4_sbp/` — LW4 + SBP-SAT viscoelastic
- `73_viscoelastic_kj_sbp/`, `75_viscoelastic_kj_lw4_sbp/`,
  `78_viscoelastic_kj_rem_sbp/` — Kjartansson-Q SBP variants

Plus a cross-reproduction dependency: the Bader 2023 reproduction
(`reproduced/doi_10.1190_geo2022-0195.1/bader_2023`) imports
`sbp_boundary_norm_weight` from this package's `sbp_sat` for its
seafloor-coupling SAT helpers.

Phase Y/2c 2026-05-28 relocates the SBP solver from
`00_common/sbp_sat.py` into this reproduction folder so the
paper claims (sw4 byte-match, SBP-property invariant) live
inside the reproduction that documents them (QED guard FM2 —
citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as
byte-checkable Fraction constants + symbolic invariants:

1. **(4, 2) D1 boundary stencil byte-match against sw4**
   (paper §2 — Strand 1994 tabulation, also in sw4's
   `src/boundaryOpc.C`). Pinned in `paper_tables.BOP_4_2` and
   cross-checked against `sbp_sat._BOP_4_2` via exact
   `Fraction` equality.

2. **Diagonal norm weights** `H_{ii}/h ∈ {17/48, 59/48, 43/48,
   49/48}` for the 4 boundary rows. Pinned in
   `paper_tables.NORM_WEIGHTS_4_2`; the boundary-row-0 weight
   `17/48` is the canonical Strand-1994 / PS-2015 value cited
   throughout the literature.

3. **Interior 4th-order centred FD stencil**
   `(1/12, -2/3, 0, 2/3, -1/12)`. Pinned in
   `paper_tables.INTERIOR_4`.

4. **SBP-property invariant** (symbolic): the boundary stencil
   + norm matrix satisfy
   `H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)`.
   This is the load-bearing constraint that makes the operator
   SBP-compatible per Strand 1994 + the Olsson 1995 energy
   method. Verified symbolically with exact `Fraction`
   arithmetic at N=16.

5. **PS-2015-compatible D2 sentinel**: the implementation MUST
   use the SBP-compatible narrow-stencil D2 (NOT centred-FD +
   boundary SAT corrections — the latter is a distinct
   surrogate per CLAUDE.md Rule 1 + the §40e Bader-2023
   surrogate-mislabel antipattern).

## File layout

```
doi_10.1016_j.jcp.2015.07.023/
├── pyproject.toml                          # declares petersson_sjogreen_2015 pkg
├── README.md                               # this file
├── petersson_sjogreen_2015/
│   ├── __init__.py                         # package docstring
│   ├── sbp_sat.py                          # the 1571-line SBP-SAT engine
│   └── paper_tables.py                     # hand-transcribed sw4 byte anchors
└── tests/
    └── test_paper_tables.py                # byte-equality + symbolic SBP property
```

The SBP solver (`sbp_sat.py`) was relocated from
`00_common/sbp_sat.py` without code changes; it implements
the Petersson-Sjögreen 2015 scheme using the Strand 1994
tabulation per its module docstring.

## Quick start

```bash
# Standalone reproduction tests (this folder only):
cd reproduced/doi_10.1016_j.jcp.2015.07.023/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo (the reproduction is wired as an
editable dep via `[tool.uv.sources]`):

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1016_j.jcp.2015.07.023/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `petersson_sjogreen_2015` package.
Parent imports use `from petersson_sjogreen_2015.sbp_sat import
...`. Phase Y/2c updated ~20 call sites across method
directories, tests, and the `_paper_faithfulness_callbacks`
registry to point at the new package.

The Bader 2023 reproduction
(`reproduced/doi_10.1190_geo2022-0195.1/bader_2023`) has a
cross-reproduction dependency on this package, declared via
its own `[tool.uv.sources]` entry — `bader_2023.acoustic_elastic_coupling`
imports `sbp_boundary_norm_weight` for its seafloor-coupling
SAT helpers.

## Reproduction provenance classification

**`published`** (Phase Y/2c initial classification 2026-05-28).
The static paper-anchor constants in `paper_tables.py` byte-match
the canonical sw4 tabulation exactly; the SBP property is
verified symbolically with exact `Fraction` arithmetic; the
integration with multiple parent method directories provides
independent empirical validation through the parent's
MMS-convergence + paper-faithfulness gates.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under
`## Graduation review`.
