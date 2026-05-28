# Bouchon (1981) — Discrete Wavenumber Method

**Paper**: Bouchon, M. (1981). *A simple method to calculate
Green's functions for elastic layered media.* Bulletin of the
Seismological Society of America 71(4): 959-971.
DOI: [10.1785/BSSA0710040959](https://doi.org/10.1785/BSSA0710040959).

**Companion review** (load-bearing for the critical-angle
discussion):
Bouchon, M. (2003). *A Review of the Discrete Wavenumber
Method.* Pure and Applied Geophysics 160: 445-465.
DOI: [10.1007/PL00012545](https://doi.org/10.1007/PL00012545).

## Why this reproduction exists

This package is the **load-bearing semi-analytical reference**
for the parent repo's homogeneous + horizontal-interface
acoustic-acoustic validation: every DWN-based test in the
parent (V5 seismogram regression, smeared-DWN trace
comparisons, the dispersion-comparison qualitative gate, the
Jiang-Zhang / Kristek cross-checks) imports the engine
implemented here.

Phase Y/2b 2026-05-28 relocates the engine from
`00_common/reflectivity_2d.py` into this reproduction folder so
the paper claims it anchors live inside the reproduction that
documents them (QED guard FM2 — citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as
byte-checkable constants + sentinel invariants:

1. **Complex-frequency regularisation α prescription**
   (paper page 962): `α = π / T_max`. Pinned in
   `paper_tables.BOUCHON_1981_ALPHA_PRESCRIPTION` and the
   `alpha_damping()` helper; byte-tested with `==` (no
   tolerance) at multiple T_max values.

2. **Periodicity-length safety margin** (paper page 962):
   strict bound `L > V_max · T_max + x_max`; implementation
   uses `1.3 × strict`. Pinned in
   `BOUCHON_1981_PERIODICITY_SAFETY_MARGIN = 1.3` +
   `periodicity_length()` helper; byte-tested via `==`.

3. **Critical-angle singularity policy** (paper page 961 +
   Bouchon 2003 page 449): the `1/q_w` singularity at the
   critical angle is regularised by the complex-frequency
   shift; the implementation MUST claim this via the sentinel
   `DWN_SOLVES_CRITICAL_ANGLE_SINGULARITY = True`.

4. **Recommended sampling parameter bounds** (Bouchon-2003
   review §3): community-practice lower bounds on `n_omega`
   and `M_wavenumbers`. Pinned as `RECOMMENDED_*_MIN`.

5. **DWN solver smoke import** — `bouchon_1981.dwn_solver`
   imports cleanly and exposes the canonical entry point
   `dwn_wavefield_acoustic_acoustic_horizontal`.

## File layout

```
doi_10.1785_BSSA0710040959/
├── pyproject.toml                          # declares bouchon_1981 pkg
├── README.md                               # this file
├── bouchon_1981/
│   ├── __init__.py                         # package docstring
│   ├── dwn_solver.py                       # the 58 kB DWN engine
│   └── paper_tables.py                     # hand-transcribed paper anchors
└── tests/
    └── test_paper_tables.py                # byte-equality regression
```

The DWN engine (`dwn_solver.py`) was relocated from
`00_common/reflectivity_2d.py` without code changes; it
implements the Bouchon-1981 method as described in its module
docstring (full math derivation + complex-frequency
regularisation + periodicity-length condition).

## Quick start

```bash
# Standalone reproduction tests (this folder only):
cd reproduced/doi_10.1785_BSSA0710040959/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo (the reproduction is wired as an
editable dep via `[tool.uv.sources]`):

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1785_BSSA0710040959/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `bouchon_1981` package. Parent
imports use `from bouchon_1981.dwn_solver import ...`. Phase
Y/2b updated 9 call sites across `tests/`, `scripts/`, and a
diagnostic note to point at the new package.

## Reproduction provenance classification

**`published`** (Phase Y/2b initial classification 2026-05-28).
The static paper-anchor constants in `paper_tables.py`
byte-match the paper's prescriptions exactly; the DWN solver
implements the paper's method per its module docstring's
detailed math citation; the integration with the parent
repo's V5 gate provides independent empirical validation
against finite-difference solver output.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under
`## Graduation review`.
