# Kristek et al. (2017) GJI orthorhombic-representation effective medium

**Primary paper**: Kristek, J., Moczo, P., Chaljub, E., & Kristeková,
M. (2017). *An orthorhombic representation of a heterogeneous medium
for the finite-difference modelling of seismic wave propagation.*
Geophysical Journal International 208(2): 1250-1264.
DOI: [10.1093/gji/ggw456](https://doi.org/10.1093/gji/ggw456).

## Why this reproduction exists

This folder gives the Kristek 2017 paper its own DOI'd reproduction
folder so the paper-faithfulness chain is discoverable from the
registry (QED guard FM2).

**The Kristek 2017 averaging is NOT implemented in this folder**;
it lives in the
[Moczo 2002 reproduction package](../doi_10.1785_0120010167/)
under the `interface_treatment='kristek2017'` dispatch of
`moczo_2002.effective_medium`. The Kristek 2017 scheme is the β=0
(axis-aligned interface) reduction of the Schoenberg-Muir
orientation-aware effective form — the production code therefore
routes it through the moczo_2002 helper rather than duplicating
the implementation.

This folder primarily:
1. **Documents the Kristek 2017 paper anchors**:
   orthorhombic-effective-medium symmetry, axis-aligned reduction
   of Schoenberg-Muir, averaging-rule inheritance from Moczo 2002.
2. **Preserves the two diagnostic scripts** that validated the
   `kristek2017` mode against the Bouchon 1981 DWN reference at
   the Petrobras config (dx=5m, dip=30°, SO=8). These previously
   lived in `notes/archived_diagnostic/` and are now relocated
   to `diag/` here as graduation-review evidence.
3. **Provides the cross-reproduction discovery pointer** to
   `moczo_2002.effective_medium` for anyone who reaches this
   folder via the Kristek DOI.

Phase Y/2i 2026-05-29 bootstrap + diag-script recovery.

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as
sentinel-string-locked structural properties:

1. **Orthorhombic effective-medium symmetry sentinel**
   (`KRISTEK_2017_EFFECTIVE_MEDIUM_SYMMETRY = "orthorhombic"`).
   Catches silent collapse to isotropic (Moczo 2002) or expansion
   to fully anisotropic (general Schoenberg-Muir).

2. **Generalises-Moczo-2002** and **included-by-Schoenberg-Muir**
   flags lock the symmetry-class hierarchy:
   isotropic ⊂ orthorhombic ⊂ general-anisotropic.

3. **Axis-aligned (β = 0) sentinel** locks the load-bearing
   reduction-from-Schoenberg-Muir-at-β=0 property.

4. **Averaging-rule inheritance** from Moczo 2002: arithmetic-ρ +
   harmonic-modulus (for the diagonal-stiffness elements).

5. **Cross-reproduction discovery pointers**:
   - `KRISTEK_2017_PRODUCTION_LOCATION` points at
     `moczo_2002.effective_medium`
     (`interface_treatment='kristek2017'` dispatch).
   - `KRISTEK_2017_VALIDATION_REFERENCE` points at
     `bouchon_1981.dwn_solver` (via `diag/d2_side_by_side.py`).

## File layout

```
doi_10.1093_gji_ggw456/
├── pyproject.toml                          # declares kristek_2017 pkg
├── README.md                               # this file
├── kristek_2017/
│   ├── __init__.py                         # paper-anchored docstring
│   └── paper_tables.py                     # hand-transcribed anchors
├── tests/
│   └── test_paper_tables.py                # 9 sentinel tests
└── diag/
    ├── d1_cij_profile.py                   # vertical-profile Cij overlay
    └── d2_side_by_side.py                  # 3-column FD-vs-DWN snapshot
```

## Diagnostic scripts (preserved evidence)

- **`diag/d1_cij_profile.py`**: visualises effective Cij in the
  mixed-cell band along a vertical profile at the Petrobras config
  (dx=5m, dip=30°, SO=8). Compares `interface_treatment='sharp'`
  vs `'kristek2017'` — renders a 7-panel figure (ρ, c11, c22,
  c12, c16, c26, c66).

- **`diag/d2_side_by_side.py`**: 3-column divergence-of-velocity
  snapshot comparison at dip=30°:
  - col 1: FD-with-sharp (post-§86 dx020 pickle)
  - col 2: FD-with-Kristek (post-§87.F1 dx020 pickle)
  - col 3: Bouchon 1981 DWN reference at the same grid

  Identifies whether the post-F1 Kristek wavefield shows interface
  scattering. This was the diagnostic that caught the §87 units
  bug (the "fp64-identical-across-distinct-configs" bug signature).

Both scripts depend on the parent's `00_common/` directory being on
`sys.path` (for `petrobras_config`, `tti_common.tti_params`, etc.)
and require the post-Phase-2 relocations (bouchon_1981 +
mallick_frazer_1990 + moczo_2002 as editable deps). They are
preserved here as **graduation-review evidence**, not as standalone
runnable tests — running them requires the full parent
configuration + the diagnostic pickle files in
`figures/dip_layered/`.

## Quick start

```bash
# Standalone paper-anchor tests:
cd reproduced/doi_10.1093_gji_ggw456/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo:

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1093_gji_ggw456/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `kristek_2017` package. The package
exposes only `kristek_2017.paper_tables` (paper anchors); there is
no parent-side import of `kristek_2017.<production_code>` because
the production averaging lives in `moczo_2002.effective_medium`.

## Reproduction provenance classification

**`published`** (Phase Y/2i initial classification 2026-05-29).
The paper anchors are all sentinel-locked; the diagnostic-script
evidence preserved in `diag/` documents the production
`interface_treatment='kristek2017'` mode's validation against the
DWN reference at the Petrobras config.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.

## Note on the "thin" reproduction scope

This folder is intentionally thin relative to the other Phase Y/2
reproductions. The reason: the production code that implements the
Kristek 2017 averaging is shared with the Moczo 2002 reproduction
(via the `'kristek2017'` dispatch mode in
`moczo_2002.effective_medium`). Duplicating the implementation here
would create two divergent copies of the same algorithm and violate
DRY at the cost of provenance clarity.

The chosen design splits the responsibilities:
- **moczo_2002 reproduction**: byte-anchors the numerical averaging
  rules (arithmetic ρ + harmonic K, μ + Voigt-basis convention +
  fluid-limit sentinels).
- **this kristek_2017 reproduction**: byte-anchors the orthorhombic-
  symmetry classification + axis-aligned-reduction sentinel + paper-
  to-production cross-reference + diag-script evidence.

Together they cover the Kristek 2017 paper-faithfulness claim
without duplicating code.
