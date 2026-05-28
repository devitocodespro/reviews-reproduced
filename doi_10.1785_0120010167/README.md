# Moczo et al. (2002) BSSA effective-medium averaging

**Primary paper**: Moczo, P., Kristek, J., Vavryčuk, V., Archuleta,
R. J., & Halada, L. (2002). *3D heterogeneous staggered-grid
finite-difference modeling of seismic motion with volume harmonic
and arithmetic averaging of elastic moduli and densities.* Bulletin
of the Seismological Society of America 92(8): 3042-3066.
DOI: [10.1785/0120010167](https://doi.org/10.1785/0120010167).

**Orientation-aware co-anchors** (kept in this folder because the
production module implements all three in one file):

- Muir, F., Dellinger, J., Etgen, J., & Nichols, D. (1992). *Modeling
  elastic fields across irregular boundaries.* Geophysics 57(9):
  1189-1193. — Schoenberg-Muir calculus foundational paper.
- Koene, E. F. M., Wittsten, J., & Robertsson, J. O. A. (2022).
  *Eliminating staircase numerical artifacts at sloped fluid-solid
  interfaces by an effective medium method.* GJI 230(3): 1922-1937.
  DOI: [10.1093/gji/ggac164](https://doi.org/10.1093/gji/ggac164). —
  Production rotate-partition-average-rotate-back form (Eqs 23-27).
- Kristek, J., Moczo, P., Chaljub, E., & Kristeková, M. (2017). *An
  orthorhombic representation of a heterogeneous medium for the FD
  modelling of seismic wave propagation.* GJI 208(2): 1250-1264.
  DOI: [10.1093/gji/ggw451](https://doi.org/10.1093/gji/ggw451). —
  Axis-aligned orthorhombic limit (β=0 reduction).

## Why this reproduction exists

This package is the **canonical heterogeneous-interface effective-medium
library** for the parent repo. It's used by `00_common/tti_params.py:
build_layered_cij_arrays` to dispatch per-cell averaging via the
`interface_treatment` kwarg, supporting:

- `'sharp'` — pre-§85 sharp per-pixel classification (no averaging);
- `'moczo2002'` — axis-aligned isotropic-effective averaging;
- `'schoenberg_muir'` — general-orientation effective via
  rotate-partition-average-rotate-back. Production target for the
  dipping Petrobras interface.

Phase Y/2e 2026-05-28 relocates the effective-medium solver from
`00_common/effective_medium.py` into this reproduction folder so
the paper claims (arithmetic ρ / harmonic K / harmonic μ, fluid-limit
sentinels, identical-medium limits, axis-aligned reduction of the SM
form) live inside the reproduction that anchors them (QED guard FM2
— citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as byte-checkable
algebraic-form rules + scalar byte-match tests + asymptotic-limit
sentinels:

1. **Moczo 2002 averaging-rule sentinels**:
   - density: `arithmetic` (`MOCZO_2002_DENSITY_AVERAGE`);
   - K = λ + μ: `harmonic` (`MOCZO_2002_K_AVERAGE`);
   - μ: `harmonic` (`MOCZO_2002_MU_AVERAGE`).
   Silent-swap to the opposite scheme would flip the Voigt-Reuss
   bound and produce a wrong large-contrast limit.

2. **Closed-form scalar anchors byte-match the solver**:
   - `moczo_2002_rho_anchor(ρ1, ρ2, f1)` matches `out['rho']` to fp64;
   - `moczo_2002_K_anchor(K1, K2, f1)` matches `(out['c11']+out['c12'])/2`;
   - `moczo_2002_mu_anchor(μ1, μ2, f1)` matches `out['c66']` (pure
     solid-solid case).

3. **2D in-plane Voigt convention** `(xx, zz, xz)` (sentinel string).
   - C11 = K + μ, C12 = K − μ (algebraic anchors).
   - C16 = C26 = 0 (isotropic-effective has no shear-coupling).
   - C11 = C22 (rotational invariance).

4. **Fluid-limit sentinels**:
   - both μ = 0 ⇒ μ_eff = 0 (production both-fluid mask);
   - `MOCZO_2002_REGULARISER_MU_EPS = 1e-30` for mixed fluid-solid.

5. **Identical-medium limit** C1 = C2 ⇒ C_eff = C1 (any f1 ∈ (0, 1)).

6. **Single-phase limits** f1 = 0 ⇒ phase-2; f1 = 1 ⇒ phase-1.

7. **Schoenberg-Muir axis-aligned reduction sentinel**:
   `SCHOENBERG_MUIR_AXIS_ALIGNED_REDUCES_TO_MOCZO_2002 = True`.
   Locks the load-bearing reduction property; empirical equivalence
   check belongs in the parent's `tests/test_effective_medium.py`
   (which has access to both forms in one process).

8. **NumPy-array vectorisation** — the solver accepts per-cell arrays
   and broadcasts elementwise (required for `build_layered_cij_arrays`
   per-cell evaluation across (Nx, Ny) grids).

## File layout

```
doi_10.1785_0120010167/
├── pyproject.toml                          # declares moczo_2002 pkg
├── README.md                               # this file
├── moczo_2002/
│   ├── __init__.py                         # paper-anchored docstring
│   ├── effective_medium.py                 # 524-line solver
│   └── paper_tables.py                     # hand-transcribed anchors
└── tests/
    └── test_paper_tables.py                # byte + form + limit gates
```

The effective-medium solver (`effective_medium.py`) was relocated
from `00_common/effective_medium.py` without code changes.

## Quick start

```bash
# Standalone reproduction tests (this folder only):
cd reproduced/doi_10.1785_0120010167/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo (the reproduction is wired as an editable
dep via `[tool.uv.sources]`):

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1785_0120010167/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `moczo_2002` package. Parent imports use
`from moczo_2002.effective_medium import ...`. Phase Y/2e updated 3
call sites:
- `00_common/tti_params.py:222` (lazy import inside
  `build_layered_cij_arrays`)
- `tests/test_effective_medium.py:37` (top-level import)
- `tests/test_effective_medium.py:939` (lazy import inside the
  `cell_volume_fractions` regression test)

## Reproduction provenance classification

**`published`** (Phase Y/2e initial classification 2026-05-28). The
algebraic-form sentinels in `paper_tables.py` lock the Moczo 2002
averaging rules; the closed-form scalar anchors byte-match the solver
to fp64; the identical-medium / single-phase / both-fluid asymptotic
limits are all sentinel-tested. The Schoenberg-Muir axis-aligned
reduction is locked as a sentinel here and empirically cross-checked
in the parent's existing test suite.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.

## Note on the multi-paper scope

`effective_medium.py` implements three averaging schemes from three
papers (Moczo 2002 + Muir 1992 + Koene-Robertsson 2022, with Kristek
2017 as the orthorhombic-reduction limit). Per the Phase Y plan's
multi-paper anchor convention, **Moczo 2002 is the primary anchor**
(folder DOI). The other three are documented as co-references in this
README and their forms are sentinel-locked in `paper_tables.py`.
Standalone reproduction folders for Muir 1992, Koene-Robertsson 2022,
and Kristek 2017 are future-work follow-ons.
