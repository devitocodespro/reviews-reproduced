# Hao et al. (2022) two-tier Kjartansson constant-Q viscoelastic TI

**Primary paper**: Hao, Q., Greenhalgh, S., Huang, X., & Li, H. (2022).
*Viscoelastic wave propagation for nearly constant-Q transverse
isotropy.* Geophysical Prospecting 70(7): 1176-1192.
DOI: [10.1111/1365-2478.13230](https://doi.org/10.1111/1365-2478.13230).

**Historical co-anchor**: Kjartansson, E. (1979). *Constant-Q wave
propagation and attenuation.* Journal of Geophysical Research: Solid
Earth 84(B9): 4737-4748.
DOI: [10.1029/JB084iB09p04737](https://doi.org/10.1029/JB084iB09p04737).

## Why this reproduction exists

This package is the **canonical Kjartansson constant-Q library** for the
parent repo. Sixteen method directories (Methods 64-67, 72-75, 77-78,
80-81 plus `00_common/stability.py` and `00_common/viscoacoustic_mms.py`)
depend on its memory-variable construction + CN time-discretisation.

Phase Y/2d 2026-05-28 relocates the Kjartansson solver from
`00_common/kjartansson.py` into this reproduction folder so the paper
claims (gamma formula, two-tier rate ratio, CN recurrence form, K=2
memory-field counts) live inside the reproduction that documents them
(QED guard FM2 — citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first` convention,
this reproduction anchors paper claims as byte-checkable constants +
closed-form formula assertions:

1. **Kjartansson 1979 gamma formula** `gamma = (1/pi) * arctan(1/Q)`
   (Kjartansson 1979 §3, Hao 2022 eq 3). Verified to fp64 at Q ∈
   {1, 10, 20, 40, 100}; the low-loss limit `gamma * pi * Q → 1`
   (Q → ∞) is also tested. The solver's `kjartansson_gamma(Q)` is
   asserted byte-equal to `paper_tables.kjartansson_gamma_anchor(Q)`.

2. **Hao 2022 two-tier rate prescription** `a_tilde = a / 2` —
   tier-2 relaxation rate is exactly half of tier-1 (per
   `kjartansson.py:94`). The test recovers `a` and `a_tilde` from the
   solver's `alpha_1` and `alpha_2` via the CN inversion
   `a = 2 * (1 - alpha) / (dt * (1 + alpha))`, then asserts the ratio
   is `0.5` to fp64. This is the defining feature of the two-tier
   method.

3. **Crank-Nicolson recurrence form** preserved:
   `alpha = (1 - a*dt/2) / (1 + a*dt/2)`,
   `beta = b*dt / (1 + a*dt/2)`.
   Sentinel string tests catch silent swaps to forward- or
   backward-Euler.

4. **K=2 memory-field counts**: exactly 6 / 4 / 2 for viscoelastic /
   viscoacoustic-coupled / viscoacoustic-scalar formulations.
   Flipping the tier count (e.g., to K=3) would break the counts.

5. **Stress-correction tier weights byte-match**:
   `c1 = 1/Q`, `c2 = gamma(Q) / (2 Q)` per `kjartansson.py:107-108`.

6. **Tier-amplitude ratio 2:1** between tier-1 (`b = 2 gamma omega_0`)
   and tier-2 (`b_tilde = gamma omega_0`). Recovered from solver
   `beta_1`/`beta_2` by inversion.

7. **Kjartansson 1979 complex-modulus form sentinel**:
   `M(omega) = M_0 * (i omega / omega_0)^{2 gamma}`.
   Flipping to an SLS L=1 viscoelastic modulus would change the
   frequency dependence (paper-defining form).

## File layout

```
doi_10.1111_1365-2478.13230/
├── pyproject.toml                          # declares hao_2022 pkg
├── README.md                               # this file
├── hao_2022/
│   ├── __init__.py                         # package docstring
│   ├── kjartansson.py                      # the 607-line Kjartansson solver
│   └── paper_tables.py                     # hand-transcribed paper anchors
└── tests/
    └── test_paper_tables.py                # byte-equality + form sentinels
```

The Kjartansson solver (`kjartansson.py`) was relocated from
`00_common/kjartansson.py` without code changes; it implements the Hao
2022 two-tier weighting-function method per its module docstring.

## Quick start

```bash
# Standalone reproduction tests (this folder only):
cd reproduced/doi_10.1111_1365-2478.13230/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo (the reproduction is wired as an editable dep
via `[tool.uv.sources]`):

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1111_1365-2478.13230/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `hao_2022` package. Parent imports use
`from hao_2022.kjartansson import ...`. Phase Y/2d updated 22 call
sites across method directories, tests, and `00_common/` to point at
the new package.

## Reproduction provenance classification

**`published`** (Phase Y/2d initial classification 2026-05-28). The
static paper-anchor constants in `paper_tables.py` byte-match the
canonical Kjartansson 1979 gamma formula and the Hao 2022 defining
ratios exactly; the K=2 memory structure is sentinel-tested; the
integration with sixteen parent method directories provides
independent empirical validation through the parent's MMS-convergence
+ paper-faithfulness gates.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.

## Note on scope of "byte-match"

The `paper_tables.py` anchors that are pure constants
(`KJARTANSSON_GAMMA_SAMPLES`, `HAO_2022_TIER2_RATE_RATIO`,
`HAO_2022_MEMORY_FIELDS_*`) byte-match the paper to fp64. The CN
coefficient formulas (`alpha`, `beta`, `c1`, `c2`) are paper-canonical
closed-form expressions; their *numerical values* depend on the
runtime triple `(Q, f0, dt)` so cannot be tabulated, but the
formulas themselves are byte-equality-asserted between the paper
anchor and the solver's implementation.

This is consistent with the project's claim-honesty discipline:
constants are byte-match; recurrence forms are sentinel-string-locked;
sample-table values are fp64-verified at canonical inputs.
