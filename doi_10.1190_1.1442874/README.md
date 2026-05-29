# Pratt (1990) 2D acoustic Green's image-method reference

**Primary paper**: Pratt, R. G. (1990). *Frequency-domain elastic
wave modeling by finite differences; a tool for crosshole seismic
imaging.* Geophysics 55(5): 626-632.
DOI: [10.1190/1.1442874](https://doi.org/10.1190/1.1442874).

## Why this reproduction exists

This package is the parent repo's **closed-form 2D acoustic Green's
function reference** — used as the V1 verification gate's
homogeneous-Green's reference in `tests/test_dwn_homogeneous.py` and
as the image-method reference in
`scripts/run_acoustic_image_reference.py`.

The Pratt 1990 paper is the seismic-modelling community's reference
for the **line-source projection** from 3D to 2D Green's functions:
the half-derivative-in-time relationship that turns the 3D
point-source Green's function

    G_3D(r, t) = δ(t − r/c) / (4π r)

into the 2D line-source equivalent

    G_2D(r, t) = H(t − r/c) / (2π √(t² − r²/c²)).

The Heaviside `H(·)` enforces causality; the integrable
inverse-square-root singularity at the wavefront is the famous 2D
"tail" — the wavefield rings after the front passes (unlike 3D
where it's impulsive).

Phase Y/2g 2026-05-28 relocates the closed-form 2D acoustic image-
method solver from `00_common/analytical_acoustic_2d.py` into this
reproduction folder so the paper claims (Green's function form,
causality, normalisation prefactor, angle-dependent reflection R(p),
post-critical evanescent |R|=1) live inside the reproduction that
anchors them (QED guard FM2 — citation hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as
sentinel-string-locked algebraic forms + scalar byte-match anchors +
asymptotic-limit guards:

1. **G_2D(r, t) form sentinel** locks the time-domain Heaviside form
   against silent swaps to the frequency-domain Hankel form
   `G_ω(r) = -i/4 · H_0^(2)(ωr/c)` (an alternative documented as a
   co-existing-form sentinel).

2. **Normalisation prefactor** `1/(2π)` byte-anchored to fp64; the
   2D/3D-prefactor ratio is exactly 2 (locks the line-source
   projection vs point-source distinction).

3. **Causality** `H(t − r/c)` sentinel + wavefront-arrival anchor
   `pratt_1990_arrival_time(r, c) = r/c`. Tests verify the solver
   produces no signal before arrival.

4. **2D-tail-vs-3D-impulsive sentinels** lock the 1/sqrt-singularity
   tail behaviour (Huygens' violation in 2D); flipping these flags
   would document an opposite-dimension Green's function.

5. **Normal-incidence acoustic R formula** `R = (Z₂ − Z₁) / (Z₂ + Z₁)`
   byte-anchored: solver matches anchor to fp64 at three canonical
   `(ρ_u, c_u, ρ_l, c_l)` configurations including the water-over-TTI
   reference value `R ≈ 0.6296` exactly.

6. **Angle-dependent R(p) formula** sentinel
   `R(p) = (Z₂ cos θ₁ − Z₁ cos θ₂) / (Z₂ cos θ₁ + Z₁ cos θ₂)`.
   Empirical verification:
   - At `p = 0` (normal incidence), the angle-dependent form
     equals the normal-incidence closed form to fp64.
   - At `p` 1% past critical (`p > 1/V_lower` for faster lower
     medium), `|R| = 1` to fp64 (post-critical total reflection
     sentinel).

7. **Mirror-source reduction at β = 0** byte-anchored:
   `(x_s, y_s) → (x_s, 2·y_anchor − y_s)`. Plus a geometric-invariant
   test that mirror image preserves perpendicular distance from the
   interface line for general β.

## File layout

```
doi_10.1190_1.1442874/
├── pyproject.toml                          # declares pratt_1990 pkg
├── README.md                               # this file
├── pratt_1990/
│   ├── __init__.py                         # paper-anchored docstring
│   ├── analytical_acoustic_2d.py           # 480-line solver
│   └── paper_tables.py                     # hand-transcribed anchors
└── tests/
    └── test_paper_tables.py                # 21 sentinel + byte-match +
                                            # asymptotic-limit + empirical-
                                            # post-critical tests
```

## Quick start

```bash
# Standalone reproduction tests:
cd reproduced/doi_10.1190_1.1442874/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo:

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1190_1.1442874/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `pratt_1990` package. Parent imports use
`from pratt_1990.analytical_acoustic_2d import ...`. Phase Y/2g
updated 2 caller sites:
- `tests/test_dwn_homogeneous.py` — V1 homogeneous Green's gate
- `scripts/run_acoustic_image_reference.py` — image-method
  reference generator

## Reproduction provenance classification

**`published`** (Phase Y/2g initial classification 2026-05-28). The
G_2D form + normalisation prefactor + causality + angle-dependent R
formula + post-critical-evanescent sentinel are all locked; the
scalar byte-match anchors (normal-incidence R, p=0 reduction)
match the solver to fp64; the asymptotic-limit empirical checks
(post-critical |R| = 1, mirror-source geometric invariant) pass.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.
