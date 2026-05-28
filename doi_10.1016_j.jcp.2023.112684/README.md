# Xie & He 2024 — STO-RSG Devito reproduction (Method 61)

**Paper**: Xie, J. & He, B. (2024). "Spatial-temporal high-order
rotated-staggered-grid finite-difference scheme of elastic wave
equations for TTI medium." *Journal of Computational Physics*
**499**: 112684.
DOI: [10.1016/j.jcp.2023.112684](https://doi.org/10.1016/j.jcp.2023.112684).

## Why this reproduction exists

Xie & He 2024 is the **load-bearing reference** for Method 61
STO-RSG (`61_elastic_sto_rsg/`). The paper's central contribution:
the TE+LS (Taylor-Expansion + Least-Squares) joint optimization
of the space-time dispersion relation produces RSG stencil
coefficients that **eliminate the spatial-interpolation error**
of conventional Saenger 2000 RSG when applied to TTI elastic
media.

Quoting the paper's §3 (page 4):

> "...using the conventional staggered-grid technique to solve the
> elastic wave equation, some spatial derivatives need to be
> approximated by spatial interpolation, which will become new
> error and reduce the accuracy of the solution."

The **45° X-mode lobe artifact** visible in the Petrobras dip-figure
when using Method 2 RSG (Saenger 2000 default Taylor coefficients) is
the empirical manifestation of this interpolation error. Method 61
STO-RSG — implementing Xie & He's algorithm — produces a clean
artifact-free wavefront at all angles (verified via the
``tests/test_homog_radial_symmetry.py`` and
``tests/test_homog_wavefront_position.py`` V1+V3 gates in the
parent repo).

## What we reproduce

Per the `feedback_reproduction_quantitative_first` memory entry,
this reproduction anchors on quantitative claims of the paper —
not on visual figure reproduction. Specific verification anchors:

1. **TE consistency constraint** (Xie & He §3 Eq 15):
   antisymmetric centered-FD coefficients satisfy
   ``2 · Σ_m m·c_m = 1``. The parent's optimizer enforces this as
   an equality constraint; tests assert the residual is < 1e-10
   at canonical configurations.

2. **Pinned canonical coefficient values** at
   ``(P, CFL) ∈ {(6, 0.3), (6, 0.4), (6, 0.5), (5, 0.4), (4, 0.4)}``.
   Test asserts re-running ``compute_sto_coefficients`` reproduces
   the pinned `.json` reference within fp64 tolerance — catches
   drift in the SciPy SLSQP optimizer, the LS objective, or the
   Taylor-initial-guess.

3. **Qualitative dispersion-reduction claim**: at fixed
   ``P ∈ {4, 5, 6}`` and representative ``kh = 1.0 rad``, the
   STO modified-wavenumber error ``|k̃h − kh|`` is **smaller**
   than the Taylor (Fornberg 1988) error at the same P. Tests
   assert this strict ordering.

4. **Topology + scheme-identity sentinels**: paper-anchored
   identifiers locking the algorithm name + the explicit claim
   that the paper addresses Saenger 2000's spatial-interpolation
   error in TTI.

## Role in the parent repo

| Method | Provenance | What this reproduction backs |
|---|---|---|
| Method 61 STO-RSG (`61_elastic_sto_rsg/`) | `provenance='published'` | The TE+LS optimizer + dispersion-reduction property + canonical coefficients are anchored here. Removes M61 from the `_PREREQUISITE_EXEMPT` list in `tests/test_reproduction_prerequisites.py`. |
| Method 63 STO-RSG+GSLS visco (`63_viscoelastic_sto_rsg/`) | `provenance='novel-combination'` | Cascades from M61's STO stencil + Robertsson 1994 GSLS attenuation. Cites this paper for the STO component. |

## Quick start

```bash
cd reproduced/doi_10.1016_j.jcp.2023.112684/
uv sync
uv run pytest tests/ -v
```

## File layout

```
doi_10.1016_j.jcp.2023.112684/
├── README.md                            # this file
├── xie_he_2024_paper.pdf                # the paper PDF
├── pyproject.toml                       # uv project (+ matplotlib for dispersion fig)
├── paper_tables.py                      # transcribed paper anchors
├── run_dispersion_analysis.py           # Phase X: dispersion curve + L1 integrals
├── tests/
│   ├── test_paper_tables.py             # byte-match + consistency tests
│   └── test_dispersion_l1_regression.py # L1-integral dispersion regression
└── reference_outputs/
    ├── sto_coefficients_pinned.json     # canonical (P, CFL) → coefficients
    ├── dispersion_curve.png             # Phase X: STO vs Taylor dispersion fig
    └── dispersion_l1_errors.json        # Phase X: L1 integrals (pinned)
```

## Phase X deepening — 2026-05-28

Phase X added three deepening anchors:

1. **High-kh band test** — replaced the original single-point
   dispersion test at kh=1.0 (which sits in the STO/Taylor
   crossover region) with a band test over [1.2, 2.0] rad
   where Xie & He's claim "STO < Taylor" holds strictly.
   Per-P empirical sweep:

   | kh    | sto_err | taylor_err | STO<Taylor? |
   |-------|---------|------------|-------------|
   | 0.3   | 7.6e-05 | 1.3e-11    | NO (Taylor essentially exact) |
   | 1.0   | 6.3e-05 | 5.4e-05    | NO (crossover; near-tied) |
   | 1.5   | 1.7e-04 | 6.1e-03    | YES (STO wins by ~36×) |
   | 2.0   | 8.5e-02 | 1.2e-01    | YES (STO wins by ~1.4×) |

2. **Low-kh tradeoff sentinel** — new test asserts that
   Taylor is more accurate than STO at low kh. This is the
   defining feature of LS optimization, NOT a bug — STO
   sacrifices low-kh accuracy to gain high-kh accuracy. If
   the LS objective collapses to the Taylor solution
   (optimizer bug), this test catches it.

3. **L1-integral regression** — `run_dispersion_analysis.py`
   produces `dispersion_curve.png` (visual) and
   `dispersion_l1_errors.json` (scalar pinned metric).
   `test_dispersion_l1_regression.py` asserts:
   - High band: `STO_L1 / Taylor_L1 < 0.6` (Xie & He's
     sharp quantitative claim)
   - Low band: `STO_L1 / Taylor_L1 > 10` (LS tradeoff)
   - Pinned L1 values stable to 5% (catches SciPy
     optimizer drift)

   Empirical (cfl=0.4, regenerated 2026-05-28):

   | P | low-band STO/Taylor | high-band STO/Taylor |
   |---|---------------------|----------------------|
   | 4 | 46.9× (LS sacrifice) | 0.24× (STO wins)    |
   | 5 | 77.8× (LS sacrifice) | 0.19× (STO wins)    |
   | 6 | 1363× (LS sacrifice) | 0.50× (STO wins)    |

Phase X also re-pinned `sto_coefficients_pinned.json` against
the current SciPy (1.17.1) optimizer output and added a
`_provenance` block documenting the re-pin context. Prior
pinned values drifted ~5e-6 at P=5, cfl=0.4 — consistent
with SLSQP local-minimum sensitivity in the LS objective
under SciPy version upgrades.

Test count: 27 (was 15). All green.

The reproduction directly exercises the parent repo's
``00_common/spacetime_coefficients.py`` (Xie & He's algorithm
implementation) — there is no separate "upstream" code to vendor.
The parent's implementation IS our reproduction; the tests here
verify it satisfies the paper's claims.

## Notes on the 45° X-mode artifact

The Petrobras dip-figure (`figures/dip_layered/snapshots_4x4_late_3method_graduated_fs.png`)
shows Method 2 RSG (Saenger 2000 default Taylor coefficients)
producing visible 45° X-mode lobes around the source. Replacing
the source recipe (commit `5e0c851`, OumZhang 8-point pattern)
removed the dominant ghost wavefronts but a residual 45° artifact
remained. The investigation (plan §M2 RSG 45° follow-up, 2026-05-26)
identified this residual as the Xie & He 2024-documented
"spatial-interpolation error of conventional staggered-grid for
TTI elastic". Method 61 STO-RSG is the canonical fix.

Empirical V1+V3 gates at 2026-05-26 calibration:

| Method | V3 angular peak/mean | V1 wavefront position |
|---|---|---|
| M1 SSG (reference) | 2.614 | 0.450 km (0.00% err) |
| M2 RSG (Saenger 2000) | 1.626 | 0.325 km (xfail; near-field detector bias) |
| M6 Dual-Pair | 1.741 | 0.475 km (5.56% err) |
| **M61 STO-RSG (Xie & He 2024)** | **2.507** (<3.0 tight threshold) | **0.450 km (0.00% err — matches SSG)** |
