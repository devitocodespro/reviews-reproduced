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
├── README.md                          # this file
├── xie_he_2024_paper.pdf              # the paper PDF
├── pyproject.toml                     # uv project
├── paper_tables.py                    # transcribed paper anchors
├── tests/
│   └── test_paper_tables.py           # byte-match + consistency tests
└── reference_outputs/
    └── sto_coefficients_pinned.json   # canonical (P, CFL) → coefficients
```

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
