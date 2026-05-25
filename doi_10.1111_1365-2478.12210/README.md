# Di Bartolo, Dors & Mansur (2015) — Equivalent Staggered-Grid Theory

The canonical TTI-elastic anchor paper for parent repo Method 1 SSG.

## Paper

> Di Bartolo, L., Dors, C. & Mansur, W.J. (2015). "Theory of
> equivalent staggered-grid schemes: application to rotated and
> standard grids in anisotropic media." *Geophysical Prospecting*
> 63(5), 1097–1125. DOI
> [10.1111/1365-2478.12210](https://doi.org/10.1111/1365-2478.12210).

## Scheme

Di Bartolo et al. develop the **Equivalent Staggered-Grid (ESG)
theory** — a unifying framework that proves SSG (Virieux 1986 /
Levander 1988), RSG (Saenger-Gold-Shapiro 2000), and NSG (Kelly
1976) schemes are algebraically equivalent under averaging
operators for anisotropic elastic media. Critically, the paper
formalises the 4-corner cross-derivative averaging that Igel et
al. 1995 introduced ad-hoc, and proves it gives the
literature-canonical SSG scheme for off-diagonal Cij coupling.

This is the paper that explains **why** the parent repo's
`run_ssg(cross_deriv='averaged')` production path is the right
one for TTI elastic — by deriving the equivalence map between
RSG (which physically averages by construction) and SSG (which
must average algebraically).

## What this reproduction validates (quantitative anchors)

Per the repo-wide
[`feedback_reproduction_quantitative_first`](../CLAUDE.md) rule:

| # | Anchor | Where in paper | Test |
|---|---|---|---|
| 1 | Table 2 VTI elastic properties for 10 rocks (vp, vs, ρ, ε, δ, ν, c11, c13, c33, c55) | Page 1117 | `test_table_2_*`, `test_water_row_byte_match`, `test_coal_row_byte_match`, `test_high_anisotropy_rocks_present` |
| 2 | Table 3 numerical parameters (Δt=100 μs, h=1 m, f₀=60 Hz, Nx=13601, Nz=2801, Ntotal=16000, t_total=1.6 s) | Page 1117 | `test_table_3_marmousi_parameters_byte_match`, `test_table_3_temporal_consistency` |
| 3 | ESG 4-corner averaging kernel = uniform 1/4 weights | Eq 42-43 (page 1104) | `test_esg_averaging_kernel_byte_match`, `test_esg_kernel_weights_sum_to_one`, `test_esg_4corner_average_preserves_constants`, `test_esg_4corner_average_linear_exactness` |
| 4 | Poisson's ratio ν computed from (vp, vs) reproduces the paper's ν column | Table 2 column | `test_poisson_ratio_matches_paper_table[0..9]` |
| 5 | Water + Coal Thomsen→Cij identities (isotropic / acoustic, strict Thomsen) | Standard Thomsen 1986 | `test_water_and_coal_thomsen_consistency_strict` |

## Known limitation: Thomsen parameterisation of Table 2

For the **anisotropic** rocks in Table 2 (Clay shale, Oil
shale, Crystal, Quartz, LS shale, Anis. shale, Gypsum), the
tabulated (c11, c13, c33, c55) do NOT strictly satisfy
Thomsen 1986 c₁₃ identity from the tabulated (vp, vs, ρ, ε, δ).
Strict Thomsen would give c₁₃ values that differ from the
table by 70-100% relative for some rocks.

The paper does not state explicitly which parameterisation was
used (extended-Tsvankin? Hybrid? Independently measured Cij?
Direct rock-physics data?). The literal table values are still
**byte-checkable** as transcribed; what's NOT byte-checkable is
the round-trip through Thomsen.

This is recorded as `test_anisotropic_rocks_thomsen_strict_consistency`
with `@pytest.mark.xfail(...)` and a docstring documenting the
known mismatch — informational, not a hard gate.

Per `feedback_reproduction_quantitative_first`, the literal
values byte-check is the authoritative reproduction gate.

## Configuration (dispersion / theoretical only)

This reproduction is **theory-anchored** rather than running a
full Devito Marmousi simulation:

- Table 2 VTI rock properties are loaded as Python constants
  and byte-checked.
- Table 3 numerical parameters are loaded + cross-validated for
  internal consistency.
- ESG ↔ SSG/RSG averaging-equivalence kernels are validated
  symbolically + numerically (linear-field exactness,
  constant-preservation).
- Poisson's ratio computation reproduces the paper's ν column
  to ~3e-3 relative.

Cross-validation against the parent repo's production
`run_ssg(cross_deriv='averaged')` path happens upstream in the
parent's faithfulness callback (Invariants 4/5, sub-steps
5b-10/11) — not duplicated here per refactor plan decision T(c).

## Usage

```bash
uv sync
uv run python run_reproduction.py
uv run pytest tests/ -v
```

## Role in parent `devitocodespro/devito-fd-survey`

This folder is the **TTI-elastic literature anchor** for parent
repo Method 1 SSG (4th reproduction folder per round-4 plan
decision U, Path 2). The parent's
`MethodSpec.reproduction_dois` for Method 1 includes:

- `doi_10.1190_1.1442147` (Virieux 1986, isotropic foundational)
- `doi_10.1190_1.1442422` (Levander 1988, 4th-order isotropic)
- `doi_10.1190_1.1443849` (Igel-Mora-Riollet 1995, generic
  anisotropic SSG, dispersion analysis with sinc-Gaussian
  stencil)
- `doi_10.1111_1365-2478.12210` (THIS folder — Di Bartolo
  2015, equivalent-SG theory for TTI elastic)

The parent's strengthened SSG paper-faithfulness callback
will:
- Verify the production cross-derivative averaging IS the
  4-corner mean per Eq 42-43 (this folder's `esg_kernel`).
- Run the production averaged-SSG path on a Table 2 VTI rock
  (Crystal: ε=1.12, δ=-1.23, extreme anisotropy) and verify
  stability + dispersion match the paper's claims.

## Graduation review

TBD — runs once parent repo's `MethodSpec.reproduction_dois[1]`
is updated to the 4-tuple and SSG callback Invariants 4/5
land (sub-step 5b-10/11).

## References

- Di Bartolo, Dors & Mansur 2015 DOI: [10.1111/1365-2478.12210](https://doi.org/10.1111/1365-2478.12210)
- Companion isotropic-SSG folders: [`../doi_10.1190_1.1442147/`](../doi_10.1190_1.1442147/), [`../doi_10.1190_1.1442422/`](../doi_10.1190_1.1442422/)
- Companion anisotropic-SSG folder: [`../doi_10.1190_1.1443849/`](../doi_10.1190_1.1443849/) (Igel 1995)
- Reproduction methodology: [`../CLAUDE.md`](../CLAUDE.md)
- Repo-wide quantitative-first convention: `feedback_reproduction_quantitative_first` (user memory)
