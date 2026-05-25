# Igel, Mora & Riollet (1995) — Anisotropic SSG with Sinc-Gaussian Derivative + Cross-Derivative Averaging

Devito-companion reproduction of the canonical SSG-on-general-
anisotropic-media paper.

## Paper

> Igel, H., Mora, P. & Riollet, B. (1995). "Anisotropic wave
> propagation through finite-difference grids." *Geophysics*
> 60(4), 1203–1216. DOI
> [10.1190/1.1443849](https://doi.org/10.1190/1.1443849).

## Scheme

Igel et al. (1995) extend the Virieux 1986 / Levander 1988
isotropic P-SV SSG framework to **general anisotropic media**.
Two mechanisms:

1. **Truncated-sinc-Gaussian derivative operator** (Eq 33 +
   tapering). The infinite-series exact coefficient at
   half-grid position `(n+½)Δx` is
   `p_∞(n) = (-1)ⁿ / (π(n+½)² Δx)`; truncated to L points and
   Gaussian-tapered to reduce edge effects. Table 1 (page 1209)
   gives the L=8 coefficients explicitly.
2. **Cross-derivative averaging** for off-diagonal Cij
   couplings (`C16`, `C26`, etc.) — Eq 25 + surrounding text.
   Single and double interpolations of the velocity-derivative
   tensor components ensure all symmetry classes higher than
   monoclinic recover faithfully when axes are aligned;
   tilted-axis triclinic media incur the per-position
   interpolation errors quantified in Fig 6 + Fig 7.

The **cross-derivative averaging methodology** is what the
parent `devitocodespro/devito-fd-survey` Method 1 SSG borrows
for its TTI extension. The truncated-sinc-Gaussian derivative
weights are NOT the Levander Taylor staggered weights used by
Method 1 — that's a separate choice. The reproduction here
implements Igel's specific scheme; the parent repo's
production uses Levander weights + Igel-style averaging.

## What this reproduction validates (quantitative anchors)

Per the repo-wide
[`feedback_reproduction_quantitative_first`](https://github.com/devitocodespro/reviews-reproduced/blob/main/CLAUDE.md)
rule, the reproduction is anchored on tables / explicit
numerical claims, NOT figure-by-eye reproduction:

| # | Anchor | Where in paper | Test |
|---|---|---|---|
| 1 | Triclinic Cij matrix (21 components) | Eq 47 (page 1210) | `test_cij_eq47_byte_match` |
| 2 | 8-point sinc-Gaussian d_T*, d_P* coefficients + stability ε | Table 1 (page 1209) | `test_table_1_L8_values_byte_match_paper` |
| 3 | Max relative phase-velocity error at 50% Nyquist | Page 1209 + Fig 6: ≤ 2% (qP), ≤ 3% (qS1), ≤ 7% (qS2) | `test_phase_velocity_error_L8_50pct_nyquist_within_paper_bounds` |

Plus reference-output regression at each `space_order ∈
{2, 4, 8, 16}` per the repo-wide so=16 convention.

## Configuration

The dispersion analysis runs over angles in the xz-plane for
the triclinic medium of Eq 47 (ρ=1.0 g/cm³, 21 independent
Cij values). At each angle:

1. Analytical phase velocities via the 2D Christoffel
   equation (eigenvalues of `c_pqrs n_q n_s / ρ`).
2. Per-direction modified wavenumber from the stencil:
   `k_eff(k) = sum(coeffs[n] · sin(k·(n+½)))`.
3. Numerical phase velocities via Christoffel with scaled
   `(k_x, k_z) → (k_eff_x, k_eff_z)` (Igel §"The numerical
   wave properties").
4. Per-wave-type max relative error over the sampled
   directions, at Nyquist fractions 30%, 50%, 70%.

This matches the analysis the paper itself performs — no
wave-propagation simulation is needed (or used) for the
quantitative verification.

## Usage

```bash
uv sync
uv run python run_reproduction.py        # all so ∈ {2, 4, 8, 16}
uv run python run_reproduction.py -s 8   # canonical L=8 only
uv run pytest tests/ -v
```

Empirical output for L=8 (canonical):
```
qP  @ 50% Nyquist: max rel-err = 0.81%   (paper bound 2%)
qS1 @ 50% Nyquist: max rel-err = 0.82%   (paper bound 3%)
qS2 @ 50% Nyquist: max rel-err = 0.82%   (paper bound 7%)
```

## Outputs

Pinned in `reference_outputs/`:

| File | Contents |
|---|---|
| `dispersion_so2.npz`, `so4.npz`, `so8.npz`, `so16.npz` | For each space_order = L: `phase_vel_analytic[wave, angle]`, `phase_vel_numerical[wave, angle, nyquist_frac]`, `rel_error_max[wave, nyquist_frac]`, plus `cij_eq47`, `stencil_coeffs`. |

## Relationship to parent `devitocodespro/devito-fd-survey`

This folder is the third reproduction prerequisite for
Method 1 SSG (per round-4 plan). The parent's
`MethodSpec.reproduction_dois` for Method 1 includes:

- `doi_10.1190_1.1442147` (Virieux 1986, foundational)
- `doi_10.1190_1.1442422` (Levander 1988, 4th-order)
- `doi_10.1190_1.1443849` (this folder — anisotropic
  extension)

The parent's `_test_ssg_virieux_levander_stencil` callback
extends to exercise the TTI cross-derivative averaging by
comparing the production SSG averaged-path output against
this reproduction's dispersion-error claims at a non-trivial
TTI configuration.

## Graduation review

TBD — runs once the parent repo's Method 1
`reproduction_dois` is updated to include this folder and the
SSG callback Invariants 4-5 land.

## References

- Igel, Mora, Riollet 1995 DOI: [10.1190/1.1443849](https://doi.org/10.1190/1.1443849)
- Companion isotropic-SSG folders: [`../doi_10.1190_1.1442147/`](../doi_10.1190_1.1442147/) (Virieux 1986), [`../doi_10.1190_1.1442422/`](../doi_10.1190_1.1442422/) (Levander 1988)
- Reproduction methodology: [`../CLAUDE.md`](../CLAUDE.md)
- Repo-wide quantitative-first convention: `feedback_reproduction_quantitative_first` (in user memory)
