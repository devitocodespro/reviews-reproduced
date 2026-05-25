# Virieux (1986) — P-SV Velocity-Stress Staggered-Grid

Devito reproduction of the foundational 2D elastic wave-propagation
scheme.

## Paper

> Virieux, J. (1986). "P-SV wave propagation in heterogeneous media;
> velocity-stress finite-difference method." *Geophysics* 51(4),
> 889–901. DOI [10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

## Scheme

2D in-plane (P-SV) elastic wave propagation written as a first-order
velocity-stress hyperbolic system on a Yee-style staggered grid:

```
∂t vx  = (1/ρ) (∂x σxx + ∂y σxy)
∂t vy  = (1/ρ) (∂x σxy + ∂y σyy)
∂t σxx = (λ + 2μ) ∂x vx + λ ∂y vy
∂t σyy = λ ∂x vx + (λ + 2μ) ∂y vy
∂t σxy = μ (∂y vx + ∂x vy)
```

Field placement (Virieux Fig. 1):

| Field | Position |
|---|---|
| `vx`  | `(i + ½, j)` |
| `vy`  | `(i,   j + ½)` |
| `σxx` | `(i,   j)` |
| `σyy` | `(i,   j)` |
| `σxy` | `(i + ½, j + ½)` |

Virieux 1986 is the 2nd-order-in-space scheme (Eq. 4); the
companion folder
[`../doi_10.1190_1.1442422/`](../doi_10.1190_1.1442422/) carries the
4th-order extension by Levander (1988).

## Implementation

This Devito reproduction uses first-class `VectorTimeFunction` /
`TensorTimeFunction` types so the half-grid staggering and cross-
stagger derivative shifts are handled by the framework's
`_eval_at` machinery (see Devito skill reference
`staggered-grids.md`). The hand-staggered version of the same
scheme using individual `TimeFunction` instances was numerically
unstable at `space_order ≥ 4` — see the git history of this
folder.

Per the repo-wide `space_order` convention
([`feedback_space_order_16_in_sweeps`](../../.claude/projects/.../feedback_space_order_16_in_sweeps.md)),
the driver parametrises `space_order ∈ {2, 4, 8, 16}`. The
**canonical Virieux 1986 claim is at `space_order=2`**; higher
orders are documented as natural Taylor-truncation extensions of
the same scheme (Levander 1988 formalises `space_order=4`; SO=8
and SO=16 are beyond either paper's formal scope but exercised
for cross-validation).

## Configuration

| Parameter | Value | Notes |
|---|---:|---|
| Domain | 4 km × 4 km | NX = NY = 401 (`dx = 10 m`) |
| Vp | 3.0 km/s | Homogeneous isotropic medium |
| Vs | 1.7 km/s | |
| ρ | 2.2 g/cm³ | |
| Source | Ricker, f₀ = 20 Hz, t₀ = 1.5/f₀ | Explosive (isotropic, injected into σxx + σyy) at domain centre |
| Final time | 0.4 s | Wavefront at ≈ 1.2 km radius — well inside the 2 km half-extent (no boundary contamination at snapshot time) |
| CFL | per `CFL_BY_ORDER` table | Saenger-Bohlen 2004 Table 2 + extrapolation for SO=16 |

Quantities are scaled to repo-standard units (km / km·s⁻¹ / g·cm⁻³),
keeping all coefficients O(1) per Devito skill numerical-precision
guidance.

## Usage

```bash
# Per-folder uv environment isolates this reproduction from the
# parent repo and other reproductions (standalone-isolation discipline
# documented in ../CLAUDE.md).
uv sync

# Generate reference outputs at every space_order:
uv run python run_reproduction.py

# Run a single space_order:
uv run python run_reproduction.py --space-order 2

# Run tests:
uv run pytest tests/ -v
```

## Tests

`tests/test_reproduction.py` covers four invariant classes:

| Class | Test(s) | What it gates |
|---|---|---|
| **Paper-faithful MMS** | `test_so2_paper_faithful_convergence_rate`, `test_so4_paper_faithful_convergence_rate`, `test_so8_convergence_rate`, `test_so16_convergence_rate` | Devito's `.dx` at each SO achieves the formal accuracy claimed by Virieux 1986 (rate ≈ 2 at SO=2) and Levander 1988 (rate ≈ 4 at SO=4). Higher orders verify saturation at near-machine precision. |
| **Reference-output regression** | `test_reference_output_matches_pin[2,4,8,16]` | Re-running the driver produces wavefields byte-identical to the pinned `reference_outputs/wavefield_so<N>.npz`. Catches Devito version drift + accidental driver edits. |
| **Physical sanity** | `test_wavefield_is_finite_and_bounded[2,4,8,16]`, `test_explosive_source_radial_p_wave_symmetry` | Wavefields are finite, bounded, and exhibit cos(φ) azimuthal symmetry for the P-wave radial particle velocity from an isotropic explosive source. |
| **CFL discipline** | `test_cfl_dt_respects_per_order_bound[2,4,8,16]` | `dt` from `cfl_dt()` respects the per-order CFL bound. |

Reference outputs are pinned by commit; each `.npz` carries the
final-step wavefield snapshot (vx, vy, σxx, σyy, σxy) along with
the numerical configuration used to produce it.

## Role in the parent `devito-fd-survey` repo

This folder is one of two reproduction prerequisites for Method 1
SSG (per the round-4 plan decision M, dual reproduction). The
parent repo's `MethodSpec.reproduction_dois` for Method 1
references both this folder and the Levander 1988 folder. The
parent's SSG `_test_ssg_virieux_levander_stencil` paper-
faithfulness callback verifies that the production SSG operator
matches the pinned reference outputs here within tolerance.

## Graduation review

TBD — runs once Method 1 is wired with `reproduction_dois` and
removed from the exemption list in
`tests/test_reproduction_prerequisites.py`.

## References

- Virieux, J. (1986). DOI [10.1190/1.1442147](https://doi.org/10.1190/1.1442147).
- Companion 4th-order folder: [`../doi_10.1190_1.1442422/`](../doi_10.1190_1.1442422/) (Levander 1988).
- Reproduction methodology: [`../CLAUDE.md`](../CLAUDE.md).
- Repo-wide convention `feedback_space_order_16_in_sweeps`.
