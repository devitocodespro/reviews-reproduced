# Levander (1988) — Fourth-Order P-SV Staggered-Grid

Devito reproduction of the canonical 4th-order extension of the
Virieux 1986 staggered-grid scheme.

## Paper

> Levander, A.R. (1988). "Fourth-order finite-difference P-SV
> seismograms." *Geophysics* 53(11), 1425–1436. DOI
> [10.1190/1.1442422](https://doi.org/10.1190/1.1442422).

## Scheme

Same 2D P-SV velocity-stress staggered-grid framework as Virieux
1986 (see [`../doi_10.1190_1.1442147/`](../doi_10.1190_1.1442147/));
Levander 1988 extends the spatial discretisation from 2nd-order
to 4th-order via Eq 4's 4-point centred-FD stencil with weights
`{1/24, -9/8, +9/8, -1/24} / Δx`.

The continuous PDE system is identical to Virieux 1986; the field
placement (Yee staggering, Virieux Fig 1) is identical; only the
spatial stencil order changes.

## Implementation

Per the standalone-isolation discipline documented in
[`../CLAUDE.md`](../CLAUDE.md), this folder duplicates the driver
code from the Virieux 1986 folder rather than importing — a
change to one folder cannot break tests in the other. The two
folders use the same Devito framework + same domain configuration
to allow direct numerical cross-validation between them at shared
`space_order` values (which gives identical wavefields by
construction).

The driver uses Devito's first-class `VectorTimeFunction` /
`TensorTimeFunction` types; cross-stagger derivative shifts are
handled by the framework's `_eval_at` machinery.

Per the repo-wide `space_order` convention, the driver parametrises
`so ∈ {2, 4, 8, 16}`. The **canonical Levander 1988 claim is at
`space_order=4`**; SO=2 is included for cross-validation against
the Virieux 1986 folder, and SO ∈ {8, 16} are exercised per the
so=16 near-spectral-baseline convention.

## Configuration

Identical to the companion Virieux 1986 reproduction:

| Parameter | Value |
|---|---:|
| Domain | 4 km × 4 km, NX = NY = 401 (dx = 10 m) |
| Vp / Vs / ρ | 3.0 km/s / 1.7 km/s / 2.2 g/cm³ |
| Source | Ricker, f₀ = 20 Hz, explosive at domain centre |
| Final time | 0.4 s (wavefront pre-boundary) |
| CFL | per-order from Saenger-Bohlen 2004 Table 2 |

Quantities scaled to repo-standard units (km / km·s⁻¹ / g·cm⁻³)
for fp64 / fp32 numerical safety.

## Usage

```bash
uv sync
uv run python run_reproduction.py           # all four space_orders
uv run python run_reproduction.py -s 4      # canonical SO=4 only
uv run pytest tests/ -v
```

## Tests

`tests/test_reproduction.py` covers the same four invariant
classes as the Virieux folder, with the canonical paper-faithful
test relocated from SO=2 to SO=4:

| Class | Test(s) | What it gates |
|---|---|---|
| **Paper-faithful MMS** | `test_so4_paper_faithful_convergence_rate` (canonical); SO=2/8/16 cross-validation | Devito's `.dx` at SO=4 achieves Levander 1988 Eq 4's claimed 4th-order accuracy. |
| **Reference-output regression** | `test_reference_output_matches_pin[2,4,8,16]` | Re-runs produce wavefields byte-identical to pinned `reference_outputs/*.npz`. |
| **Physical sanity** | `test_wavefield_is_finite_and_bounded`, `test_explosive_source_radial_p_wave_symmetry` | Wavefield finite, bounded, cos(φ) symmetric. |
| **CFL discipline** | `test_cfl_dt_respects_per_order_bound` | Per-order CFL bounds respected. |

## Relationship to Virieux 1986 reproduction

The two folders are intentionally redundant at the code-and-data
level (identical configuration → identical numerical output at
shared space_orders). They differ in:

- **Canonical claim**: Virieux folder pins SO=2 as the formally-
  reproduced scheme; Levander folder pins SO=4.
- **Test thresholds**: Each folder asserts the formal-order
  convergence rate at its OWN canonical SO (Virieux at SO=2;
  Levander at SO=4) — the same test framework, different
  emphasis.
- **Failure isolation**: A Devito API change that breaks one
  folder's SO=2 stencil leaves the other folder's SO=4 stencil
  passing — useful diagnostic for narrowing the regression
  scope.

## Role in parent `devito-fd-survey`

This folder is the second of two reproduction prerequisites for
Method 1 SSG (round-4 plan decision M, dual reproduction).

## Graduation review

TBD — runs once Method 1's `reproduction_dois` is wired to
`("doi_10.1190_1.1442147", "doi_10.1190_1.1442422")`.

## References

- Levander 1988 DOI: [10.1190/1.1442422](https://doi.org/10.1190/1.1442422)
- Companion 2nd-order folder: [`../doi_10.1190_1.1442147/`](../doi_10.1190_1.1442147/)
- Reproduction methodology: [`../CLAUDE.md`](../CLAUDE.md)
