# Mallick & Frazer (1990) anisotropic R/T reflection coefficient

**Primary paper**: Mallick, S., & Frazer, L. N. (1990). *Computation
of synthetic seismograms for stratified azimuthally anisotropic
media.* Journal of Geophysical Research 95(B6): 8513-8526. DOI:
[10.1029/JB095iB06p08513](https://doi.org/10.1029/JB095iB06p08513).

**Companion paper** (R/T coefficients): Mallick, S., & Frazer, L. N.
(1991). *Reflection/transmission coefficients and azimuthal
anisotropy in marine seismic studies.* GJI 105(1): 241-252.

## Why this reproduction exists

This package is the parent repo's **anisotropic acoustic-elastic R/T
reflection-coefficient solver** for planar interfaces where the
lower medium is TTI elastic. It's the load-bearing component of
the parent's DWN reference solver (`bouchon_1981.dwn_solver`) for
the acoustic-elastic-TTI Petrobras geometry.

Per the paper's three-component continuity at the interface
(normal displacement, normal traction, vanishing tangential
traction), each (p_x, ω) pair yields a 3×3 linear system with
unknowns (R, T_qP, T_qSV).

Phase Y/2h 2026-05-28 relocates the anisotropic R/T solver from
`00_common/anisotropic_rt.py` into this reproduction folder so the
paper claims (3×3 system structure, Christoffel quartic-in-q_z,
Snell-slowness conservation, normal-incidence reduction) live
inside the reproduction that anchors them (QED guard FM2 — citation
hallucination).

## What we reproduce

Per the project's `feedback_reproduction_quantitative_first`
convention, this reproduction anchors paper claims as sentinel-
string-locked structural properties + asymptotic-limit empirical
checks:

1. **3×3 acoustic-elastic continuity system** sentinel locks the
   dimension and the unknown ordering `(R, T_qP, T_qSV)` and the
   BC list (`u_z` continuity, `σ_zz` continuity, `σ_xz = 0` on
   the elastic side). Catches silent reduction to 2×2
   (acoustic-acoustic) or expansion to 4×4 (elastic-elastic).

2. **Christoffel quartic structure** — the Christoffel equation
   reduces to a degree-4 polynomial in q_z; exactly 2 downgoing
   modes (qP + qSV) selected with `Im(q_z) ≤ 0` per Conv A.
   The solver's `christoffel_qz_quartic_coeffs` returns exactly
   5 coefficients (verified by test).

3. **Snell-slowness conservation** `p_x_upper = p_x_lower`
   (sentinel string).

4. **q_w² = 1/V_U² − p_x²** sentinel for the acoustic upper-layer
   vertical slowness. Silent swap to `1/V_U² + p_x²` (Helmholtz
   form) would flip evanescence direction.

5. **Normal-incidence reduction** sentinel-locked
   (`MALLICK_FRAZER_1990_NORMAL_INCIDENCE_DECOUPLES_QSV = True`,
   `..._REDUCES_TO_ACOUSTIC_IMPEDANCE = True`). The numerical
   verification of R against the acoustic-impedance value
   `(Z_TTI − Z_water) / (Z_TTI + Z_water)` lives in the parent's
   `tests/test_anisotropic_rt.py` where the
   displacement-vs-pressure-amplitude convention conversion is
   handled against the DWN reference. The reproduction folder's
   standalone tests verify structural / contract properties
   (smoke + return-keys + finite outputs) instead.

6. **2D Voigt naming convention sentinel** locks the
   `(13, 15, 33, 35, 55)` seismological standard
   (`1=x, 3=z, 5=xz`). The production module uses this naming,
   distinct from the older `12/16/22/26/66` naming used by other
   `Cij` utilities — silent confusion between the two conventions
   would corrupt the Christoffel matrix elements.

7. **Conv A Fourier-sign sentinel** (`exp(-iωt)` exponent).
   Flipping the sign would 180°-rotate every R(p_x, ω) coefficient.

8. **`stiffness_from_thomsen` returns standard keys** `{C_11, C_13,
   C_15, C_33, C_35, C_55}` — locks the production helper's output
   contract.

## File layout

```
doi_10.1029_JB095iB06p08513/
├── pyproject.toml                          # declares mallick_frazer_1990 pkg
├── README.md                               # this file
├── mallick_frazer_1990/
│   ├── __init__.py                         # paper-anchored docstring
│   ├── anisotropic_rt.py                   # 724-line solver
│   └── paper_tables.py                     # hand-transcribed anchors
└── tests/
    └── test_paper_tables.py                # 20 sentinel + structural +
                                            # smoke + finite-output tests
```

## Quick start

```bash
# Standalone reproduction tests:
cd reproduced/doi_10.1029_JB095iB06p08513/
uv sync
uv run pytest tests/ -v
```

Or from the parent repo:

```bash
cd ~/projects/reviews
uv run pytest reproduced/doi_10.1029_JB095iB06p08513/tests/ -v
```

## Parent-repo wiring

The parent's `pyproject.toml` registers this folder under
`[tool.uv.sources]` as the `mallick_frazer_1990` package. Parent
imports use `from mallick_frazer_1990.anisotropic_rt import ...`.
Phase Y/2h updated 9 caller sites in the parent:
- `tests/test_smeared_dwn.py`
- `tests/test_dwn_homogeneous.py`
- `notes/archived_diagnostic/diag_kristek_d2_side_by_side.py`
- 6 scripts: `run_jiang_zhang_tier4_dwn_check.py`,
  `run_jz_test2_iso_vti.py`, `run_jz_smeared_dwn_reference.py`,
  `run_dwn_reference_phase2d.py`, `run_kristek_tier4_dwn_check.py`,
  `compare_fd_vs_dwn_errors.py`

**Cross-reproduction dependency**: the Bouchon 1981 reproduction
package (`reproduced/doi_10.1785_BSSA0710040959/`) also imports
this module — `bouchon_1981.dwn_solver` calls
`mallick_frazer_1990.acoustic_elastic_reflection` and
`...acoustic_elastic_elastic_stack_reflection` for the anisotropic
R/T per ω at each DWN summation k_x tap. The cross-dep is declared
via `bouchon_1981`'s own `[tool.uv.sources]` editable-dep pointing
at `../doi_10.1029_JB095iB06p08513`.

## Reproduction provenance classification

**`published`** (Phase Y/2h initial classification 2026-05-28). The
3×3 system structure + Christoffel quartic + Snell conservation +
normal-incidence reduction are all sentinel-locked; the asymptotic-
limit empirical checks (`p_x = 0` reduces R to acoustic impedance,
qSV decouples for isotropic lower) pass to numerical tolerance.

A future QED-extended review-pool graduation review (per
`~/.claude/plans/twinkling-popping-wadler.md` Reviewer Protocol
section) will record AGREE/DISAGREE here under `## Graduation review`.
