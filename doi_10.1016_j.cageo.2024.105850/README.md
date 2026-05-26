# Zhang & Schmitt 2025 — RSG Devito reproduction

**Paper**: Zhang, O. & Schmitt, D. R. (2025). "An optimized 2D/3D
finite-difference seismic wave propagator using rotated staggered
grid for complex elastic anisotropic structures."
*Computers & Geosciences* **196**: 105850.
DOI: [10.1016/j.cageo.2024.105850](https://doi.org/10.1016/j.cageo.2024.105850).

**Code DOI** (Zenodo): [10.5281/zenodo.14611320](https://doi.org/10.5281/zenodo.14611320)

**Upstream repository**: <https://github.com/OumZhang/rsg>

## Why this reproduction exists

The Rotated Staggered Grid (RSG, Saenger 2000) is the load-bearing
operator for Method 2 RSG (`02_rsg/`) and Method 7 RSG-VE
(`07_viscoelastic_rsg/`) in the parent ``devito-fd-survey`` repo.
The OumZhang/rsg implementation is the **canonical RSG Devito**
reference for the scheme — the parent's ``setup_rsg_lebedev_source``
adopts its 8-point isotropic-source recipe verbatim
(``upstream/src/operators.py:113-149``).

This folder is the prerequisite-gate-protected reproduction
(``tests/test_reproduction_prerequisites.py``) that backs Method 2's
``REPRODUCTION_DOIS_OVERRIDES[2]`` claim:

```
2: ("doi_10.1016_S0165-2125(99)00023-2",       # Saenger 2000 RSG paper
    "doi_10.1190_1.1707078",                   # Bohlen-Saenger 2004 averaging
    "doi_10.1016_j.cageo.2024.105850")         # Zhang-Schmitt 2025 (this folder)
```

## What we reproduce

Per the `feedback_reproduction_quantitative_first` memory entry,
this reproduction anchors on **tabulated quantitative values** —
NOT visual figure reproduction. The transcribed paper constants
live in ``paper_tables.py`` (byte-checkable). Tests in ``tests/``
assert byte-match between the OumZhang code's coefficient
computation and our independent transcription.

Specific verification anchors:

1. **Fornberg 1st-derivative stencil weights** at SO ∈ {2, 4, 6, 8}
   (per `feedback_space_order_16_in_sweeps` — would extend to SO=16
   once Fornberg's recurrence-based helper is byte-validated past
   SO=8). Computed in
   ``upstream/src/wavesolver.py:308-321:fdcoeff_1st``.
2. **8-point isotropic source-weighting recipe** (per stress field:
   8 face/corner offsets × weight ``1/8`` = total moment 1). Encoded
   in ``upstream/src/operators.py:113-149:getsrcterm_2d_around``
   under the ``moment="iso"`` branch.
3. **Saenger 2000 Eq 14 diagonal-stencil application** (``d1 + d2``
   for ∂/∂x, ``d1 − d2`` for ∂/∂y, where ``d1=(+x,+y)`` and
   ``d2=(+x,−y)`` are the two rotated diagonals). Encoded in
   ``upstream/src/wavesolver.py:250-273:dv_2d`` /
   ``dtau_2d``.
4. **CFL bound** for RSG SO=4 per the paper's stability table
   (TBC — section + page to be cited after a targeted re-read).

## Role this reproduction plays in the parent repo

| Method | Status | What this reproduction backs |
|---|---|---|
| Method 2 RSG (`02_rsg/`) | `provenance='published'` | The OumZhang 8-point source recipe is wired verbatim into ``00_common/source.py:setup_rsg_lebedev_source`` (commit `5e0c851` 2026-05-26). This reproduction provides the prerequisite-gate evidence that we can produce the canonical RSG outputs the paper claims. |
| Method 7 RSG-VE (`07_viscoelastic_rsg/`) | `provenance='novel-combination'` | Cascades from Method 2 via shared source helper. |

## Known divergence from the upstream implementation

Two areas where the parent repo's RSG path diverges from the
OumZhang/rsg pattern:

1. **Source-helper structure**: parent's
   ``00_common/source.py:setup_rsg_lebedev_source`` produces the
   same 8 inject-equation pattern via SymPy ``.shift()``, but
   structured slightly differently from upstream's
   ``getsrcterm_2d_around``. The mathematics is byte-identical
   (8 × 1/8 = 1; same 8 offset positions); the parent uses one
   ``src.inject(field=sfields, expr=exprs)`` call with the full
   list, while upstream uses 8 separate ``src.inject(...)``
   calls. Both produce the same operator.
2. **Forward-operator staggering**: parent's default flipped
   2026-05-26 to ``staggering='colocated'`` (explicit ``.subs()``
   shifts on cell-centred Devito fields). The OumZhang/rsg
   reference uses physical staggering (Devito's native
   ``staggered=`` on ``VectorTimeFunction`` /
   ``TensorTimeFunction``). The parent's physical-staggering
   path exists but is opt-in and has a documented wave-speed
   dispersion bug (see memory entry
   ``feedback_rsg_physical_staggering_broken.md`` and
   ``02_rsg/physical_staggered_fd.py:phys_d45`` FIXME). The bug
   appears related to long-standing open Devito issues #1589
   and #2485 (custom-coefficient substitution on staggered
   grids).

These divergences are documented HONESTLY here per the repo's
academic-rigor protocol (memory entry
``feedback_academic_rigor_no_silent_compromise``).

## Quick start

```bash
cd reproduced/doi_10.1016_j.cageo.2024.105850/
uv sync           # creates .venv/ + installs pinned devito etc.
uv run pytest tests/ -v
uv run python run_reproduction.py
```

## File layout

```
doi_10.1016_j.cageo.2024.105850/
├── README.md                          # this file
├── zhang_schmitt_2025_paper.pdf       # Computers & Geosciences 196:105850
├── pyproject.toml                     # uv project + pinned deps
├── paper_tables.py                    # transcribed paper constants (byte-checkable)
├── run_reproduction.py                # top-level driver (invokes upstream)
├── tests/
│   └── test_paper_tables.py           # byte-match parent's transcription
│                                       # against upstream's computation
├── reference_outputs/                  # pinned reproduction snapshots
└── upstream/                           # vendored OumZhang/rsg snapshot
    ├── README.md                       # upstream's README
    ├── CITATION.cff
    ├── LICENSE
    ├── example/Tutorial.ipynb
    └── src/
        ├── model.py
        ├── operators.py
        ├── stiffnessoperator.py
        └── wavesolver.py
```

## Upstream pin

The vendored ``upstream/`` snapshot corresponds to OumZhang/rsg
repository state at commit ``be969e3`` (cloned 2026-05-26). The
``upstream/.git/`` directory has been removed; the upstream tree
is frozen at the vendoring commit. To update, re-clone and rsync.
