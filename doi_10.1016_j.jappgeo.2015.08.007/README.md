# Yang, Yan & Liu (2015) — Optimal RSG-TTI elastic FD reproduction

**Paper anchor**: Yang, L., Yan, H. & Liu, H. (2015). "Optimal rotated
staggered-grid finite-difference schemes for elastic wave modeling in
TTI media." *Journal of Applied Geophysics* 122:40–52.
DOI [10.1016/j.jappgeo.2015.08.007](https://doi.org/10.1016/j.jappgeo.2015.08.007).
BibTeX key: `yang2015` (in `~/projects/reviews/references.bib`).

**Local PDF**: `yang2015.pdf` (self-contained submodule copy of
`~/projects/reviews/papers/yang2015__*.pdf`).

**PDF SHA256**: `72eeef5f063b4e3d70b6e2446fb5ce9a64a69d1d652e99b89fe13d93038e8741`
(pinned in `tests/test_paper_tables.py` as a provenance gate).

## Role in the parent repo

This reproduction is the **literature anchor for RSG-TTI optimised
coefficients** referenced by the parent's Method 2 RSG cascade
graduation (`devitocodespro/devito-fd-survey:refactor`, plan
§"Method 2 RSG cascade Rounds 8-12"). Yang 2015 explicitly identifies
that **Taylor-series-expansion (TE) based RSG is suboptimal for TTI
media** — the same observation that drove M2's graduation to
`surrogate` provenance. The paper proposes two optimal alternatives:

1. **SA (sampling approximation)** — based on Liu Y. (2014) GJI
   197:1033 method, extended to RSG-TTI.
2. **LS (least-squares)** — least-squares optimisation in the
   wavenumber-error functional.

Reproducing Yang 2015 unlocks the literature-anchored upgrade path
for M2 RSG (and cascade methods M7, M60, M61, M62, M63) from
`surrogate` toward `novel-combination` or `published` provenance.

## What we reproduce

Per the `feedback_reproduction_quantitative_first` repo rule:
anchor on tabulated constants + scalar bounds + algorithmic
recipes, NOT on figure reproduction.

### 1. Elastic PDE for 2D RSG-TTI (paper Eq 1)

2D, 3-component velocity-stress system in TTI media using the d_ij
coefficient matrix (rotated stiffness D = R·C·Rᵀ).

### 2. Stiffness matrix C in VTI form (paper Eq 3)

6×6 symmetric matrix with 5 independent entries (c11, c12, c13, c33,
c44) and the constraint c66 = (c11-c12)/2.

### 3. Bond rotation matrix R (paper Eq 4)

6×6 matrix in the azimuth (ϕ) and tilt (θ) angles. Transforms
stiffness from natural to observation coordinate system per
Winterstein 1990.

### 4. TE / SA / LS RSG coefficient tables (paper Sections 3-4)

Per-stencil-length tables to be byte-transcribed under the
side-by-side user-confirmed review protocol.

### 5. Dispersion-comparison qualitative + envelope claims (paper §4)

Numerical-accuracy claims at small vs large wavenumbers, validated
via qualitative ordering + envelope check (NOT k_cross byte-match,
per pre-flight dual-reviewer Codex+Gemini finding YF3).

## Transcription provenance

All entries verified under the user-mandated arXiv-first /
side-by-side review protocol
(`feedback_transcription_workflow_arxiv_then_side_by_side`).
arXiv check: **no preprint with LaTeX source** for this paper
(journal-only). PDF-side-by-side mandatory for every entry.

| Entry | Source | Status |
|---|---|---|
| Eq 1 PDE | paper PDF p. 41 | (pending side-by-side) |
| Eq 2 D = R·C·Rᵀ | paper PDF p. 41 | (pending side-by-side) |
| Eq 3 C VTI form | paper PDF p. 41 | (pending side-by-side) |
| Eq 4 R Bond rotation | paper PDF p. 41 | (pending side-by-side) |
| TE coefficients per L | paper PDF (sections 3-4, TBD) | (pending) |
| SA coefficients per L | paper PDF (sections 3-4, TBD) | (pending) |
| LS coefficients per L | paper PDF (sections 3-4, TBD) | (pending) |

Side-by-side review PDFs preserved at
`transcription_review/yang2015_*.tex` (transcription LaTeX) +
re-generable via the Irakarama / LP04 `side_by_side.py` pattern.

## Status

🟡 **PHASE Y.1 BOOTSTRAP COMPLETE (2026-05-27)** — folder
skeleton, pyproject.toml, README. Tests and paper_tables.py
populated incrementally as side-by-side review confirms each
load-bearing entry.

🟡 PHASES Y.2-Y.10 pending (estimated ~16-24 h focused work
per pre-flight dual-reviewer-refined plan at
`~/.claude/plans/twinkling-popping-wadler.md` §"Yang 2015
reproduction — faithful bootstrap").

## Quick start

```bash
cd reproduced/doi_10.1016_j.jappgeo.2015.08.007/
uv sync
uv run pytest tests/ -v
```

## File layout

```
doi_10.1016_j.jappgeo.2015.08.007/
├── pyproject.toml
├── uv.lock                              # committed
├── README.md                            # this file
├── yang2015.pdf                         # self-contained PDF copy
├── paper_tables.py                      # byte-transcribed constants
├── run_reproduction.py                  # top-level driver
├── transcription_review/                # side-by-side review LaTeX/PDFs
├── tests/
│   └── test_paper_tables.py             # byte-match + invariant gates
├── reference_outputs/                   # pinned dispersion-comparison outputs
└── figures/                             # output figures (dispersion, modeling)
```

## Reproduction provenance classification

To be finalised post-Y.9 dual reviewer graduation review.
Anticipated: `published` if all byte-match + cross-check tests
pass; `novel-combination` if independent re-derivation differs
within paper's print precision; `surrogate` if implementation
diverges from paper.

## Plan reference

Full execution plan at
`~/.claude/plans/twinkling-popping-wadler.md` §"Yang 2015
reproduction — faithful bootstrap (2026-05-27)" + the
pre-flight dual-reviewer findings YF1-YF4 captured in the
plan's "Pre-flight dual-reviewer verdict" subsection.
