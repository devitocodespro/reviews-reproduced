# Irakarama, Luo, Etgen, Shen & O'Brien (2026) — Dual-Pair FD

Reproduction of *"Accelerating anisotropic elastic subsurface
imaging with dual-pair finite differences"*, Fifth International
Meeting for Applied Geoscience & Energy (IMAGE 2025) Expanded
Abstracts pp. 825-829, Society of Exploration Geophysicists, 2026
([DOI 10.1190/image2025-4313380.1](https://doi.org/10.1190/image2025-4313380.1)).

**Status**: ported from `legacy` branch of
`devitocodespro/devito-fd-survey` (formerly the engagement's
peer-review project under `image2025-4313380/`), 2026-05-26.
Anchors the **parent-repo Method 6 / Method 12 (Dual-Pair)
reproduction-as-prerequisite gate** (round-4 plan decision L).

## Paper

Modeste Irakarama (BP), Simon Luo (BP), John Etgen (BP), Xukai
Shen (BP), Michael O'Brien (BP). The paper introduces:

1. **Dual-pair architecture**: asymmetric forward / backward
   1st-derivative operators $D^+$ and $D^-$ satisfying
   $D^- = -(D^+)^\top$, so the displacement spatial operator
   $L u = D^- [\,C\, D^+ u\,]$ is self-adjoint and discrete-energy
   conservative on a single non-staggered grid. The 9-point
   forward stencil offsets are $\{-3,-2,-1,0,+1,\dots,+5\}$
   (asymmetric: 4 left + 5 right of the evaluation point).
2. **Optimised "Proposed" coefficients** (paper Table 1):
   least-squares-minimised dispersion error of the individual
   $D^+$ operator over a wavenumber range
   ($kh \in [0, \pi/2]$, i.e. down to ~4 PPWL) following Liu
   2014, with fixed first-order constraints $\sum c_m = 0$,
   $\sum m\,c_m = 1$. **Paper's defining low-PPWL contribution.**
3. **Selective 2D product filter** (paper Table 2): 11-point
   symmetric high-wavenumber damping filter, applied as
   $\tilde u = u - \sigma D_f[u]$ each timestep with
   $\sigma \in [0,1]$. Paper's "Proposed" filter minimises
   variation in the damping profile vs Bogey-Bailly 2004.
4. **Image-method free-surface + two-step CPML** absorbing
   boundaries (paper §"image method", refs Robertsson 1996,
   Fang et al. 2022).

The paper presents only the **9-point** stencil coefficients
explicitly; higher-order stencils (13/15/17 point) are
mentioned but not tabulated.

## Reference implementation — 5-stage peer review

This reproduction was originally developed as an **independent
peer review of the paper** (legacy `image2025-4313380/`
2026-02-23, see `review_findings.qmd`):

| Stage | Scope | Script |
|:-:|---|---|
| 0 | Operator coefficient verification (Taylor + Proposed) + dispersion/dissipation analysis (reproduces paper Fig 2 + Fig 3) | `stage0_operator_analysis.py` |
| 1 | 2D acoustic dual-pair (convergence test, two-layer model) | `stage1_acoustic_dual_pair.py` |
| 2 | 2D isotropic elastic displacement | `stage2_elastic_isotropic.py` |
| 3 | Selective filtering + source injection (reproduces paper Fig 1) | `stage3_filtering_and_source.py` |
| 4 | 2D TTI elastic with Bond rotation — basis adapted into parent repo's Method 6 (`06_dualpair/dp_elastic_tti.py`) | `stage4_tti_elastic.py` |
| 5 | MMS convergence (legacy stage5 only on local checkout) | `stage5_mms_convergence.py` |

Outputs reproduced: 24 figures under `figures/` including
paper Fig 2 / Fig 3 / Fig 1 reproductions (`fig2_*.png`,
`fig3_filter_response.png`, `stage{1-4}_*.png`). The full
peer-review write-up is at `review_findings.qmd`.

## Quantitative reference — Tables 1 + 2

The byte-checkable load-bearing artifacts (per the user's
standing rule `feedback_reproduction_quantitative_first.md` —
anchor tests on tables + scalar bounds, not visual figure
reproduction) live at `paper_tables.py`:

- `TABLE_1_TAYLOR_9PT` — paper-published Taylor row (matches
  SymPy `finite_diff_weights` at order=8 byte-identically to
  ~1e-6 within the paper's print precision).
- `TABLE_1_PROPOSED_9PT` — paper-published least-squares
  Proposed row (the defining low-PPWL coefficients).
- `TABLE_2_TAYLOR_FILTER`, `TABLE_2_BOGEY_BAILLY_2004_FILTER`,
  `TABLE_2_PROPOSED_FILTER` — 11-point symmetric selective
  filter rows (j..j+5 shown).

Tests at `tests/test_paper_tables.py` verify:

- Stencil offset sets per paper $nl = (n-1)/2 - 1$,
  $nr = (n-1)/2 + 1$ formula
- Taylor row matches SymPy recurrence
- Both rows satisfy consistency ($\sum c_m = 0$) and first-order
  ($\sum m\,c_m = 1$) constraints
- Proposed row is non-trivially distinct from Taylor (paper's
  optimisation claim)
- Filter rows satisfy DC pass-through ($\sum d_m = 1$)
- Transpose-adjoint property $D^- = -(D^+)^\top$ holds for both
  rows

## Run

```bash
cd reviews-reproduced/doi_10.1190_image2025-4313380.1/
uv sync          # creates .venv/ in-place
uv run pytest tests/          # runs the byte-match suite
uv run python stage0_operator_analysis.py  # regenerates Fig 2 + Fig 3
```

Or `make` the full Quarto peer-review PDF:

```bash
make pdf         # rebuilds review_findings.pdf
```

(Requires Quarto + LuaLaTeX.)

## Role in parent repo

Parent repo `devitocodespro/devito-fd-survey` adapted
`stage4_tti_elastic.py` into `06_dualpair/dp_elastic_tti.py`
(Method 6 — Dual-Pair Elastic TTI; Method 12 cascades to
viscoelastic). The parent's
`tests/_paper_faithfulness_callbacks.py::_test_dualpair_irakarama_2026`
callback byte-checks the parent's `dp_operators.py` against
the `TABLE_1_*` + `TABLE_2_*` constants here.

Until 2026-05-26 the parent only carried the **Taylor row**
in production (`get_taylor_dp_coefficients`) plus the
paper's Proposed filter; the load-bearing Proposed derivative
coefficients were missing. Bootstrapping this reproduction
folder + wiring the Proposed coefficients into the parent's
`06_dualpair/dp_operators.py` is the path back to
`provenance='published'` (see parent repo's
`06_dualpair/README.md` § "Graduation review").
