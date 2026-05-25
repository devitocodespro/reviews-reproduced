# reviews-reproduced — Reproduction Methodology

This repository is a companion to
[`devitocodes/reviews`](https://github.com/devitocodes/reviews) — the
TTI finite-difference methods review. Each subdirectory in this repo
is a **standalone Devito-based reproduction** of one published paper,
named by DOI, with its own report, tests, and dependency stack.

The methodology is inspired by:

- The **QED multi-agent proof system** (An, Ye, Pan, Zhang, arXiv
  2604.24021v2 2026-04-29) — particularly the 7 failure modes (FM1
  context contamination, FM2 citation hallucination, FM3 hand-waving,
  FM4 unstable proof plans, FM5 unfocused verification, FM6 problem
  modification, FM7 single-model bottleneck) and the
  REVISE_PROOF / REVISE_PLAN / REWRITE retry hierarchy.
- The **FD rigour-gated task graph** documented at
  `~/Downloads/fd-playbook(1).md` (14 stages S0-S13 with explicit
  pass/fail gates per stage).
- The **Implementing methods that claim to reproduce published work**
  rules in `~/CLAUDE.md` (Rules 1-14: faithfulness vs surrogate
  classification, paper-faithfulness invariants, reference-
  implementation search, failure-kind taxonomy).

## Repository structure

```
reviews-reproduced/
├── CLAUDE.md                          # this file
├── README.md                          # user-facing overview
├── doi_<paper_DOI>/                   # one folder per reproduced paper
│   ├── README.md / report.qmd         # Devito implementation considerations
│   ├── pyproject.toml                 # ISOLATED dependencies (uv-managed)
│   ├── uv.lock                        # resolved lockfile (committed)
│   ├── run_reproduction.py            # main reproduction driver
│   ├── tests/                         # standalone pytest subsuite
│   └── reference_outputs/             # pinned reference outputs (.npz, .png)
└── ...
```

### Standalone discipline

**Each `doi_*/` folder is hermetically standalone.** Reasoning:

1. A change in one paper's helper code MUST NOT break tests in
   another paper's folder.
2. A reproduction frozen to a specific paper at a specific Devito
   version must remain frozen — even when the parent
   `devitocodes/reviews` repo evolves.
3. The reproduction's `pyproject.toml` pins the EXACT Devito commit
   that produces the reference outputs. A reader can `uv sync` and
   immediately reproduce the figures.

Concretely:

- **No cross-folder imports.** If two reproductions need the same
  helper code, that helper is duplicated (not symlinked) in both
  folders. Convergence on a shared abstraction happens upstream in
  `devitocodes/reviews/core/`, not here.
- **Per-folder venv.** `uv sync` in `doi_<DOI>/` creates `.venv/` in
  that folder. Tests run as `uv run pytest tests/`.
- **No top-level imports.** The repo root has no Python package; it
  carries only `README.md`, `CLAUDE.md`, and per-folder
  subdirectories.

### Per-folder file conventions

| File | Required | Purpose |
|---|---|---|
| `README.md` or `report.qmd` | YES | Devito implementation considerations: paper summary, key equations, BC + IC choices, parameter values, deviations from the paper with justification, reproduction outputs |
| `pyproject.toml` | YES | Exact dependency pinning incl. Devito commit |
| `uv.lock` | YES | Resolved lockfile, committed |
| `run_reproduction.py` | YES | Main driver that produces the reference outputs |
| `tests/test_reproduction.py` | YES | Standalone pytest subsuite, including paper-faithfulness invariants (Rule 3) |
| `reference_outputs/*.npz` | YES | Pinned reference output arrays (small enough to commit) |
| `reference_outputs/figs/*.png` | YES | Reproduction figures + side-by-side with paper |

## Reproduction workflow (the methodology proper)

For each new paper to reproduce, follow this protocol. Each step
maps to one or more `fd-playbook(1).md` stages.

### Step 1 — Paper acquisition (FM2 citation hallucination guard)

Required tooling (in priority order):

1. **Zotero** via the `pyzotero` skill — check whether the paper is
   already in `~/Zotero/`. Use `pyzotero` skill commands to locate the
   PDF attachment.
2. **Zotero RAG** (graph DB) — search by keyword if not located by
   DOI/title.
3. **Crossref MCP** — `mcp__crossref__get_work_metadata` with the
   DOI to verify metadata (title, authors, journal, volume, pages,
   year) BEFORE writing any code referencing the paper.
4. **OpenAlex MCP** — fallback for metadata if Crossref incomplete.
5. **arxiv MCP** — preprint version if behind paywall.
6. **GitHub reference-implementation search** (per `~/CLAUDE.md`
   2026-05-22 protocol):
   - search by author username
   - search by paper title / DOI
   - check for supplementary-materials repos
   - **if a reference implementation exists, read it BEFORE writing
     our own**. Bader 2023 lesson: paper-text alone produces an
     unstable scheme; the GitHub reference reveals the implicit
     coupling.

### Step 2 — Folder bootstrap

```bash
mkdir reviews-reproduced/doi_<encoded_DOI>/
cd reviews-reproduced/doi_<encoded_DOI>/
uv init --no-package
# Edit pyproject.toml: pin devito @ git+https://github.com/devitocodes/devito.git@<commit>
uv add numpy scipy sympy matplotlib pytest
uv sync
```

DOI encoding for folder name: replace `/` with `_` (e.g., DOI
`10.1190/geo2022-0195.1` → folder `doi_10.1190_geo2022-0195.1`).

### Step 3 — Devito implementation

Always invoke the `/devito` skill before writing Devito DSL. The
skill documents critical patterns (e.g., `implicit_dims` for
Function-only dependency chains, source injection conventions,
staggered grids, `space_order` on intermediate Functions for
composed derivatives) that are not obvious from the Devito API
alone. Failing to consult it has previously caused silent
correctness bugs.

### Step 4 — Paper-faithfulness invariants (FM6 problem modification guard)

Per `~/CLAUDE.md` Rule 3: a `provenance='published'` claim requires
**paper-level invariant tests** that exercise the defining feature
at the operator-algebra level. Acceptable invariants:

- Stencil weights byte-match the paper's tabulated coefficients;
- Symbolic SBP / skew-adjoint / energy-norm property;
- Cross-grid coupling structure verified at the operator level;
- Specific dispersion / CFL bounds cited in the paper.

**MMS convergence rate alone is insufficient** — a surrogate scheme
can pass MMS at the same rate as the published method on smooth
homogeneous solutions. Always combine MMS with the defining-feature
test.

### Step 5 — Failure-kind classification on each loss (FM4)

Per `~/CLAUDE.md` Rule 14: every failed attempt during the
reproduction must be tagged as:

- `EXECUTION` — local fix; resume same plan (QED `REVISE_PROOF`).
- `PLAN` — structural flaw within the same framework; revise the
  plan (QED `REVISE_PLAN`).
- `APPROACH` — fundamental framing wrong; discard the plan (QED
  `REWRITE`).

Escalation: 3 consecutive `EXECUTION`-class fixes promote the
diagnosis to `PLAN`; 2 consecutive `PLAN`-class revisions promote
to `APPROACH`. Prevents the "random walk through proof space"
failure.

### Step 6 — Verification gate sequence (FM5 unfocused verification guard)

Run gates in this order:

| Gate | Source-of-truth | Time |
|---|---|---|
| 0. Smoke import | `python -c "import run_reproduction"` | seconds |
| 1. Paper-faithfulness invariants | `tests/test_reproduction.py::test_paper_faithfulness_*` | ~10 s |
| 2. MMS convergence (spatial + temporal SEPARATELY) | `tests/test_reproduction.py::test_mms_*` | minutes |
| 3. Reference-output reproduction | `tests/test_reproduction.py::test_reference_match` | depends |
| 4. (If applicable) high-precision oracle | `tests/test_reproduction.py::test_oracle_*` | minutes |

Each gate FAILS FAST — do not advance to gate N+1 until gate N is
green.

### Step 7 — Graduation review (FM7 single-model bottleneck guard)

Before declaring a reproduction "done", run a graduation review
using the parent repo's tooling:

```bash
# From the parent devitocodes/reviews repo:
make graduation_review METHOD=<corresponding_method_id>
# OR for a standalone reproduction:
python scripts/graduation_review.py --path reviews-reproduced/doi_<DOI>/
```

This invokes Codex (or another second-model reviewer) against the
implementation diff + the paper-faithfulness callback + the
reference implementation if one exists. Verdict is `AGREE` /
`DISAGREE` + objections. **Do NOT mark as `provenance='published'`
until the graduation review returns AGREE.**

Record the verdict in the reproduction's `README.md` under a
`## Graduation review` section with date + reviewer model.

## Tooling reference

### Required MCPs (from parent CLAUDE.md)

- **Crossref MCP** (`mcp__crossref__get_work_metadata`): authoritative
  paper metadata
- **OpenAlex MCP** (`mcp__openalex__*`): author/work fallback
- **arxiv MCP** (`mcp__arxiv__*`): preprint access
- **Opencitations MCP** (`mcp__opencitations__*`): citation graph

### Required skills

- **`pyzotero`** skill — local Zotero library access (cited papers,
  attached PDFs, tags/notes)
- **`devito`** skill — Devito DSL patterns (always invoke before
  writing solver code)

### Optional tooling

- **`gh` CLI** — paper-companion GitHub repo cloning
- **`uv`** — per-folder Python environment isolation
- **`quarto`** — `report.qmd` rendering

## Adding a new reproduction

1. Create the folder per Step 2.
2. Add the folder name + paper title to the table in `README.md`.
3. Follow Steps 3-7 above.
4. Open a PR to this repo with the reproduction.
5. Once merged, bump the submodule pin in the parent
   `devitocodes/reviews` repo to point at the new commit.

## Anti-patterns (NEVER)

- **NEVER** declare a reproduction `provenance='published'` without:
  (a) reading the paper's reference implementation if one exists,
  (b) writing operator-level paper-faithfulness invariants,
  (c) passing a graduation review.
- **NEVER** import code from another `doi_<DOI>/` folder. Vendor it
  locally if needed.
- **NEVER** share a single `pyproject.toml` across reproductions
  (defeats isolation).
- **NEVER** commit `.venv/` (gitignored).
- **NEVER** track `CLAUDE.md` files inside reproduction folders
  beyond this top-level one — per `~/CLAUDE.md` recursive rule.
