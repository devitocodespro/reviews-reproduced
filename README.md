# reviews-reproduced

Standalone reproductions of computational geophysics papers in
[Devito](https://www.devitoproject.org). Companion repository to
[`devitocodes/reviews`](https://github.com/devitocodes/reviews) (the
TTI finite-difference methods review).

Each subdirectory reproduces one paper from scratch with:

- A standalone [`uv`](https://docs.astral.sh/uv/) Python environment
  pinning the exact Devito commit used.
- A standalone test suite including paper-faithfulness invariants.
- A `report.qmd` (or `README.md`) covering Devito implementation
  considerations + deviations + reproduction outputs.
- Reference outputs (`.npz` arrays + figures) pinned alongside the
  code that produced them.

The reproductions are hermetically isolated — changes to one folder
cannot break tests in another. See [`CLAUDE.md`](CLAUDE.md) for the
detailed reproduction methodology (QED-inspired multi-stage workflow
with paper-faithfulness invariants, failure-kind classification, and
graduation review).

## Reproductions

| DOI | Paper | Status |
|---|---|---|
| [`10.1190/geo2022-0195.1`](doi_10.1190_geo2022-0195.1/) | Bader, Almquist, Dunham 2023 — Energy-stable SBP-SAT acoustic-elastic coupling | in-progress |
| `10.1190/1.1707078` | Bohlen, Saenger 2004 — Rayleigh waves on staggered grids | stub |
| `10.1016/S0165-2125(99)00023-2` | Saenger, Gold, Shapiro 2000 — Rotated-staggered grid | stub |
| `10.1190/geo2023-0515.1` | Caunt, Nelson, Luporini, Gorman 2024 — Immersed boundary for irregular topography | stub |
| `(gmd-2022-48)` | Dolci et al. 2022 — ABC comparison for FWI | stub (salvage from `legacy` branch of parent repo) |

## Quick start (per reproduction)

```bash
git clone https://github.com/devitocodespro/reviews-reproduced
cd reviews-reproduced/doi_<DOI>/
uv sync
uv run python run_reproduction.py
uv run pytest tests/ -v
```

## Adding a new reproduction

See [`CLAUDE.md`](CLAUDE.md) for the full protocol. Summary:

1. Run reference-implementation search on GitHub before writing
   any code (per `~/CLAUDE.md` 2026-05-22 rule).
2. Create `doi_<encoded_DOI>/` folder; `uv init`; pin Devito by
   commit in `pyproject.toml`.
3. Implement; write paper-faithfulness invariants; verify MMS
   spatially and temporally separately.
4. Run a graduation review using the parent repo's tooling.
5. Open a PR.

## License

MIT (matching parent repo Devito ecosystem).
