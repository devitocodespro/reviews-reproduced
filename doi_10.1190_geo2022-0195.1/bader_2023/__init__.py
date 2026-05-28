"""Bader, Almquist & Dunham (2023) — SBP-SAT acoustic-elastic coupling
reproduction package.

Paper: Bader M., Almquist M., Dunham E.M. (2023). "Modeling and
inversion in acoustic-elastic coupled media using energy-stable
summation-by-parts operators." *Geophysics* 88(3) T137-T150.
DOI: 10.1190/geo2022-0195.1.

This package collects the paper-anchored modules that the parent
TTI-FD-survey repo imports. Each module is a hand-implementation
of the paper's algorithms keyed to specific equations / sections;
see individual module docstrings for the load-bearing citations.

Modules:
- `acoustic_elastic_coupling`: seafloor-coupling SAT helpers
  (Bader 2023 eqs 3, 9-10)
- `bader_canonical_runners`: canonical Fig 4 / Fig 7 runners
  for two-zone SBP-SAT comparisons

Phase Y/2a 2026-05-28: relocated from `00_common/` to this
reproduction folder per the paper-faithful-modules-curation
sweep; wired to the parent via `[tool.uv.sources]` editable-dep.
"""
