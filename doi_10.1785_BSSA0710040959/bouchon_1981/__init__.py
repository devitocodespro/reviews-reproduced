"""Bouchon (1981) Discrete Wavenumber Method reproduction package.

Paper: Bouchon, M. (1981). "A simple method to calculate Green's
functions for elastic layered media." *Bulletin of the
Seismological Society of America* 71(4): 959-971.
DOI: 10.1785/BSSA0710040959.

Plus the modern review by the same author:

Bouchon, M. (2003). "A Review of the Discrete Wavenumber Method."
*Pure and Applied Geophysics* 160: 445-465.
DOI: 10.1007/PL00012545.

This package implements the DWN for the parent TTI-FD-survey
repo's 2D acoustic-acoustic horizontal-interface semi-analytical
reference, used to validate finite-difference solver accuracy at
horizontal fluid-solid interfaces.

Modules:
- `dwn_solver`: full DWN engine — frequency-domain
  Green's-function evaluation + complex-frequency
  regularisation + inverse FFT
- `paper_tables`: hand-transcribed paper anchors — α prescription
  constant, periodicity-length safety margin, critical-angle
  regularisation policy

Phase Y/2b 2026-05-28: relocated from `00_common/reflectivity_2d.py`
to this reproduction folder per the paper-faithful-modules-curation
sweep; wired to the parent via `[tool.uv.sources]` editable-dep.
"""
