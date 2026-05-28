"""Petersson & Sjögreen (2015) sw4 SBP-SAT reproduction package.

Paper: Petersson, N. A., & Sjögreen, B. (2015). "Wave propagation in
anisotropic elastic materials and curvilinear coordinates using a
summation-by-parts finite difference method." *Journal of Computational
Physics* 299: 820-841.
DOI: 10.1016/j.jcp.2015.07.023.

Antecedent (the original (4,2) diagonal-norm SBP D1 tabulation):
Strand, B. (1994). "Summation by parts for finite difference
approximations for d/dx." *Journal of Computational Physics* 110: 47-67.

This package collects the SBP-SAT modules that implement the
Petersson-Sjögreen 2015 scheme as published in their sw4
(Seismic Waves 4th-order) reference implementation. The
(4, 2) diagonal-norm boundary stencils tabulated here byte-match
the sw4 source file `src/boundaryOpc.C` (open-source at
github.com/geodynamics/sw4).

Modules:
- `sbp_sat`: SBP D1/D2 boundary stencil weights + norm matrix +
  Devito-side `Eq` constructors for the displacement-formulation
  PS-2015 elastic-TTI scheme.
- `paper_tables`: hand-transcribed sw4 byte anchors for the
  (4, 2) D1 boundary rows + diagonal norm + interior stencil +
  SBP-property sentinel.

Phase Y/2c 2026-05-28: relocated from `00_common/sbp_sat.py` to
this reproduction folder per the paper-faithful-modules-curation
sweep; wired to the parent via `[tool.uv.sources]` editable-dep.
"""
