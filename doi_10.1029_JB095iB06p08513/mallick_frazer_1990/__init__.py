"""Mallick & Frazer (1990) anisotropic R/T reproduction package.

Primary paper:
    Mallick, S., & Frazer, L. N. (1990). "Computation of synthetic
    seismograms for stratified azimuthally anisotropic media."
    *Journal of Geophysical Research* 95(B6): 8513-8526.
    DOI: 10.1029/JB095iB06p08513.

Companion (R/T coefficients):
    Mallick, S., & Frazer, L. N. (1991). "Reflection/transmission
    coefficients and azimuthal anisotropy in marine seismic studies."
    *Geophysical Journal International* 105(1): 241-252.

This package implements the Mallick-Frazer 1990 machinery for
computing reflection coefficients `R(p_x, ω)` at a planar
acoustic-elastic interface where the lower medium is TTI elastic.

Per the paper's three-component continuity at the interface
(normal displacement, normal traction, vanishing tangential
traction), each (p_x, ω) pair yields a 3×3 linear system that
the production code solves numerically.

Modules:
- `anisotropic_rt`: NumPy production implementation
    * `stiffness_from_thomsen(Vp, Vs, rho, eps, delta, theta)` —
      build 2D TTI stiffness in standard 13/15/33/35/55 naming
    * `christoffel_qz_quartic_coeffs(p_x, C)` — 4th-order
      polynomial in q_z whose roots are the qP, qSV vertical
      slownesses at horizontal slowness p_x
    * `solve_quartic_roots(coeffs)`, `select_downgoing_qz`,
      `tti_polarisation`
    * `acoustic_elastic_reflection(p_x, omega, qw, ...)` — 3×3
      solve for R, T_qP, T_qSV at a single interface
    * `acoustic_elastic_elastic_stack_reflection(...)` — extension
      to a multi-layer elastic stack via Aki-Richards layer
      product
- `paper_tables`: hand-transcribed paper anchors:
    * 3×3 acoustic-elastic continuity-equations form sentinel
    * Christoffel quartic structure (degree 4 in q_z)
    * Down-going-mode selection rule (Im(q_z) ≤ 0)
    * Snell-slowness conservation (p_x continuous across interface)
    * Normal-incidence reduction (p_x = 0 ⇒ R reduces to acoustic
      impedance R_normal)

Phase Y/2h 2026-05-28: relocated from `00_common/anisotropic_rt.py`
into this reproduction folder per the paper-faithful-modules
curation sweep; wired to the parent via `[tool.uv.sources]`
editable-dep. The Bouchon 1981 reproduction package
(`reproduced/doi_10.1785_BSSA0710040959/`) declares this as a
cross-reproduction dependency via its own `[tool.uv.sources]`
because `bouchon_1981.dwn_solver` calls
`mallick_frazer_1990.acoustic_elastic_reflection` and
`...acoustic_elastic_elastic_stack_reflection` for the anisotropic
R/T per ω at each DWN summation kx tap.
"""
