"""Hao et al. (2022) two-tier Kjartansson constant-Q reproduction package.

Primary paper:
    Hao, Q., Greenhalgh, S., Huang, X., & Li, H. (2022). "Viscoelastic
    wave propagation for nearly constant-Q transverse isotropy."
    *Geophysical Prospecting* 70(7): 1176-1192.
    DOI: 10.1111/1365-2478.13230.

Historical co-anchor:
    Kjartansson, E. (1979). "Constant-Q wave propagation and
    attenuation." *Journal of Geophysical Research: Solid Earth*
    84(B9): 4737-4748. DOI: 10.1029/JB084iB09p04737.

This package collects the Hao 2022 two-tier weighting-function /
memory-variable implementation of Kjartansson's exact constant-Q
complex modulus, plus the Crank-Nicolson time-discretisation
recurrences and the Devito-side Eq builders used by the parent
repo's KJ method directories (Methods 64-67, 72-75, 77-78, 80-81).

Modules:
- `kjartansson`: Hao 2022 CN coefficient computation
  (`kjartansson_cn_coefficients(Q, f0, dt)`) + two-tier memory
  state builders for displacement (SBP / Dual-Pair / LW4 / REM)
  and velocity-stress (RSG / Lebedev) families.
- `paper_tables`: hand-transcribed Hao 2022 + Kjartansson 1979
  byte anchors (gamma formula, CN recurrence form, two-tier
  K=2 memory-field counts, asymptotic limits).

Phase Y/2d 2026-05-28: relocated from `00_common/kjartansson.py`
into this reproduction folder per the paper-faithful-modules
curation sweep; wired to the parent via `[tool.uv.sources]`
editable-dep.
"""
