"""Moczo et al. (2002) effective-medium averaging reproduction package.

Primary paper:
    Moczo, P., Kristek, J., Vavryčuk, V., Archuleta, R. J., & Halada, L.
    (2002). "3D heterogeneous staggered-grid finite-difference modeling
    of seismic motion with volume harmonic and arithmetic averaging of
    elastic moduli and densities." *Bulletin of the Seismological
    Society of America* 92(8): 3042-3066.
    DOI: 10.1785/0120010167.

Co-anchors (orientation-aware extensions, kept inside this folder
because the production module implements all three in one file):

    Muir, F., Dellinger, J., Etgen, J., & Nichols, D. (1992).
    "Modeling elastic fields across irregular boundaries."
    *Geophysics* 57(9): 1189-1193. (Schoenberg-Muir calculus
    foundational paper.)

    Koene, E. F. M., Wittsten, J., & Robertsson, J. O. A. (2022).
    "Eliminating staircase numerical artifacts at sloped fluid-solid
    interfaces by an effective medium method." *Geophysical
    Journal International* 230(3): 1922-1937.
    DOI: 10.1093/gji/ggac164. (Production rotate-partition-
    average-rotate-back formula, Eqs 23-27.)

    Kristek, J., Moczo, P., Chaljub, E., & Kristeková, M. (2017).
    "An orthorhombic representation of a heterogeneous medium for
    the finite-difference modelling of seismic wave propagation."
    *Geophysical Journal International* 208(2): 1250-1264.
    DOI: 10.1093/gji/ggw456. (Axis-aligned orthorhombic limit;
    reduces from the Schoenberg-Muir form at β=0.)

Module structure:
- `effective_medium`: NumPy production implementation of three
  averaging schemes:
    * `moczo2002_average(...)` — axis-aligned isotropic-effective
      (Moczo 2002 BSSA).
    * `schoenberg_muir_average_scalar(...)` /
      `schoenberg_muir_average_grid(...)` — general-orientation
      effective via rotate-partition-average-rotate-back (Muir 1992
      + Koene-Robertsson 2022).
    * `cell_volume_fractions(...)` — sub-pixel area sampling for
      f1 and interface-normal direction.
- `paper_tables`: hand-transcribed paper anchors:
    * Moczo 2002 isotropic-effective form: arithmetic ρ, harmonic
      K and μ;
    * fluid-limit μ=0 / both-fluid sentinel;
    * Schoenberg-Muir rotate-partition-average-rotate-back form
      (Muir 1992 / Koene-Robertsson 2022 Eq 23-27).

Phase Y/2e 2026-05-28: relocated from `00_common/effective_medium.py`
into this reproduction folder per the paper-faithful-modules
curation sweep; wired to the parent via `[tool.uv.sources]`
editable-dep.
"""
