"""Kristek, Moczo, Chaljub & Kristeková (2017) GJI reproduction package.

Primary paper:
    Kristek, J., Moczo, P., Chaljub, E., & Kristeková, M. (2017).
    "An orthorhombic representation of a heterogeneous medium for
    the finite-difference modelling of seismic wave propagation."
    *Geophysical Journal International* 208(2): 1250-1264.
    DOI: 10.1093/gji/ggw456.

Cross-reproduction note: the Kristek 2017 **axis-aligned
orthorhombic averaging** is implemented in the parent's
`moczo_2002.effective_medium` module (relocated to
`reproduced/doi_10.1785_0120010167/` in Phase Y/2e) under the
`interface_treatment='kristek2017'` dispatch. The Kristek 2017
scheme is the β=0 (axis-aligned interface) reduction of the
Schoenberg-Muir orientation-aware effective form; the production
code therefore exposes it as a mode of the moczo_2002 helper
rather than a separate module.

This reproduction folder primarily:
- documents the Kristek 2017 paper anchors (orthorhombic-
  representation key claim, axis-aligned reduction sentinel);
- preserves the two diagnostic scripts that validated the
  `interface_treatment='kristek2017'` mode against the
  Bouchon 1981 DWN reference in production figures (now in
  `diag/` rather than under `notes/archived_diagnostic/`);
- gives Kristek 2017 its own DOI'd reproduction folder so the
  paper-faithfulness chain is discoverable from the registry
  (QED guard FM2).

Modules:
- `paper_tables`: hand-transcribed Kristek 2017 paper anchors:
    * orthorhombic-representation form sentinel;
    * axis-aligned (β=0) reduction-of-Schoenberg-Muir sentinel;
    * arithmetic-density + harmonic-modulus averaging rules
      (inherited from Moczo 2002 → Kristek 2017 extends to
      orthorhombic).

Phase Y/2i 2026-05-29: bootstrap + recovery of diag scripts
from `notes/archived_diagnostic/`. The diag scripts are
preserved as graduation-review evidence of the production
`interface_treatment='kristek2017'` mode's validation against
the DWN reference at the Petrobras config (dx=5m, dip=30°,
SO=8).
"""
