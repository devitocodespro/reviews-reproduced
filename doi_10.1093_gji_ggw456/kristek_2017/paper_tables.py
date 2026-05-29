"""Hand-transcribed paper anchors for Kristek et al. (2017) GJI.

Primary paper:
    Kristek, J., Moczo, P., Chaljub, E., & Kristeková, M. (2017).
    "An orthorhombic representation of a heterogeneous medium for
    the finite-difference modelling of seismic wave propagation."
    *Geophysical Journal International* 208(2): 1250-1264.
    DOI: 10.1093/gji/ggw456.

Cross-reproduction: the Kristek 2017 axis-aligned orthorhombic
averaging is implemented in `moczo_2002.effective_medium` (Phase
Y/2e relocation) under the `interface_treatment='kristek2017'`
dispatch — the Kristek 2017 scheme is the β=0 reduction of the
Schoenberg-Muir orientation-aware effective form, so the
production code routes it through the same helper.

The anchors here are load-bearing structural sentinels of the
Kristek 2017 method; the numerical Cij and per-cell averaging are
already byte-anchored at the `moczo_2002.paper_tables` level.
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Orthorhombic representation
# ──────────────────────────────────────────────────────────────────


# Kristek 2017's central claim: a heterogeneous medium can be
# represented by an **effective orthorhombic medium** for the
# purpose of finite-difference modelling at the cell scale. This
# is a strictly more general representation than the isotropic-
# effective Moczo 2002 form — orthorhombic admits direction-
# dependent stiffness eigenvalues.
KRISTEK_2017_EFFECTIVE_MEDIUM_SYMMETRY = "orthorhombic"
KRISTEK_2017_GENERALISES_MOCZO_2002 = True  # isotropic ⊂ orthorhombic
KRISTEK_2017_INCLUDED_BY_SCHOENBERG_MUIR = True  # orthorhombic ⊂ Schoenberg-Muir-general


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Axis-aligned (β = 0) limit
# ──────────────────────────────────────────────────────────────────


# The Kristek 2017 orthorhombic-representation form is the β = 0
# (axis-aligned interface) reduction of the Schoenberg-Muir
# orientation-aware effective form. The production code recovers
# this reduction by passing β = 0 to the Schoenberg-Muir grid
# helper, OR by dispatching the `interface_treatment='kristek2017'`
# mode in `moczo_2002.effective_medium`.
KRISTEK_2017_AXIS_ALIGNED_INTERFACE = True
KRISTEK_2017_REDUCES_FROM_SCHOENBERG_MUIR_AT_BETA_ZERO = True


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Averaging-rule inheritance from Moczo 2002
# ──────────────────────────────────────────────────────────────────


# Per moczo_2002.paper_tables: density averaged arithmetically,
# moduli averaged harmonically. Kristek 2017 inherits these for
# the diagonal components of the orthorhombic stiffness; the
# off-diagonal components are computed via the Schoenberg-Muir
# partitioning (rotate-partition-average-rotate-back) at β = 0.
KRISTEK_2017_DENSITY_AVERAGE = "arithmetic"
KRISTEK_2017_DIAGONAL_MODULUS_AVERAGE = "harmonic"


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Production-code location pointer (not paper claim)
# ──────────────────────────────────────────────────────────────────


# The Kristek 2017 averaging is implemented in
# `moczo_2002.effective_medium` as the
# `interface_treatment='kristek2017'` mode. This sentinel string
# is the cross-reproduction pointer; the moczo_2002 reproduction
# folder anchors the byte-level averaging rules.
KRISTEK_2017_PRODUCTION_LOCATION = (
    "moczo_2002.effective_medium "
    "(interface_treatment='kristek2017' dispatch)"
)
KRISTEK_2017_VALIDATION_REFERENCE = (
    "bouchon_1981.dwn_solver "
    "(via diag/d2_side_by_side.py at Petrobras dip=30 config)"
)
