"""Hand-transcribed-constants regression tests for Kristek (2017) GJI.

Asserts:
  (1) Orthorhombic-effective-medium symmetry sentinel locked;
      reduces-from-Schoenberg-Muir and generalises-Moczo-2002 flags
      pinned.
  (2) Axis-aligned (β = 0) sentinel locked.
  (3) Averaging-rule inheritance from Moczo 2002 sentinels.
  (4) Production-code location pointer + validation-reference
      sentinels locked (cross-reproduction discovery).
"""
from __future__ import annotations

from kristek_2017.paper_tables import (
    KRISTEK_2017_AXIS_ALIGNED_INTERFACE,
    KRISTEK_2017_DENSITY_AVERAGE,
    KRISTEK_2017_DIAGONAL_MODULUS_AVERAGE,
    KRISTEK_2017_EFFECTIVE_MEDIUM_SYMMETRY,
    KRISTEK_2017_GENERALISES_MOCZO_2002,
    KRISTEK_2017_INCLUDED_BY_SCHOENBERG_MUIR,
    KRISTEK_2017_PRODUCTION_LOCATION,
    KRISTEK_2017_REDUCES_FROM_SCHOENBERG_MUIR_AT_BETA_ZERO,
    KRISTEK_2017_VALIDATION_REFERENCE,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Orthorhombic-effective-medium symmetry
# ──────────────────────────────────────────────────────────────────


def test_effective_medium_symmetry_orthorhombic():
    """The Kristek 2017 effective medium is orthorhombic — NOT
    isotropic (Moczo 2002) and NOT fully anisotropic (general
    Schoenberg-Muir). Silent collapse to isotropic would hide the
    direction-dependent stiffness."""
    assert KRISTEK_2017_EFFECTIVE_MEDIUM_SYMMETRY == "orthorhombic"


def test_generalises_moczo_2002_sentinel():
    """Kristek 2017 generalises Moczo 2002 (isotropic ⊂ orthorhombic)."""
    assert KRISTEK_2017_GENERALISES_MOCZO_2002 is True


def test_included_by_schoenberg_muir_sentinel():
    """The Kristek 2017 orthorhombic representation is a subset of
    the Schoenberg-Muir general-orientation form."""
    assert KRISTEK_2017_INCLUDED_BY_SCHOENBERG_MUIR is True


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Axis-aligned (β = 0) limit
# ──────────────────────────────────────────────────────────────────


def test_axis_aligned_interface_sentinel():
    assert KRISTEK_2017_AXIS_ALIGNED_INTERFACE is True


def test_reduces_from_schoenberg_muir_at_beta_zero():
    """The Kristek 2017 form is the β = 0 limit of the
    Schoenberg-Muir orientation-aware effective. Sentinel locks
    this load-bearing reduction."""
    assert KRISTEK_2017_REDUCES_FROM_SCHOENBERG_MUIR_AT_BETA_ZERO is True


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Averaging-rule inheritance from Moczo 2002
# ──────────────────────────────────────────────────────────────────


def test_density_average_arithmetic():
    """Kristek 2017 inherits arithmetic-ρ from Moczo 2002."""
    assert KRISTEK_2017_DENSITY_AVERAGE == "arithmetic"


def test_diagonal_modulus_average_harmonic():
    """Diagonal-stiffness elements use harmonic averaging
    (inherited from Moczo 2002)."""
    assert KRISTEK_2017_DIAGONAL_MODULUS_AVERAGE == "harmonic"


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Cross-reproduction discovery pointers
# ──────────────────────────────────────────────────────────────────


def test_production_location_pointer():
    """The Kristek 2017 averaging is implemented in
    `moczo_2002.effective_medium` as the kristek2017 mode.
    Locks the cross-reproduction discovery path."""
    assert KRISTEK_2017_PRODUCTION_LOCATION == (
        "moczo_2002.effective_medium "
        "(interface_treatment='kristek2017' dispatch)"
    )


def test_validation_reference_pointer():
    """The Kristek mode is validated against the Bouchon 1981 DWN
    reference via the d2_side_by_side diagnostic at the Petrobras
    config. Locks the cross-reproduction validation chain."""
    assert KRISTEK_2017_VALIDATION_REFERENCE == (
        "bouchon_1981.dwn_solver "
        "(via diag/d2_side_by_side.py at Petrobras dip=30 config)"
    )
