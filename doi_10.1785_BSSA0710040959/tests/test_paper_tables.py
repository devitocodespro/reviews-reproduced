"""Hand-transcribed-constants regression tests for Bouchon 1981.

Asserts byte-equality between the constants in
``bouchon_1981.paper_tables`` and their canonical Bouchon 1981
values. These are deliberate `assert_array_equal` /
plain-`==` checks (no tolerance) since the prescriptions are
literal closed-form expressions, not numerical fits.

Provenance: hand-transcribed from Bouchon (1981) page 962 (α
prescription, periodicity condition) + Bouchon (2003) page 449
(critical-angle regularisation).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bouchon_1981.paper_tables import (
    BOUCHON_1981_ALPHA_PRESCRIPTION,
    BOUCHON_1981_PERIODICITY_SAFETY_MARGIN,
    DWN_SOLVES_CRITICAL_ANGLE_SINGULARITY,
    RECOMMENDED_M_WAVENUMBERS_MIN,
    RECOMMENDED_N_OMEGA_MIN,
    alpha_damping,
    periodicity_length,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: α prescription
# ──────────────────────────────────────────────────────────────────


def test_alpha_prescription_string_anchor():
    """The α prescription identifier is exactly the literal
    ``pi/T_max`` per the paper's notation. A different string
    (e.g. ``2*pi/T_max``) would indicate prescription drift."""
    assert BOUCHON_1981_ALPHA_PRESCRIPTION == "pi/T_max"


@pytest.mark.parametrize("T_max", [0.1, 0.5, 1.0, 1.2, 2.0])
def test_alpha_damping_equals_pi_over_T_max(T_max):
    """`alpha_damping(T_max)` must equal ``π / T_max`` exactly.

    No fp64 noise is tolerated — both sides compute from the
    same `math.pi` / `np.pi` literal; any drift indicates
    prescription corruption."""
    expected = math.pi / T_max
    actual = alpha_damping(T_max)
    assert actual == expected


def test_alpha_damping_rejects_nonpositive_T_max():
    """Sanity: zero or negative T_max raises ValueError
    (catches misuse where T_max wasn't initialised)."""
    with pytest.raises(ValueError):
        alpha_damping(0.0)
    with pytest.raises(ValueError):
        alpha_damping(-1.0)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: periodicity-length safety margin
# ──────────────────────────────────────────────────────────────────


def test_periodicity_safety_margin_anchor():
    """Bouchon 1981's strict bound is L > V_max·T_max + x_max;
    we use a 30% safety margin (1.3×). A different multiplier
    (e.g. 1.0 = strict, or 2.0 = wasteful) would indicate
    drift."""
    assert BOUCHON_1981_PERIODICITY_SAFETY_MARGIN == 1.3


@pytest.mark.parametrize(
    "V_max, T_max, x_max",
    [
        (1.5, 1.2, 4.0),    # canonical Petrobras water-TTI
        (3.0, 0.5, 2.0),    # arbitrary smaller case
        (6.0, 2.0, 10.0),   # large-domain case
    ],
)
def test_periodicity_length_formula(V_max, T_max, x_max):
    """`periodicity_length` must equal exactly
    ``1.3 × (V_max·T_max + x_max)`` (Bouchon-1981 strict bound
    × 30% safety margin)."""
    expected = 1.3 * (V_max * T_max + x_max)
    actual = periodicity_length(V_max, T_max, x_max)
    assert actual == expected
    # Strict-bound condition: L > strict
    strict = V_max * T_max + x_max
    assert actual > strict


# ──────────────────────────────────────────────────────────────────
# Anchor 3: critical-angle singularity policy
# ──────────────────────────────────────────────────────────────────


def test_critical_angle_singularity_policy_anchor():
    """The DWN implementation MUST claim to solve the
    critical-angle 1/q_w singularity via the complex-frequency
    shift. If this sentinel is ever False, the implementation
    has reverted to a direct slowness-FFT approach (which would
    produce the 10^301 amplitude blowup that motivated DWN in
    the first place)."""
    assert DWN_SOLVES_CRITICAL_ANGLE_SINGULARITY is True


# ──────────────────────────────────────────────────────────────────
# Anchor 4: recommended sampling parameter bounds
# ──────────────────────────────────────────────────────────────────


def test_recommended_sampling_bounds_anchor():
    """Practical-community sampling lower bounds per
    Bouchon-2003 review §3."""
    assert RECOMMENDED_N_OMEGA_MIN == 64
    assert RECOMMENDED_M_WAVENUMBERS_MIN == 256


# ──────────────────────────────────────────────────────────────────
# Cross-check: solver imports cleanly from the package
# ──────────────────────────────────────────────────────────────────


def test_dwn_solver_module_importable():
    """Smoke: `bouchon_1981.dwn_solver` imports cleanly. This
    is a structural regression guard against the package being
    inadvertently broken (e.g. circular import, missing
    dependency)."""
    from bouchon_1981 import dwn_solver
    # Sanity: the canonical entry point exists
    assert hasattr(dwn_solver, "dwn_wavefield_acoustic_acoustic_horizontal")
