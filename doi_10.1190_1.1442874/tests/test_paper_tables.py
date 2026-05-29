"""Hand-transcribed-constants regression tests for Pratt (1990).

Asserts:
  (1) G_2D form sentinel + normalisation prefactor 1/(2π);
  (2) Causality: G_2D = 0 before wavefront arrival; arrival at t = r/c;
  (3) Normal-incidence acoustic reflection R = (Z2 − Z1) / (Z2 + Z1)
      byte-match between solver and paper-tables anchor;
  (4) Canonical water-over-TTI R ≈ 0.6296 byte-match;
  (5) Angle-dependent R(p) formula form + post-critical |R| = 1
      sentinel; verified empirically with the solver for V_lower
      > V_upper case at p slightly > 1/V_lower;
  (6) Mirror source reduces to (x_s, 2*y_anchor - y_s) at β = 0.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pratt_1990.analytical_acoustic_2d import (
    acoustic_impedance_reflection,
    angle_dependent_acoustic_R,
    mirror_source_across_dipping_line,
)
from pratt_1990.paper_tables import (
    PRATT_1990_AXIS_ALIGNED_MIRROR_FORM,
    PRATT_1990_CAUSALITY_NAME,
    PRATT_1990_CRITICAL_SLOWNESS_FORMULA,
    PRATT_1990_GREEN_3D_POINT_PREFACTOR,
    PRATT_1990_GREEN_FUNCTION_DOMAIN,
    PRATT_1990_GREEN_FUNCTION_FORM,
    PRATT_1990_GREEN_FUNCTION_FREQ_DOMAIN_ALT,
    PRATT_1990_GREEN_PREFACTOR,
    PRATT_1990_HAS_2D_TAIL,
    PRATT_1990_3D_HAS_TAIL,
    PRATT_1990_LATE_TIME_DECAY_RATE,
    PRATT_1990_POST_CRITICAL_TOTAL_REFLECTION,
    PRATT_1990_R_FORMULA_NAME,
    PRATT_1990_WAVEFRONT_ARRIVAL_CONDITION,
    WATER_TTI_CANONICAL_R,
    pratt_1990_R_normal_incidence_anchor,
    pratt_1990_arrival_time,
    pratt_1990_is_causal_before_arrival,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: G_2D form + normalisation prefactor
# ──────────────────────────────────────────────────────────────────


def test_green_function_form_sentinel():
    """Locking the form string catches a silent swap from the
    time-domain Heaviside form to the frequency-domain Hankel form."""
    assert PRATT_1990_GREEN_FUNCTION_FORM == (
        "G_2D(r, t) = H(t - r/c) / (2*pi * sqrt(t^2 - r^2/c^2))"
    )
    assert PRATT_1990_GREEN_FUNCTION_DOMAIN == "time-domain"


def test_green_function_freq_domain_alt_documented():
    """The frequency-domain alternative is documented as a sentinel
    so future readers can identify it; the production code uses
    the time-domain form."""
    assert PRATT_1990_GREEN_FUNCTION_FREQ_DOMAIN_ALT == (
        "G_omega(r) = -i/4 * H_0^(2)(omega * r / c)"
    )


def test_green_prefactor_2d_is_one_over_two_pi():
    """2D line-source prefactor = 1/(2π). Distinct from the 3D
    point-source prefactor 1/(4π)."""
    assert PRATT_1990_GREEN_PREFACTOR == pytest.approx(1.0 / (2.0 * math.pi),
                                                       abs=1e-15)
    assert PRATT_1990_GREEN_3D_POINT_PREFACTOR == pytest.approx(
        1.0 / (4.0 * math.pi), abs=1e-15)
    # Ratio: 2D / 3D = 4π / (2π) = 2 (exactly)
    assert PRATT_1990_GREEN_PREFACTOR / PRATT_1990_GREEN_3D_POINT_PREFACTOR == 2.0


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Causality
# ──────────────────────────────────────────────────────────────────


def test_causality_sentinel_name():
    assert PRATT_1990_CAUSALITY_NAME == "Heaviside step H(t - r/c)"
    assert PRATT_1990_WAVEFRONT_ARRIVAL_CONDITION == "t = r / c"


@pytest.mark.parametrize("r,c", [(1.0, 1.5), (2.5, 3.0), (0.5, 2.0)])
def test_arrival_time_anchor(r, c):
    """Wavefront arrives at t = r / c."""
    assert pratt_1990_arrival_time(r, c) == r / c


@pytest.mark.parametrize("r,c,t,is_before", [
    (1.0, 1.5, 0.1, True),    # t = 0.1 < r/c = 0.667
    (1.0, 1.5, 0.6666, True),  # just before arrival
    (1.0, 1.5, 0.6667, False),  # just after
    (1.0, 1.5, 1.0, False),     # well after
])
def test_causality_before_after_arrival(r, c, t, is_before):
    assert pratt_1990_is_causal_before_arrival(r, c, t) == is_before


# ──────────────────────────────────────────────────────────────────
# Anchor 3: 2D-tail-vs-3D-impulsive sentinels
# ──────────────────────────────────────────────────────────────────


def test_2d_tail_vs_3d_impulsive_sentinels():
    """2D has the wavefield 'tail' (1/sqrt(t² − r²/c²) decay after
    the wavefront); 3D is impulsive (δ-function). Flipping these
    flags would document an opposite-dimension Green's function."""
    assert PRATT_1990_HAS_2D_TAIL is True
    assert PRATT_1990_3D_HAS_TAIL is False
    assert PRATT_1990_LATE_TIME_DECAY_RATE == "1 / t (algebraic)"


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Normal-incidence R formula byte-match against solver
# ──────────────────────────────────────────────────────────────────


def test_R_normal_incidence_form_sentinel():
    assert PRATT_1990_R_FORMULA_NAME == (
        "R(p) = (Z2 cos_th1 - Z1 cos_th2) / (Z2 cos_th1 + Z1 cos_th2)"
    )


@pytest.mark.parametrize("rho_u,c_u,rho_l,c_l", [
    (1.0, 1.5, 2.2, 3.0),   # canonical water-over-TTI
    (1.0, 1.5, 2.5, 4.0),   # higher-impedance contrast
    (2.5, 4.0, 1.0, 1.5),   # inverted (faster upper)
])
def test_R_normal_incidence_byte_match(rho_u, c_u, rho_l, c_l):
    """Solver's normal-incidence R MUST byte-match the closed-form
    anchor to fp64."""
    solver = acoustic_impedance_reflection(rho_u, c_u, rho_l, c_l)
    anchor = pratt_1990_R_normal_incidence_anchor(rho_u, c_u, rho_l, c_l)
    assert solver == anchor


def test_canonical_water_tti_R_anchor():
    """Water (ρ=1, V=1.5) over TTI (ρ=2.2, vertical V_p=3.0):
    R = (6.6 - 1.5) / (6.6 + 1.5) = 5.1 / 8.1 ≈ 0.6296."""
    R = acoustic_impedance_reflection(1.0, 1.5, 2.2, 3.0)
    assert R == WATER_TTI_CANONICAL_R
    assert R == pytest.approx(0.6296, abs=1e-4)


def test_R_at_zero_contrast_is_zero():
    """Identical-impedance layers ⇒ R = 0 (no reflection)."""
    R = acoustic_impedance_reflection(2.2, 3.0, 2.2, 3.0)
    assert R == 0.0


def test_R_sign_inverts_when_impedance_inverts():
    """Swapping (upper, lower) flips R sign."""
    R_forward = acoustic_impedance_reflection(1.0, 1.5, 2.2, 3.0)
    R_inverse = acoustic_impedance_reflection(2.2, 3.0, 1.0, 1.5)
    assert R_forward == -R_inverse


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Angle-dependent R(p) — normal-incidence p=0 + post-critical
# ──────────────────────────────────────────────────────────────────


def test_angle_dependent_R_at_zero_slowness_matches_normal():
    """At p = 0 (normal incidence), the angle-dependent R(p) MUST
    match the normal-incidence closed form."""
    rho_u, V_u = 1.0, 1.5
    rho_l, V_l = 2.2, 3.0
    R_p0 = angle_dependent_acoustic_R(0.0, rho_u, V_u, rho_l, V_l)
    R_normal = acoustic_impedance_reflection(rho_u, V_u, rho_l, V_l)
    assert R_p0.real == pytest.approx(R_normal, abs=1e-15)
    assert R_p0.imag == pytest.approx(0.0, abs=1e-15)


def test_post_critical_unit_magnitude_sentinel():
    """The sentinel is locked + the empirical post-critical |R| = 1
    is verified at p just inside the post-critical regime for a
    faster lower medium."""
    assert PRATT_1990_POST_CRITICAL_TOTAL_REFLECTION is True
    assert PRATT_1990_CRITICAL_SLOWNESS_FORMULA == (
        "p_critical = 1 / V_lower (when V_lower > V_upper)"
    )
    # Empirical check: faster lower (V_lower=3.0 > V_upper=1.5),
    # p just above 1/V_lower = 1/3.0 = 0.333... → |R| should be 1.
    rho_u, V_u = 1.0, 1.5
    rho_l, V_l = 2.2, 3.0
    p_critical = 1.0 / V_l
    p_post_critical = p_critical * 1.01  # 1% past critical
    R = angle_dependent_acoustic_R(p_post_critical, rho_u, V_u, rho_l, V_l)
    assert abs(R) == pytest.approx(1.0, abs=1e-12)


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Mirror-source reduction at β = 0
# ──────────────────────────────────────────────────────────────────


def test_mirror_source_axis_aligned_sentinel():
    assert PRATT_1990_AXIS_ALIGNED_MIRROR_FORM == (
        "(x_s, y_s) -> (x_s, 2*y_anchor - y_s) at beta=0"
    )


def test_mirror_source_axis_aligned_byte_match():
    """At β = 0 the mirror is exactly (x_s, 2*y_anchor − y_s)."""
    x_s, y_s = 1.0, 0.5
    x_a, y_a = 0.0, 2.0
    x_m, y_m = mirror_source_across_dipping_line(
        x_s, y_s, x_a, y_a, beta_rad=0.0)
    assert x_m == x_s
    assert y_m == 2.0 * y_a - y_s


def test_mirror_source_preserves_distance_from_interface():
    """The mirror image preserves the perpendicular distance from
    the interface line (a geometric invariant of reflection
    across any line)."""
    # Interface passes through (0, 0.5) at β = 30°
    x_a, y_a = 0.0, 0.5
    beta = math.radians(30.0)
    # Source above the interface
    x_s, y_s = 0.2, 1.0
    x_m, y_m = mirror_source_across_dipping_line(x_s, y_s, x_a, y_a, beta)

    # Perpendicular distance from source to interface line
    # Line in form ax + by + c = 0 where a = tan β, b = 1, c = -y_a - tan β * x_a
    # Wait: interface is y = y_a - tan β (x - x_a), so:
    #   tan β (x - x_a) + y - y_a = 0
    #   a = tan β,  b = 1,  c = -tan β x_a - y_a
    a = math.tan(beta)
    b = 1.0
    c = -math.tan(beta) * x_a - y_a
    denom = math.sqrt(a * a + b * b)
    d_source = abs(a * x_s + b * y_s + c) / denom
    d_mirror = abs(a * x_m + b * y_m + c) / denom
    assert d_source == pytest.approx(d_mirror, abs=1e-12)


# ──────────────────────────────────────────────────────────────────
# Anchor 7: Solver self-consistency at canonical config
# ──────────────────────────────────────────────────────────────────


def test_arrival_at_typical_water_offset():
    """At offset r = 1 km in water (c = 1.5 km/s), arrival = 2/3 s.
    Sanity check: wavefront arrives at expected time given a
    canonical V1 verification config."""
    r = 1.0
    c = 1.5
    assert pratt_1990_arrival_time(r, c) == pytest.approx(2.0 / 3.0, abs=1e-15)
