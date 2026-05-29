"""Hand-transcribed paper anchors for Pratt (1990).

Primary paper:
    Pratt, R. G. (1990). "Frequency-domain elastic wave modeling by
    finite differences; a tool for crosshole seismic imaging."
    *Geophysics* 55(5): 626-632. DOI: 10.1190/1.1442874.

The Pratt 1990 paper is the canonical seismic-modelling reference
for the line-source projection from 3D to 2D Green's functions.

The anchors here are:
  - The closed-form G_2D(r, t) formula (causal time-domain form);
  - Heaviside causality + wavefront-arrival condition;
  - Angle-dependent acoustic R(p) formula (Z and cos θ structure);
  - Normal-incidence reflection limit;
  - Post-critical evanescent |R| = 1 sentinel;
  - Mirror-source reduction at β = 0 (axis-aligned interface).
"""
from __future__ import annotations

import math


# ──────────────────────────────────────────────────────────────────
# Anchor 1: G_2D(r, t) closed-form (time-domain line-source projection)
# ──────────────────────────────────────────────────────────────────


# The 2D acoustic Green's function for the scalar wave equation
#     ∂²u/∂t² − c² ∇²u = δ(x) δ(t)
# in time-domain form (Pratt 1990; vanderHijden 1987; Aki-Richards
# eq 4.18). DIFFERENT from the frequency-domain Hankel form
# G_ω(r) = -i/4 · H_0^(2)(ωr/c) — the time-domain closed form is the
# one the production code implements.
PRATT_1990_GREEN_FUNCTION_FORM = (
    "G_2D(r, t) = H(t - r/c) / (2*pi * sqrt(t^2 - r^2/c^2))"
)
PRATT_1990_GREEN_FUNCTION_DOMAIN = "time-domain"

# Frequency-domain alternative (NOT used by the production code,
# but documented here as a co-existing form).
PRATT_1990_GREEN_FUNCTION_FREQ_DOMAIN_ALT = (
    "G_omega(r) = -i/4 * H_0^(2)(omega * r / c)"
)


# Normalisation prefactor: 1 / (2*pi). This is the load-bearing
# constant that distinguishes the 2D line-source projection from
# the 3D point-source 1/(4*pi) normalisation.
PRATT_1990_GREEN_PREFACTOR = 1.0 / (2.0 * math.pi)
PRATT_1990_GREEN_3D_POINT_PREFACTOR = 1.0 / (4.0 * math.pi)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Causality + wavefront-arrival condition
# ──────────────────────────────────────────────────────────────────


# The Heaviside H(t - r/c) factor enforces causality: the Green's
# function vanishes BEFORE the wavefront arrival t = r/c. Silent
# removal of the Heaviside would produce acausal pre-front signal.
PRATT_1990_CAUSALITY_NAME = "Heaviside step H(t - r/c)"
PRATT_1990_WAVEFRONT_ARRIVAL_CONDITION = "t = r / c"


def pratt_1990_arrival_time(r: float, c: float) -> float:
    """Wavefront arrival time t_arrival = r / c (paper-canonical)."""
    return r / c


def pratt_1990_is_causal_before_arrival(r: float, c: float, t: float) -> bool:
    """G_2D(r, t) = 0 for t < r/c (paper-canonical causality)."""
    return t < r / c


# ──────────────────────────────────────────────────────────────────
# Anchor 3: 2D acoustic tail (vs 3D impulsive wavefront)
# ──────────────────────────────────────────────────────────────────


# In 2D, the wavefield "rings" after the wavefront passes (the
# 1/sqrt(t² − r²/c²) factor decays as 1/t at late times). In 3D the
# wavefield is impulsive — δ(t − r/c) / (4πr) — and vanishes
# immediately after the wavefront. This is the 2D "tail" / Huygens'
# violation famous in scalar acoustics.
PRATT_1990_HAS_2D_TAIL = True
PRATT_1990_3D_HAS_TAIL = False
PRATT_1990_LATE_TIME_DECAY_RATE = "1 / t (algebraic)"


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Angle-dependent acoustic R(p) formula
# ──────────────────────────────────────────────────────────────────


# The angle-dependent acoustic-acoustic reflection coefficient at
# horizontal slowness p:
#
#     R(p) = (Z2 cos θ1 − Z1 cos θ2) / (Z2 cos θ1 + Z1 cos θ2),
#     cos θi = sqrt(1 − Vi² p²),  Zi = ρi Vi.
#
# Silent swap of cos θ1 ↔ cos θ2 inverts the angle dependence (the
# Snell-conformal cos θ factor at each side weights the impedance
# of the OTHER side).
PRATT_1990_R_FORMULA_NAME = (
    "R(p) = (Z2 cos_th1 - Z1 cos_th2) / (Z2 cos_th1 + Z1 cos_th2)"
)


def pratt_1990_R_normal_incidence_anchor(rho_upper: float, c_upper: float,
                                          rho_lower: float,
                                          c_lower: float) -> float:
    """Closed-form normal-incidence acoustic reflection coefficient
        R(p=0) = (Z2 − Z1) / (Z2 + Z1)
    where Z = ρ c."""
    Z1 = rho_upper * c_upper
    Z2 = rho_lower * c_lower
    return (Z2 - Z1) / (Z2 + Z1)


# Canonical water-over-TTI sample (water ρ=1, V=1.5; TTI ρ=2.2,
# vertical V_p=3.0):
#     Z_w = 1 * 1.5 = 1.5
#     Z_TTI = 2.2 * 3.0 = 6.6
#     R = (6.6 - 1.5) / (6.6 + 1.5) ≈ 0.62962962...
# Computed in the same order as the solver (Z computed via
# ρ * c then differenced / summed) so the byte-equality test is
# unambiguous to fp64.
WATER_TTI_CANONICAL_R = (2.2 * 3.0 - 1.0 * 1.5) / (2.2 * 3.0 + 1.0 * 1.5)


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Post-critical evanescent regime |R| = 1
# ──────────────────────────────────────────────────────────────────


# For incidence into a FASTER lower medium (V_lower > V_upper) at
# horizontal slowness p > 1/V_lower, the lower-medium cos θ2 becomes
# imaginary; the reflection coefficient has |R| = 1 (total
# reflection). The production code handles this via complex-cast
# sqrt, returning a complex R with unit magnitude.
PRATT_1990_POST_CRITICAL_TOTAL_REFLECTION = True
PRATT_1990_CRITICAL_SLOWNESS_FORMULA = "p_critical = 1 / V_lower (when V_lower > V_upper)"


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Mirror-source reduction at β = 0
# ──────────────────────────────────────────────────────────────────


# For a horizontal interface (β = 0) at y = y_anchor, the mirror
# image of (x_s, y_s) is (x_s, 2*y_anchor - y_s). The dipping-
# interface generalisation reduces to this at β → 0.
PRATT_1990_AXIS_ALIGNED_MIRROR_FORM = (
    "(x_s, y_s) -> (x_s, 2*y_anchor - y_s) at beta=0"
)
