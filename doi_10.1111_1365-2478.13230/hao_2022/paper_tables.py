"""Hand-transcribed paper anchors for Hao et al. (2022) + Kjartansson (1979).

Primary paper:
    Hao, Q., Greenhalgh, S., Huang, X., & Li, H. (2022). "Viscoelastic
    wave propagation for nearly constant-Q transverse isotropy."
    *Geophysical Prospecting* 70(7): 1176-1192.
    DOI: 10.1111/1365-2478.13230.

Historical co-anchor:
    Kjartansson, E. (1979). "Constant-Q wave propagation and
    attenuation." *Journal of Geophysical Research* 84(B9): 4737-4748.
    DOI: 10.1029/JB084iB09p04737.

The constants exposed here are the **symbolic / structural** load-bearing
facts about the Hao 2022 two-tier method. They are not pure floating-point
coefficient tables (the CN coefficients depend on (Q, f0, dt) and are
therefore data-driven, not paper-print constants). Instead, the byte-anchors
are:

(1) The exact Kjartansson 1979 gamma formula `gamma = (1/pi) * arctan(1/Q)`
    — verifiable to fp64 with the `gamma` sample table below.
(2) The Hao 2022 tier-2 rate-prescription invariant: `a_tilde = a / 2`
    (slower tier-2 relaxation). Implementation-defining constant; flipping
    this ratio reduces to a single-tier system.
(3) The K=2 memory-field counts (6 for viscoelastic TI, 4 for viscoacoustic
    coupled, 2 for viscoacoustic scalar).
(4) The Crank-Nicolson algebraic form of the recurrence (sentinel string),
    distinguishing CN from forward-Euler / backward-Euler / RK variants.
(5) The Kjartansson 1979 constant-Q complex-modulus form
    `M(omega) = M_0 * (i omega / omega_0)^{2 gamma}` as a sentinel.
"""
from __future__ import annotations

import math


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Kjartansson 1979 gamma formula
#     gamma = (1/pi) * arctan(1/Q)
# ──────────────────────────────────────────────────────────────────


def kjartansson_gamma_anchor(Q: float) -> float:
    """Closed-form Kjartansson attenuation parameter (paper-canonical).

    Reproduces Kjartansson 1979 §3 / Hao 2022 eq (3):
        gamma = (1/pi) * arctan(1/Q)

    This is the constant-Q-power-law exponent in the complex modulus
        M(omega) = M_0 * (i omega / omega_0)^{2 gamma}.

    In the low-loss limit (Q → ∞), gamma → 1/(pi Q); in the lossless
    limit, gamma → 0 and the modulus reduces to its elastic value.
    """
    return math.atan(1.0 / Q) / math.pi


# Reference gamma values at canonical Q. Verifiable to fp64 against
# `kjartansson_gamma_anchor(Q)` AND against the parent solver's
# `kjartansson_gamma(Q)` (both are byte-identical: pure math.atan).
KJARTANSSON_GAMMA_SAMPLES: dict[float, float] = {
    # math.atan(1.0)   / pi  (Q=1, 45-degree atan)
    1.0:    0.25,
    # math.atan(0.1)  / pi
    10.0:   0.031725517430553574,
    # math.atan(0.05) / pi
    20.0:   0.015902251256176378,
    # math.atan(0.025) / pi
    40.0:   0.007956089912025814,
    # math.atan(0.01) / pi   ≈ 1/(pi Q) in the low-loss limit
    100.0:  0.003182992764908255,
}


# Low-loss asymptotic sentinel:
# In the Q → ∞ limit, gamma * Q * pi → 1.
# Sentinel for verifying the formula has not been swapped for a wrong
# small-angle approximation.
KJARTANSSON_LOW_LOSS_LIMIT_PRODUCT = 1.0  # lim_{Q→∞} (gamma * pi * Q)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Hao 2022 two-tier rate prescription
#     a_tilde = a / 2  (tier-2 relaxation = half of tier-1 rate)
# ──────────────────────────────────────────────────────────────────


# The defining-feature ratio of the Hao 2022 two-tier method.
# Forcing this ratio to 1 collapses the system to a single-rate
# response; forcing it to 0 freezes the tier-2 memory.
HAO_2022_TIER2_RATE_RATIO = 0.5  # a_tilde / a per kjartansson.py:94

# Hao 2022 tier-1 amplitude prescription:
#     b = 2 * gamma * omega_0
# (vs tier-2: b_tilde = gamma * omega_0; ratio 2:1)
HAO_2022_TIER1_AMPLITUDE_RATIO = 2.0  # b / (gamma * omega_0)
HAO_2022_TIER2_AMPLITUDE_RATIO = 1.0  # b_tilde / (gamma * omega_0)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: K=2 memory-field counts
# ──────────────────────────────────────────────────────────────────


# Per kjartansson.py docstring (lines 30-33). These are the K=2
# memory-field totals — compare to a GSLS L=3 baseline (9 fields
# viscoelastic, 6 viscoacoustic coupled, 3 viscoacoustic scalar).
HAO_2022_MEMORY_FIELDS_VISCOELASTIC = 6   # w_xx, w_yy, w_xy, wt_xx, wt_yy, wt_xy
HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_COUPLED = 4  # w_p, w_r, wt_p, wt_r
HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_SCALAR = 2   # w_p, wt_p

GSLS_BASELINE_L = 3
GSLS_MEMORY_FIELDS_VISCOELASTIC = 9
GSLS_MEMORY_FIELDS_VISCOACOUSTIC_COUPLED = 6
GSLS_MEMORY_FIELDS_VISCOACOUSTIC_SCALAR = 3


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Crank-Nicolson time-discretisation form
# ──────────────────────────────────────────────────────────────────


# The recurrence form is:
#     w^{n+1} = alpha_1 * w^n + beta_1 * d_epsilon
# where alpha_1 = (1 - a*dt/2) / (1 + a*dt/2)  (Crank-Nicolson, NOT
# forward-Euler `1 - a*dt` and NOT backward-Euler `1 / (1 + a*dt)`).
# This sentinel locks the time-integrator class.
HAO_2022_TIME_INTEGRATOR_NAME = "Crank-Nicolson"
HAO_2022_ALPHA_FORM = "(1 - a*dt/2) / (1 + a*dt/2)"
HAO_2022_BETA_FORM = "b*dt / (1 + a*dt/2)"


def hao_2022_alpha_anchor(a: float, dt: float) -> float:
    """Closed-form CN decay coefficient for the Hao 2022 recurrence."""
    return (1.0 - a * dt * 0.5) / (1.0 + a * dt * 0.5)


def hao_2022_beta_anchor(a: float, b: float, dt: float) -> float:
    """Closed-form CN forcing coefficient for the Hao 2022 recurrence."""
    return b * dt / (1.0 + a * dt * 0.5)


# CN consistency check: lim_{dt→0} alpha → 1 (memory frozen at small dt).
HAO_2022_CN_ALPHA_DT_ZERO_LIMIT = 1.0
# CN consistency check: alpha(a, dt) + alpha(-a, dt) = 2 / (1 - (a*dt/2)^2)
# diverges as a*dt → 2; the recurrence is unconditionally stable in dt
# only because of damping (a > 0).


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Kjartansson 1979 complex-modulus form (symbolic sentinel)
# ──────────────────────────────────────────────────────────────────


# The continuous constant-Q complex modulus from Kjartansson 1979 §3:
#     M(omega) = M_0 * (i omega / omega_0)^{2 gamma}
# This is the load-bearing definition the Hao 2022 two-tier system
# approximates near omega_0. The sentinel string locks the form
# against silent swaps (e.g., to an SLS L=1 viscoelastic modulus
# which has a different frequency dependence).
KJARTANSSON_1979_COMPLEX_MODULUS_FORM = "M_0 * (i * omega / omega_0)**(2 * gamma)"
KJARTANSSON_1979_CONSTANT_Q_PROPERTY = True  # Q is frequency-independent
KJARTANSSON_1979_CAUSALITY_NAME = "Kramers-Kronig consistent (causal)"


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Stress-correction tier weights
#     c1 = 1 / Q     (primary correction)
#     c2 = gamma / (2 Q)  (second-order correction)
# ──────────────────────────────────────────────────────────────────


def hao_2022_stress_correction_c1_anchor(Q: float) -> float:
    """Tier-1 stress-correction weight: c1 = 1 / Q (paper-canonical)."""
    return 1.0 / Q


def hao_2022_stress_correction_c2_anchor(Q: float) -> float:
    """Tier-2 stress-correction weight: c2 = gamma(Q) / (2 Q)."""
    return kjartansson_gamma_anchor(Q) / (2.0 * Q)
