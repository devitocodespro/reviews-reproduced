"""Hand-transcribed-constants regression tests for Hao 2022 + Kjartansson 1979.

Asserts:
  (1) Kjartansson 1979 gamma formula matches paper-canonical values at
      canonical Q to fp64.
  (2) The closed-form `kjartansson_gamma_anchor(Q)` in paper_tables
      byte-matches the solver's `kjartansson.kjartansson_gamma(Q)`.
  (3) Hao 2022 two-tier rate prescription `a_tilde = a / 2` is preserved
      by the solver's `kjartansson_cn_coefficients(Q, f0, dt)`.
  (4) The Crank-Nicolson recurrence form has been preserved (not swapped
      for forward / backward Euler).
  (5) The K=2 memory-field counts are exactly 6 / 4 / 2 for the three
      formulation families (locks the two-tier defining feature).
  (6) Stress-correction weights c1 = 1/Q and c2 = gamma/(2Q) are
      byte-matched.
  (7) Asymptotic limits (Q → ∞ low-loss; dt → 0 CN-freeze).
"""
from __future__ import annotations

import math

import pytest

from hao_2022.kjartansson import (
    kjartansson_cn_coefficients,
    kjartansson_gamma,
)
from hao_2022.paper_tables import (
    HAO_2022_ALPHA_FORM,
    HAO_2022_BETA_FORM,
    HAO_2022_CN_ALPHA_DT_ZERO_LIMIT,
    HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_COUPLED,
    HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_SCALAR,
    HAO_2022_MEMORY_FIELDS_VISCOELASTIC,
    HAO_2022_TIER1_AMPLITUDE_RATIO,
    HAO_2022_TIER2_AMPLITUDE_RATIO,
    HAO_2022_TIER2_RATE_RATIO,
    HAO_2022_TIME_INTEGRATOR_NAME,
    KJARTANSSON_1979_COMPLEX_MODULUS_FORM,
    KJARTANSSON_1979_CONSTANT_Q_PROPERTY,
    KJARTANSSON_GAMMA_SAMPLES,
    KJARTANSSON_LOW_LOSS_LIMIT_PRODUCT,
    hao_2022_alpha_anchor,
    hao_2022_beta_anchor,
    hao_2022_stress_correction_c1_anchor,
    hao_2022_stress_correction_c2_anchor,
    kjartansson_gamma_anchor,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: Kjartansson 1979 gamma formula
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Q,expected_gamma", list(KJARTANSSON_GAMMA_SAMPLES.items()))
def test_kjartansson_gamma_canonical_sample(Q, expected_gamma):
    """Paper-canonical gamma values to fp64."""
    assert kjartansson_gamma_anchor(Q) == pytest.approx(expected_gamma, abs=1e-15)


def test_kjartansson_gamma_q1_is_quarter():
    """Q=1 → gamma = arctan(1)/pi = pi/4 / pi = 1/4 exactly."""
    assert kjartansson_gamma_anchor(1.0) == 0.25


@pytest.mark.parametrize("Q", [1.0, 10.0, 40.0, 100.0])
def test_kjartansson_gamma_solver_matches_paper_anchor(Q):
    """The solver's `kjartansson_gamma(Q)` must byte-match
    `paper_tables.kjartansson_gamma_anchor(Q)`. Both are pure
    `math.atan(1/Q) / math.pi` (no asymptotic expansion swap)."""
    assert kjartansson_gamma(Q) == kjartansson_gamma_anchor(Q)


def test_kjartansson_gamma_low_loss_limit():
    """Q=10^8 sentinel: gamma * pi * Q → 1 (low-loss limit)."""
    Q = 1e8
    product = kjartansson_gamma_anchor(Q) * math.pi * Q
    assert product == pytest.approx(KJARTANSSON_LOW_LOSS_LIMIT_PRODUCT,
                                    abs=1e-10)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Hao 2022 two-tier rate prescription a_tilde = a / 2
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Q,f0,dt", [
    (40.0, 25.0, 1e-3),
    (100.0, 50.0, 5e-4),
    (10.0, 10.0, 2e-3),
])
def test_two_tier_rate_ratio_preserved(Q, f0, dt):
    """The Hao 2022 tier-2 rate = (1/2) * tier-1 rate. Recover it
    from the CN coefficients via alpha inversion:
        alpha = (1 - a dt/2) / (1 + a dt/2)
        ⇒ a = 2 * (1 - alpha) / (dt * (1 + alpha))
    """
    kj = kjartansson_cn_coefficients(Q, f0, dt)

    def _recover_a(alpha):
        return 2.0 * (1.0 - alpha) / (dt * (1.0 + alpha))

    a1 = _recover_a(kj['alpha_1'])
    a2 = _recover_a(kj['alpha_2'])
    assert a2 / a1 == pytest.approx(HAO_2022_TIER2_RATE_RATIO, rel=1e-12)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Crank-Nicolson recurrence form preserved
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("a,dt", [
    (0.1, 1e-3),
    (5.0, 5e-4),
    (50.0, 1e-4),
])
def test_cn_alpha_byte_match(a, dt):
    """Closed-form CN alpha matches the documented expression."""
    expected = (1.0 - a * dt * 0.5) / (1.0 + a * dt * 0.5)
    assert hao_2022_alpha_anchor(a, dt) == expected


def test_cn_form_strings_sentinel():
    """The form-name strings sentinel the time-integrator class.
    Flipping to forward-Euler or backward-Euler trips this."""
    assert HAO_2022_TIME_INTEGRATOR_NAME == "Crank-Nicolson"
    assert HAO_2022_ALPHA_FORM == "(1 - a*dt/2) / (1 + a*dt/2)"
    assert HAO_2022_BETA_FORM == "b*dt / (1 + a*dt/2)"


def test_cn_alpha_dt_zero_limit():
    """dt → 0 freezes memory: alpha → 1 (no decay over zero time)."""
    assert hao_2022_alpha_anchor(1.0, 0.0) == HAO_2022_CN_ALPHA_DT_ZERO_LIMIT


# ──────────────────────────────────────────────────────────────────
# Anchor 4: K=2 memory-field counts
# ──────────────────────────────────────────────────────────────────


def test_memory_field_counts_anchor():
    """The K=2 two-tier method has exactly 6 / 4 / 2 memory fields
    for the three formulation families. Flipping the tier count
    (e.g., to K=3) would break the counts."""
    assert HAO_2022_MEMORY_FIELDS_VISCOELASTIC == 6
    assert HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_COUPLED == 4
    assert HAO_2022_MEMORY_FIELDS_VISCOACOUSTIC_SCALAR == 2


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Stress-correction weights byte-match
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Q,f0,dt", [
    (40.0, 25.0, 1e-3),
    (100.0, 50.0, 5e-4),
])
def test_stress_correction_c1_byte_match(Q, f0, dt):
    """c1 = 1/Q (tier-1 correction weight). Byte-match the solver."""
    kj = kjartansson_cn_coefficients(Q, f0, dt)
    assert kj['c1'] == hao_2022_stress_correction_c1_anchor(Q)


@pytest.mark.parametrize("Q,f0,dt", [
    (40.0, 25.0, 1e-3),
    (100.0, 50.0, 5e-4),
])
def test_stress_correction_c2_byte_match(Q, f0, dt):
    """c2 = gamma(Q) / (2*Q). Byte-match the solver."""
    kj = kjartansson_cn_coefficients(Q, f0, dt)
    assert kj['c2'] == hao_2022_stress_correction_c2_anchor(Q)


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Tier-amplitude ratio (b vs b_tilde)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("Q,f0,dt", [
    (40.0, 25.0, 1e-3),
    (10.0, 10.0, 2e-3),
])
def test_tier_amplitude_ratio_2_to_1(Q, f0, dt):
    """Tier-1 forcing amplitude b = 2 * gamma * omega_0; tier-2
    b_tilde = gamma * omega_0. Recover ratio via beta inversion:
        beta = b * dt / (1 + a*dt/2)
    """
    kj = kjartansson_cn_coefficients(Q, f0, dt)
    # Recover b from beta_1, knowing a from alpha_1
    a1 = 2.0 * (1.0 - kj['alpha_1']) / (dt * (1.0 + kj['alpha_1']))
    a2 = 2.0 * (1.0 - kj['alpha_2']) / (dt * (1.0 + kj['alpha_2']))
    b1 = kj['beta_1'] * (1.0 + a1 * dt * 0.5) / dt
    b2 = kj['beta_2'] * (1.0 + a2 * dt * 0.5) / dt
    # gamma * omega_0 from b2 directly
    gamma_omega0_2 = b2 / HAO_2022_TIER2_AMPLITUDE_RATIO
    gamma_omega0_1 = b1 / HAO_2022_TIER1_AMPLITUDE_RATIO
    assert gamma_omega0_1 == pytest.approx(gamma_omega0_2, rel=1e-12)


# ──────────────────────────────────────────────────────────────────
# Anchor 7: Kjartansson 1979 complex-modulus form sentinel
# ──────────────────────────────────────────────────────────────────


def test_complex_modulus_form_sentinel():
    """The Kjartansson 1979 constant-Q complex modulus form is
    `M_0 * (i omega / omega_0)^{2 gamma}`. Flipping to an SLS L=1
    form would change the frequency dependence."""
    assert (KJARTANSSON_1979_COMPLEX_MODULUS_FORM
            == "M_0 * (i * omega / omega_0)**(2 * gamma)")
    assert KJARTANSSON_1979_CONSTANT_Q_PROPERTY is True
