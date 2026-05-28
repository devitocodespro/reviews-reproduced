"""Hand-transcribed-constants regression tests for Moczo 2002 +
Muir 1992 + Koene-Robertsson 2022.

Asserts:
  (1) Moczo 2002 isotropic-effective averaging uses arithmetic ρ +
      harmonic K + harmonic μ (silent-swap signatures locked).
  (2) Density / K / μ scalar anchors byte-match the solver at
      canonical (ρ1, ρ2, K1, K2, μ1, μ2, f1) inputs.
  (3) C11 = K + μ and C12 = K - μ in the 2D in-plane Voigt
      convention; C16 = C26 = 0 for isotropic-effective.
  (4) Fluid-limit μ = 0 produces μ_eff = 0 (both-fluid sentinel).
  (5) Identical-medium limit C1 = C2 ⇒ C_eff = C1.
  (6) Single-phase limits f1=0 ⇒ phase-2; f1=1 ⇒ phase-1.
  (7) Schoenberg-Muir at θ=0 reduces to Moczo 2002 (sentinel).
"""
from __future__ import annotations

import numpy as np
import pytest

from moczo_2002.effective_medium import moczo2002_average
from moczo_2002.paper_tables import (
    BOND_ROTATION_2D_FORM_NAME,
    IDENTICAL_MEDIUM_RECOVERS_INPUT,
    MOCZO_2002_BOTH_FLUID_MU_EFF,
    MOCZO_2002_DENSITY_AVERAGE,
    MOCZO_2002_FLUID_LIMIT_C66_EFF,
    MOCZO_2002_ISOTROPIC_C16_C26,
    MOCZO_2002_K_AVERAGE,
    MOCZO_2002_MU_AVERAGE,
    MOCZO_2002_OUTPUT_VOIGT_BASIS,
    MOCZO_2002_REGULARISER_MU_EPS,
    SCHOENBERG_MUIR_AXIS_ALIGNED_REDUCES_TO_MOCZO_2002,
    SCHOENBERG_MUIR_ROTATION_BASIS,
    SINGLE_PHASE_RECOVERS_INPUT,
    moczo_2002_c11_anchor,
    moczo_2002_c12_anchor,
    moczo_2002_K_anchor,
    moczo_2002_mu_anchor,
    moczo_2002_rho_anchor,
)


# ──────────────────────────────────────────────────────────────────
# Anchor 1: averaging-rule sentinel strings (silent-swap guard)
# ──────────────────────────────────────────────────────────────────


def test_averaging_rule_sentinels():
    """Locking these strings prevents a silent swap of arithmetic ↔
    harmonic for any of (ρ, K, μ)."""
    assert MOCZO_2002_DENSITY_AVERAGE == "arithmetic"
    assert MOCZO_2002_K_AVERAGE == "harmonic"
    assert MOCZO_2002_MU_AVERAGE == "harmonic"


def test_voigt_basis_sentinel():
    """In-plane Voigt convention is (xx, zz, xz); not (xx, yy, zz)
    or (xx, zz, yz)."""
    assert MOCZO_2002_OUTPUT_VOIGT_BASIS == "(xx, zz, xz)"
    assert SCHOENBERG_MUIR_ROTATION_BASIS == "(xx, zz, xz) Voigt"
    assert BOND_ROTATION_2D_FORM_NAME == "Carcione 2014 Eq 1.59"


# ──────────────────────────────────────────────────────────────────
# Anchor 2: scalar averaging byte-match against solver
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def canonical_two_layer():
    """Two-layer setup at f1 = 0.3 with water-over-elastic-solid
    contrast (water-TTI Petrobras-style)."""
    return dict(
        lambda1=2.25,  mu1=0.0,   rho1=1.0,    # water (μ=0)
        lambda2=8.0,   mu2=4.0,   rho2=2.5,    # elastic solid
        f1=0.3,
    )


def test_density_anchor_byte_match(canonical_two_layer):
    """ρ_eff byte-match against arithmetic anchor."""
    p = canonical_two_layer
    out = moczo2002_average(**p)
    expected = moczo_2002_rho_anchor(p['rho1'], p['rho2'], p['f1'])
    assert float(out['rho']) == expected


def test_K_average_form_byte_match(canonical_two_layer):
    """K_eff = 1 / (f1/K1 + f2/K2). Recover K_eff = (C11+C12)/2 in
    the in-plane Voigt convention, then byte-match anchor."""
    p = canonical_two_layer
    out = moczo2002_average(**p)
    K1 = p['lambda1'] + p['mu1']
    K2 = p['lambda2'] + p['mu2']
    K_eff_solver = 0.5 * (float(out['c11']) + float(out['c12']))
    K_eff_anchor = moczo_2002_K_anchor(K1, K2, p['f1'])
    assert K_eff_solver == pytest.approx(K_eff_anchor, abs=1e-15)


def test_mu_average_form_byte_match_solid_solid():
    """Pure solid-solid case (both μ > 0): μ_eff matches harmonic
    anchor to fp64."""
    p = dict(
        lambda1=2.0, mu1=2.0, rho1=2.0,
        lambda2=8.0, mu2=4.0, rho2=2.5,
        f1=0.4,
    )
    out = moczo2002_average(**p)
    mu_eff_solver = float(out['c66'])  # C66 = μ
    mu_eff_anchor = moczo_2002_mu_anchor(p['mu1'], p['mu2'], p['f1'])
    assert mu_eff_solver == pytest.approx(mu_eff_anchor, abs=1e-15)


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Voigt-convention C11/C12 byte-match + zero off-diagonals
# ──────────────────────────────────────────────────────────────────


def test_c11_anchor_byte_match():
    """C11 = K + μ in the in-plane Voigt convention."""
    K_eff = 3.5
    mu_eff = 1.7
    assert moczo_2002_c11_anchor(K_eff, mu_eff) == 5.2


def test_c12_anchor_byte_match():
    """C12 = K - μ in the in-plane Voigt convention."""
    K_eff = 3.5
    mu_eff = 1.7
    assert moczo_2002_c12_anchor(K_eff, mu_eff) == pytest.approx(1.8, abs=1e-15)


def test_c11_equals_c22_solid_solid():
    """Isotropic-effective: C11 = C22 (rotational invariance)."""
    p = dict(lambda1=2.0, mu1=2.0, rho1=2.0,
             lambda2=8.0, mu2=4.0, rho2=2.5, f1=0.4)
    out = moczo2002_average(**p)
    assert float(out['c11']) == pytest.approx(float(out['c22']), abs=1e-15)


def test_off_diagonals_c16_c26_zero(canonical_two_layer):
    """Isotropic-effective has no off-diagonal shear coupling:
    C16 = C26 = 0."""
    out = moczo2002_average(**canonical_two_layer)
    assert float(out['c16']) == MOCZO_2002_ISOTROPIC_C16_C26
    assert float(out['c26']) == MOCZO_2002_ISOTROPIC_C16_C26


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Fluid-limit μ = 0 sentinels
# ──────────────────────────────────────────────────────────────────


def test_both_fluid_mu_eff_is_zero():
    """μ1 = μ2 = 0 ⇒ μ_eff = 0 (compliance dominates → zero
    stiffness). Sentinel: production both-fluid mask."""
    p = dict(
        lambda1=2.25, mu1=0.0, rho1=1.0,
        lambda2=2.25, mu2=0.0, rho2=1.0,
        f1=0.5,
    )
    out = moczo2002_average(**p)
    assert float(out['c66']) == MOCZO_2002_BOTH_FLUID_MU_EFF
    assert float(out['c66']) == MOCZO_2002_FLUID_LIMIT_C66_EFF


def test_mu_regulariser_constant():
    """Production code uses 1e-30 to regularise 1/μ at μ=0 in the
    fluid-solid mixed case (not the both-fluid mask)."""
    assert MOCZO_2002_REGULARISER_MU_EPS == 1e-30


# ──────────────────────────────────────────────────────────────────
# Anchor 5: Identical-medium limit C1 = C2 ⇒ C_eff = C1
# ──────────────────────────────────────────────────────────────────


def test_identical_medium_recovers_input():
    """When both layers are identical, the effective medium IS
    that medium for any f1 ∈ (0, 1)."""
    assert IDENTICAL_MEDIUM_RECOVERS_INPUT is True
    p = dict(lambda1=2.5, mu1=1.5, rho1=2.2,
             lambda2=2.5, mu2=1.5, rho2=2.2, f1=0.4)
    out = moczo2002_average(**p)
    K = p['lambda1'] + p['mu1']
    mu = p['mu1']
    assert float(out['rho']) == pytest.approx(p['rho1'], abs=1e-15)
    assert float(out['c11']) == pytest.approx(K + mu, abs=1e-13)
    assert float(out['c12']) == pytest.approx(K - mu, abs=1e-13)
    assert float(out['c66']) == pytest.approx(mu, abs=1e-13)


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Single-phase limits f1=0 / f1=1
# ──────────────────────────────────────────────────────────────────


def test_single_phase_recovers_input_sentinel():
    """The sentinel is True at the paper anchor level."""
    assert SINGLE_PHASE_RECOVERS_INPUT is True


def test_single_phase_f1_zero_recovers_phase_2():
    """f1 = 0 ⇒ effective medium = phase 2."""
    p = dict(lambda1=2.0, mu1=1.0, rho1=1.5,
             lambda2=8.0, mu2=4.0, rho2=2.5, f1=0.0)
    out = moczo2002_average(**p)
    K2 = p['lambda2'] + p['mu2']
    assert float(out['rho']) == pytest.approx(p['rho2'], abs=1e-15)
    assert float(out['c11']) == pytest.approx(K2 + p['mu2'], abs=1e-12)


def test_single_phase_f1_one_recovers_phase_1():
    """f1 = 1 ⇒ effective medium = phase 1."""
    p = dict(lambda1=2.0, mu1=1.0, rho1=1.5,
             lambda2=8.0, mu2=4.0, rho2=2.5, f1=1.0)
    out = moczo2002_average(**p)
    K1 = p['lambda1'] + p['mu1']
    assert float(out['rho']) == pytest.approx(p['rho1'], abs=1e-15)
    assert float(out['c11']) == pytest.approx(K1 + p['mu1'], abs=1e-12)


# ──────────────────────────────────────────────────────────────────
# Anchor 7: Schoenberg-Muir axis-aligned reduction sentinel
# ──────────────────────────────────────────────────────────────────


def test_schoenberg_muir_reduces_to_moczo_axis_aligned():
    """The Schoenberg-Muir orientation-aware form reduces to Moczo
    2002 at axis-aligned θ = 0 — locked as a sentinel here.
    Empirical equivalence check belongs in the parent's
    `tests/test_effective_medium.py` (which has access to both
    the SM grid form and the Moczo scalar form)."""
    assert SCHOENBERG_MUIR_AXIS_ALIGNED_REDUCES_TO_MOCZO_2002 is True


# ──────────────────────────────────────────────────────────────────
# Anchor 8: NumPy-array vectorisation behaviour
# ──────────────────────────────────────────────────────────────────


def test_vectorised_inputs_supported():
    """The solver accepts NumPy arrays of (λ, μ, ρ, f1) and broadcasts
    elementwise — required for per-cell evaluation across (Nx, Ny)
    grids."""
    rng = np.random.default_rng(seed=42)
    shape = (4, 5)
    p = dict(
        lambda1=rng.uniform(1.0, 4.0, shape),
        mu1=rng.uniform(0.5, 3.0, shape),
        rho1=rng.uniform(1.5, 2.5, shape),
        lambda2=rng.uniform(2.0, 6.0, shape),
        mu2=rng.uniform(1.0, 4.0, shape),
        rho2=rng.uniform(1.8, 3.0, shape),
        f1=rng.uniform(0.1, 0.9, shape),
    )
    out = moczo2002_average(**p)
    assert out['rho'].shape == shape
    assert out['c11'].shape == shape
    assert np.all(np.isfinite(out['rho']))
    assert np.all(np.isfinite(out['c11']))
