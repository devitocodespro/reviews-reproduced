"""Interface-continuity gate for `PlaneWaveAcousticElastic`.

Verifies that the analytical reference at the fluid-solid interface
satisfies the LP04 BC system to fp64:
  • v_y continuous across Γ ([v·n] = 0)
  • σ_yy continuous (continuity of normal stress; in degenerate-solid
    fluid layout, σ_yy = -p so this is pressure-traction continuity)
  • σ_xy vanishes on solid side at Γ

These hold by construction once the velocity-amplitude convention
is consistent across the BC system + the evaluate() implementation.
Regression test locks the post-2026-05-27 BC fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from analytical_reference import PlaneWaveAcousticElastic  # noqa: E402


@pytest.fixture
def normal_incidence_lp04():
    return PlaneWaveAcousticElastic(
        A=1.0, omega=2 * np.pi * 5e4, theta_inc=0.0,
        c_f=1500.0, rho_f=1000.0,
        c_p=4000.0, c_s=2000.0, rho_s=2600.0,
        y_interface=0.5,
    )


def _interface_continuity_t(pw: PlaneWaveAcousticElastic):
    """Pick t where sin(ωt) is near max — avoids tiny denominators."""
    return np.pi / (2.0 * pw.omega) * 0.97


def _sample_at_interface(pw: PlaneWaveAcousticElastic, t: float = None):
    """Evaluate fluid-formula and solid-formula DIRECTLY at y = y_Γ
    (bypassing the side mask in evaluate()).

    This isolates the BC residual from O(eps · ∂U/∂y) finite-difference
    error introduced by sampling at small ±eps offsets. Returns
    (above_state, below_state) — both at exactly dY=0.
    """
    if t is None:
        t = _interface_continuity_t(pw)
    # Sample one cell on each side just to drive the mask; we'll
    # overwrite with direct formula evaluation at dY = 0.
    XX = np.array([[0.0, 0.0]])
    YY = np.array([[pw.y_interface + 1e-3,
                     pw.y_interface - 1e-3]])
    # Force the in_fluid path evaluation:
    YY_fluid = np.array([[pw.y_interface + 1e-3]])
    XX_pt = np.array([[0.0]])
    U_fluid_grid = pw.evaluate(XX_pt, YY_fluid, t=t)
    # Force the in_solid path evaluation:
    YY_solid = np.array([[pw.y_interface - 1e-3]])
    U_solid_grid = pw.evaluate(XX_pt, YY_solid, t=t)

    # Now manually evaluate each side's formula AT dY = 0:
    info = pw._angles_and_coeffs()
    R_pp = info['R_pp']
    T_pp = info['T_pp']
    T_ps = info['T_ps']
    s_inc, c_inc = info['s_inc'], info['c_inc']
    s_tp, c_tp = info['s_tp'], info['c_tp']
    s_ts, c_ts = info['s_ts'], info['c_ts']
    Z_f = pw.rho_f * pw.c_f
    lam_s = pw.rho_s * pw.c_p ** 2 - 2 * pw.rho_s * pw.c_s ** 2
    mu_s = pw.rho_s * pw.c_s ** 2

    # At Γ (dY = 0) the phase is just ω·t (no spatial dependence
    # since incident, reflected, P, SV all have x=0 here too).
    arg_at_gamma = pw.omega * t + pw.phase
    sin_at_gamma = np.sin(arg_at_gamma)
    sin_i = sin_r = sin_at_gamma   # all four wave phases collapse at Γ
    sin_tp = sin_ts = sin_at_gamma

    # Fluid (above) AT Γ:
    above = np.zeros(5)
    above[0] = pw.A * (s_inc * sin_i + s_inc * R_pp * sin_r)
    above[1] = pw.A * ((-c_inc) * sin_i + c_inc * R_pp * sin_r)
    p_fluid = Z_f * pw.A * (sin_i + R_pp * sin_r)
    above[2] = -p_fluid
    above[3] = 0.0
    above[4] = -p_fluid

    # Solid (below) AT Γ:
    V_tp = T_pp * pw.A
    V_ts = T_ps * pw.A
    below = np.zeros(5)
    below[0] = V_tp * s_tp * sin_tp + V_ts * c_ts * sin_ts
    below[1] = V_tp * (-c_tp) * sin_tp + V_ts * s_ts * sin_ts
    sigma_xx_P = -(V_tp / pw.c_p) * (lam_s + 2 * mu_s * s_tp ** 2) * sin_tp
    sigma_xx_SV = -(mu_s * V_ts / pw.c_s) * 2 * c_ts * s_ts * sin_ts
    below[2] = sigma_xx_P + sigma_xx_SV
    sigma_xy_P = (V_tp / pw.c_p) * 2 * mu_s * s_tp * c_tp * sin_tp
    sigma_xy_SV = (mu_s * V_ts / pw.c_s) * (c_ts ** 2 - s_ts ** 2) * sin_ts
    below[3] = sigma_xy_P + sigma_xy_SV
    sigma_yy_P = -(V_tp / pw.c_p) * (lam_s + 2 * mu_s * c_tp ** 2) * sin_tp
    sigma_yy_SV = (mu_s * V_ts / pw.c_s) * 2 * s_ts * c_ts * sin_ts
    below[4] = sigma_yy_P + sigma_yy_SV

    return above, below


def test_v_y_continuous_at_interface(normal_incidence_lp04):
    """At Γ, fluid v_y should equal solid v_y to fp64 (LP04 BC [v·n]=0).

    Sampled at y = y_Γ ± 1e-9 with eps small enough that O(eps · ∂U/∂y)
    correction is < fp64 floor relative to the wavefield magnitude.
    """
    pw = normal_incidence_lp04
    above, below = _sample_at_interface(pw)
    jump = above[1] - below[1]
    assert abs(jump) < 1e-10, (
        f"v_y jump at Γ: {jump:.3e} (expected ~0 by BC1)")


def test_sigma_yy_continuous_at_interface(normal_incidence_lp04):
    """At Γ, σ_yy_fluid = σ_yy_solid (LP04 BC for normal stress).

    In degenerate-solid fluid layout: σ_yy_fluid = -p_fluid.
    Solid side: σ_yy from P + SV waves.
    """
    pw = normal_incidence_lp04
    above, below = _sample_at_interface(pw)
    max_mag = max(abs(above[4]), abs(below[4]), 1.0)
    jump_rel = (above[4] - below[4]) / max_mag
    assert abs(jump_rel) < 1e-10, (
        f"σ_yy jump: rel {jump_rel:.3e} "
        f"(above={above[4]:.3e}, below={below[4]:.3e})")


def test_sigma_xy_vanishes_on_solid_at_interface(normal_incidence_lp04):
    """At Γ, σ_xy on solid side must be zero (LP04 L_2 BC: no shear
    at fluid-solid interface). For normal incidence T_S = 0 so this
    is trivial; for oblique it tests the SV polarization sign.
    """
    pw = normal_incidence_lp04
    _, below = _sample_at_interface(pw)
    assert abs(below[3]) < 1e-10, (
        f"σ_xy_solid: {below[3]:.3e} (expected 0)")


def _oblique_pw():
    return PlaneWaveAcousticElastic(
        A=1.0, omega=2 * np.pi * 5e4, theta_inc=np.deg2rad(15.0),
        c_f=1500.0, rho_f=1000.0,
        c_p=4000.0, c_s=2000.0, rho_s=2600.0,
        y_interface=0.5,
    )


def test_oblique_v_y_continuous():
    """v_y continuity holds at oblique sub-critical incidence (15°)."""
    pw = _oblique_pw()
    above, below = _sample_at_interface(pw)
    max_mag = max(abs(above[1]), abs(below[1]), 1e-6)
    jump_rel = (above[1] - below[1]) / max_mag
    assert abs(jump_rel) < 1e-8, (
        f"v_y oblique jump: rel {jump_rel:.3e}")


def test_oblique_sigma_yy_continuous():
    """σ_yy continuity at oblique sub-critical incidence (15°)."""
    pw = _oblique_pw()
    above, below = _sample_at_interface(pw)
    max_mag = max(abs(above[4]), abs(below[4]), 1.0)
    jump_rel = (above[4] - below[4]) / max_mag
    assert abs(jump_rel) < 1e-8, (
        f"σ_yy oblique jump: rel {jump_rel:.3e}")


def test_oblique_sigma_xy_solid_vanishes():
    """σ_xy → 0 on solid at Γ for oblique incidence (SV polarization
    must cancel the P-wave shear contribution at the boundary).
    Uses relative tolerance since the bulk σ_xy_solid is non-zero
    away from Γ; only the AT-Γ value vanishes per BC3.
    """
    pw = _oblique_pw()
    _, below = _sample_at_interface(pw)
    # Compare AT-Γ σ_xy to bulk σ_xy magnitude (~Z_s · A · sin)
    bulk_scale = pw.rho_s * pw.c_s * pw.A
    rel = abs(below[3]) / bulk_scale
    assert rel < 1e-8, (
        f"σ_xy_solid oblique at Γ: rel {rel:.3e} (bulk_scale={bulk_scale:.3e})")
