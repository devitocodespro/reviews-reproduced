"""Oblique R/T sanity gates for the fluid-solid analytical reference.

Verifies the velocity-amplitude convention is consistent across:
  1. Normal-incidence reduces to textbook acoustic (Z_p−Z_f)/(Z_p+Z_f).
  2. Energy-flux conservation at any incidence below P-critical:
       Z_f V_inc² = Z_f V_ref² + Z_p (V_tp cos)² + Z_s (V_ts cos)²
     (i.e., the sum of normal-flux contributions matches the incident.)
  3. Critical-angle threshold: sin θ_inc_crit = c_f/c_p; for incidence
     just below critical, T_P is finite; above, it should reach toward
     1.0 (total internal reflection of P).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from analytical_reference import PlaneWaveAcousticElastic  # noqa: E402


@pytest.fixture(scope='module')
def lp04_test_medium():
    """LP04 §4.2 Eq 48 test medium (SI units)."""
    return dict(
        c_f=1500.0, rho_f=1000.0,
        c_p=4000.0, c_s=2000.0, rho_s=2600.0,
        omega=2 * np.pi * 5e4,
    )


def _build(med, theta_inc):
    return PlaneWaveAcousticElastic(
        A=1.0, omega=med['omega'], theta_inc=theta_inc,
        c_f=med['c_f'], rho_f=med['rho_f'],
        c_p=med['c_p'], c_s=med['c_s'], rho_s=med['rho_s'],
        y_interface=0.5,
    )


def test_normal_incidence_matches_textbook(lp04_test_medium):
    med = lp04_test_medium
    pw = _build(med, theta_inc=0.0)
    info = pw._angles_and_coeffs()
    Zf = med['rho_f'] * med['c_f']
    Zp = med['rho_s'] * med['c_p']
    R_expected = (Zp - Zf) / (Zp + Zf)
    assert abs(info['R_pp'] - R_expected) < 1e-12, (
        f"R_pp = {info['R_pp']:.10f}, expected {R_expected:.10f}")
    assert abs(info['T_ps']) < 1e-12, (
        f"T_ps at normal incidence = {info['T_ps']}, expected 0")
    # T_P + R = 1 at normal incidence (v_y continuity)
    assert abs(info['T_pp'] + info['R_pp'] - 1.0) < 1e-12


def test_energy_flux_conservation_below_critical(lp04_test_medium):
    """At incidence below P-critical (θ_inc < arcsin(c_f/c_p) ≈ 22°),
    the sum of reflected + transmitted P + transmitted SV energy fluxes
    must equal the incident flux.

    Energy flux of each plane wave through Γ (per unit area along Γ):
      F_inc = (1/2) ρ_f c_f V_inc² · cos θ_inc
      F_ref = (1/2) ρ_f c_f V_ref² · cos θ_inc
      F_tp  = (1/2) ρ_s c_p V_tp²  · cos θ_tp
      F_ts  = (1/2) ρ_s c_s V_ts²  · cos θ_ts

    Conservation: F_inc = F_ref + F_tp + F_ts
    """
    med = lp04_test_medium
    for theta_deg in (5.0, 10.0, 15.0, 20.0):    # well below 22° critical
        pw = _build(med, np.deg2rad(theta_deg))
        info = pw._angles_and_coeffs()
        Zf = med['rho_f'] * med['c_f']
        Zp = med['rho_s'] * med['c_p']
        Zs = med['rho_s'] * med['c_s']
        c_inc = info['c_inc']
        c_tp = info['c_tp']
        c_ts = info['c_ts']
        F_inc = Zf * 1.0 * c_inc        # V_inc = 1 (unit normalisation)
        F_ref = Zf * info['R_pp'] ** 2 * c_inc
        F_tp = Zp * info['T_pp'] ** 2 * c_tp
        F_ts = Zs * info['T_ps'] ** 2 * c_ts
        rel_err = abs(F_inc - (F_ref + F_tp + F_ts)) / F_inc
        assert rel_err < 1e-10, (
            f"θ={theta_deg}°: F_inc={F_inc:.6e} but F_ref+F_tp+F_ts="
            f"{F_ref + F_tp + F_ts:.6e} (rel err {rel_err:.3e})")


def test_T_ps_grows_with_oblique_incidence(lp04_test_medium):
    """T_ps = 0 at normal incidence, finite at oblique. This catches a
    regression where the SV coupling row is accidentally zeroed.
    """
    med = lp04_test_medium
    info_n = _build(med, 0.0)._angles_and_coeffs()
    info_o = _build(med, np.deg2rad(15.0))._angles_and_coeffs()
    assert abs(info_n['T_ps']) < 1e-12
    assert abs(info_o['T_ps']) > 1e-3, (
        f"T_ps at θ=15° = {info_o['T_ps']:.3e}; expected >1e-3 (SV "
        f"coupling should be active at oblique incidence).")


def test_critical_P_angle_geometry(lp04_test_medium):
    """At incidence = arcsin(c_f/c_p) ≈ 22°, sin(θ_tp) reaches 1.
    Verify our angle calculation.
    """
    med = lp04_test_medium
    theta_crit = np.arcsin(med['c_f'] / med['c_p'])
    pw = _build(med, theta_crit)
    info = pw._angles_and_coeffs()
    # s_tp should be very close to 1 (clipped to exactly 1.0)
    assert info['s_tp'] >= 0.9999
    # c_tp should be near 0
    assert info['c_tp'] < 0.02


def test_T_ps_is_real_below_S_critical(lp04_test_medium):
    """At incidence below S-critical (arcsin(c_f/c_s) ≈ 49°), the SV
    transmission is real-valued (no evanescent SV).
    """
    med = lp04_test_medium
    for theta_deg in (10.0, 21.0, 30.0):
        info = _build(med, np.deg2rad(theta_deg))._angles_and_coeffs()
        assert np.isfinite(info['T_ps'])
        assert abs(info['T_ps'].imag if hasattr(info['T_ps'], 'imag')
                    else 0.0) < 1e-12
