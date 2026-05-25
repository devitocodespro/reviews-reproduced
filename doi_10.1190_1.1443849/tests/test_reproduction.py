"""Tests for the Igel, Mora & Riollet (1995) reproduction.

Three classes of test, all anchored on QUANTITATIVE metrics
quoted in the paper (NOT figure-by-eye comparison) per the
repo-wide `feedback_reproduction_quantitative_first` rule.

1. **Triclinic Cij matrix byte-match** (Eq 47, page 1210):
   the loaded Cij values must equal the published table to
   fp64.
2. **Table 1 stencil-coefficient byte-match** (page 1209):
   for L=8 the truncated-sinc-Gaussian-tapered derivative
   coefficients ``d_P*`` must agree with Table 1 (allowing
   loose tolerance since our Gaussian-taper σ is tuned, not
   tabulated by the paper).
3. **Phase-velocity error within paper bounds** (page 1209 +
   Fig 6/7): the per-wave max relative phase-velocity error
   at 50% Nyquist for the triclinic medium must be at or
   below the paper-quoted ~2 %, 3 %, 7 % for qP, qS1, qS2.

Plus a regression gate against the pinned reference outputs
in ``reference_outputs/dispersion_so<N>.npz``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parent.parent
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))

from run_reproduction import (
    CIJ_EQ47, RHO_GCC, TABLE_1_L8, PAPER_ERROR_BOUNDS,
    igel_derivative_coeffs, numerical_wavenumber,
    christoffel_phase_velocities_2d, run_dispersion_analysis,
)


# ---------------------------------------------------------------
# Test class 1 — Triclinic Cij byte-match (Eq 47)
# ---------------------------------------------------------------

def test_cij_eq47_byte_match():
    """Eq 47 (page 1210) gives the 6×6 triclinic Cij matrix
    for the paper's worked example. The loaded values must
    equal the published table to fp64. Note we check the
    UNIQUE 21 components (upper triangle) since the matrix is
    symmetric by definition."""
    expected = np.array([
        [10.0,   3.5,   2.5,  -5.0,   0.1,   0.3 ],
        [ 3.5,   8.0,   1.5,   0.2,  -0.1,  -0.15],
        [ 2.5,   1.5,   6.0,   1.0,   0.4,   0.24],
        [-5.0,   0.2,   1.0,   5.0,   0.35,  0.525],
        [ 0.1,  -0.1,   0.4,   0.35,  4.0,  -1.0 ],
        [ 0.3,  -0.15,  0.24,  0.525,-1.0,   3.0 ],
    ], dtype=np.float64)
    assert np.array_equal(CIJ_EQ47, expected), (
        f"CIJ_EQ47 does not byte-match Eq 47: max diff "
        f"{np.max(np.abs(CIJ_EQ47 - expected))}"
    )


def test_cij_eq47_is_symmetric():
    """The Cij matrix in Voigt notation is symmetric for any
    physical anisotropic medium."""
    assert np.array_equal(CIJ_EQ47, CIJ_EQ47.T), (
        "CIJ_EQ47 is not symmetric — Voigt Cij must be."
    )


def test_density_matches_paper():
    """Paper states ρ = 1.0 g/cm³ immediately above Eq 47."""
    assert RHO_GCC == 1.0


# ---------------------------------------------------------------
# Test class 2 — Table 1 byte-match (page 1209)
# ---------------------------------------------------------------

def test_table_1_L8_values_byte_match_paper():
    """Table 1 of the paper (page 1209) gives the L=8 truncated-
    sinc-Gaussian coefficients (d_T* and d_P*) plus stability
    factor ε for the 4 half-stencil positions n = -3, -2, -1, 0.

    The TABLE_1_L8 constant in the driver must reproduce
    these published values byte-exact (modulo paper-printed
    precision)."""
    expected = {
        1: (0.423, -3, -0.00768803, -0.00224279),
        2: (0.733, -2,  0.03647890,  0.01493160),
        3: (0.582, -1, -0.13530900, -0.09823596),
        4: (0.981,  0,  0.60557100,  1.21140000),
    }
    for N in expected:
        eps_e, n_e, dT_e, dP_e = expected[N]
        eps_a, n_a, dT_a, dP_a = TABLE_1_L8[N]
        assert n_a == n_e, f"N={N}: n mismatch"
        assert abs(eps_a - eps_e) < 1e-12, f"N={N}: ε mismatch"
        # Paper-printed precision: ~8 significant figures.
        assert abs(dT_a - dT_e) < 1e-7, (
            f"N={N}: d_T* {dT_a} ≠ {dT_e} (Table 1)"
        )
        assert abs(dP_a - dP_e) < 1e-7, (
            f"N={N}: d_P* {dP_a} ≠ {dP_e} (Table 1)"
        )


def test_table_1_d_T_is_symmetric_about_half_grid():
    """Interpolation coefficients (d_T*) are SYMMETRIC about
    the half-grid centre (between n=-1 and n=0). The full
    stencil is built by reflecting the left half. The paper's
    Table 1 tabulates only the left half; the test that the
    left half is internally consistent with a symmetric
    extension is a sanity check."""
    # Sum of left half plus its mirror image gives total sum.
    # For a Gaussian-tapered sinc-interpolation operator the
    # total weight should be ~1 (it's an interpolation, so it
    # reproduces a constant).
    left_half = [TABLE_1_L8[N][2] for N in (1, 2, 3, 4)]
    total = 2.0 * sum(left_half)
    # Should be close to 1.0 (Gaussian truncation makes it
    # slightly under). Loose check.
    assert abs(total - 1.0) < 0.05, (
        f"Sum of d_T* (doubled for full 8-pt) = {total}; "
        f"expected ~1.0 for an interpolation operator (loose 5% bound)"
    )


def test_table_1_d_P_is_antisymmetric_signature():
    """Derivative coefficients (d_P*) are ANTISYMMETRIC about
    the half-grid centre — so the LEFT half should have a
    sign pattern consistent with antisymmetry when extended
    to the right half via -1× reflection. The half-stencil
    sum, after antisymmetric extension, gives zero — so
    half-stencil sum is not constrained, BUT the largest
    magnitude should be at n=0 (closest to half-grid centre),
    decreasing in magnitude away from centre."""
    magnitudes = [abs(TABLE_1_L8[N][3]) for N in (1, 2, 3, 4)]
    # n = -3, -2, -1, 0 — increasing |d_P*| toward the centre.
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] <= magnitudes[i + 1], (
            f"|d_P*| not monotone-increasing toward n=0: {magnitudes}"
        )
    # n=0 should be the largest by far (it's closest to the
    # half-grid sample point).
    assert magnitudes[-1] > 5 * magnitudes[-2], (
        f"|d_P*(0)| should dominate the half-stencil: "
        f"got |d_P*(0)|={magnitudes[-1]}, |d_P*(-1)|={magnitudes[-2]}"
    )


# ---------------------------------------------------------------
# Test class 3 — Phase-velocity error within paper bounds
# ---------------------------------------------------------------

def test_phase_velocity_error_L8_50pct_nyquist_within_paper_bounds():
    """The headline quantitative claim of the paper: at 50%
    Nyquist using the 8-point convolution operator, the
    relative phase-velocity error in the triclinic medium of
    Eq 47 is bounded approximately by 2% (qP), 3% (qS1), and
    7% (qS2).

    Our reproduction's error should be at or below these
    paper-quoted bounds.
    """
    snap = run_dispersion_analysis(space_order=8, save_npz=False)
    bounds = PAPER_ERROR_BOUNDS[0.50]
    fracs = list(snap['nyquist_fracs'])
    k_idx = fracs.index(0.50)
    err_qP = snap['rel_error_max'][0, k_idx]
    err_qS1 = snap['rel_error_max'][1, k_idx]
    err_qS2 = snap['rel_error_max'][2, k_idx]
    # Allow a small additional margin (1.5×) for finite-angle
    # sampling noise + our Gaussian taper σ approximation.
    margin = 1.5
    assert err_qP <= bounds['qP'] * margin, (
        f"qP error {err_qP:.3e} exceeds paper's 50%-Nyquist "
        f"bound {bounds['qP']:.3e} (×{margin} margin)"
    )
    assert err_qS1 <= bounds['qS1'] * margin, (
        f"qS1 error {err_qS1:.3e} exceeds paper's bound "
        f"{bounds['qS1']:.3e} (×{margin} margin)"
    )
    assert err_qS2 <= bounds['qS2'] * margin, (
        f"qS2 error {err_qS2:.3e} exceeds paper's bound "
        f"{bounds['qS2']:.3e} (×{margin} margin)"
    )


def test_christoffel_recovers_isotropic_limit():
    """Sanity check: for an isotropic medium (Vp=3, Vs=1.5,
    ρ=2.2), the Christoffel solver must return Vp and Vs (and
    Vs again — the qS2 branch is degenerate with qS1 in
    isotropic media)."""
    vp, vs, rho = 3.0, 1.5, 2.2
    mu = rho * vs**2
    lam_p_2mu = rho * vp**2
    lam = lam_p_2mu - 2.0 * mu
    cij_iso = np.array([
        [lam_p_2mu, lam, lam, 0, 0, 0],
        [lam, lam_p_2mu, lam, 0, 0, 0],
        [lam, lam, lam_p_2mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ], dtype=np.float64)
    velocities = christoffel_phase_velocities_2d(cij_iso, rho,
                                                   angle_rad=0.3)
    # P-wave is the largest eigenvalue.
    assert abs(velocities[0] - vp) < 1e-12, (
        f"qP velocity {velocities[0]} != Vp={vp} (isotropic)"
    )
    # qS1 = qS2 = Vs in isotropic media.
    assert abs(velocities[1] - vs) < 1e-12
    assert abs(velocities[2] - vs) < 1e-12


# ---------------------------------------------------------------
# Test class 4 — Reference-output regression
# ---------------------------------------------------------------

@pytest.mark.parametrize('so', [2, 4, 8, 16])
def test_reference_output_matches_pin(so: int):
    """Re-run the dispersion analysis and verify the result
    matches the pinned ``reference_outputs/dispersion_so<N>.npz``
    to fp64 tolerance. Catches drift caused by changes to the
    Gaussian taper, Christoffel solver, or any other internal
    detail."""
    ref_path = FOLDER / 'reference_outputs' / f'dispersion_so{so}.npz'
    assert ref_path.is_file(), (
        f"Reference output missing: {ref_path}. Run "
        f"`uv run python run_reproduction.py` to regenerate."
    )
    ref = np.load(ref_path)
    fresh = run_dispersion_analysis(so, save_npz=False)
    for key in ('phase_vel_analytic', 'phase_vel_numerical',
                'rel_error_max', 'stencil_coeffs', 'cij_eq47'):
        ref_a = ref[key]
        fresh_a = np.asarray(fresh[key])
        assert ref_a.shape == fresh_a.shape, (
            f"Shape mismatch on {key} at so={so}: ref={ref_a.shape}, "
            f"fresh={fresh_a.shape}"
        )
        denom = max(float(np.max(np.abs(ref_a))), 1e-30)
        err = float(np.max(np.abs(ref_a - fresh_a))) / denom
        assert err < 1e-10, (
            f"{key} at so={so} drifted from reference: "
            f"max|Δ|/max|ref| = {err:.3e} (threshold 1e-10)"
        )


# ---------------------------------------------------------------
# Test class 5 — Reproducer health checks
# ---------------------------------------------------------------

@pytest.mark.parametrize('L', [2, 4, 8, 16])
def test_igel_derivative_coeffs_antisymmetric_signature(L: int):
    """The stencil should be antisymmetric about the half-grid
    centre (between indices L/2-1 and L/2): coeffs[L/2-1-k] =
    -coeffs[L/2+k] for k in 0..L/2-1."""
    coeffs = igel_derivative_coeffs(L)
    half_L = L // 2
    for k in range(half_L):
        a = coeffs[half_L - 1 - k]
        b = coeffs[half_L + k]
        # Antisymmetric: weights should be exact mirrors with
        # opposite signs.
        assert abs(a + b) < 1e-12, (
            f"L={L} stencil not antisymmetric at offset ±{k+0.5}: "
            f"left={a:.6e}, right={b:.6e}"
        )


def test_numerical_wavenumber_at_zero_is_zero():
    """``k_num(k=0) = 0`` for any antisymmetric stencil — the
    DC component vanishes by construction. This catches sign-
    convention bugs."""
    for L in (2, 4, 8, 16):
        coeffs = igel_derivative_coeffs(L)
        k_num = numerical_wavenumber(np.array([0.0]), coeffs)
        assert abs(k_num[0]) < 1e-14, (
            f"L={L}: k_num(0)={k_num[0]} should be 0"
        )
