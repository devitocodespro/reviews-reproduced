"""Byte-match transcribed paper anchors against parent implementation.

Verifies that the parent repo's ``00_common/spacetime_coefficients.py``
(used by Method 61 STO-RSG) reproduces the Xie & He 2024 algorithm
+ pinned canonical coefficient values + dispersion-reduction
property.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent  # reviews/

# Make both the parent's 00_common and the reproduction's paper_tables
# importable.
sys.path.insert(0, str(REPO_ROOT / "00_common"))
sys.path.insert(0, str(ROOT))

import paper_tables as pt  # noqa: E402
from spacetime_coefficients import (  # noqa: E402
    compute_sto_coefficients,
    sto_modified_wavenumber,
)
from drp_coefficients import centered_fd_coefficients  # noqa: E402


# ─── (1) Consistency constraint (Xie & He Eq 15) ────────────────────


@pytest.mark.parametrize("P,cfl", [
    (6, 0.3), (6, 0.4), (6, 0.5),
    (5, 0.4),
    (4, 0.4),
])
def test_te_ls_consistency_constraint(P, cfl):
    """Per Xie & He §3 Eq (15): ``2·Σ_m m·c_m = 1`` for the
    antisymmetric centered FD coefficients. The SLSQP optimizer
    enforces this as an equality constraint; verify it holds to
    fp64."""
    c = compute_sto_coefficients(P, cfl)
    pos = {m: c[m] for m in c if m > 0}
    residual = pt.consistency_residual(pos)
    assert abs(residual) < 1e-10, (
        f"Consistency residual {residual:.3e} for P={P} cfl={cfl} — "
        f"the TE constraint Σ_m m·c_m = 0.5 is not satisfied. "
        f"SLSQP optimization may have failed to converge."
    )


# ─── (2) Pinned canonical-coefficient reproducibility ────────────────


PINNED = {
    k: v
    for k, v in json.loads(
        (ROOT / "reference_outputs" / "sto_coefficients_pinned.json").read_text()
    ).items()
    if not k.startswith("_")  # skip _provenance metadata
}


@pytest.mark.parametrize("key,expected", PINNED.items())
def test_pinned_sto_coefficients(key, expected):
    """Re-running ``compute_sto_coefficients`` must reproduce the
    pinned canonical values within fp64 tolerance. This locks the
    parent's STO implementation against drift (e.g., a future SciPy
    SLSQP regression or an inadvertent edit to the optimization
    objective)."""
    # Parse "P6_cfl0.4" → (6, 0.4)
    P_str, cfl_str = key.split("_cfl")
    P = int(P_str.lstrip("P"))
    cfl = float(cfl_str)
    c = compute_sto_coefficients(P, cfl)
    for m_str, c_expected in expected.items():
        m = int(m_str)
        c_actual = c[m]
        assert abs(c_actual - c_expected) < 1e-9, (
            f"Pinned coefficient drift at P={P} cfl={cfl} m={m}: "
            f"expected {c_expected!r}, got {c_actual!r}. Either "
            f"the optimizer converged differently (check scipy "
            f"version + LS objective) or this is a deliberate "
            f"change requiring a pin update."
        )


# ─── (3) Dispersion-error reduction vs Taylor (qualitative claim) ────


def _taylor_modified_wavenumber(taylor: dict[int, float], P: int,
                                 kh: float) -> float:
    """Helper: same modified-wavenumber form as sto_modified_wavenumber.

    For antisymmetric centered FD with positive coefficients c_m at
    offsets m=1..P:  k_h ≈ 2·Σ c_m·sin(m·kh).
    """
    return 2 * sum(taylor[m] * np.sin(m * kh) for m in range(1, P + 1))


@pytest.mark.parametrize("P", [4, 5, 6])
def test_sto_dispersion_lower_than_taylor_high_kh_band(P):
    """Xie & He's central qualitative claim, anchored in the band
    where it actually holds.

    Per Phase X deepening sweep (2026-05-28): the LS-optimized
    stencil trades low-kh accuracy for high-kh accuracy. The
    claim "STO < Taylor" holds strictly in the high-kh band
    [1.2, 2.0] (away from Nyquist, away from the low-kh
    crossover region). The paper's §4 dispersion analysis
    presents the comparison at wavenumbers approaching Nyquist,
    which sits inside this band."""
    cfl = 0.4  # canonical
    sto = compute_sto_coefficients(P, cfl)
    taylor = centered_fd_coefficients(2 * P)

    lo, hi = pt.XIE_HE_2024_DISPERSION_HIGH_KH_BAND
    kh_grid = np.linspace(lo, hi, 9)
    for kh in kh_grid:
        sto_kh = sto_modified_wavenumber(sto, kh)
        taylor_kh = _taylor_modified_wavenumber(taylor, P, kh)
        sto_err = abs(sto_kh - kh)
        taylor_err = abs(taylor_kh - kh)
        assert sto_err < taylor_err, (
            f"STO dispersion error {sto_err:.4e} >= Taylor "
            f"{taylor_err:.4e} at P={P} kh={kh:.3f} (in the "
            f"high-kh band {pt.XIE_HE_2024_DISPERSION_HIGH_KH_BAND} "
            f"where Xie & He's claim should hold). Check the LS "
            f"optimization objective."
        )


@pytest.mark.parametrize("P", [4, 5, 6])
def test_sto_low_kh_tradeoff_vs_taylor(P):
    """Sentinel-style test: documents the defining feature of the
    LS-optimized stencil — at LOW kh, Taylor is more accurate
    than STO, because Taylor has more polynomial-order-equivalent
    terms matched while STO spreads its budget over the
    high-kh band.

    This is NOT a bug. The tradeoff IS the algorithm. The test
    fails if STO is suspiciously accurate at low kh — that would
    indicate the LS objective is mis-weighted or the optimizer
    is collapsing to the Taylor solution."""
    cfl = 0.4
    sto = compute_sto_coefficients(P, cfl)
    taylor = centered_fd_coefficients(2 * P)

    lo, hi = pt.XIE_HE_2024_DISPERSION_LOW_KH_BAND
    # Sample a few points; require Taylor < STO at least most of them
    # (allow a single near-tie point in case the optimizer's low-kh
    # tail oscillates slightly).
    kh_grid = np.linspace(lo, hi, 7)
    taylor_wins = 0
    for kh in kh_grid:
        sto_kh = sto_modified_wavenumber(sto, kh)
        taylor_kh = _taylor_modified_wavenumber(taylor, P, kh)
        sto_err = abs(sto_kh - kh)
        taylor_err = abs(taylor_kh - kh)
        if taylor_err < sto_err:
            taylor_wins += 1
    # Strong assertion: Taylor MUST win the majority of low-kh
    # sample points. If STO matches or beats Taylor at low kh,
    # the LS optimization isn't doing its job.
    assert taylor_wins >= 5, (
        f"P={P}: Taylor wins only {taylor_wins}/7 low-kh points — "
        f"expected ≥ 5. The LS-optimized stencil should be WORSE "
        f"than Taylor at low kh (this is the defining tradeoff of "
        f"the algorithm)."
    )


# ─── (4) RSG topology + interpolation-error claim ────────────────────


def test_xie_he_2024_solves_interpolation_error():
    """Sentinel constant — Xie & He §3 explicitly states that
    their TE+LS RSG addresses the spatial-interpolation error of
    conventional staggered-grid (SSG-Yee) when applied to TTI
    elastic. This sentinel locks the claim in our reproduction."""
    assert pt.XIE_HE_2024_SOLVES_TTI_INTERPOLATION_ERROR is True
    assert pt.XIE_HE_2024_PRIMARY_P == 6, (
        "Paper's primary configuration is P=6 (12th order)")


def test_xie_he_2024_method_name_anchor():
    """Lock the algorithm-identity sentinel."""
    assert pt.XIE_HE_2024_METHOD_NAME == "TE+LS RSG"
    assert "Taylor-Expansion + Least-Squares" in \
        pt.XIE_HE_2024_FORMULATION
    assert "Saenger 2000" in pt.XIE_HE_2024_FORMULATION
