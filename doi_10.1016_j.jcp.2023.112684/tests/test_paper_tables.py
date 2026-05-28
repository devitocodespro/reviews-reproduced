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


PINNED = json.loads(
    (ROOT / "reference_outputs" / "sto_coefficients_pinned.json").read_text()
)


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


@pytest.mark.parametrize("P", [4, 5, 6])
def test_sto_dispersion_lower_than_taylor(P):
    """Xie & He's central qualitative claim: STO coefficients
    produce LOWER dispersion error than Taylor (Fornberg) at the
    same P, particularly in the high-kh band.

    Verify at a representative kh = 1.0 rad (paper §4
    demonstration range)."""
    cfl = 0.4  # canonical
    sto = compute_sto_coefficients(P, cfl)
    taylor = centered_fd_coefficients(2 * P)

    kh = pt.XIE_HE_2024_DISPERSION_REDUCTION_KH

    sto_kh = sto_modified_wavenumber(sto, kh)
    # Compute Taylor modified wavenumber in the same form
    taylor_kh = 2 * sum(
        taylor[m] * np.sin(m * kh)
        for m in range(1, P + 1)
    )

    sto_err = abs(sto_kh - kh)
    taylor_err = abs(taylor_kh - kh)
    assert sto_err < taylor_err, (
        f"STO dispersion error {sto_err:.4e} >= Taylor "
        f"{taylor_err:.4e} at P={P} kh={kh}. Xie & He's "
        f"qualitative claim (STO < Taylor) doesn't hold — "
        f"check the LS optimization objective."
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
