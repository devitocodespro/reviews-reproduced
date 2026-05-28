"""L1-integral dispersion regression test.

Per Phase X deepening (2026-05-28): single-point dispersion tests
at a fixed kh are sensitive to small kh choices near the
crossover region. The L1 integral of |modified_wavenumber_error|
over a kh band is a much more robust quantitative gate.

This test pins the L1 integrals produced by
`run_dispersion_analysis.py` and asserts (a) Taylor < STO in the
low-kh band (defining LS-tradeoff), and (b) STO < Taylor in the
high-kh band (Xie & He's central claim — sharp version).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

sys.path.insert(0, str(REPO_ROOT / "00_common"))
sys.path.insert(0, str(ROOT))

L1_PIN_PATH = ROOT / "reference_outputs" / "dispersion_l1_errors.json"


@pytest.fixture(scope="module")
def l1_pinned():
    """Load pinned L1 integrals; regenerate via
    `python run_dispersion_analysis.py` if missing."""
    if not L1_PIN_PATH.exists():
        pytest.skip(
            f"{L1_PIN_PATH.name} not found — run "
            f"`python run_dispersion_analysis.py` to generate."
        )
    raw = json.loads(L1_PIN_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@pytest.mark.parametrize("P", [4, 5, 6])
def test_high_band_sto_l1_dominates_taylor(P, l1_pinned):
    """Xie & He's central quantitative claim: in the high-kh
    band [1.2, 2.0], the L1 integral of STO's dispersion error
    is materially SMALLER than Taylor's at the same P."""
    entry = l1_pinned[f"P{P}"]["high_band"]
    sto, taylor = entry["sto_l1"], entry["taylor_l1"]
    ratio = sto / taylor
    assert ratio < 0.6, (
        f"P={P}: high-band STO_L1/Taylor_L1 = {ratio:.3f} "
        f"(STO={sto:.4e}, Taylor={taylor:.4e}). Expected < 0.6 "
        f"per Xie & He §4 dispersion analysis. If this fails, "
        f"the LS-optimization objective may have drifted."
    )


@pytest.mark.parametrize("P", [4, 5, 6])
def test_low_band_taylor_l1_dominates_sto(P, l1_pinned):
    """Defining-feature sentinel: in the low-kh band [0.1, 0.7],
    Taylor IS more accurate than STO. STO sacrifices low-kh
    accuracy to gain high-kh accuracy — this is the algorithm.
    If STO matches Taylor at low kh, the LS objective may have
    collapsed to the Taylor solution."""
    entry = l1_pinned[f"P{P}"]["low_band"]
    sto, taylor = entry["sto_l1"], entry["taylor_l1"]
    ratio = sto / taylor
    assert ratio > 10.0, (
        f"P={P}: low-band STO_L1/Taylor_L1 = {ratio:.3f} "
        f"(STO={sto:.4e}, Taylor={taylor:.4e}). Expected > 10 — "
        f"if STO is matching Taylor at low kh, the LS "
        f"optimization isn't trading low-kh for high-kh accuracy."
    )


@pytest.mark.parametrize("P", [4, 5, 6])
def test_pinned_l1_values_stable(P, l1_pinned):
    """Lock the pinned L1 integrals to the regenerated values from
    `run_dispersion_analysis.py`. Catches drift in the SciPy
    SLSQP optimizer or the LS objective definition."""
    # These were regenerated 2026-05-28 on scipy 1.17.1. The
    # tolerance is 5% relative — wider than fp64 because the
    # optimizer can land in slightly different local minima
    # under scipy upgrades (see also _provenance note in
    # sto_coefficients_pinned.json).
    expected = {
        4: {"low": (3.9512e-06, 1.8543e-04),
            "high": (6.4301e-02, 1.5480e-02)},
        5: {"low": (3.5248e-07, 2.7427e-05),
            "high": (3.6223e-02, 6.7931e-03)},
        6: {"low": (3.2809e-08, 4.4725e-05),
            "high": (2.1037e-02, 1.0533e-02)},
    }[P]
    rtol = 0.05
    for band, (exp_taylor, exp_sto) in expected.items():
        band_key = "low_band" if band == "low" else "high_band"
        got_taylor = l1_pinned[f"P{P}"][band_key]["taylor_l1"]
        got_sto = l1_pinned[f"P{P}"][band_key]["sto_l1"]
        assert abs(got_taylor - exp_taylor) / exp_taylor < rtol, (
            f"P={P} {band}-band taylor L1 drift: "
            f"expected {exp_taylor:.4e}, got {got_taylor:.4e}"
        )
        assert abs(got_sto - exp_sto) / exp_sto < rtol, (
            f"P={P} {band}-band sto L1 drift: "
            f"expected {exp_sto:.4e}, got {got_sto:.4e}"
        )
