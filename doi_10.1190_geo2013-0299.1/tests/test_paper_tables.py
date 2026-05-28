"""Byte-match the transcribed constants in paper_tables.py against the
load-bearing claims of Vishnevsky-Lisitsa-Tcheverda-Reshetova 2014.

These are regression tests that catch any future edit drift in the
transcribed constants.

Provenance: the Cij matrices (Eqs 14, 15, 16) and the AS1/AS2/AS3
densities were verified on 2026-05-28 via the review-pool vision-LLM
protocol (Azure GPT-5 + Ollama gemma4:31b, both AGREE). The IS1/IS2/IS3/IF
parameters and SOURCE_IC were transcribed directly from page T223 (plain
prose, unambiguous typography).
"""
from __future__ import annotations

import pytest
import numpy as np

import paper_tables as pt


# ----------------------------------------------------------------------------
# Test medium parameters (paper page T223 §"Numerical experiments")
# ----------------------------------------------------------------------------


def test_IS1_constants():
    """Item 1: isotropic solid 1, ρ=1800, Vp=1900, Vs=1200."""
    assert pt.IS1 == {'rho': 1800.0, 'Vp': 1900.0, 'Vs': 1200.0}


def test_IS2_constants():
    """Item 2: ρ=2200, Vp=2400, Vs=1400."""
    assert pt.IS2 == {'rho': 2200.0, 'Vp': 2400.0, 'Vs': 1400.0}


def test_IS3_constants():
    """Item 3: ρ=1600, Vp=1600, Vs=900."""
    assert pt.IS3 == {'rho': 1600.0, 'Vp': 1600.0, 'Vs': 900.0}


def test_IF_constants():
    """Item 4: ideal fluid, ρ=1000, Vp=1500, Vs=0."""
    assert pt.IF == {'rho': 1000.0, 'Vp': 1500.0, 'Vs': 0.0}


# ----------------------------------------------------------------------------
# Cij matrices Eqs 14, 15, 16 — verified by review-pool vision-LLM
# protocol 2026-05-28
# ----------------------------------------------------------------------------


def test_AS1_Cij_matches_Eq14():
    """Eq 14: 3×3 symmetric, ρ=1800 kg/m³.

    [[ 3.6,  1.8, -0.9],
     [ 1.8, 3.24,  0.0],
     [-0.9,  0.0,  2.7]] × 10⁹
    """
    assert pt.AS1['rho'] == 1800.0
    expected = np.array([
        [3.6, 1.8, -0.9],
        [1.8, 3.24, 0.0],
        [-0.9, 0.0, 2.7],
    ]) * 1e9
    np.testing.assert_array_equal(pt.AS1['C'], expected)
    # Symmetry
    np.testing.assert_array_equal(pt.AS1['C'], pt.AS1['C'].T)


def test_AS2_Cij_matches_Eq15():
    """Eq 15: 3×3 symmetric, ρ=2200 kg/m³.

    [[4.4, 2.2, 2.2],
     [2.2, 6.6, 2.2],
     [2.2, 2.2, 4.4]] × 10⁹
    """
    assert pt.AS2['rho'] == 2200.0
    expected = np.array([
        [4.4, 2.2, 2.2],
        [2.2, 6.6, 2.2],
        [2.2, 2.2, 4.4],
    ]) * 1e9
    np.testing.assert_array_equal(pt.AS2['C'], expected)
    np.testing.assert_array_equal(pt.AS2['C'], pt.AS2['C'].T)


def test_AS3_Cij_matches_Eq16():
    """Eq 16: 3×3 symmetric, ρ=1600 kg/m³.

    [[1.6, 0.8, 0.8],
     [0.8, 2.4, 0.8],
     [0.8, 0.8, 1.6]] × 10⁹
    """
    assert pt.AS3['rho'] == 1600.0
    expected = np.array([
        [1.6, 0.8, 0.8],
        [0.8, 2.4, 0.8],
        [0.8, 0.8, 1.6],
    ]) * 1e9
    np.testing.assert_array_equal(pt.AS3['C'], expected)
    np.testing.assert_array_equal(pt.AS3['C'], pt.AS3['C'].T)


# ----------------------------------------------------------------------------
# Source IC (Eq 17)
# ----------------------------------------------------------------------------


def test_SOURCE_IC_constants():
    """Eq 17 source IC: Gaussian at (1500, 750) on 3000×3000 domain."""
    assert pt.SOURCE_IC == {
        'x_s': 1500.0,
        'z_s': 750.0,
        'sigma_scale': 0.1,
        'domain_x': 3000.0,
        'domain_z': 3000.0,
    }


def test_source_ic_peaks_at_xs_zs():
    """source_ic(xs, zs) should equal 1.0 (Gaussian peak)."""
    X, Z = np.meshgrid([pt.SOURCE_IC['x_s']], [pt.SOURCE_IC['z_s']])
    val = pt.source_ic(X, Z)
    assert val.item() == 1.0


# ----------------------------------------------------------------------------
# Convergence indicator thresholds (paper Eqs 12-13)
# ----------------------------------------------------------------------------


def test_convergence_thresholds():
    """δ_k → 4 for 2nd order, → 2 for 1st order (h/2 refinement)."""
    assert pt.CONVERGENCE_THRESHOLDS['second_order_delta'] == 4.0
    assert pt.CONVERGENCE_THRESHOLDS['first_order_delta'] == 2.0


# ----------------------------------------------------------------------------
# Load-bearing prediction: fluid-solid degrades RSGS/LS to 1st order
# ----------------------------------------------------------------------------


def test_petrobras_relevant_prediction_horizontal_fluid_isotropic_solid():
    """The load-bearing anchor for the Petrobras cohort ranking.

    Per paper Figure 5 + §"Conclusion":
    - SSGS preserves 2nd-order at horizontal fluid-isotropic-solid
    - RSGS and LS degrade to 1st-order
    """
    pred = pt.CONVERGENCE_PREDICTIONS['horizontal_fluid_isotropic_solid']
    assert pred['SSGS'] == 4.0    # 2nd order
    assert pred['RSGS'] == 2.0    # 1st order
    assert pred['LS'] == 2.0      # 1st order
    # The paper quote MUST be present for documentation
    assert 'fluid-solid' in pred['paper_quote'].lower()


def test_horizontal_solid_solid_all_2nd_order():
    """Eq pred: all schemes preserve 2nd-order at solid-solid (Figure 4)."""
    pred = pt.CONVERGENCE_PREDICTIONS['horizontal_solid_solid_iso_modified']
    assert pred['SSGS'] == 4.0
    assert pred['RSGS'] == 4.0
    assert pred['LS'] == 4.0


def test_inclined_solid_solid_iso_degrades_to_1st_order():
    """Per §"Conclusion" + Figure 8: all schemes degrade to 1st at
    inclined SOLID-SOLID interfaces (staircase approximation)."""
    pred = pt.CONVERGENCE_PREDICTIONS['inclined_solid_solid_iso']
    assert pred['SSGS'] == 2.0
    assert pred['RSGS'] == 2.0
    assert pred['LS'] == 2.0


def test_inclined_fluid_solid_below_first_order_per_paper():
    """Per page T225: at inclined fluid/solid interfaces, the
    paper explicitly states first-order convergence was NOT
    observed. This is BELOW the project's ≈1st-order indicator
    threshold (δ=2.0), so the corresponding scheme cells must
    encode `None` and carry the verbatim paper_quote.

    Field-test 2026-05-28 caught the prior numeric (2.0)
    encoding as a Rule 1 silent-strengthening violation; this
    test locks in the corrected encoding (Phase Y/1.5a)."""
    verbatim = (
        "did not even observe a convergence of the first order "
        "for the fluid/solid interface"
    )
    for key in (
        'inclined_fluid_isotropic_solid',
        'inclined_fluid_anisotropic_solid',
    ):
        pred = pt.CONVERGENCE_PREDICTIONS[key]
        for scheme in pred:
            if scheme in {'SSGS', 'RSGS', 'LS'}:
                assert pred[scheme] is None, (
                    f"{key}.{scheme} = {pred[scheme]!r} but paper "
                    f"says < 1st-order; must encode as None to avoid "
                    f"silent strengthening to ≈1st-order indicator."
                )
        assert 'paper_quote' in pred, (
            f"{key} missing required paper_quote field — verbatim "
            f"paper language is the load-bearing artifact for a "
            f"None-encoded cell."
        )
        assert verbatim in pred['paper_quote'], (
            f"{key}.paper_quote missing verbatim claim from page "
            f"T225: {verbatim!r}"
        )


def test_abstract_load_bearing_claim_present():
    """The paper's abstract quotes must be preserved in the module so
    downstream readers can verify the framing."""
    assert 'SSGS' in pt.ABSTRACT_LOAD_BEARING_CLAIM
    assert 'RSGS' in pt.ABSTRACT_LOAD_BEARING_CLAIM
    assert 'fluid-solid' in pt.ABSTRACT_LOAD_BEARING_CLAIM
    assert 'first order' in pt.ABSTRACT_LOAD_BEARING_CLAIM.lower()
