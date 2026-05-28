"""Paper-anchored transcription of Vishnevsky-Lisitsa-Tcheverda-Reshetova 2014
quantitative claims.

Vishnevsky, D., Lisitsa, V., Tcheverda, V. & Reshetova, G. (2014).
"Numerical study of the interface errors of finite-difference simulations
of seismic waves." *Geophysics* 79(4):T219-T232.
DOI: 10.1190/geo2013-0299.1.

Per the project's `feedback_reproduction_quantitative_first` convention,
this module hand-transcribes the paper's quantitative claims so the
reproduction tests can byte-check the parent's interpretation against
the paper.

Anchors transcribed here:

1. **Test medium parameters** (paper §"Numerical experiments",
   page T223): IS1, IS2, IS3, IF, AS1, AS2, AS3.

2. **Source initial condition** (Eq 17): Gaussian on σ_xx, σ_zz.

3. **Convergence indicator** (Eqs 12-13): δ_k = ε_k/ε_{k-1} with
   asymptotic 2^γ scaling.

4. **The qualitative convergence-rate predictions** extracted from
   Figures 4-16 + §"Conclusion".

All transcriptions made from the PDF at
`vishnevsky_lisitsa_2014_paper.pdf` (Zotero key RNNG7GC5).

Transcription provenance
------------------------

The Cij matrices (Eqs 14, 15, 16) + the AS1/AS2/AS3 densities were
verified on 2026-05-28 via the review-pool vision-LLM protocol
(see `~/.claude/projects/-Users-ggorman-projects-reviews/memory/feedback_transcription_workflow_arxiv_then_side_by_side.md`).

Reviewers (both AGREE):
- Azure GPT-5 (vision multimodal) — 31s wall
- Ollama gemma4:31b (vision capability) — 29s wall

Combined verdict: AGREE on Q1 (Eq 14), Q2 (Eq 15), Q3 (Eq 16), Q4
(densities). Reviewer transcripts:
- /tmp/review_pool_1779979639_01_*_azure_gpt5_*.txt
- /tmp/review_pool_1779979639_02_*_ollama_*.txt

The IS1/IS2/IS3/IF parameters (textual, prose-only) were
transcribed directly from page T223 (Vp, Vs, ρ are plain-text
constants with unambiguous typography; no LLM verification
necessary). Same for SOURCE_IC (Eq 17) and the convergence
indicator predictions (extracted from Figure captions Fig 4-16
+ §"Conclusion").
"""

import numpy as np

# ----------------------------------------------------------------------------
# Paper §"Numerical experiments", page T223 — Test medium parameters
# ----------------------------------------------------------------------------

# Isotropic solid 1 (paper page T223 item 1)
IS1 = {
    'rho': 1800.0,  # kg/m^3
    'Vp': 1900.0,   # m/s
    'Vs': 1200.0,   # m/s
}

# Isotropic solid 2 (item 2)
IS2 = {
    'rho': 2200.0,
    'Vp': 2400.0,
    'Vs': 1400.0,
}

# Isotropic solid 3 (item 3)
IS3 = {
    'rho': 1600.0,
    'Vp': 1600.0,
    'Vs': 900.0,
}

# Ideal fluid (item 4)
IF = {
    'rho': 1000.0,
    'Vp': 1500.0,
    'Vs': 0.0,
}

# Anisotropic solid 1 — Eq 14 (page T223), ρ = 1800 kg/m^3
# C in units of 10^9 Pa (paper writes "× 10^9 kg × s / m" — a typographic
# error; the dimensionally-correct unit for stiffness is N/m² = Pa.
# Plausibility check: C11 of 3.6e9 Pa ≈ Vp²·ρ with Vp ≈ 1414, ρ = 1800
# → Vp² ≈ C11/ρ = 2e6, Vp ≈ 1414, which is in the geophysical range).
AS1 = {
    'rho': 1800.0,
    'C': np.array([
        [3.6, 1.8, -0.9],
        [1.8, 3.24, 0.0],
        [-0.9, 0.0, 2.7],
    ]) * 1e9,  # Pa
}

# Anisotropic solid 2 — Eq 15, ρ = 2200 kg/m^3
AS2 = {
    'rho': 2200.0,
    'C': np.array([
        [4.4, 2.2, 2.2],
        [2.2, 6.6, 2.2],
        [2.2, 2.2, 4.4],
    ]) * 1e9,
}

# Anisotropic solid 3 — Eq 16, ρ = 1600 kg/m^3
AS3 = {
    'rho': 1600.0,
    'C': np.array([
        [1.6, 0.8, 0.8],
        [0.8, 2.4, 0.8],
        [0.8, 0.8, 1.6],
    ]) * 1e9,
}

# ----------------------------------------------------------------------------
# Eq 17 — Source initial condition (page T223)
#
#   u_x = 0; u_z = 0;
#   σ_xz = 0;
#   σ_xx = σ_zz = exp{-0.1·((x - x_s)² + (z - z_s)²)}
#
# Centred at x_s = 1500 m, z_s = 750 m on a 3000 × 3000 m domain.
# (Paper §"Numerical experiments", page T223.)
# ----------------------------------------------------------------------------

SOURCE_IC = {
    'x_s': 1500.0,   # m
    'z_s': 750.0,    # m
    'sigma_scale': 0.1,  # exponent prefactor in exp{-0.1·r²}
    'domain_x': 3000.0,  # m
    'domain_z': 3000.0,  # m
}


def source_ic(X, Z):
    """Return σ_xx = σ_zz at (X, Z) coordinates per Eq 17.

    X, Z are arrays of coordinates in meters.
    """
    return np.exp(-SOURCE_IC['sigma_scale'] *
                  ((X - SOURCE_IC['x_s']) ** 2 +
                   (Z - SOURCE_IC['z_s']) ** 2))


# ----------------------------------------------------------------------------
# Eqs 12-13 — Convergence indicator
#
#   δ_k = ε_k / ε_{k-1}                                       (Eq 12)
#   δ_k ≈ (h_k / h_{k+1})^γ                                   (Eq 13)
#
# where γ is the convergence rate. For grid refinement h_k → h_k/2:
#   δ_k ≈ 2^γ.
#
# For 2nd-order convergence (γ = 2): δ_k → 4
# For 1st-order convergence (γ = 1): δ_k → 2
# ----------------------------------------------------------------------------

CONVERGENCE_THRESHOLDS = {
    'second_order_delta': 4.0,  # δ_k → 4 for γ = 2
    'first_order_delta': 2.0,   # δ_k → 2 for γ = 1
    # Acceptance tolerances for empirical δ_k in the reproduction's
    # h-refinement study (chosen to allow some variation across
    # experiment number per Figures 4-16):
    'second_order_band': (3.5, 4.5),  # γ ≈ 2
    'first_order_band': (1.5, 2.5),   # γ ≈ 1
}


# ----------------------------------------------------------------------------
# Qualitative convergence-rate table (extracted from Figures 4-16 +
# §"Discussion" + §"Conclusion", pages T223-T231).
#
# Each entry is the expected δ_k value (asymptotic).
#
# Schemes:
#   SSGS = Standard Staggered-Grid Scheme (Virieux 1986)
#   RSGS = Rotated Staggered-Grid Scheme (Saenger 2000)
#   LS   = Lebedev Scheme (Lebedev 1964; Davydycheva 2003)
# Modifier "w" prefix = "without parameter modification" (no balance
# technique / finely-layered averaging applied at the interface).
# ----------------------------------------------------------------------------

CONVERGENCE_PREDICTIONS = {
    # Horizontal solid-solid contacts — modified
    'horizontal_solid_solid_iso_modified': {
        'SSGS': 4.0, 'RSGS': 4.0, 'LS': 4.0,
        'source': 'Figure 4 (page T223)',
    },
    'horizontal_solid_solid_aniso_modified': {
        'RSGS': 4.0, 'LS': 4.0,
        'source': 'Figure 6 (page T224)',
    },

    # Horizontal fluid-solid contacts — THE LOAD-BEARING ANCHOR for
    # the Petrobras water-TTI cohort ranking.
    'horizontal_fluid_isotropic_solid': {
        'SSGS': 4.0,   # 2nd order — only scheme that preserves it
        'RSGS': 2.0,   # 1st order — degrades at fluid-solid
        'LS': 2.0,     # 1st order — degrades at fluid-solid
        'source': 'Figure 5 (page T224)',
        'paper_quote': (
            'The existence of a fluid-solid interface reduces the '
            'convergence for the LS and the RSGS to a first order '
            'of convergence, whereas the SSGS remains a second '
            'order of convergence.'),
    },
    'horizontal_fluid_anisotropic_solid': {
        'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 7 (page T224)',
    },

    # Inclined interfaces — staircase approximation forces 1st order
    # for ALL schemes regardless of interface class.
    'inclined_solid_solid_iso': {
        'SSGS': 2.0, 'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 8 (page T224)',
        'paper_quote': (
            'A staircase approximation of the inclined interface '
            'was used, which resulted in the interface being '
            'approximated only with the first order. Therefore, '
            'the wavefield could not converge faster than the '
            'model, which was confirmed by the numerical '
            'experiments.'),
    },
    'inclined_fluid_isotropic_solid': {
        'SSGS': 2.0, 'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 9 (page T225)',
    },
    'inclined_solid_solid_aniso': {
        'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 10 (page T225)',
    },
    'inclined_fluid_anisotropic_solid': {
        'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 11 (page T225)',
    },

    # Corner problems (3 adjacent media meeting at a vertex)
    'corner_three_isotropic_solids': {
        'SSGS': 4.0, 'RSGS': 4.0, 'LS': 4.0,
        'source': 'Figure 13 (page T226)',
    },
    'corner_fluid_two_isotropic_solids': {
        'SSGS': 4.0,    # 2nd order preserved (corner case)
        'RSGS': 2.0,    # 1st order — fluid-solid still degrades
        'LS': 2.0,
        'source': 'Figure 14 (page T226)',
    },
    'corner_three_anisotropic_solids': {
        'RSGS': 4.0, 'LS': 4.0,
        'source': 'Figure 15 (page T226)',
    },
    'corner_fluid_two_anisotropic_solids': {
        'RSGS': 2.0, 'LS': 2.0,
        'source': 'Figure 16 (page T226)',
    },

    # Without parameter modification (baseline — no balance technique)
    'unmodified_any_interface': {
        'SSGS_w': '< 2',
        'LS_w': '< 2',
        'source': 'Figures 4-9 dotted curves (wSSGS, wLS)',
        'paper_quote': (
            "For either the isotropic or anisotropic solid-solid "
            "contact, all three schemes possess a second-order "
            "convergence if the density is modified by the "
            "arithmetic averaging of the adjoining grid cells and "
            "the components of a stiffness tensor are constructed "
            "by the formulas for averaging the finely layered "
            "media. For the fluid-solid interface, only the SSGS "
            "attains a second-order convergence with c_33 = 0 at "
            "the interface, whereas the LS and RSGS produce a "
            "first-order convergence."),
    },
}


# ----------------------------------------------------------------------------
# Plausibility check — paper quotes the abstract claim verbatim
# ----------------------------------------------------------------------------

ABSTRACT_LOAD_BEARING_CLAIM = (
    "We determined that a standard staggered-grid scheme (SSGS) "
    "(also known as the Virieux scheme), a rotated staggered-grid "
    "scheme (RSGS), and a Lebedev scheme (LS) preserve the second "
    "order of convergence at horizontal/vertical solid-solid "
    "interfaces when the medium parameters have been properly "
    "modified, such as by harmonic averaging of finely layered "
    "media for the stiffness tensor and arithmetic mean for the "
    "density. However, for a fluid-solid interface aligned with "
    "the grid line, a second-order convergence can only be "
    "achieved by an SSGS. In addition, the presence of a "
    "fluid-solid interface reduces the order of convergence for "
    "the LS and the RSGS to a first order of convergence. The "
    "presence of inclined interfaces makes high-order (second "
    "and more) convergence impossible."
)


if __name__ == '__main__':
    # Quick sanity check — print key constants
    print('Test media:')
    for name, m in [('IS1', IS1), ('IS2', IS2), ('IS3', IS3), ('IF', IF)]:
        print(f"  {name}: ρ={m['rho']} Vp={m['Vp']} Vs={m['Vs']}")
    print()
    print('Anisotropic stiffness tensors:')
    for name, m in [('AS1', AS1), ('AS2', AS2), ('AS3', AS3)]:
        print(f"  {name}: ρ={m['rho']}")
        print(f"    C = {m['C'] / 1e9} × 10^9 Pa")
    print()
    print(f"Convergence indicator predictions: {len(CONVERGENCE_PREDICTIONS)} configs")
    print(f"Petrobras-relevant: horizontal_fluid_isotropic_solid")
    print(f"  SSGS → δ ≈ {CONVERGENCE_PREDICTIONS['horizontal_fluid_isotropic_solid']['SSGS']}")
    print(f"  RSGS → δ ≈ {CONVERGENCE_PREDICTIONS['horizontal_fluid_isotropic_solid']['RSGS']}")
    print(f"  LS   → δ ≈ {CONVERGENCE_PREDICTIONS['horizontal_fluid_isotropic_solid']['LS']}")
