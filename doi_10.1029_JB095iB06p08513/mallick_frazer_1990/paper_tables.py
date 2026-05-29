"""Hand-transcribed paper anchors for Mallick & Frazer (1990).

Primary paper:
    Mallick, S., & Frazer, L. N. (1990). "Computation of synthetic
    seismograms for stratified azimuthally anisotropic media."
    *Journal of Geophysical Research* 95(B6): 8513-8526.
    DOI: 10.1029/JB095iB06p08513.

The anchors here are the load-bearing structural / algebraic
properties of the Mallick-Frazer 1990 acoustic-elastic interface
R/T machinery — not floating-point coefficient tables (the solver
operates on arbitrary continuous (p_x, ω, Cij, ρ) inputs).
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────
# Anchor 1: 3×3 acoustic-elastic continuity system
# ──────────────────────────────────────────────────────────────────


# At a planar acoustic-elastic interface, three continuity
# conditions yield a 3×3 linear system per (p_x, ω):
#   (i)   continuity of normal displacement u_z;
#   (ii)  continuity of normal traction σ_zz = -p_acoustic;
#   (iii) vanishing tangential traction σ_xz = 0 on the elastic side.
# The 3 unknowns are R (acoustic reflection), T_qP (transmitted
# qP), T_qSV (transmitted qSV).
#
# Locking these flags catches silent reductions (e.g., to a 2×2
# acoustic-acoustic system) or expansions (e.g., to a 4×4
# elastic-elastic system).
MALLICK_FRAZER_1990_INTERFACE_SYSTEM_DIM = 3
MALLICK_FRAZER_1990_UNKNOWNS = ("R", "T_qP", "T_qSV")
MALLICK_FRAZER_1990_BCS = (
    "continuity of u_z",
    "continuity of sigma_zz",
    "vanishing sigma_xz on elastic side",
)


# ──────────────────────────────────────────────────────────────────
# Anchor 2: Christoffel quartic structure
# ──────────────────────────────────────────────────────────────────


# At horizontal slowness p_x, the Christoffel equation
# (det(Γ - ρ ω² I) = 0 with Γ_ij = C_ikjl p_k p_l) reduces to a
# **quartic polynomial in q_z**. Two roots correspond to the
# downgoing qP and qSV modes; the other two are upgoing. The
# quartic structure is load-bearing — silently solving a cubic or
# quintic would index the wrong wave modes.
MALLICK_FRAZER_1990_CHRISTOFFEL_DEGREE_IN_QZ = 4


# Number of distinct downgoing modes selected per (p_x, ω):
# exactly 2 (qP + qSV in the 2D SV plane).
MALLICK_FRAZER_1990_DOWNGOING_MODE_COUNT = 2


# Downgoing-mode selection rule: the imaginary part of q_z must be
# ≤ 0 (decaying as z increases for evanescent modes; positive-real
# for propagating modes with energy flowing in +z). Conv A
# convention matches the rest of the parent DWN code; flipping the
# sign would invert energy flow direction.
MALLICK_FRAZER_1990_DOWNGOING_QZ_SELECTION_RULE = "Im(q_z) <= 0"


# ──────────────────────────────────────────────────────────────────
# Anchor 3: Snell-slowness conservation
# ──────────────────────────────────────────────────────────────────


# Horizontal slowness p_x is CONTINUOUS across the interface
# (Snell's law generalisation for anisotropy). The acoustic upper
# wave has q_w² = 1/V_U² - p_x²; the elastic lower modes have q_z
# from the Christoffel quartic at the same p_x.
MALLICK_FRAZER_1990_SNELL_CONSERVATION = "p_x_upper = p_x_lower"


# Acoustic upper-layer vertical slowness:
#   q_w² = 1/V_U² - p_x²,   Im(q_w) ≤ 0
# Locking the formula sentinel.
MALLICK_FRAZER_1990_QW_FORMULA = "q_w^2 = 1/V_U^2 - p_x^2"


# ──────────────────────────────────────────────────────────────────
# Anchor 4: Normal-incidence reduction (p_x = 0)
# ──────────────────────────────────────────────────────────────────


# At p_x = 0 (normal incidence), the qSV mode decouples (T_qSV = 0
# because the polarisation has no z-component) and R reduces to the
# acoustic impedance formula
#   R(p_x=0) = (Z_TTI_vertical - Z_water) / (Z_TTI_vertical + Z_water)
# where Z_TTI_vertical = sqrt(rho_L * C_33) (the vertical P-wave
# impedance of the TTI medium).
MALLICK_FRAZER_1990_NORMAL_INCIDENCE_DECOUPLES_QSV = True
MALLICK_FRAZER_1990_NORMAL_INCIDENCE_REDUCES_TO_ACOUSTIC_IMPEDANCE = True


# ──────────────────────────────────────────────────────────────────
# Anchor 5: 2D Voigt-naming convention sentinel
# ──────────────────────────────────────────────────────────────────


# The production module uses the **standard 13/15/33/35/55**
# naming (1=x, 3=z, 5=xz), which is the seismological convention
# Mallick-Frazer use throughout. NOT the codebase's older
# 12/16/22/26/66 naming used by other Cij utilities — those
# require conversion via the `stiffness_from_thomsen` helper.
# Silent confusion between the two conventions would corrupt the
# Christoffel matrix elements.
MALLICK_FRAZER_1990_VOIGT_NAMING = "(13, 15, 33, 35, 55)"
MALLICK_FRAZER_1990_VOIGT_AXES = "1=x, 3=z, 5=xz"


# ──────────────────────────────────────────────────────────────────
# Anchor 6: Conv A Fourier sign convention (cross-link to DWN)
# ──────────────────────────────────────────────────────────────────


# The Mallick-Frazer R/T machinery in this module follows Conv A
# (matching `bouchon_1981.dwn_solver`):
#   hat u(omega) = ∫ u(t) exp(-i*omega*t) dt
#   Outgoing/causal: G ∝ exp(-i*q*|z - z_s|) / (2j q),
#   with Im(q) ≤ 0.
# Flipping the sign convention would propagate as a 180-degree
# phase rotation through every R(p_x, ω) coefficient.
MALLICK_FRAZER_1990_FOURIER_CONVENTION = "Conv A (- i omega t exponent)"
