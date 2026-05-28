"""Yang, Yan & Liu (2015) Geophys. Prospect. — Sampling Approximation
(SA) method for implicit SGFD.

Self-contained reproduction of Yang, L., Yan, H. & Liu, H. (2015).
"Optimal implicit staggered-grid finite-difference schemes based on
the sampling approximation method for seismic modelling."
*Geophysical Prospecting* 64:595-610.
DOI 10.1111/1365-2478.12325 (accepted 2015-07, online 2016).

ROLE: this module is the **antecedent** for the SA method that
Yang 2015 *J. Appl. Geophys.* 122:40-52 extends to RSG-TTI. Y.5.2
cross-check calls this module's `solve_yang_yan_liu_2015_sa(M, u)`
as an INDEPENDENT re-derivation of the SA-method coefficients —
closing pre-flight dual-reviewer YF2 (Codex+Gemini DISAGREE on
test set being insufficient) at the SA-antecedent level.

SCOPE CAVEAT (Codex YF2 + YF3 follow-up)
----------------------------------------
This antecedent paper covers **IMPLICIT** SGFD (Padé-form
rational approximation with adjustable parameter b). Yang 2015
*J. Appl. Geophys.* (the paper being reproduced) applies the SA
method to **EXPLICIT** RSG. The SA principle generalises but
the specific linear system differs:

- IMPLICIT SA (here): (M+1) sample points, (M+1) unknowns
  (a_1..a_M, b), with b as a free parameter.
- EXPLICIT SA (Yang 2015 J.Appl.Geophys. — to be reproduced
  separately): M sample points, M unknowns (a_1..a_M only;
  no b since explicit operator has no adjustable parameter).

This module reproduces the IMPLICIT antecedent. Verifying it
matches the antecedent's Table 1 validates the SA principle.
Yang 2015 J.Appl.Geophys. SA coefficients will then use the
same idea adapted to the RSG-TTI dispersion relation.

SA recipe (Yang/Yan/Liu 2015 GP Eq 7-8)
--------------------------------------

Implicit SGFD operator (Eq 1):

    ∂p/∂x ≈ (1/h) Σ_{m=1}^M a_m [p(x+(m-0.5)h) - p(x-(m-0.5)h)]
             /  [1 + b·h²·(δ²/δx²)]

After plane-wave substitution + simplification (Eq 7):

    [1 - 2b + 2b·cos(2β)] β ≈ Σ_{m=1}^M a_m sin((2m-1) β)

where β = kh/2 ∈ [0, π/2].

SA principle: choose u ∈ (0, π/2] (max wavenumber for accuracy
band), sample (M+1) evenly-spaced points β(k) = k·u/(M+1) for
k=1..M+1, force Eq 7 to hold exactly at each → (M+1)×(M+1)
linear system in (a_1..a_M, b). Solve via numpy.linalg.solve
(Eq 8 in paper).

Test cross-check via Table 1 (p. 598) at u=1.25 for M=2..11.
"""
from __future__ import annotations

import numpy as np


def solve_yang_yan_liu_2015_sa(M: int, u: float
                                ) -> tuple[np.ndarray, float]:
    """Solve the SA-optimised implicit SGFD coefficients per
    Yang/Yan/Liu 2015 *Geophys. Prospect.* §"Optimal implicit
    staggered-grid finite-difference coefficients based on the
    sampling approximation method".

    Parameters
    ----------
    M : int
        Number of half-stencil coefficients (operator has 2M
        points). M ≥ 2.
    u : float
        Sampling band upper bound (β ∈ (0, u]). The (M+1)
        sample points are evenly spaced at β(k) = k·u/(M+1)
        for k=1..M+1. Per paper Table 1 the canonical value is
        u = 1.25.

    Returns
    -------
    a : ndarray of shape (M,)
        SA-optimised implicit SGFD coefficients a_1..a_M.
    b : float
        SA-optimised adjustable parameter (Padé denominator
        parameter in Eq 1).

    Method (Yang/Yan/Liu 2015 GP Eq 7-8)
    -----------------------------------
    1. Sample (M+1) evenly-spaced β values: β(k) = k·u/(M+1)
       for k = 1, 2, ..., M+1.
    2. Build (M+1)×(M+1) linear system A·x = b_rhs where
       x = (a_1, a_2, ..., a_M, b) and
       A[i, j] = sin((2(j+1)-1)·β(i+1))      for j = 0..M-1
       A[i, M] = 2β(i+1) - 2β(i+1)·cos(2β(i+1))
       b_rhs[i] = β(i+1)
       for i = 0..M.
    3. Solve via numpy.linalg.solve (paper's Eq 8 Gaussian
       elimination).
    """
    if M < 2:
        raise ValueError(f"M must be ≥ 2 (got {M})")
    if not (0.0 < u <= 0.5 * np.pi):
        raise ValueError(f"u must be in (0, π/2] (got {u})")

    n_eqs = M + 1
    betas = np.array([k * u / (M + 1) for k in range(1, n_eqs + 1)],
                      dtype=np.float64)

    A = np.zeros((n_eqs, n_eqs), dtype=np.float64)
    rhs = np.zeros(n_eqs, dtype=np.float64)

    for i, beta in enumerate(betas):
        for j in range(M):
            # m = j + 1 (1-indexed)
            A[i, j] = np.sin((2 * (j + 1) - 1) * beta)
        A[i, M] = 2.0 * beta - 2.0 * beta * np.cos(2.0 * beta)
        rhs[i] = beta

    sol = np.linalg.solve(A, rhs)
    a = sol[:M]
    b = float(sol[M])
    return a, b


# ─── Yang/Yan/Liu 2015 GP Table 1 (SA-based ISFD coeffs at u=1.25) ────

# Source: Yang/Yan/Liu 2015 GP Table 1 (p. 598).
# Format: YYL_2015_GP_TABLE_1[M] = {"b": ..., "a": (a_1, ..., a_M)}
# Precision: 7 sig figs as printed in the paper (xxx.xxxxxxE±NN).
# Side-by-side transcription protocol applies — these are direct
# reads from the paper PDF at the cited page.

YYL_2015_GP_TABLE_1_U_1P25: dict[int, dict] = {
    2: {
        "b": 1.629639e-1,
        "a": (
            6.207188e-1,
            1.274681e-1,
        ),
    },
    3: {
        "b": 1.933497e-1,
        "a": (
            5.036975e-1,
            1.728199e-1,
           -4.506886e-3,
        ),
    },
    4: {
        "b": 2.081935e-1,
        "a": (
            4.463404e-1,
            1.952571e-1,
           -7.146062e-3,
            5.254613e-4,
        ),
    },
    5: {
        "b": 2.169254e-1,
        "a": (
            4.124880e-1,
            2.085767e-1,
           -8.830505e-3,
            9.650428e-4,
           -9.244449e-5,
        ),
    },
    6: {
        "b": 2.226601e-1,
        "a": (
            3.901945e-1,
            2.173847e-1,
           -9.994879e-3,
            1.305876e-3,
           -1.928768e-4,
            2.027148e-5,
        ),
    },
    7: {
        "b": 2.267095e-1,
        "a": (
            3.744186e-1,
            2.236371e-1,
           -1.084760e-2,
            1.572850e-3,
           -2.840291e-4,
            4.730046e-5,
           -5.108892e-6,
        ),
    },
    8: {
        "b": 2.297191e-1,
        "a": (
            3.626732e-1,
            2.283037e-1,
           -1.149919e-2,
            1.786399e-3,
           -3.634971e-4,
            7.531349e-5,
           -1.316850e-5,
            1.417839e-6,
        ),
    },
    9: {
        "b": 2.320426e-1,
        "a": (
            3.535919e-1,
            2.319190e-1,
           -1.201343e-2,
            1.960731e-3,
           -4.322492e-4,
            1.021496e-4,
           -2.253939e-5,
            3.997559e-6,
           -4.223790e-7,
        ),
    },
    10: {
        "b": 2.338902e-1,
        "a": (
            3.463623e-1,
            2.348019e-1,
           -1.242971e-2,
            2.105613e-3,
           -4.918707e-4,
            1.270783e-4,
           -3.230835e-5,
            7.317850e-6,
           -1.292258e-6,
            1.328562e-7,
        ),
    },
    11: {
        "b": 2.353942e-1,
        "a": (
            3.404715e-1,
            2.371542e-1,
           -1.277363e-2,
            2.227877e-3,
           -5.438686e-4,
            1.499410e-4,
           -4.199549e-5,
            1.105552e-5,
           -2.518748e-6,
            4.381472e-7,
           -4.362636e-8,
        ),
    },
}
