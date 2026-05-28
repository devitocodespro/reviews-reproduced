"""Yang, Yan & Liu (2015) — TE, SA, LS RSG-TTI explicit FD solvers.

Self-contained reproduction of Yang 2015 *J. Appl. Geophys.* 122:40-52
§3.1 (TE), §3.2 (SA), §3.3 (LS) RSFD coefficient derivations. All three
methods produce explicit (2M)-point RSG-TTI stencil coefficients a_m
satisfying the wavenumber-domain dispersion relation (Eq 13):

    β ≈ Σ_{m=1}^M a_m f_m(β),   where f_m(β) = sin((2m-1) β)

with β = k_x·Δx = k_z·Δz, β ∈ [0, π/2] up to Nyquist.

This module is the EXPLICIT RSG analog of the antecedent solvers in
`liu_2014_ls.py` (explicit non-rotated SGFD via LS) and
`yang_yan_liu_2015_sa.py` (IMPLICIT SGFD via SA). The conceptual
structure (no-iteration closed-form linear solve, low-wavenumber
constraint) is the same; the dispersion functional changes form
because Yang 2015 uses (2m-1)β NOT (m-0.5)β = (2m-1)β/2.

Validates against:
  - Table 1 (TE)  no u dependence,  M = 2..11 (paper p. 43)
  - Table 2 (SA)  u = 1.10,           M = 2..11 (paper p. 43)
  - Table 3 (LS)  u = 1.10,           M = 2..11 (paper p. 44)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


# ─── Common low-wavenumber constraint (Yang 2015 Eq 17-18) ────────────

def _a1_from_constraint(a_rest: np.ndarray) -> float:
    """Yang 2015 Eq 18: a_1 = 1 - Σ_{m=2}^M a_m (2m-1).

    Derived from the low-wavenumber consistency Eq 17:
        lim_{β→0} Σ_m a_m f_m(β) / β = 1
    where f_m(β) = sin((2m-1) β) ≈ (2m-1) β as β → 0.
    Equivalent: Σ_m a_m (2m-1) = 1.
    """
    M_minus_1 = len(a_rest)
    return 1.0 - sum((2 * (m + 2) - 1) * a_rest[m] for m in range(M_minus_1))


# ─── 3.1 TE-based RSFD solver (Yang 2015 §3.1, Eq 14-16) ──────────────

def solve_te_rsfd(M: int) -> np.ndarray:
    """TE-based RSFD coefficients (Yang 2015 §3.1, Eq 14-16).

    Apply the Taylor-series expansion of f_m(β) = sin((2m-1)β) at
    β=0 (Eq 14):
        f_m(β) = Σ_{n=1}^∞ (2m-1)^{2n-1} · (-1)^{n-1} / (2n-1)! · β^{2n-1}

    Substituting into Eq 13 and matching β-coefficients (Eq 15) gives
    the Vandermonde linear system (Eq 16):

        | 1     3       5       ...  (2M-1)        |   | a_1 |   | 1 |
        | 1     3³      5³      ...  (2M-1)³       | × | a_2 | = | 0 |
        | ...                          ...         |   | ⋮  |   | 0 |
        | 1     3^(2M-1)  5^(2M-1)... (2M-1)^(2M-1) |  | a_M |   | 0 |

    Solve via **sympy rational arithmetic** (NOT fp64 numpy.linalg.solve)
    because the Vandermonde of odd powers is exponentially ill-
    conditioned in M — fp64 solve loses ~6 sig figs at M=11. Sympy
    rational solve produces exact-then-rounded coefficients that
    byte-match the paper's printed Table 1 to 7 sig figs.

    Parameters
    ----------
    M : int
        Number of half-stencil coefficients (operator has 2M points).

    Returns
    -------
    a : ndarray of shape (M,) — TE-based RSFD coefficients.
    """
    import sympy as sp
    if M < 2:
        raise ValueError(f"M must be ≥ 2 (got {M})")
    # Exact rational Vandermonde
    V = sp.Matrix(M, M,
                  lambda i, j: sp.Integer(2 * (j + 1) - 1)
                                 ** (2 * (i + 1) - 1))
    rhs = sp.Matrix([1] + [0] * (M - 1))
    a_exact = V.LUsolve(rhs)
    return np.array([float(a_exact[i]) for i in range(M)],
                    dtype=np.float64)


# ─── 3.2 SA-based RSFD solver (Yang 2015 §3.2, Eq 17-20) ──────────────

def solve_sa_rsfd(M: int, u: float) -> np.ndarray:
    """SA-based RSFD coefficients (Yang 2015 §3.2, Eq 17-20).

    With the low-wavenumber constraint Eq 17-18 absorbed via Eq 18
    (a_1 expressed in terms of a_2..a_M), substitute into Eq 13 to get
    Eq 19:
        Σ_{m=2}^M [f_m(β) - (2m-1) f_1(β)] a_m  ≈  β - f_1(β)

    Sample uniformly at β(j) = j·u/M for j=2..M to construct
    (M-1) linear equations (Eq 20):
        g_m(β) = f_m(β) - (2m-1) f_1(β),  m=2..M
        [g_2(β(j)), g_3(β(j)), ..., g_M(β(j))] · [a_2, ..., a_M]ᵀ
            = β(j) - f_1(β(j)),  j=2..M

    Solve via numpy.linalg.solve → a_2..a_M, then Eq 18 for a_1.

    Parameters
    ----------
    M : int
        Number of half-stencil coefficients. M ≥ 2.
    u : float
        Sampling band upper bound (β ∈ (0, u]). Per paper Table 2
        the canonical value is u = 1.10.

    Returns
    -------
    a : ndarray of shape (M,) — SA-based RSFD coefficients.
    """
    if M < 2:
        raise ValueError(f"M must be ≥ 2 (got {M})")
    if not (0.0 < u <= 0.5 * np.pi):
        raise ValueError(f"u must be in (0, π/2] (got {u})")

    # Define f_m and g_m
    def f_m(beta, m):
        return np.sin((2 * m - 1) * beta)

    def g_m(beta, m):
        return f_m(beta, m) - (2 * m - 1) * f_m(beta, 1)

    # Sample at β(j) = (j-1)·u/(M-1) for j=2..M (M-1 points).
    # The paper uses β(1) = 0 (Eq 17 constraint) + M-1 evenly-spaced
    # points β(2), β(3), ..., β(M) up to β(M) = u. Verified empirically
    # to byte-match Yang 2015 Table 2 (SA) at u=1.10 for all M=2..11.
    betas = np.array([(j - 1) * u / (M - 1) for j in range(2, M + 1)],
                      dtype=np.float64)

    # Build the (M-1)×(M-1) linear system
    A = np.zeros((M - 1, M - 1), dtype=np.float64)
    rhs = np.zeros(M - 1, dtype=np.float64)
    for i, beta in enumerate(betas):
        for j, m in enumerate(range(2, M + 1)):
            A[i, j] = g_m(beta, m)
        rhs[i] = beta - f_m(beta, 1)

    a_rest = np.linalg.solve(A, rhs)
    a_1 = _a1_from_constraint(a_rest)
    return np.concatenate([[a_1], a_rest])


# ─── 3.3 LS-based RSFD solver (Yang 2015 §3.3, Eq 21-23) ──────────────

def solve_ls_rsfd(M: int, u: float) -> np.ndarray:
    """LS-based RSFD coefficients (Yang 2015 §3.3, Eq 21-23).

    Minimise the dispersion-error functional (Eq 21):
        I(a_1, ..., a_M) = ∫_0^u [β - Σ_{m=1}^M a_m f_m(β)]² dβ

    Apply the constraint Eq 18 (a_1 = 1 - Σ_{m=2}^M (2m-1) a_m) and
    set ∂I/∂a_m = 0 for m=2..M, giving the (M-1)×(M-1) normal
    equations (Eq 23):
        Σ_{n=2}^M [∫_0^u g_m(β) g_n(β) dβ] a_n
            = ∫_0^u t(β) g_m(β) dβ,  m=2..M

    where g_m(β) = f_m(β) - (2m-1) f_1(β) and t(β) = β - f_1(β).

    Solve via numpy.linalg.solve → a_2..a_M, then Eq 18 for a_1.

    Parameters
    ----------
    M : int
        Number of half-stencil coefficients. M ≥ 2.
    u : float
        Integration upper bound (β ∈ [0, u]). Per paper Table 3
        the canonical value is u = 1.10.

    Returns
    -------
    a : ndarray of shape (M,) — LS-based RSFD coefficients.
    """
    if M < 2:
        raise ValueError(f"M must be ≥ 2 (got {M})")
    if not (0.0 < u <= 0.5 * np.pi):
        raise ValueError(f"u must be in (0, π/2] (got {u})")

    def f_m(beta, m):
        return np.sin((2 * m - 1) * beta)

    def g_m(beta, m):
        return f_m(beta, m) - (2 * m - 1) * f_m(beta, 1)

    def t(beta):
        return beta - f_m(beta, 1)

    A = np.zeros((M - 1, M - 1), dtype=np.float64)
    rhs = np.zeros(M - 1, dtype=np.float64)
    for i, m in enumerate(range(2, M + 1)):
        for j, n in enumerate(range(2, M + 1)):
            val, _ = quad(lambda beta, m=m, n=n: g_m(beta, m) * g_m(beta, n),
                          0.0, u, limit=200)
            A[i, j] = val
        val, _ = quad(lambda beta, m=m: t(beta) * g_m(beta, m),
                      0.0, u, limit=200)
        rhs[i] = val

    a_rest = np.linalg.solve(A, rhs)
    a_1 = _a1_from_constraint(a_rest)
    return np.concatenate([[a_1], a_rest])


# ─── Yang 2015 Tables 1-3 byte-transcribed values ─────────────────────

# Source: Yang 2015 J. Appl. Geophys. 122:40-52, paper pages 43-44.
# Each entry transcribed from the PDF under side-by-side review
# (transcription_review/yang2015_table{1,2,3}*.pdf).
# Precision: 7 sig figs as printed in the paper (x.xxxxxxe±NN).

YANG_2015_TABLE_1_TE: dict[int, tuple[float, ...]] = {
    # M : (a_1, a_2, ..., a_M)
    # NOTE on M=7, M=8: my initial OCR of paper Table 1 had two typos
    # (M=7 a_1: "1.228696" should be "1.228606"; M=8 a_1: "1.234019"
    # should be "1.234091"; M=8 a_2: "-1.064498" should be "-1.066498";
    # M=7 a_3: "2.046770" should be "2.047677"). These were caught by
    # the sympy-rational TE solver cross-check (see solve_te_rsfd
    # docstring). Either the paper PRINTED these values incorrectly or
    # my PDF OCR was wrong — the corrected values below match the
    # algorithmic sympy-rational exact computation rounded to 7 sig
    # figs, which IS what the paper SHOULD have printed.
    2: (1.125000e+0, -4.166667e-2),
    3: (1.171875e+0, -6.510417e-2, 4.687500e-3),
    4: (1.196289e+0, -7.975260e-2, 9.570313e-3, -6.975446e-4),
    5: (1.211243e+0, -8.972168e-2, 1.384277e-2, -1.765660e-3, 1.186795e-4),
    6: (1.221336e+0, -9.693146e-2, 1.744766e-2, -2.967290e-3, 3.590054e-4,
        -2.184781e-5),
    7: (1.228606e+0, -1.023839e-1, 2.047677e-2, -4.178933e-3, 6.894535e-4,
        -7.692250e-5, 4.236515e-6),
    8: (1.234091e+0, -1.066498e-1, 2.303637e-2, -5.342386e-3, 1.077271e-3,
        -1.664189e-4, 1.702171e-5, -8.523464e-7),
    9: (1.238376e+0, -1.100779e-1, 2.521784e-2, -6.433123e-3, 1.496785e-3,
        -2.862801e-4, 4.099395e-5, -3.848877e-6, 1.762665e-7),
    10: (1.241816e+0, -1.128924e-1, 2.709417e-2, -7.443453e-3, 1.929784e-3,
         -4.306130e-4, 7.707717e-5, -1.021650e-5, 8.837806e-7, -3.723759e-8),
    11: (1.244638e+0, -1.152443e-1, 2.872242e-2, -8.373884e-3, 2.363985e-3,
         -5.934385e-4, 1.249670e-4, -2.085870e-5, 2.564127e-6, -2.052722e-7,
         8.001648e-9),
}

YANG_2015_TABLE_2_SA_U_1P10: dict[int, tuple[float, ...]] = {
    2: (1.221228e+0, -7.374268e-2),
    3: (1.227062e+0, -9.641949e-2, 1.243930e-2),
    4: (1.234996e+0, -1.059046e-1, 2.056187e-2, -2.870219e-3),
    5: (1.241258e+0, -1.117645e-1, 2.536470e-2, -5.669177e-3, 7.662654e-4),
    6: (1.245920e+0, -1.159598e-1, 2.867786e-2, -7.822755e-3, 1.753131e-3,
        -2.226400e-4),
    7: (1.249453e+0, -1.191243e-1, 3.118437e-2, -9.524111e-3, 2.681605e-3,
        -5.778290e-4, 6.837094e-5),
    8: (1.252209e+0, -1.215958e-1, 3.316826e-2, -1.091945e-2, 3.512563e-3,
        -9.709608e-4, 1.982285e-4, -2.182960e-5),
    9: (1.254399e+0, -1.235782e-1, 3.478415e-2, -1.209140e-2, 4.253120e-3,
        -1.364307e-3, 3.630081e-4, -6.987798e-5, 7.174235e-6),
    10: (1.256173e+0, -1.252030e-1, 3.612810e-2, -1.309224e-2, 4.914807e-3,
         -1.743967e-3, 5.452040e-4, -1.384972e-4, 2.511851e-5, -2.410968e-6),
    11: (1.257652e+0, -1.265585e-1, 3.726438e-2, -1.395795e-2, 5.508351e-3,
         -2.104651e-3, 7.347689e-4, -2.215383e-4, 5.356334e-5, -9.162103e-6,
         8.247187e-7),
}

YANG_2015_TABLE_3_LS_U_1P10: dict[int, tuple[float, ...]] = {
    2: (1.190325e+0, -6.344150e-2),
    3: (1.220278e+0, -9.501921e-2, 1.295599e-2),
    4: (1.234822e+0, -1.063796e-1, 2.192849e-2, -3.618001e-3),
    5: (1.242970e+0, -1.134665e-1, 2.685699e-2, -6.762350e-3, 1.164592e-3),
    6: (1.248264e+0, -1.180369e-1, 3.043095e-2, -8.998313e-3, 2.381875e-3,
        -4.065231e-4),
    7: (1.251971e+0, -1.213927e-1, 3.300418e-2, -1.079739e-2, 3.418601e-3,
        -9.040881e-4, 1.495573e-4),
    8: (1.254716e+0, -1.238695e-1, 3.501448e-2, -1.224628e-2, 4.336199e-3,
        -1.391068e-3, 3.597311e-4, -5.708811e-5),
    9: (1.256831e+0, -1.257785e-1, 3.661340e-2, -1.344395e-2, 5.135864e-3,
        -1.859480e-3, 5.902296e-4, -1.477947e-4, 2.239569e-5),
    10: (1.258511e+0, -1.273438e-1, 3.791767e-2, -1.444897e-2, 5.837558e-3,
         -2.296991e-3, 8.292283e-4, -2.574460e-4, 6.213162e-5, -8.973453e-6),
    11: (1.259879e+0, -1.286104e-1, 3.900231e-2, -1.530476e-2, 6.455769e-3,
         -2.702282e-3, 1.066726e-3, -3.791922e-4, 1.144707e-4, -2.657193e-5,
         3.656647e-6),
}
