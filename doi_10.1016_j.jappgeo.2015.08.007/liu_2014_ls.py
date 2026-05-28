"""Liu (2014) GJI 197:1033 — closed-form LS solver for SGFD coefficients.

Self-contained reproduction of Liu Y. (2014) "Optimal staggered-grid
finite-difference schemes based on least-squares for wave equation
modelling" §2.1 (Eq 1-12). DOI 10.1093/gji/ggu032.

ROLE: this module is the **antecedent** for the LS method that Yang
2015 extends to RSG-TTI. Y.5.3 cross-check (per Yang 2015 plan)
calls Liu 2014's LS solver as an INDEPENDENT re-derivation of the
LS-RSG-TTI coefficients that Yang 2015 publishes — closing the
dual-reviewer YF2 finding (tests must be non-tautological).

LS recipe (Liu 2014 Eq 1-12)
----------------------------

ESGFD operator (Eq 1):
    ∂p/∂x ≈ (1/h) Σ_{m=1}^M c_m [p(x + (m-0.5)h) - p(x - (m-0.5)h)]

In wavenumber space (Eq 3) with β = kh ∈ [0, π]:
    2 Σ_{m=1}^M c_m sin((m-0.5) β)  ≈  β

LS functional (Eq 6) on β ∈ [0, b] with b < π:
    E = ∫_0^b [Σ_m c_m φ_m(β) - f(β)]² dβ
where φ_m(β) = 2 sin((m-0.5) β) and f(β) = β.

Low-wavenumber constraint (Eq 8-9) — preserves first-derivative
accuracy at β → 0:
    Σ_m (2m - 1) c_m = 1
    ⇒  c_1 = 1 - Σ_{m=2}^M (2m - 1) c_m       (Eq 9)

Substituting Eq 9 into Eq 6 yields the reduced LS system on
(M-1) unknowns (c_2, ..., c_M), solved via the normal equations
(Eq 12):
    Σ_{m=2}^M [∫_0^b ψ_m(β) ψ_n(β) dβ] c_m
        = ∫_0^b g(β) ψ_n(β) dβ,   for n = 2, ..., M

where (Eq 11)
    ψ_m(β) = 2 sin((m-0.5) β) - 2 (2m-1) sin(0.5 β)
    g(β)   = β - 2 sin(0.5 β)

NOTE on Eq 11 derivation
------------------------
Substituting c_1 = 1 - Σ_{m=2}^M (2m-1) c_m into
Σ_{m=1}^M c_m · 2 sin((m-0.5)β):

  = [1 - Σ_{m=2}^M (2m-1) c_m] · 2 sin(0.5β) + Σ_{m=2}^M c_m · 2 sin((m-0.5)β)
  = 2 sin(0.5β) + Σ_{m=2}^M c_m · [2 sin((m-0.5)β) - 2(2m-1) sin(0.5β)]

so ψ_m(β) = 2 sin((m-0.5)β) - 2(2m-1) sin(0.5β) and the residual
to minimise is g(β) := β - 2 sin(0.5β).

NO ITERATION — the LS system is a closed-form (M-1)×(M-1) linear
solve. The paper emphasises "no iterations involved" (abstract).

Reproduction of Liu 2014 Table 3
---------------------------------

Table 3 (p. 1036) lists optimised c_m for M ∈ {2, ..., 10} at
maximum relative dispersion error η = 10⁻⁴. The b values used
are from Table 2 (p. 1035) at η = 10⁻⁴:
  M=2:  b=1.08      M=3:  b=1.76      M=4:  b=2.16
  M=5:  b=2.40      M=6:  b=2.55      M=7:  b=2.66
  M=8:  b=2.74      M=9:  b=2.80      M=10: b=2.85

Test `tests/test_liu_2014_ls.py` calls this module's
`solve_liu_2014_ls(M, b)` and asserts the result matches the
paper-byte-transcribed Table 3 to 7 sig figs (paper print
precision).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def _phi_m(beta: float, m: int) -> float:
    """φ_m(β) = 2 sin((m - 0.5) β).  Per Liu 2014 Eq 4."""
    return 2.0 * np.sin((m - 0.5) * beta)


def _psi_m(beta: float, m: int) -> float:
    """ψ_m(β) = 2 sin((m-0.5) β) - 2 (2m-1) sin(0.5 β).

    Per Liu 2014 Eq 11 (derived from substituting Eq 9 constraint
    into Eq 6 LS functional).
    """
    return 2.0 * np.sin((m - 0.5) * beta) - 2.0 * (2 * m - 1) * np.sin(0.5 * beta)


def _g(beta: float) -> float:
    """g(β) = β - 2 sin(0.5 β).  Per Liu 2014 Eq 11."""
    return beta - 2.0 * np.sin(0.5 * beta)


def solve_liu_2014_ls(M: int, b: float) -> np.ndarray:
    """Solve the LS-optimised ESGFD coefficients per Liu 2014 §2.1.

    Parameters
    ----------
    M : int
        Number of half-stencil coefficients (operator has 2M points).
        M ≥ 2 (M=1 is the trivial centred difference c_1 = 1).
    b : float
        Upper bound of the wavenumber integration interval [0, b].
        b ∈ (0, π); choose per Liu 2014 Table 2 for a target max
        dispersion error η.

    Returns
    -------
    c : ndarray of shape (M,)
        ESGFD coefficients [c_1, c_2, ..., c_M] satisfying:
          (a) Liu 2014 Eq 9 constraint: Σ_m (2m-1) c_m = 1
          (b) LS-minimised dispersion error on [0, b]

    Method (Liu 2014 Eq 10-12)
    -------
    1. For m, n ∈ {2, ..., M}: compute the matrix entries
         A[m,n] = ∫_0^b ψ_m(β) ψ_n(β) dβ
       and vector entries
         r[n]   = ∫_0^b g(β) ψ_n(β) dβ
    2. Solve linear system A · (c_2, ..., c_M) = r  via numpy.linalg.solve.
    3. Compute c_1 = 1 - Σ_{m=2}^M (2m-1) c_m  per Eq 9.
    """
    if M < 2:
        raise ValueError(f"M must be ≥ 2 (got {M})")
    if not (0.0 < b < np.pi):
        raise ValueError(f"b must be in (0, π) (got {b})")

    # Build the (M-1)×(M-1) LS system on c_2..c_M
    A = np.zeros((M - 1, M - 1), dtype=np.float64)
    r = np.zeros(M - 1, dtype=np.float64)

    for i, m in enumerate(range(2, M + 1)):
        for j, n in enumerate(range(2, M + 1)):
            val, _ = quad(lambda beta, m=m, n=n: _psi_m(beta, m) * _psi_m(beta, n),
                          0.0, b, limit=200)
            A[i, j] = val
        val, _ = quad(lambda beta, n=m: _g(beta) * _psi_m(beta, n),
                      0.0, b, limit=200)
        r[i] = val

    # Solve for c_2..c_M
    c_2_to_M = np.linalg.solve(A, r)

    # Recover c_1 from Eq 9 constraint
    c_1 = 1.0 - np.sum([(2 * m - 1) * c_2_to_M[i]
                        for i, m in enumerate(range(2, M + 1))])

    return np.concatenate([[c_1], c_2_to_M])


# ─── Liu 2014 Table 2 (b values for η = 10⁻⁴) ────────────────────────────

LIU_2014_TABLE_2_B_ETA_1E_MINUS_4: dict[int, float] = {
    # M : b
    # Source: Liu 2014 Table 2 (p. 1035), η = 10⁻⁴ column.
    # Verified: row M=6 gives b=2.17, matching Liu 2014 Fig 2 caption
    # ("For the LSM, b = 2.17 is used in eq (12)... M = 6").
    2: 0.61,
    3: 1.21,
    4: 1.65,
    5: 1.95,
    6: 2.17,
    7: 2.32,
    8: 2.44,
    9: 2.53,
    10: 2.60,
}


# ─── Liu 2014 Table 3 (LS-optimised ESGFD coefficients at η = 10⁻⁴) ────

# Source: Liu 2014 Table 3 (p. 1036).
# Format: LIU_2014_TABLE_3[M] = (c_1, c_2, ..., c_M)
# Precision: 7 sig figs as printed in the paper.
# Side-by-side transcription protocol applies — these are direct
# reads from the paper PDF at the cited page.

LIU_2014_TABLE_3: dict[int, tuple[float, ...]] = {
    2: (
        0.1129136E+1,   # c_1 =  1.129136
       -0.4304542E-1,   # c_2 = -0.04304542
    ),
    3: (
        0.1186247E+1,   # c_1 =  1.186247
       -0.7266808E-1,   # c_2 = -0.07266808
        0.6351497E-2,   # c_3 =  0.006351497
    ),
    4: (
        0.1218159E+1,
       -0.9397218E-1,
        0.1519043E-1,
       -0.1742128E-2,
    ),
    5: (
        0.1236425E+1,
       -0.1081130E+0,
        0.2339911E-1,
       -0.5061550E-2,
        0.7054313E-3,
    ),
    6: (
        0.1247576E+1,
       -0.1174969E+0,
        0.2997288E-1,
       -0.8741572E-2,
        0.2262285E-2,
       -0.3745306E-3,
    ),
    7: (
        0.1254380E+1,
       -0.1235307E+0,
        0.3467231E-1,
       -0.1192915E-1,
        0.4057090E-2,
       -0.1191005E-2,
        0.2263204E-3,
    ),
    8: (
        0.1259012E+1,
       -0.1277647E+0,
        0.3820715E-1,
       -0.1458251E-1,
        0.5845385E-2,
       -0.2213861E-2,
        0.7243880E-3,
       -0.1566173E-3,
    ),
    9: (
        0.1262147E+1,
       -0.1306967E+0,
        0.4075792E-1,
       -0.1665221E-1,
        0.7377057E-2,
       -0.3258150E-2,
        0.1336259E-2,
       -0.4775830E-3,
        0.1151664E-3,
    ),
    10: (
        0.1264362E+1,
       -0.1327958E+0,
        0.4264687E-1,
       -0.1824918E-1,
        0.8656223E-2,
       -0.4200034E-2,
        0.1989180E-2,
       -0.8686637E-3,
        0.3342741E-3,
       -0.8854090E-4,
    ),
}
