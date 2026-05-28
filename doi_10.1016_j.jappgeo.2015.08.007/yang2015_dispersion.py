"""Yang 2015 §4 dispersion-error analysis — qualitative + envelope gates.

Closes pre-flight dual-reviewer YF3 (Codex Q4 + Gemini Q4 DISAGREE on
k_cross byte-match — use qualitative ordering + envelope instead).

Implements Yang 2015 J.Appl.Geophys. §4 Eq 24-25:

    ε(M, u; scheme)  =  sqrt[ (1/u) ∫_0^u δ_M(β)² dβ ]
    δ_M(β)          =  Σ_{m=1}^M a_m sin((2m-1)β) / β  −  1

where ε is the RMS dispersion error over [0, u] for a given scheme +
operator length M. The paper's quantitative anchor is **Table 4**
(p. 45): u values at which ε first hits the thresholds {10⁻⁶, 10⁻⁵,
10⁻⁴, 10⁻³}, tabulated across M = 2..11 × {TE, SA, LS} = 120 entries.

Y.6 byte-match strategy
-----------------------
For each (ε_target, M, scheme):
  1. Compute a_m via `solve_te/sa/ls_rsfd` (already byte-matched
     against Tables 1-3).
  2. Find the u where ε(M, u; scheme) = ε_target via binary search
     on u ∈ (0, π/2].
  3. Byte-match against paper Table 4 value (paper precision: 2
     decimal places ≡ 0.005 tolerance).

Qualitative invariants per YF3:
  - u_TE < u_SA < u_LS at every (ε, M)
  - u(ε, M) monotonically increases with M (longer operator → wider
    accurate band)
  - u(ε, M) monotonically increases with ε (larger tolerated error →
    wider band)

Note: when computing ε(M, u; scheme) for SA / LS, the a_m coefficients
THEMSELVES are functions of u (since the SA/LS solvers optimise the
coefficients for a specific u). Per the paper's framing, Table 4 column
u_SA(ε, M) is the u for which the SA-optimised coefficients at that u
yield ε(M, u; SA) = ε_target. So we have a fixed-point condition:
solve u such that SA-optimisation-at-u produces ε = ε_target.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def dispersion_error_rms(a: np.ndarray, u: float) -> float:
    """RMS dispersion error ε(a, u) per Yang 2015 Eq 24.

        ε = sqrt[ (1/u) ∫_0^u δ_M(β)² dβ ]
        δ_M(β) = Σ_m a_m sin((2m-1)β)/β  −  1

    Parameters
    ----------
    a : ndarray of shape (M,)
        RSFD coefficients (TE, SA, or LS).
    u : float
        Integration upper bound (β ∈ [0, u]).

    Returns
    -------
    epsilon : float
        RMS dispersion error.
    """
    M = len(a)

    def delta(beta):
        # Σ_m a_m sin((2m-1)β) / β  − 1
        # At β=0 limit: numerator → β·Σ_m a_m (2m-1) = β·1 = β, so
        # delta(0) → 0/0 limit = lim (Σ a_m (2m-1) β + O(β^3))/β - 1
        #        = Σ a_m (2m-1) − 1 = 1 − 1 = 0 (by Yang 2015 Eq 17).
        if abs(beta) < 1e-14:
            return 0.0
        total = sum(a[m] * np.sin((2 * (m + 1) - 1) * beta)
                    for m in range(M))
        return total / beta - 1.0

    integral, _ = quad(lambda b: delta(b) ** 2, 0.0, u, limit=200)
    return float(np.sqrt(integral / u))


def find_u_for_target_error(M: int,
                              eps_target: float,
                              scheme: str,
                              u_min: float = 0.05,
                              u_max: float = 1.55) -> float:
    """For (M, ε_target, scheme), find the u such that the scheme's
    RMS error ε equals ε_target.

    For SA/LS, the coefficients are themselves u-dependent, so this
    is a fixed-point: solve u such that scheme(M, u) yields ε(u) =
    ε_target. For TE, the coefficients are u-independent, so the
    error increases monotonically with u and we just find the u where
    the TE error first hits ε_target.

    Implementation: binary search on log10(ε) - log10(ε_target).

    Parameters
    ----------
    M : int
        Operator length.
    eps_target : float
        Target RMS error (e.g., 1e-6, 1e-5, 1e-4, 1e-3).
    scheme : str
        'TE', 'SA', or 'LS'.
    u_min, u_max : float
        Bracketing range for u.

    Returns
    -------
    u : float
        The u-value satisfying ε(M, u; scheme) = eps_target.
    """
    from yang2015_rsfd_solvers import (
        solve_te_rsfd, solve_sa_rsfd, solve_ls_rsfd)

    def epsilon_at_u(u: float) -> float:
        if scheme == "TE":
            a = solve_te_rsfd(M)
        elif scheme == "SA":
            a = solve_sa_rsfd(M, u)
        elif scheme == "LS":
            a = solve_ls_rsfd(M, u)
        else:
            raise ValueError(f"scheme must be TE/SA/LS, got {scheme!r}")
        return dispersion_error_rms(a, u)

    def f(u: float) -> float:
        # Use log10 difference for better numerical scaling
        return np.log10(epsilon_at_u(u)) - np.log10(eps_target)

    # Find a sign change in (u_min, u_max) via expansion if needed
    fmin = f(u_min)
    fmax = f(u_max)
    if fmin * fmax > 0:
        # No sign change in the default bracket — return u_max (the
        # scheme's error never reaches the target in this band).
        return float(u_max if fmin > 0 else u_min)
    return float(brentq(f, u_min, u_max, xtol=1e-4, maxiter=100))


# ─── Table 4 (Yang 2015 p. 45) byte-transcribed ───────────────────────

# Format: YANG_2015_TABLE_4[eps_target][M] = {"TE": u_TE, "SA": u_SA, "LS": u_LS}
# Precision: 2 decimal places as printed.

YANG_2015_TABLE_4: dict[float, dict[int, dict[str, float]]] = {
    1e-6: {
        2:  {"TE": 0.08, "SA": 0.10, "LS": 0.11},
        3:  {"TE": 0.21, "SA": 0.28, "LS": 0.31},
        4:  {"TE": 0.33, "SA": 0.46, "LS": 0.51},
        5:  {"TE": 0.44, "SA": 0.62, "LS": 0.68},
        6:  {"TE": 0.52, "SA": 0.74, "LS": 0.81},
        7:  {"TE": 0.59, "SA": 0.84, "LS": 0.92},
        8:  {"TE": 0.65, "SA": 0.92, "LS": 1.00},
        9:  {"TE": 0.70, "SA": 0.98, "LS": 1.07},
        10: {"TE": 0.75, "SA": 1.04, "LS": 1.12},
        11: {"TE": 0.79, "SA": 1.08, "LS": 1.17},   # Codex G1 fix 2026-05-27
    },
    1e-5: {
        2:  {"TE": 0.14, "SA": 0.17, "LS": 0.19},
        3:  {"TE": 0.31, "SA": 0.41, "LS": 0.45},
        4:  {"TE": 0.45, "SA": 0.62, "LS": 0.67},
        5:  {"TE": 0.55, "SA": 0.78, "LS": 0.84},   # Codex G1 fix 2026-05-27
        6:  {"TE": 0.64, "SA": 0.90, "LS": 0.97},
        7:  {"TE": 0.71, "SA": 0.99, "LS": 1.06},
        8:  {"TE": 0.77, "SA": 1.06, "LS": 1.13},
        9:  {"TE": 0.82, "SA": 1.12, "LS": 1.19},
        10: {"TE": 0.86, "SA": 1.17, "LS": 1.23},
        11: {"TE": 0.90, "SA": 1.21, "LS": 1.27},
    },
    1e-4: {
        2:  {"TE": 0.25, "SA": 0.30, "LS": 0.34},
        3:  {"TE": 0.45, "SA": 0.60, "LS": 0.66},
        4:  {"TE": 0.60, "SA": 0.82, "LS": 0.89},
        5:  {"TE": 0.71, "SA": 0.97, "LS": 1.04},
        6:  {"TE": 0.79, "SA": 1.08, "LS": 1.15},
        7:  {"TE": 0.86, "SA": 1.17, "LS": 1.22},
        8:  {"TE": 0.91, "SA": 1.23, "LS": 1.28},
        9:  {"TE": 0.96, "SA": 1.28, "LS": 1.32},
        10: {"TE": 1.00, "SA": 1.32, "LS": 1.35},
        11: {"TE": 1.03, "SA": 1.35, "LS": 1.38},
    },
    1e-3: {
        2:  {"TE": 0.45, "SA": 0.53, "LS": 0.60},
        3:  {"TE": 0.68, "SA": 0.87, "LS": 0.95},
        4:  {"TE": 0.83, "SA": 1.09, "LS": 1.15},
        5:  {"TE": 0.93, "SA": 1.22, "LS": 1.27},
        6:  {"TE": 1.00, "SA": 1.31, "LS": 1.35},
        7:  {"TE": 1.06, "SA": 1.37, "LS": 1.40},
        8:  {"TE": 1.10, "SA": 1.41, "LS": 1.43},
        9:  {"TE": 1.14, "SA": 1.44, "LS": 1.46},
        10: {"TE": 1.17, "SA": 1.47, "LS": 1.48},
        11: {"TE": 1.20, "SA": 1.49, "LS": 1.50},
    },
}
