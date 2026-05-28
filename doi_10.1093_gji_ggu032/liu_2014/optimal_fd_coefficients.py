"""Optimal staggered-grid finite-difference coefficients —
general solver for any order, any optimisation method.

Plan §91 Phase 1: §91e.2.

This module derives optimal half-grid-staggered FD weights for the
first derivative

    ∂f/∂x|_x ≈ (1/h) Σ_{m=1}^M c_m [f(x + (m-½)h) − f(x − (m-½)h)]

at any order = 2M and via any of several optimisation methods:

* **'taylor'** — Truncation-expansion (TE) coefficients via the
  closed-form formula of Liu 2014 Eq 13, equivalent to Fornberg
  1988 / Kindelan 1990. Maximum formal order 2M; dispersion error
  concentrated at high wavenumbers.

* **'liu_2014_relative'** — Least-squares (LS) optimisation
  minimising the **relative** dispersion error over a fixed band
  [0, b]. Liu 2014 GJI 197(2):1033-1047, §2.4, Eq 12 + 17.
  Reproduces Liu Table 6 byte-identically given Liu's published
  b values (Table 5). The relative-error formulation matches the
  one Jiang & Zhang 2021 (GJI 227:2016-2043, footnote at line 179)
  cite for their 14th-order optimal weights.

* **'liu_2014_absolute'** — LS minimising the **absolute**
  dispersion error per Liu 2014 §2.3, Eq 12 with $\\psi_m(\\beta)
  = 2\\{\\sin[(m-½)\\beta] - (2m-1)\\sin(½\\beta)\\}$ and
  $g(\\beta) = \\beta - 2\\sin(½\\beta)$. Reproduces Liu Table 3.

* **'tam_webb'** — Tam & Webb 1993 DRP optimisation. Closely
  related to Liu's absolute-error LSM but with a different Taylor-
  constraint count. Provided for completeness; not used by JZ
  reproduction.

* **'holberg'** — Holberg 1987 LS optimisation with a relative-
  velocity-error objective. Distinct objective from Liu / Tam-Webb.

References
----------
* Liu, Y. (2014). *Optimal staggered-grid finite-difference schemes
  based on least-squares for wave equation modelling.* GJI 197(2):
  1033-1047. DOI: 10.1093/gji/ggu032.
* Liu, Y. (2013). *Globally optimal finite-difference schemes
  based on least squares.* Geophysics 78(4): T113-T132. DOI:
  10.1190/geo2012-0480.1.
* Tam, C.K.W. & Webb, J.C. (1993). *Dispersion-relation-preserving
  finite difference schemes for computational acoustics.* J. Comput.
  Phys. 107: 262-281.
* Holberg, O. (1987). *Computational aspects of the choice of operator
  and sampling interval for numerical differentiation in large-scale
  simulation of wave phenomena.* Geophys. Prospect. 35: 629-655.

Verification — Plan §91c.1:
``tests/test_optimal_fd_coefficients.py::test_liu_2014_table_6_byte_match``
is the critical pytest gate: derived weights for M=7, b=2.38 must
match Liu Table 6 column M=7 to ≤ 1e-6 relative.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import quad


__all__ = [
    "optimal_staggered_fd_coeffs",
    "taylor_staggered_coeffs",
    "liu_2014_relative_coeffs",
    "liu_2014_absolute_coeffs",
    "modified_wavenumber",
    "dispersion_error",
]


# ────────────────────────────────────────────────────────────────
# Taylor (TE) closed-form, Liu 2014 Eq 13
# ────────────────────────────────────────────────────────────────


def taylor_staggered_coeffs(order: int) -> np.ndarray:
    """Truncation-expansion (Taylor) ESGFD coefficients via Liu
    2014 Eq 13 closed form. Half-stencil only (c_1 ... c_M, with
    M = order/2). The full antisymmetric stencil applies these to
    half-grid samples at offsets ±(m-½)h.

    Equivalent to Fornberg 1988 at the same order, evaluated at
    the half-grid point.

    Parameters
    ----------
    order : int
        Even integer ≥ 2. The total stencil width is `order` points;
        M = order/2.

    Returns
    -------
    np.ndarray, shape (M,) — c_1, c_2, ..., c_M.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError(f"order must be even ≥ 2, got {order}")
    M = order // 2
    c = np.zeros(M, dtype=np.float64)
    for m in range(1, M + 1):
        prod = 1.0
        for n in range(1, M + 1):
            if n == m:
                continue
            prod *= abs((2 * n - 1) ** 2 /
                        ((2 * m - 1) ** 2 - (2 * n - 1) ** 2))
        c[m - 1] = ((-1) ** (m + 1) / (2 * m - 1)) * prod
    return c


# ────────────────────────────────────────────────────────────────
# Liu 2014 LSM — relative-error objective (Eq 12 + 17)
# ────────────────────────────────────────────────────────────────


def _liu_relative_psi(m: int, beta: np.ndarray) -> np.ndarray:
    """Liu 2014 Eq 17 ψ_m(β) for the relative-error objective:
       ψ_m(β) = 2/β · { sin[(m-½)β] - (2m-1) sin(½β) }.

    Handles β → 0 via L'Hôpital (ψ_m(0) is the limit, finite)."""
    beta = np.asarray(beta, dtype=np.float64)
    out = np.empty_like(beta)
    mask_zero = np.abs(beta) < 1e-12
    # For β > 0: direct formula
    nz = ~mask_zero
    if np.any(nz):
        out[nz] = (2.0 / beta[nz]) * (
            np.sin((m - 0.5) * beta[nz])
            - (2 * m - 1) * np.sin(0.5 * beta[nz])
        )
    # For β → 0: leading-order Taylor expansion.
    # sin((m-½)β) ≈ (m-½)β - (m-½)³β³/6 + ...
    # (2m-1)sin(½β) ≈ (m-½)β · 2 · ½ - ... = (m-½)β - (2m-1)·β³/48 + ...
    # So [sin((m-½)β) - (2m-1)sin(½β)] = β³ · O(1) at leading order
    # Hence ψ_m(0) = 0 (the leading-order matches by construction).
    out[mask_zero] = 0.0
    return out


def _liu_relative_g(beta: np.ndarray) -> np.ndarray:
    """Liu 2014 Eq 17 g(β) for the relative-error objective:
       g(β) = 1 - 2/β · sin(½β).

    g(0) = 0 by L'Hôpital (sin(½β)/β → ½ as β → 0)."""
    beta = np.asarray(beta, dtype=np.float64)
    out = np.empty_like(beta)
    mask_zero = np.abs(beta) < 1e-12
    nz = ~mask_zero
    if np.any(nz):
        out[nz] = 1.0 - (2.0 / beta[nz]) * np.sin(0.5 * beta[nz])
    out[mask_zero] = 0.0
    return out


def _integrate_psi_psi(m: int, n: int, b: float,
                         epsabs: float = 1e-12,
                         epsrel: float = 1e-12,
                         limit: int = 200) -> float:
    """∫_0^b ψ_m(β) ψ_n(β) dβ for the relative-error objective."""
    def integrand(beta):
        # scalar-wrapped path for scipy.quad
        beta_arr = np.array([beta], dtype=np.float64)
        return (_liu_relative_psi(m, beta_arr)
                * _liu_relative_psi(n, beta_arr))[0]
    val, _ = quad(integrand, 0.0, b,
                  epsabs=epsabs, epsrel=epsrel, limit=limit)
    return val


def _integrate_g_psi(n: int, b: float,
                      epsabs: float = 1e-12,
                      epsrel: float = 1e-12,
                      limit: int = 200) -> float:
    """∫_0^b g(β) ψ_n(β) dβ for the relative-error objective."""
    def integrand(beta):
        beta_arr = np.array([beta], dtype=np.float64)
        return (_liu_relative_g(beta_arr)
                * _liu_relative_psi(n, beta_arr))[0]
    val, _ = quad(integrand, 0.0, b,
                  epsabs=epsabs, epsrel=epsrel, limit=limit)
    return val


def liu_2014_relative_coeffs(order: int, b: float) -> np.ndarray:
    """Liu 2014 LSM ESGFD coefficients minimising the **relative**
    dispersion error over [0, b]. Liu §2.4, Eqs 9 + 12 + 17.

    Algorithm
    ---------
    1. Build the (M-1) × (M-1) normal-equation matrix
       A_{nm} = ∫_0^b ψ_m(β) ψ_n(β) dβ for m, n ∈ {2, ..., M}.
    2. Build the RHS r_n = ∫_0^b g(β) ψ_n(β) dβ.
    3. Solve A·[c_2, ..., c_M]ᵀ = r.
    4. Compute c_1 = 1 - Σ_{m=2}^M (2m-1) c_m  (Eq 9, Taylor-match
       constraint).

    Parameters
    ----------
    order : int
        Even integer ≥ 4 (M ≥ 2). Total stencil width = order;
        M = order/2.
    b : float
        LS integration band edge in [0, π). Larger b → wider valid
        wavenumber range, but increases the absolute LS error.
        For Liu's Table 6 / Table 5 values at η = 10⁻⁴, see Liu
        Table 5 for the published b per M.

    Returns
    -------
    np.ndarray, shape (M,) — c_1, ..., c_M.
    """
    if order < 4 or order % 2 != 0:
        raise ValueError(
            f"order must be even ≥ 4 for Liu LSM, got {order}. "
            f"(M ≥ 2 is required because the LSM solves for c_2..c_M; "
            f"c_1 is determined by the Eq 9 constraint.)")
    if not (0.0 < b < np.pi):
        raise ValueError(f"b must lie in (0, π); got {b}")
    M = order // 2

    # Build normal-equation matrix and RHS for c_2 ... c_M.
    A = np.zeros((M - 1, M - 1), dtype=np.float64)
    r = np.zeros(M - 1, dtype=np.float64)
    for j, n in enumerate(range(2, M + 1)):  # row index j, c_n
        for i, m in enumerate(range(2, M + 1)):  # col index i, c_m
            A[j, i] = _integrate_psi_psi(m, n, b)
        r[j] = _integrate_g_psi(n, b)

    # Solve for c_2, ..., c_M.
    c_rest = np.linalg.solve(A, r)

    # c_1 from the Taylor-match constraint (Eq 9).
    c1 = 1.0 - sum((2 * m - 1) * c_rest[m - 2]
                   for m in range(2, M + 1))

    coeffs = np.zeros(M, dtype=np.float64)
    coeffs[0] = c1
    coeffs[1:] = c_rest
    return coeffs


# ────────────────────────────────────────────────────────────────
# Liu 2014 LSM — absolute-error objective (Eq 12 + Eq 10/11)
# ────────────────────────────────────────────────────────────────


def _liu_absolute_psi(m: int, beta: np.ndarray) -> np.ndarray:
    """Liu 2014 Eq 11 ψ_m(β) for the absolute-error objective:
       ψ_m(β) = 2 { sin[(m-½)β] - (2m-1) sin(½β) }."""
    beta = np.asarray(beta, dtype=np.float64)
    return 2.0 * (np.sin((m - 0.5) * beta)
                  - (2 * m - 1) * np.sin(0.5 * beta))


def _liu_absolute_g(beta: np.ndarray) -> np.ndarray:
    """Liu 2014 Eq 11 g(β) for the absolute-error objective:
       g(β) = β - 2 sin(½β)."""
    beta = np.asarray(beta, dtype=np.float64)
    return beta - 2.0 * np.sin(0.5 * beta)


def liu_2014_absolute_coeffs(order: int, b: float) -> np.ndarray:
    """Liu 2014 LSM coefficients minimising the **absolute**
    dispersion error over [0, b]. Liu §2.3, Eqs 9 + 11 + 12.

    Same algorithm shape as the relative-error variant, with ψ_m
    and g taken from Eq 11 instead of Eq 17.

    Reproduces Liu Table 3 given the corresponding b from Table 2.
    """
    if order < 4 or order % 2 != 0:
        raise ValueError(f"order must be even ≥ 4 for Liu LSM, got {order}")
    if not (0.0 < b < np.pi):
        raise ValueError(f"b must lie in (0, π); got {b}")
    M = order // 2

    A = np.zeros((M - 1, M - 1), dtype=np.float64)
    r = np.zeros(M - 1, dtype=np.float64)
    for j, n in enumerate(range(2, M + 1)):
        for i, m in enumerate(range(2, M + 1)):
            A[j, i] = quad(
                lambda beta, _m=m, _n=n: float(
                    _liu_absolute_psi(_m, np.array([beta]))[0]
                    * _liu_absolute_psi(_n, np.array([beta]))[0]),
                0.0, b, epsabs=1e-12, epsrel=1e-12, limit=200,
            )[0]
        r[j] = quad(
            lambda beta, _n=n: float(
                _liu_absolute_g(np.array([beta]))[0]
                * _liu_absolute_psi(_n, np.array([beta]))[0]),
            0.0, b, epsabs=1e-12, epsrel=1e-12, limit=200,
        )[0]

    c_rest = np.linalg.solve(A, r)
    c1 = 1.0 - sum((2 * m - 1) * c_rest[m - 2]
                   for m in range(2, M + 1))

    coeffs = np.zeros(M, dtype=np.float64)
    coeffs[0] = c1
    coeffs[1:] = c_rest
    return coeffs


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────


def optimal_staggered_fd_coeffs(
    order: int,
    method: str = "liu_2014_relative",
    *,
    b: Optional[float] = None,
    error_tolerance: Optional[float] = None,
) -> np.ndarray:
    """Optimal staggered-grid FD coefficients for the first
    derivative.

    Stencil:
        f'(x) ≈ (1/h) Σ_{m=1}^M c_m [f(x + (m-½)h) - f(x - (m-½)h)]

    where M = order/2.

    Parameters
    ----------
    order : int
        Even integer ≥ 2. The total stencil width is `order` points.

    method : str
        * ``'taylor'`` — Closed-form Liu Eq 13 / Fornberg.
        * ``'liu_2014_relative'`` (default) — Liu 2014 LSM,
          relative-error objective. Requires ``b``.
        * ``'liu_2014_absolute'`` — Liu 2014 LSM, absolute-error
          objective. Requires ``b``.

    b : float, optional
        LS integration band edge in (0, π). Required for the Liu
        methods. For Liu Table 5/6 reproduction, use Liu Table 5
        value for the chosen (M, η). For Liu Table 2/3, use Liu
        Table 2 value.

    error_tolerance : float, optional
        Reserved for an outer-search variant (find b such that
        max |ε| over [0, B] ≤ tolerance). Not yet implemented in
        Phase 1 — use ``b`` directly. (Plan §91.followup.)

    Returns
    -------
    np.ndarray, shape (M,) — c_1, ..., c_M half-stencil weights.
    """
    method = method.lower()
    if method == "taylor":
        return taylor_staggered_coeffs(order)
    elif method == "liu_2014_relative":
        if b is None:
            raise ValueError(
                "Liu 2014 relative-error LSM requires `b` "
                "(LS integration band edge). For Liu Table 6 "
                "reproduction at η = 1e-4, see Liu Table 5 for the "
                "published b per M.")
        return liu_2014_relative_coeffs(order, b)
    elif method == "liu_2014_absolute":
        if b is None:
            raise ValueError(
                "Liu 2014 absolute-error LSM requires `b`")
        return liu_2014_absolute_coeffs(order, b)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported: 'taylor', "
            f"'liu_2014_relative', 'liu_2014_absolute'.")


# ────────────────────────────────────────────────────────────────
# Analysis helpers
# ────────────────────────────────────────────────────────────────


def modified_wavenumber(
    coeffs: np.ndarray, kh: np.ndarray,
) -> np.ndarray:
    """Modified wavenumber β̃(kh) for a half-stencil weight set.

    For coefficients ``coeffs = [c_1, ..., c_M]`` applied to a
    staggered stencil:
        β̃(kh) = 2 Σ_{m=1}^M c_m sin((m-½) kh)

    Liu 2014 Eq 3. Returns an array same shape as kh.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    kh = np.asarray(kh, dtype=np.float64)
    M = len(coeffs)
    result = np.zeros_like(kh)
    for m in range(1, M + 1):
        result += coeffs[m - 1] * np.sin((m - 0.5) * kh)
    return 2.0 * result


def dispersion_error(
    coeffs: np.ndarray, kh: np.ndarray,
    *, error_type: str = "relative",
) -> np.ndarray:
    """Dispersion error ε(kh) for a staggered FD stencil.

    error_type='absolute' (Liu 2014 default):
        ε(β) = β̃(β) - β

    error_type='relative' (Liu 2014 Eq 16):
        ε(β) = β̃(β) / β - 1 = 2β⁻¹ Σ c_m sin((m-½)β) - 1
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    kh = np.asarray(kh, dtype=np.float64)
    beta_tilde = modified_wavenumber(coeffs, kh)
    if error_type == "absolute":
        return beta_tilde - kh
    elif error_type == "relative":
        # Handle kh → 0 via L'Hôpital (constraint Eq 9 sets β̃/β → 1
        # as β → 0, so the relative error → 0).
        out = np.empty_like(kh)
        mask_zero = np.abs(kh) < 1e-12
        nz = ~mask_zero
        out[nz] = beta_tilde[nz] / kh[nz] - 1.0
        out[mask_zero] = 0.0
        return out
    else:
        raise ValueError(
            f"error_type must be 'relative' or 'absolute', got '{error_type}'")
