"""Anisotropic acoustic-elastic R/T coefficient at a planar
horizontal interface (acoustic upper / TTI elastic lower) — plan
§81 Stage C / Phase 2d.

This module implements the Mallick-Frazer 1990 [@mallick_frazer_1990]
machinery needed to compute the reflection coefficient $R(p_x,
\\omega)$ for an incident downgoing acoustic plane wave hitting a
TTI elastic lower layer. The three-component continuity at the
interface (normal displacement, normal traction, vanishing
tangential traction) gives a 3×3 linear system per (p_x, ω) pair
that we solve numerically.

Notation
--------
2D Voigt indices map as: 1 = x, 2 = z, 6 = xz. Standard
seismological convention uses 1=x, 3=z, 5=xz; we follow the
**standard 13/15/33/35/55** naming (NOT the codebase's 12/16/22/26/66
naming used by `compute_tti_stiffness_2d`). Inputs to this module
should pass the standard names; we provide a convenience wrapper
`stiffness_from_thomsen` that converts.

Convention
----------
Conv A (matching the rest of `00_common/reflectivity_2d.py` after
plan §81a fixes):

- $\\hat u(\\omega) = \\int u(t) e^{-i\\omega t} dt$,
  $u(t) = (1/(2\\pi)) \\int \\hat u(\\omega) e^{+i\\omega t} d\\omega$.
- Outgoing/causal plane wave: $u \\propto e^{-i\\omega(p_x x + q_z z)
  - i\\omega t}$? No wait, let me match the DWN code:
  $\\tilde G \\propto e^{-i q_w |z - z_s|} / (2j q_w)$ with
  $\\mathrm{Im}(q_w) \\le 0$.

Stage C / Phase 2d acoustic-elastic interface:

* Upper layer (acoustic, $\\rho_U, V_U$): incident downgoing
  potential $\\phi = e^{i\\omega(p_x x - q_w (z - z_a))}$ + reflected
  upgoing $R e^{i\\omega(p_x x + q_w (z - z_a))}$ where
  $q_w^2 = 1/V_U^2 - p_x^2$ with $\\mathrm{Im}(q_w) \\le 0$.
* Lower layer (TTI elastic, $\\rho_L, C_{ij}^L$): two transmitted
  downgoing modes (qP + qSV) with vertical slownesses $q_m$ and
  polarisations $(U_x^m, U_z^m)$ derived from the Christoffel
  matrix.
* Boundary conditions at $z = z_a$: continuity of $u_z$,
  continuity of normal traction $-p^U = \\sigma_{zz}^L$, vanishing
  tangential traction $\\sigma_{xz}^L = 0$.
* 3 unknowns: $R, T_{qP}, T_{qSV}$.
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Stiffness helpers
# ──────────────────────────────────────────────────────────────────


def stiffness_from_thomsen(Vp: float, Vs: float, rho: float,
                            epsilon: float, delta: float,
                            theta_rad: float) -> dict[str, float]:
    """TTI stiffness components in 2D standard naming (C11, C13,
    C15, C33, C35, C55) from Thomsen parameters + tilt.

    Wraps `bond_rotation.compute_tti_stiffness_2d` which returns
    the codebase's 12/16/22/26/66 naming; converts to the standard
    13/15/33/35/55 used in this module.

    Returns
    -------
    dict with keys "C11", "C13", "C15", "C33", "C35", "C55", "rho".
    """
    from bond_rotation import compute_tti_stiffness_2d
    # `compute_tti_stiffness_2d` returns (C11, C22, C12, C16, C26,
    # C66) where their indices use 1=x, 2=z, 6=xz. Convert:
    C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
        Vp, Vs, rho, epsilon, delta, theta_rad,
    )
    return {
        "C11": float(C11), "C33": float(C22),
        "C13": float(C12), "C15": float(C16),
        "C35": float(C26), "C55": float(C66),
        "rho": float(rho),
        # Stash original Thomsen params so the dipping-interface
        # wrapper can apply an additional Bond rotation by -β
        # without re-deriving the inverse transformation.
        "_thomsen": (float(Vp), float(Vs), float(rho),
                     float(epsilon), float(delta), float(theta_rad)),
    }


# ──────────────────────────────────────────────────────────────────
# Christoffel quartic for vertical slowness $q_z$
# ──────────────────────────────────────────────────────────────────


def christoffel_qz_quartic_coeffs(p_x, C: dict) -> np.ndarray:
    """For a 2D TTI medium with given $C_{ij}$ and density $\\rho$, return
    the 5 coefficients of the quartic polynomial in $q_z$ obtained
    from $\\det(\\Gamma - \\rho I) = 0$. Coefficients are in
    **ascending order** [D_0, D_1, D_2, D_3, D_4] (D_4 is the $q_z^4$
    term).

    The Christoffel matrix elements (standard 2D Voigt indexing):
        Γ_xx = C11 p_x² + 2 C15 p_x q_z + C55 q_z²
        Γ_zz = C55 p_x² + 2 C35 p_x q_z + C33 q_z²
        Γ_xz = C15 p_x² + (C13 + C55) p_x q_z + C35 q_z²

    Returns
    -------
    coeffs : ndarray with the last axis of length 5. Other axes
        broadcast over p_x. E.g. if p_x has shape (N,) the result
        has shape (N, 5).
    """
    p_x = np.asarray(p_x)
    C11 = C["C11"]; C13 = C["C13"]; C15 = C["C15"]
    C33 = C["C33"]; C35 = C["C35"]; C55 = C["C55"]
    rho = C["rho"]

    p2 = p_x * p_x

    # A = Γ_xx - ρ : A_0 + A_1 q + A_2 q² with
    A0 = C11 * p2 - rho
    A1 = 2 * C15 * p_x
    A2 = np.broadcast_to(np.asarray(C55, dtype=A0.dtype), A0.shape).copy()

    # B = Γ_zz - ρ
    B0 = C55 * p2 - rho
    B1 = 2 * C35 * p_x
    B2 = np.broadcast_to(np.asarray(C33, dtype=A0.dtype), A0.shape).copy()

    # G = Γ_xz : G_0 + G_1 q + G_2 q²
    G0 = C15 * p2
    G1 = (C13 + C55) * p_x
    G2 = np.broadcast_to(np.asarray(C35, dtype=A0.dtype), A0.shape).copy()

    # AB = A * B (polynomial product, degree 4)
    AB_0 = A0 * B0
    AB_1 = A0 * B1 + A1 * B0
    AB_2 = A0 * B2 + A1 * B1 + A2 * B0
    AB_3 = A1 * B2 + A2 * B1
    AB_4 = A2 * B2

    # G² = G * G (polynomial product, degree 4)
    G2_0 = G0 * G0
    G2_1 = 2 * G0 * G1
    G2_2 = 2 * G0 * G2 + G1 * G1
    G2_3 = 2 * G1 * G2
    G2_4 = G2 * G2

    # det = AB - G²
    D0 = AB_0 - G2_0
    D1 = AB_1 - G2_1
    D2 = AB_2 - G2_2
    D3 = AB_3 - G2_3
    D4 = AB_4 - G2_4

    return np.stack([D0, D1, D2, D3, D4], axis=-1)


def solve_quartic_roots(coeffs: np.ndarray) -> np.ndarray:
    """Batched quartic root solver via companion-matrix eigenvalues.

    Parameters
    ----------
    coeffs : ndarray of shape (..., 5), ascending order
        [D_0, D_1, D_2, D_3, D_4] for ``D_0 + D_1 q + ... + D_4 q^4
        = 0``.

    Returns
    -------
    roots : ndarray of shape (..., 4) complex.
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    flat_shape = coeffs.shape[:-1]
    flat = coeffs.reshape(-1, 5)  # (N, 5)
    N = flat.shape[0]

    # Normalise: divide by D_4 so monic. If D_4 ≈ 0, the polynomial
    # is degenerate (lower-degree). We assume D_4 ≠ 0 for our 2D
    # TTI case (D_4 = C_33 C_55 - C_35² > 0 for physical media).
    D4 = flat[:, 4]
    monic = flat[:, :4] / D4[:, None]
    # monic = [d_0, d_1, d_2, d_3] with poly = q^4 + d_3 q^3 + d_2 q^2 + d_1 q + d_0

    # Companion matrix: for poly q^4 + d_3 q^3 + d_2 q^2 + d_1 q + d_0,
    # companion is
    #   [[0, 0, 0, -d_0],
    #    [1, 0, 0, -d_1],
    #    [0, 1, 0, -d_2],
    #    [0, 0, 1, -d_3]]
    companion = np.zeros((N, 4, 4), dtype=np.complex128)
    companion[:, 0, 3] = -monic[:, 0]
    companion[:, 1, 3] = -monic[:, 1]
    companion[:, 2, 3] = -monic[:, 2]
    companion[:, 3, 3] = -monic[:, 3]
    companion[:, 1, 0] = 1.0
    companion[:, 2, 1] = 1.0
    companion[:, 3, 2] = 1.0

    # Batched eigenvalues
    roots = np.linalg.eigvals(companion)  # (N, 4) complex
    return roots.reshape(*flat_shape, 4)


def select_downgoing_qz(roots: np.ndarray,
                         omega: np.ndarray | None = None,
                         ) -> np.ndarray:
    """From 4 complex roots, select the 2 downgoing modes.

    Convention (matches Conv A in `reflectivity_2d.py`): a wave
    propagating downward from the interface ($z_\\text{iface} - z
    > 0$) has propagator $e^{-i\\omega q_z (z_\\text{iface} - z)}$
    with magnitude $e^{\\mathrm{Im}(\\omega q_z)(z_\\text{iface}
    - z)}$. The **physical** (decaying or oscillating, not
    growing) modes have $\\mathrm{Im}(\\omega q_z) \\le 0$.

    For real $\\omega > 0$ this reduces to $\\mathrm{Im}(q_z)
    \\le 0$ (which the original implementation used). For
    $\\omega < 0$ — which occurs in the negative-frequency half of
    the DFT — the same criterion in $q_z$ alone picks the WRONG
    branch (a growing mode); the correct criterion **requires
    accounting for $\\omega$'s sign**. Plan §82 Stage E1.

    Parameters
    ----------
    roots : ndarray of shape (..., 4) — complex roots from
        `solve_quartic_roots`.
    omega : ndarray of shape (...) or broadcastable to ``roots``
        without the last axis — complex frequency. If ``None``
        (legacy callers), falls back to the original $\\mathrm{Im}
        (q_z) \\le 0$ rule which only works for $\\omega > 0$.

    Returns
    -------
    sorted_roots[..., :2] : the 2 roots with the smallest (most
        negative) $\\mathrm{Im}(\\omega q_z)$ — i.e., the
        decaying/outgoing-downward modes.
    """
    roots = np.asarray(roots)
    if omega is None:
        # Legacy fallback: pick by Im(q_z) only. Correct for ω > 0
        # but wrong for ω < 0. Kept for backward compatibility with
        # callers that don't yet pass omega.
        idx = np.argsort(roots.imag, axis=-1)
    else:
        omega_b = np.broadcast_to(
            np.asarray(omega)[..., None], roots.shape)
        idx = np.argsort((omega_b * roots).imag, axis=-1)
    sorted_roots = np.take_along_axis(roots, idx, axis=-1)
    return sorted_roots[..., :2]


def tti_polarisation(p_x, q_z, C: dict) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvector (U_x, U_z) of (Γ - ρ I) corresponding to the
    given q_z root. Vectorised: broadcasts over leading axes.

    Computes the 2×2 Christoffel matrix and finds the null-space
    direction via the row of (Γ - ρ I) with larger magnitude
    diagonal element (for numerical stability).

    Returns (U_x, U_z) with unit Euclidean norm.
    """
    C11 = C["C11"]; C13 = C["C13"]; C15 = C["C15"]
    C33 = C["C33"]; C35 = C["C35"]; C55 = C["C55"]
    rho = C["rho"]

    Gxx = C11 * p_x ** 2 + 2 * C15 * p_x * q_z + C55 * q_z ** 2 - rho
    Gzz = C55 * p_x ** 2 + 2 * C35 * p_x * q_z + C33 * q_z ** 2 - rho
    Gxz = C15 * p_x ** 2 + (C13 + C55) * p_x * q_z + C35 * q_z ** 2

    # Use row with larger absolute pivot for stability.
    use_row0 = np.abs(Gxx) >= np.abs(Gzz)
    # Row 0: (Gxx, Gxz). Null: U ∝ (-Gxz, Gxx).
    Ux0 = -Gxz
    Uz0 = Gxx
    # Row 1: (Gxz, Gzz). Null: U ∝ (Gzz, -Gxz).
    Ux1 = Gzz
    Uz1 = -Gxz

    Ux = np.where(use_row0, Ux0, Ux1)
    Uz = np.where(use_row0, Uz0, Uz1)

    # Normalise
    norm = np.sqrt(np.abs(Ux) ** 2 + np.abs(Uz) ** 2)
    norm = np.where(norm > 0, norm, 1.0)  # avoid /0 for nullspace == 0
    return Ux / norm, Uz / norm


# ──────────────────────────────────────────────────────────────────
# 3×3 acoustic-elastic R/T system
# ──────────────────────────────────────────────────────────────────


def acoustic_elastic_reflection(p_x, omega, qw,
                                 rho_upper: float, V_upper: float,
                                 C_lower: dict,
                                 return_transmitted: bool = False):
    """Compute the reflection coefficient $R$ for an incident
    downgoing acoustic plane wave hitting a TTI elastic lower
    layer.

    Solves the 3×3 system at the horizontal interface $z=0$
    (3 boundary conditions, 3 unknowns: $R, T_{qP}, T_{qSV}$).

    Derivation (plan §81 Stage C2; corrected 2026-05-16):

    Convention A time dep $e^{+i\\omega t}$ throughout; downgoing
    plane waves have spatial $e^{+ik_z z}$ with $k_z > 0$. Upper
    layer parameterised in **displacement** amplitude (incident
    $u_z = 1$); lower layer in mode-displacement amplitude.

    * BC 1 ($u_z$ continuity): $1 - R = U_z^{qP} T_{qP} + U_z^{qSV} T_{qSV}$
    * BC 2 ($\\sigma_{zz}^L = -p^U$): with the strain expressions
      $\\epsilon_{zz} = ik_z U_z$, $2\\epsilon_{xz} = i(k_x U_z + k_z U_x)$,
      $\\epsilon_{xx} = ik_x U_x$ — yields
      $k_w \\Sigma_{zz}^{qP} T_{qP} + k_w \\Sigma_{zz}^{qSV} T_{qSV} -
      \\rho_U \\omega^2 R = \\rho_U \\omega^2$
      where $\\Sigma_{zz}^m = C_{13} k_x U_x + C_{33} k_z U_z
      + C_{35}(k_x U_z + k_z U_x)$ and $k_w = \\omega q_w$.
    * BC 3 ($\\sigma_{xz}^L = 0$):
      $\\Sigma_{xz}^{qP} T_{qP} + \\Sigma_{xz}^{qSV} T_{qSV} = 0$
      where $\\Sigma_{xz}^m = C_{15} k_x U_x + C_{35} k_z U_z
      + C_{55}(k_x U_z + k_z U_x)$.

    Verified against the acoustic-acoustic normal-incidence
    formula $R = (Z_L - Z_U)/(Z_L + Z_U)$ in the isotropic limit
    (see `tests/test_anisotropic_rt.py`).

    Parameters
    ----------
    p_x : ndarray (or scalar) — horizontal slowness, complex
        allowed.
    omega : ndarray (or scalar) — complex frequency
        (Bouchon-damped, $\\omega - ia$).
    qw : ndarray (or scalar) — upper-layer vertical slowness, with
        $\\mathrm{Im}(q_w) \\le 0$ for outgoing convention.
    rho_upper, V_upper : scalar — upper-layer density and acoustic
        speed.
    C_lower : dict with keys C11, C13, C15, C33, C35, C55, rho —
        TTI lower-layer stiffness (standard Voigt naming).

    Returns
    -------
    R : ndarray (default; same shape as broadcast of p_x, omega, qw)
        — complex reflection coefficient.
    full : dict (when ``return_transmitted=True``)
        Keys: ``R``, ``T_qP``, ``T_qSV`` (reflection + two
        transmission amplitudes), ``qz_qP``, ``qz_qSV`` (lower-
        layer vertical slownesses for the two downgoing modes),
        ``Ux_qP``, ``Uz_qP``, ``Ux_qSV``, ``Uz_qSV`` (mode
        polarisation components). Lets the DWN engine (plan §82
        Stage E1) synthesise the lower-layer transmitted wavefield
        without redoing the Christoffel quartic + 3×3 solve.
    """
    p_x_arr = np.asarray(p_x, dtype=np.complex128)
    omega_arr = np.asarray(omega, dtype=np.complex128)
    qw_arr = np.asarray(qw, dtype=np.complex128)
    # Broadcast all to common shape
    p_x_b, omega_b, qw_b = np.broadcast_arrays(p_x_arr, omega_arr, qw_arr)

    # Find lower-layer vertical slowness roots and select 2 downgoing.
    # The selector needs ω to apply the correct Im(ω q_z) ≤ 0
    # criterion (handles ω < 0 in the DFT correctly).
    coeffs = christoffel_qz_quartic_coeffs(p_x_b, C_lower)  # (..., 5)
    roots = solve_quartic_roots(coeffs)  # (..., 4)
    qz_pair = select_downgoing_qz(roots, omega=omega_b)  # (..., 2)
    qz_qP = qz_pair[..., 0]
    qz_qSV = qz_pair[..., 1]

    # Polarisations (U_x, U_z) per mode
    Ux_qP, Uz_qP = tti_polarisation(p_x_b, qz_qP, C_lower)
    Ux_qSV, Uz_qSV = tti_polarisation(p_x_b, qz_qSV, C_lower)

    # Wavenumber form: k_x = ω p_x, k_z^m = ω q_z^m, k_w = ω q_w
    k_x = omega_b * p_x_b
    kz_qP = omega_b * qz_qP
    kz_qSV = omega_b * qz_qSV
    k_w = omega_b * qw_b

    # Stress amplitude coefficients per mode (wavenumber form):
    # Σ_zz^m = C13 k_x U_x + C33 k_z U_z + C35 (k_x U_z + k_z U_x)
    # Σ_xz^m = C15 k_x U_x + C35 k_z U_z + C55 (k_x U_z + k_z U_x)
    C13 = C_lower["C13"]; C15 = C_lower["C15"]
    C33 = C_lower["C33"]; C35 = C_lower["C35"]; C55 = C_lower["C55"]

    def stress_coeffs(kx, kz, Ux, Uz):
        s_zz = (C13 * kx * Ux + C33 * kz * Uz
                + C35 * (kx * Uz + kz * Ux))
        s_xz = (C15 * kx * Ux + C35 * kz * Uz
                + C55 * (kx * Uz + kz * Ux))
        return s_zz, s_xz

    Szz_qP, Sxz_qP = stress_coeffs(k_x, kz_qP, Ux_qP, Uz_qP)
    Szz_qSV, Sxz_qSV = stress_coeffs(k_x, kz_qSV, Ux_qSV, Uz_qSV)

    # Build 3×3 system. Variables: x = (R, T_qP, T_qSV)^T
    # BC 1 (u_z continuity at z=0):
    #     1 R + U_z^qP T_qP + U_z^qSV T_qSV = 1
    # BC 2 (σ_zz^L = -p^U at z=0):
    #     -ρ_U ω² R + k_w Σ_zz^qP T_qP + k_w Σ_zz^qSV T_qSV = ρ_U ω²
    # BC 3 (σ_xz^L = 0 at z=0):
    #     0 R + Σ_xz^qP T_qP + Σ_xz^qSV T_qSV = 0
    shape = p_x_b.shape
    A = np.zeros(shape + (3, 3), dtype=np.complex128)
    b = np.zeros(shape + (3,), dtype=np.complex128)

    rho_omega2 = rho_upper * omega_b * omega_b

    A[..., 0, 0] = 1.0
    A[..., 0, 1] = Uz_qP
    A[..., 0, 2] = Uz_qSV
    A[..., 1, 0] = -rho_omega2
    A[..., 1, 1] = k_w * Szz_qP
    A[..., 1, 2] = k_w * Szz_qSV
    A[..., 2, 0] = 0.0
    A[..., 2, 1] = Sxz_qP
    A[..., 2, 2] = Sxz_qSV

    b[..., 0] = 1.0
    b[..., 1] = rho_omega2
    b[..., 2] = 0.0

    # Batched 3×3 solve. Numpy's solve gufunc requires b padded to
    # (..., 3, 1) to avoid shape ambiguity over broadcast axes.
    sol = np.linalg.solve(A, b[..., None])[..., 0]  # (..., 3)
    R = sol[..., 0]

    if return_transmitted:
        return {
            "R": R,
            "T_qP": sol[..., 1],
            "T_qSV": sol[..., 2],
            "qz_qP": qz_qP,
            "qz_qSV": qz_qSV,
            "Ux_qP": Ux_qP,
            "Uz_qP": Uz_qP,
            "Ux_qSV": Ux_qSV,
            "Uz_qSV": Uz_qSV,
        }
    return R


def _select_upgoing_qz(roots, omega):
    """Pick the 2 'upgoing' roots from the 4 Christoffel qz values.

    Dual of ``select_downgoing_qz``: returns roots with
    ``Im(omega * qz) >= 0`` (the modes whose propagator
    ``exp(+i omega qz z)`` is bounded for the up-propagation
    parameterisation in plan §90.b).

    The 4 quartic roots come in 2 down + 2 up pairs (qP + qSV
    on each side); we keep the 2 not already in the downgoing
    selection.

    Parameters
    ----------
    roots : ndarray (..., 4) — 4 Christoffel qz roots.
    omega : ndarray (...,) — complex frequency (Bouchon-damped).

    Returns
    -------
    qz_pair : ndarray (..., 2) — 2 upgoing qz values, sorted by
        descending |U_z·U_x| ratio (qP-like first, qSV-like second).
        Convention matches ``select_downgoing_qz`` ordering.
    """
    # Identify which roots are downgoing (Im(ω q) ≤ 0)
    omega_b = np.broadcast_to(omega[..., None], roots.shape)
    is_down = np.imag(omega_b * roots) <= 0
    # Each (p_x, ω) bin should have exactly 2 down + 2 up; pick the
    # 2 with Im(ω q) > 0 strictly. Use lexsort to keep the up modes
    # in the same order as the matching down pair.
    n_down_per_bin = is_down.sum(axis=-1)
    # Most bins have 2-down/2-up; for edge cases where roots are on
    # the imaginary axis, fall back to the complement of the
    # downgoing selection.
    qz_down = select_downgoing_qz(roots, omega=omega)  # (..., 2)
    # For each root, mark whether it's in qz_down.
    diff_close = np.abs(roots[..., :, None] - qz_down[..., None, :])  # (..., 4, 2)
    is_in_down = (diff_close < 1e-12).any(axis=-1)  # (..., 4)
    # Upgoing roots = not in downgoing.
    is_up = ~is_in_down
    # For each (...,) bin pick 2 up roots; we expect exactly 2.
    # Use sort-then-pick with the up mask as the primary key.
    sort_key = np.where(is_up, 0, 1)   # 0 for up, 1 for down
    sort_idx = np.argsort(sort_key, axis=-1, kind='stable')
    sorted_roots = np.take_along_axis(roots, sort_idx, axis=-1)
    qz_up_pair = sorted_roots[..., :2]   # first 2 are up
    return qz_up_pair


def acoustic_elastic_elastic_stack_reflection(
    p_x, omega, qw,
    rho_upper: float, V_upper: float,
    C_middle: dict, h_middle: float,
    C_lower: dict,
    return_full: bool = False,
):
    """Plan §90 — 3-region acoustic-elastic-elastic stack R/T.

    Solves the 7×7 boundary system at two interfaces:
    - Interface 1 at z=h_middle (acoustic upper ↔ elastic VTI middle)
    - Interface 2 at z=0 (elastic VTI middle ↔ elastic TTI lower)

    Region structure (z increases UPWARD; interface 1 at top, 2 at bottom):
    - Region 1 (z > h_middle): acoustic upper, semi-infinite.
      Incident-down + reflected-up; 1 unknown (R).
    - Region 2 (0 < z < h_middle): elastic VTI middle, finite thickness.
      4 modes (qP↓, qSV↓, qP↑, qSV↑); 4 unknowns.
    - Region 3 (z < 0): elastic TTI lower, semi-infinite.
      2 transmitted-down modes (qP, qSV); 2 unknowns.

    Total: 7 unknowns + 7 BCs (3 at interface 1: u_z, σ_zz, σ_xz=0;
    4 at interface 2: u_z, u_x, σ_zz, σ_xz).

    Amplitudes referenced to the **near edge** of each mode for numeric
    stability:
    - Down modes in Region 2: amplitude at z=0 (bottom of middle).
      Propagator to z=h: exp(-i ω qz↓ h) — bounded since Im(ω qz↓) ≤ 0.
    - Up modes in Region 2: amplitude at z=h (top of middle).
      Propagator to z=0: exp(+i ω qz↑ h) — bounded since Im(ω qz↑) ≥ 0.

    h_middle → 0 limit (validated by `test_smeared_dwn.py`):
    when C_middle == C_lower, the system reduces to the single-interface
    `acoustic_elastic_reflection` result to fp64 precision.

    Parameters
    ----------
    p_x : ndarray — horizontal slowness, complex allowed.
    omega : ndarray — complex frequency (Bouchon-damped).
    qw : ndarray — upper-layer vertical slowness, Im(qw) ≤ 0.
    rho_upper, V_upper : scalar — upper-layer (acoustic) parameters.
    C_middle : dict with keys C11, C13, C15, C33, C35, C55 — middle-layer
        (VTI or TTI) Voigt stiffness in lab frame. For the §90 hybrid
        path, this is the JZ Bond-rotated VTI cell-effective tensor.
    h_middle : float — middle-layer thickness (in same units as ω/V).
    C_lower : dict with same keys — lower-layer TTI stiffness.

    Returns
    -------
    R : ndarray (default; same shape as broadcast of p_x, omega, qw).
    full : dict (when ``return_full=True``) with keys:
        'R', 'A_pd', 'A_sd', 'A_pu', 'A_su', 'T_qP', 'T_qSV',
        'qz_p_d', 'qz_s_d', 'qz_p_u', 'qz_s_u',
        'qz_qP_lower', 'qz_qSV_lower',
        'Ux_*', 'Uz_*' (for all 6 modes — 4 middle + 2 lower).
    """
    p_x_arr = np.asarray(p_x, dtype=np.complex128)
    omega_arr = np.asarray(omega, dtype=np.complex128)
    qw_arr = np.asarray(qw, dtype=np.complex128)
    p_x_b, omega_b, qw_b = np.broadcast_arrays(p_x_arr, omega_arr, qw_arr)
    shape = p_x_b.shape

    # ── Christoffel quartic roots: 4 for middle, 4 for lower ──
    coeffs_m = christoffel_qz_quartic_coeffs(p_x_b, C_middle)
    coeffs_l = christoffel_qz_quartic_coeffs(p_x_b, C_lower)
    roots_m = solve_quartic_roots(coeffs_m)
    roots_l = solve_quartic_roots(coeffs_l)

    # Pair into 2 down + 2 up for the middle; only 2 down needed for lower
    qz_m_down = select_downgoing_qz(roots_m, omega=omega_b)
    qz_m_up = _select_upgoing_qz(roots_m, omega=omega_b)
    qz_l_down = select_downgoing_qz(roots_l, omega=omega_b)

    qz_p_d = qz_m_down[..., 0]
    qz_s_d = qz_m_down[..., 1]
    qz_p_u = qz_m_up[..., 0]
    qz_s_u = qz_m_up[..., 1]
    qz_p_l = qz_l_down[..., 0]
    qz_s_l = qz_l_down[..., 1]

    # ── Polarisations (U_x, U_z) per mode ──
    Ux_p_d, Uz_p_d = tti_polarisation(p_x_b, qz_p_d, C_middle)
    Ux_s_d, Uz_s_d = tti_polarisation(p_x_b, qz_s_d, C_middle)
    Ux_p_u, Uz_p_u = tti_polarisation(p_x_b, qz_p_u, C_middle)
    Ux_s_u, Uz_s_u = tti_polarisation(p_x_b, qz_s_u, C_middle)
    Ux_p_l, Uz_p_l = tti_polarisation(p_x_b, qz_p_l, C_lower)
    Ux_s_l, Uz_s_l = tti_polarisation(p_x_b, qz_s_l, C_lower)

    # ── Wavenumbers ──
    k_x = omega_b * p_x_b
    k_w = omega_b * qw_b
    kz_p_d = omega_b * qz_p_d
    kz_s_d = omega_b * qz_s_d
    kz_p_u = omega_b * qz_p_u
    kz_s_u = omega_b * qz_s_u
    kz_p_l = omega_b * qz_p_l
    kz_s_l = omega_b * qz_s_l

    # ── Stress coefficient helpers ──
    def _stress(C, kx, kz, Ux, Uz):
        s_zz = (C["C13"] * kx * Ux + C["C33"] * kz * Uz
                + C["C35"] * (kx * Uz + kz * Ux))
        s_xz = (C["C15"] * kx * Ux + C["C35"] * kz * Uz
                + C["C55"] * (kx * Uz + kz * Ux))
        return s_zz, s_xz

    Szz_m_pd, Sxz_m_pd = _stress(C_middle, k_x, kz_p_d, Ux_p_d, Uz_p_d)
    Szz_m_sd, Sxz_m_sd = _stress(C_middle, k_x, kz_s_d, Ux_s_d, Uz_s_d)
    Szz_m_pu, Sxz_m_pu = _stress(C_middle, k_x, kz_p_u, Ux_p_u, Uz_p_u)
    Szz_m_su, Sxz_m_su = _stress(C_middle, k_x, kz_s_u, Ux_s_u, Uz_s_u)
    Szz_l_p, Sxz_l_p = _stress(C_lower, k_x, kz_p_l, Ux_p_l, Uz_p_l)
    Szz_l_s, Sxz_l_s = _stress(C_lower, k_x, kz_s_l, Ux_s_l, Uz_s_l)

    # ── Phase factors and parameterisation ──
    # Match Phase 2d's convention (downgoing transmitted-mode
    # profile $\propto e^{+i\omega q^\downarrow z}$ in lab frame
    # z). Within the middle slab, in primed coords z' = z -
    # z_iface_lower ∈ [0, h]:
    #
    #   down-mode profile: e^{+iω q^↓ z'} — magnitude grows with z'
    #     (larger near top; source side). Reference at z'=h (top).
    #     Propagator from ref (z'=h) to z'=0: e^{-iω q^↓ h} (≤ 1).
    #
    #   up-mode profile: e^{+iω q^↑ z'} — magnitude decays with z'
    #     (larger near bottom; energy from below). Reference at
    #     z'=0 (bottom). Propagator from ref to z'=h: e^{+iω q^↑ h}
    #     (≤ 1 since Im(ω q^↑) ≥ 0).
    e_pd = np.exp(-1j * omega_b * qz_p_d * h_middle)  # down: at z'=0
    e_sd = np.exp(-1j * omega_b * qz_s_d * h_middle)
    e_pu = np.exp(+1j * omega_b * qz_p_u * h_middle)  # up: at z'=h
    e_su = np.exp(+1j * omega_b * qz_s_u * h_middle)

    # ── 7×7 system assembly ──
    # Unknowns: x = (R, A_pd, A_sd, A_pu, A_su, T_qP, T_qSV)^T
    # Down-mode reference at z'=h (top); up-mode reference at
    # z'=0 (bottom).
    # Rows: BC1-3 at z=h, BC4-7 at z=0.
    A = np.zeros(shape + (7, 7), dtype=np.complex128)
    b = np.zeros(shape + (7,), dtype=np.complex128)
    rho_omega2 = rho_upper * omega_b * omega_b

    # BC1: u_z continuity at z=h. Down-mode contributes
    # A_pd · U_z (no propagator at ref); up-mode contributes
    # A_pu · U_z · e_pu (propagator from z'=0 ref to z'=h).
    A[..., 0, 0] = 1.0
    A[..., 0, 1] = Uz_p_d
    A[..., 0, 2] = Uz_s_d
    A[..., 0, 3] = Uz_p_u * e_pu
    A[..., 0, 4] = Uz_s_u * e_su
    b[..., 0] = 1.0

    # BC2: σ_zz^(2)(z=h) = -p^(1)(z=h)
    A[..., 1, 0] = -rho_omega2
    A[..., 1, 1] = k_w * Szz_m_pd
    A[..., 1, 2] = k_w * Szz_m_sd
    A[..., 1, 3] = k_w * Szz_m_pu * e_pu
    A[..., 1, 4] = k_w * Szz_m_su * e_su
    b[..., 1] = rho_omega2

    # BC3: σ_xz^(2)(z=h) = 0
    A[..., 2, 1] = Sxz_m_pd
    A[..., 2, 2] = Sxz_m_sd
    A[..., 2, 3] = Sxz_m_pu * e_pu
    A[..., 2, 4] = Sxz_m_su * e_su

    # BC4: u_z continuity at z=0. Down-mode contributes
    # A_pd · U_z · e_pd (propagator from z'=h ref to z'=0);
    # up-mode contributes A_pu · U_z (no propagator at ref).
    A[..., 3, 1] = Uz_p_d * e_pd
    A[..., 3, 2] = Uz_s_d * e_sd
    A[..., 3, 3] = Uz_p_u
    A[..., 3, 4] = Uz_s_u
    A[..., 3, 5] = -Uz_p_l
    A[..., 3, 6] = -Uz_s_l

    # BC5: u_x continuity at z=0
    A[..., 4, 1] = Ux_p_d * e_pd
    A[..., 4, 2] = Ux_s_d * e_sd
    A[..., 4, 3] = Ux_p_u
    A[..., 4, 4] = Ux_s_u
    A[..., 4, 5] = -Ux_p_l
    A[..., 4, 6] = -Ux_s_l

    # BC6: σ_zz continuity at z=0
    A[..., 5, 1] = Szz_m_pd * e_pd
    A[..., 5, 2] = Szz_m_sd * e_sd
    A[..., 5, 3] = Szz_m_pu
    A[..., 5, 4] = Szz_m_su
    A[..., 5, 5] = -Szz_l_p
    A[..., 5, 6] = -Szz_l_s

    # BC7: σ_xz continuity at z=0
    A[..., 6, 1] = Sxz_m_pd * e_pd
    A[..., 6, 2] = Sxz_m_sd * e_sd
    A[..., 6, 3] = Sxz_m_pu
    A[..., 6, 4] = Sxz_m_su
    A[..., 6, 5] = -Sxz_l_p
    A[..., 6, 6] = -Sxz_l_s

    sol = np.linalg.solve(A, b[..., None])[..., 0]
    R = sol[..., 0]

    if return_full:
        return {
            "R": R,
            "A_pd": sol[..., 1],
            "A_sd": sol[..., 2],
            "A_pu": sol[..., 3],
            "A_su": sol[..., 4],
            "T_qP": sol[..., 5],
            "T_qSV": sol[..., 6],
            "qz_p_d": qz_p_d, "qz_s_d": qz_s_d,
            "qz_p_u": qz_p_u, "qz_s_u": qz_s_u,
            "qz_qP_lower": qz_p_l, "qz_qSV_lower": qz_s_l,
            "Ux_p_d": Ux_p_d, "Uz_p_d": Uz_p_d,
            "Ux_s_d": Ux_s_d, "Uz_s_d": Uz_s_d,
            "Ux_p_u": Ux_p_u, "Uz_p_u": Uz_p_u,
            "Ux_s_u": Ux_s_u, "Uz_s_u": Uz_s_u,
            "Ux_qP_lower": Ux_p_l, "Uz_qP_lower": Uz_p_l,
            "Ux_qSV_lower": Ux_s_l, "Uz_qSV_lower": Uz_s_l,
        }
    return R


__all__ = [
    "stiffness_from_thomsen",
    "christoffel_qz_quartic_coeffs",
    "solve_quartic_roots",
    "select_downgoing_qz",
    "tti_polarisation",
    "acoustic_elastic_reflection",
    "acoustic_elastic_elastic_stack_reflection",
]
