"""Closed-form 2D acoustic semi-analytical reference (plan §80 Phase 2a).

This is the **acoustic image-method** reference — a strictly
closed-form analytical solution for the upper-water-layer wavefield
under a dipping rigid-impedance interface. It does NOT include
lower-layer TTI transmitted-qP / converted-qSV (those require the
full Phase 2 reflectivity engine in `00_common/reflectivity_2d.py`,
which is not yet built).

What this module computes
-------------------------

For an acoustic source in 2D water (homogeneous, isotropic,
$V_p$ = 1.5 km/s) with a planar dipping interface at angle $\\beta$
to the lower TTI:

* **Direct wave** $u_d(x, y, t)$ = source-wavelet-convolved 2D
  acoustic Green's function $G_{2D}(r, t)$, where $r$ is the
  source-receiver distance.
* **Reflected wave** $u_r(x, y, t)$ = same Green's function but
  evaluated at the **mirror image** of the source across the
  dipping interface, scaled by the acoustic-acoustic
  impedance-contrast reflection coefficient
  $R = (Z_\\text{TTI} - Z_\\text{water})/(Z_\\text{TTI} + Z_\\text{water})$.
* **Total upper-layer wavefield**: $u(x, y, t) = u_d + u_r$,
  evaluated only above the interface (below = lower TTI, not
  covered by this module).

Mathematical foundation
-----------------------

The 2D acoustic Green's function for
$\\partial_t^2 u - c^2 \\nabla^2 u = \\delta(\\mathbf{x})\\delta(t)$
is the **half-derivative of the 3D Green's function** along $t$
(@pratt_1990 line-source projection). Equivalently, the closed
form is

$$G_{2D}(r, t) = \\frac{1}{2\\pi} \\cdot \\frac{H(t - r/c)}{\\sqrt{t^2 - r^2/c^2}}$$

with $H$ the Heaviside step function. The convolution with a source
wavelet $s(\\tau)$ over $\\tau \\in [0, t - r/c]$ has an integrable
$\\tau^{-1/2}$ singularity at the upper limit; we resolve it
analytically via the substitution
$\\tau = t - \\sqrt{u^2 + r^2/c^2}$, which renders the integrand
smooth:

$$u(\\mathbf{x}, t) = \\frac{1}{2\\pi} \\int_0^{U}
   \\frac{s(t - \\sqrt{u^2 + r^2/c^2})}
        {\\sqrt{u^2 + r^2/c^2}} \\, du,
   \\quad U = \\sqrt{t^2 - r^2/c^2}.$$

This integral is computed by fully vectorised trapezoidal rule
on a fixed $u$-grid; no quadrature singularity remains.

Mirror image across a dipping line in 2D Cartesian (with positive
$y$ upward, dip $\\beta$ measured from horizontal, interface
passing through anchor point $(x_a, y_a)$ and tilting *down to
the right* for $\\beta > 0$):

Interface line: $y = y_a - \\tan\\beta \\cdot (x - x_a)$, i.e.
$\\tan\\beta \\cdot x + y - (y_a + \\tan\\beta \\cdot x_a) = 0$.

Signed perpendicular distance of source $(x_s, y_s)$ from the
line: $d = \\cos\\beta \\cdot (y_s - y_a + \\tan\\beta (x_s - x_a))$.
Mirror image: $(x_s', y_s') = (x_s - 2 d \\sin\\beta,
y_s - 2 d \\cos\\beta)$.

Plan reference: §80 Phase 2a (the cheap closed-form precursor to
the full reflectivity engine).
"""
from __future__ import annotations

import numpy as np


# ─── 2D acoustic Green's function snapshot ────────────────────────


def ricker_wavelet(t: np.ndarray, f0: float,
                   t_delay: float | None = None) -> np.ndarray:
    """Standard Ricker wavelet at peak frequency ``f0`` (Hz).

    $s(t) = (1 - 2\\pi^2 f_0^2 (t - t_d)^2) \\exp(-\\pi^2 f_0^2 (t - t_d)^2)$.

    Default $t_d = 1.5/f_0$ matches the convention used in
    `00_common/source.py`.
    """
    if t_delay is None:
        t_delay = 1.5 / f0
    arg = np.pi * f0 * (t - t_delay)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def acoustic_2d_green_snapshot(x_grid: np.ndarray,
                               y_grid: np.ndarray,
                               t: float,
                               x_source: float,
                               y_source: float,
                               c: float,
                               source_func,
                               n_u: int = 400) -> np.ndarray:
    """Snapshot of the 2D acoustic Green's function $G_{2D} * s$ at
    fixed time ``t`` over a 2D rectangular grid of receiver points.

    Parameters
    ----------
    x_grid, y_grid : 1D arrays — receiver coordinates (km).
    t : snapshot time (s).
    x_source, y_source : source location (km).
    c : acoustic velocity (km/s).
    source_func : callable, ``source_func(tau) -> s(tau)`` for
        ``tau`` in seconds. Must accept arrays; should return 0
        for ``tau < 0`` and ``tau > t_max_source``. Use
        ``ricker_wavelet`` or a similar smooth wavelet.
    n_u : int — number of $u$-grid points for the trapezoidal
        quadrature. Higher = more accurate, slower. 400 is ample
        for Ricker at ~15 Hz.

    Returns
    -------
    u_field : 2D array of shape ``(len(x_grid), len(y_grid))``
        — the wavefield at time ``t``.

    Notes
    -----
    The integration variable transformation
    $\\tau = t - \\sqrt{u^2 + r^2/c^2}$ regularises the
    $\\tau^{-1/2}$ singularity at the wavefront. The resulting
    integrand is smooth, suitable for trapezoidal quadrature.
    """
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    r = np.sqrt((X - x_source) ** 2 + (Y - y_source) ** 2)
    # Mask out the source singularity: r → 0 makes the Green's
    # function blow up. Use a small floor at half the typical grid
    # spacing.
    dx_typ = float(np.diff(x_grid).mean()) if x_grid.size > 1 else 1e-3
    dy_typ = float(np.diff(y_grid).mean()) if y_grid.size > 1 else 1e-3
    r_floor = 0.5 * max(dx_typ, dy_typ)
    near_source = r < r_floor
    r_safe = np.where(near_source, r_floor, r)
    r_over_c = r_safe / c

    # Wavefront arrival mask: receivers reached at time t have t > r/c.
    arrived = t > r_over_c

    # U(x, y) = sqrt(t^2 - (r/c)^2) for arrived points; 0 elsewhere.
    U = np.where(arrived, np.sqrt(np.maximum(t ** 2 - r_over_c ** 2,
                                              0.0)),
                 0.0)
    U_max = float(U.max()) if U.size else 0.0
    if U_max <= 0:
        return np.zeros_like(r)

    # Use a fixed u-grid scaled per-cell by U(x, y) for accuracy at
    # near-wavefront cells. Simpler: use a global u-grid up to U_max
    # and zero out the integrand where u > U(x, y).
    u_grid = np.linspace(0.0, U_max, n_u)
    du = u_grid[1] - u_grid[0]

    # τ(u) = t - sqrt(u² + r²/c²), shape (n_u, Nx, Ny)
    u_b = u_grid[:, None, None]
    sqrt_arg = u_b ** 2 + r_over_c[None, :, :] ** 2
    sqrt_term = np.sqrt(sqrt_arg)
    tau = t - sqrt_term

    # Source wavelet at tau (vectorised)
    s_tau = source_func(tau)

    # Integrand: s(τ(u)) / sqrt(u² + r²/c²), with zero contribution
    # where u > U(x, y) (i.e., where τ < 0 in our convention with
    # source supported on [0, ∞))
    mask = (u_b <= U[None, :, :]) & arrived[None, :, :]
    integrand = np.where(mask, s_tau / sqrt_term, 0.0)

    # Trapezoidal rule over u (axis 0). `np.trapezoid` is the
    # NumPy 2.0+ name for what used to be `np.trapz`.
    u_field = np.trapezoid(integrand, dx=du, axis=0) / (2.0 * np.pi)
    # Mask out source-singularity cells (they would be plotted
    # white; the analytical wavefield is undefined at the source).
    u_field = np.where(near_source, 0.0, u_field)
    return u_field


# ─── Image-method dipping-interface helpers ───────────────────────


def mirror_source_across_dipping_line(x_source: float, y_source: float,
                                      x_anchor: float, y_anchor: float,
                                      beta_rad: float) -> tuple[float, float]:
    """Mirror image of $(x_s, y_s)$ across the line
    $y = y_a - \\tan\\beta (x - x_a)$.

    Convention: positive $y$ upward; positive $\\beta$ tilts the
    interface *down to the right*.

    For $\\beta = 0$ this reduces to vertical reflection
    $(x_s, y_s) \\mapsto (x_s, 2 y_a - y_s)$.

    Returns
    -------
    (x_s_image, y_s_image) : mirror-image source coordinates.
    """
    if beta_rad == 0.0:
        return x_source, 2.0 * y_anchor - y_source
    # Signed perpendicular distance from source to the interface
    # line: positive if source is on the "above" side.
    cos_b = float(np.cos(beta_rad))
    sin_b = float(np.sin(beta_rad))
    tan_b = float(np.tan(beta_rad))
    d = cos_b * (y_source - y_anchor + tan_b * (x_source - x_anchor))
    # Mirror displacement: the perpendicular from source to the
    # interface points in the direction (-sin β, -cos β) (down and
    # to the left for positive dip β with interface tilting down to
    # the right). Mirror image = source + 2 d * (-sin β, -cos β).
    x_image = x_source - 2.0 * d * sin_b
    y_image = y_source - 2.0 * d * cos_b
    return x_image, y_image


def angle_dependent_acoustic_R(p: np.ndarray | float,
                                rho_upper: float, V_upper: float,
                                rho_lower: float, V_lower: float
                                ) -> np.ndarray:
    """Angle-dependent acoustic-acoustic reflection coefficient
    $R(p)$ at horizontal slowness $p$. Closed-form, handles
    post-critical evanescent regime via complex $\\cos\\theta$.

    $$R(p) = \\frac{Z_2 \\cos\\theta_1 - Z_1 \\cos\\theta_2}
                   {Z_2 \\cos\\theta_1 + Z_1 \\cos\\theta_2}, \\quad
       \\cos\\theta_i = \\sqrt{1 - V_i^2 p^2}, \\quad Z_i = \\rho_i V_i.$$

    For $p > 1/V_\\text{lower}$ (post-critical from upper side
    into a faster lower layer) the wave totally reflects;
    $|R| = 1$.

    Returns
    -------
    R : complex ndarray same shape as p. For real $R$ the result
        is also real-valued (imaginary part = 0); for post-
        critical $R$ has both real and imaginary parts.
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    arg1 = 1.0 - (V_upper * p_arr) ** 2
    arg2 = 1.0 - (V_lower * p_arr) ** 2
    cos_t1 = np.sqrt(arg1.astype(np.complex128))
    cos_t2 = np.sqrt(arg2.astype(np.complex128))
    Z1 = rho_upper * V_upper
    Z2 = rho_lower * V_lower
    R = (Z2 * cos_t1 - Z1 * cos_t2) / (Z2 * cos_t1 + Z1 * cos_t2)
    if np.isscalar(p):
        return complex(R[0])
    return R


def acoustic_image_method_wavefield_angle_dependent(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    x_source: float,
    y_source: float,
    interface_anchor: tuple[float, float],
    beta_rad: float,
    rho_upper: float, V_upper: float,
    rho_lower: float, V_lower: float,
    source_func,
    n_u: int = 400,
    clip_below_interface: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Angle-dependent image-method reflectivity (Phase 2b).

    Improves on ``acoustic_image_method_wavefield`` by computing
    the **incidence angle at the geometric bounce point** for each
    receiver and applying the angle-dependent reflection
    coefficient $R(\\theta)$. This is exact for plane waves and
    asymptotically exact for spherical-wave reflections at high
    frequency. It captures:

    * Angle-dependent reflection (vs constant normal-incidence $R$).
    * Post-critical total reflection $|R| = 1$ for incidence
      angles beyond the critical angle.
    * Real part of $R$ (sign change near critical angle if any).

    It does NOT capture:

    * Head waves (the lateral wave that travels along the interface
      at the lower-layer P-velocity and re-emerges as an upgoing
      wavefront after the critical angle).
    * The phase shift in the post-critical reflected wave (the
      complex $R$ introduces a frequency-dependent phase that
      mixes amplitude across the Ricker bandwidth; this method
      uses the absolute value $|R|$ for simplicity).

    For our Petrobras config (water 1.5 km/s over TTI vertical
    3.0 km/s), the critical slowness is $p_\\text{crit} = 1/V_p^\\text{TTI}
    = 0.333$ s/km, corresponding to a critical angle of
    $\\arcsin(V_w/V_p^\\text{TTI}) = \\arcsin(0.5) = 30°$.
    For our 0.2 km source-interface depth, receivers beyond
    $\\sim 0.115$ km horizontal offset from the source see a
    post-critical reflection ($|R|=1$). Phase 2a's normal-incidence
    $R = 0.63$ underestimates the reflection at those angles by
    $\\sim 35\\%$.

    Returns
    -------
    u_total : direct + reflected wavefield (same shape as
        ``acoustic_image_method_wavefield``).
    R_field : 2D array of the |R(θ)| value applied at each
        receiver (informative; same shape as ``u_total``).
    """
    direct = acoustic_2d_green_snapshot(
        x_grid, y_grid, t,
        x_source, y_source,
        V_upper, source_func, n_u=n_u,
    )

    x_image, y_image = mirror_source_across_dipping_line(
        x_source, y_source,
        interface_anchor[0], interface_anchor[1],
        beta_rad,
    )
    reflected_kernel = acoustic_2d_green_snapshot(
        x_grid, y_grid, t,
        x_image, y_image,
        V_upper, source_func, n_u=n_u,
    )

    # Per-receiver geometric incidence angle: the reflected ray
    # from the mirror image S' to receiver (X, Y) crosses the
    # interface. The angle between this ray and the interface
    # normal is the incidence angle θ.
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    dxs = X - x_image
    dys = Y - y_image
    r_img = np.sqrt(dxs ** 2 + dys ** 2)
    # Interface normal (pointing "above" = up-and-to-the-right for
    # dip > 0): (sin β, cos β).
    n_x = np.sin(beta_rad)
    n_y = np.cos(beta_rad)
    # cos θ = |d · n| / |d|; ensure positive (the ray is a vector
    # from below to above the interface, so its alignment with n
    # is positive).
    eps = 1e-30
    cos_theta = np.abs(dxs * n_x + dys * n_y) / np.maximum(r_img, eps)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    # Horizontal slowness at the bounce point: p = sin θ / V_upper
    p = sin_theta / V_upper

    # Reflection coefficient at the geometric incidence angle
    R = angle_dependent_acoustic_R(p, rho_upper, V_upper,
                                    rho_lower, V_lower)
    # Apply real(R) for the reflected wavefield. (See docstring on
    # the complex-R phase-shift caveat.)
    R_real = R.real

    total = direct + R_real * reflected_kernel

    if clip_below_interface:
        # Interface line: y = y_a - tan(β)(x - x_a). Above = y > line.
        y_line = (interface_anchor[1]
                  - np.tan(beta_rad) * (X - interface_anchor[0]))
        above = Y > y_line
        total = np.where(above, total, 0.0)

    return total, np.abs(R)


def acoustic_image_method_wavefield(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t: float,
    x_source: float,
    y_source: float,
    c_upper: float,
    R: float,
    interface_anchor: tuple[float, float],
    beta_rad: float,
    source_func,
    n_u: int = 400,
    clip_below_interface: bool = True,
) -> np.ndarray:
    """Total upper-layer wavefield via the image method:
    direct wave + dipping-interface reflection.

    $u_\\text{total}(\\mathbf{x}, t)
       = u_d(\\mathbf{x}, t) + R \\cdot u_d(\\mathbf{x}, t;
         \\mathbf{x}_s')$

    where $\\mathbf{x}_s'$ is the mirror image of the source
    across the dipping interface and the second term uses the same
    Green's function evaluated at the mirror-source-to-receiver
    distance.

    Parameters
    ----------
    x_grid, y_grid : receiver coordinates (km).
    t : snapshot time (s).
    x_source, y_source : actual source location (km).
    c_upper : acoustic velocity in the upper layer (km/s).
    R : reflection coefficient at the interface.
    interface_anchor : (x_a, y_a) point through which the interface
        line passes.
    beta_rad : dip angle (rad), positive tilts interface down to
        the right.
    source_func : Ricker or other smooth source wavelet.
    n_u : quadrature granularity.
    clip_below_interface : if True, sets the wavefield to zero at
        grid cells *below* the interface line (in the lower TTI
        region not covered by this model).

    Returns
    -------
    u_total : 2D array of shape ``(len(x_grid), len(y_grid))``
        — direct + reflected wavefield in the upper water layer.
    """
    direct = acoustic_2d_green_snapshot(
        x_grid, y_grid, t,
        x_source, y_source,
        c_upper, source_func, n_u=n_u,
    )

    x_image, y_image = mirror_source_across_dipping_line(
        x_source, y_source,
        interface_anchor[0], interface_anchor[1],
        beta_rad,
    )
    reflected = acoustic_2d_green_snapshot(
        x_grid, y_grid, t,
        x_image, y_image,
        c_upper, source_func, n_u=n_u,
    )

    total = direct + R * reflected

    if clip_below_interface:
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        # Interface line: y = y_a - tan(β)(x - x_a). Above = y > line.
        y_line = (interface_anchor[1]
                  - np.tan(beta_rad) * (X - interface_anchor[0]))
        above = Y > y_line
        total = np.where(above, total, 0.0)

    return total


# ─── Acoustic-acoustic impedance reflection coefficient ───────────


def acoustic_impedance_reflection(rho_upper: float, c_upper: float,
                                  rho_lower: float, c_lower_vert: float
                                  ) -> float:
    """Acoustic-acoustic impedance reflection coefficient at
    normal incidence:

    $R = (Z_\\text{lower} - Z_\\text{upper})
         / (Z_\\text{lower} + Z_\\text{upper})$

    where $Z = \\rho \\cdot c$.

    For water-over-TTI ($\\rho_\\text{w} = 1.0$,
    $c_\\text{w} = 1.5$, $\\rho_\\text{TTI} = 2.2$, vertical
    $V_p^\\text{TTI} = 3.0$) this gives $R \\approx 0.63$.

    This is the *acoustic* approximation; the actual elastic R
    depends on incidence angle and includes mode conversion. For
    the image-method reference this approximation captures the
    dominant reflection sign and order-of-magnitude amplitude.
    """
    Z_u = rho_upper * c_upper
    Z_l = rho_lower * c_lower_vert
    return (Z_l - Z_u) / (Z_l + Z_u)


__all__ = [
    "ricker_wavelet",
    "acoustic_2d_green_snapshot",
    "mirror_source_across_dipping_line",
    "acoustic_image_method_wavefield",
    "acoustic_impedance_reflection",
]
