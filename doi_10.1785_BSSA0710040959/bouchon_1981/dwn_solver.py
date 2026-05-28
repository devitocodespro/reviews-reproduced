"""2D acoustic-acoustic reflectivity reference via Discrete Wavenumber
Method (plan §80 Phase 2c-DWN).

Replaces the slowness-FFT engine of the earlier draft (which hit
the critical-angle $1/q_w$ singularity and produced $10^{301}$
amplitude blowup) with the Discrete Wavenumber Method of @bouchon_1981.

Mathematical foundation
-----------------------

Working in the rotated frame where the interface is horizontal at
$z' = 0$ and the source is at $(x_s, z_s)$ in the upper water layer
($z' > z'_\\text{interface}$).

The 2D Green's function for the upper layer in slowness-frequency
representation is

$$\\tilde G(k_x, z; z_s, \\omega)
   = \\frac{i}{2 q_w(\\omega, k_x)}
   \\Big[ e^{i q_w |z - z_s|}
         + R(\\omega, k_x) \\, e^{i q_w (z + z_s - 2 z_\\text{iface})} \\Big]$$

with $q_w(\\omega, k_x) = \\sqrt{\\omega^2/V_p^2 - k_x^2}$ (positive
imaginary part for evanescent modes). The acoustic-acoustic
reflection coefficient $R(\\omega, k_x)$ is reformulated in terms
of $q_w$ and $q_z^\\text{lower}$ (avoiding the $k_x/\\omega$
quotient that overflows near $\\omega \\to 0$):

$$R = \\frac{Z_2 V_1 q_w - Z_1 V_2 q_z^\\text{lower}}
            {Z_2 V_1 q_w + Z_1 V_2 q_z^\\text{lower}}.$$

**Discrete Wavenumber Method** (@bouchon_1981): the wavenumber
integral $\\int dk_x$ is replaced by a finite discrete sum over
$k_m = 2\\pi m / L$ for $m \\in \\{-M, \\ldots, +M\\}$, with $L$
chosen large enough that periodic copies of the source don't
reach the receivers within the time window $T_\\text{max}$:

$$L > V_\\text{max} \\cdot T_\\text{max} + x_\\text{max}.$$

The frequency integral $\\int d\\omega$ remains an inverse FFT;
the **critical-angle $1/q_w$ singularity** at $q_w = 0$ is
regularised by a **complex-frequency shift** $\\omega \\to
\\omega - i a$ where $a > 0$ is a small damping. The shift moves
all $q_w$ values off zero, then the time-domain wavefield is
back-corrected by multiplying by $e^{a t}$.

Standard Bouchon-1981 parameter choice:

* $a = \\pi / T_\\text{max}$ — gives ~25 dB damping at
  $t = T_\\text{max}$, recovered by the back-correction.
* $L = 1.3 \\cdot (V_\\text{max} T_\\text{max} + x_\\text{max})$
  — 30% safety margin past the strict Bouchon-1981 condition.

Plan reference: §80d-DWN. References to read: @bouchon_1981
(seminal), @bouchon_2003 (modern review), @bouchon_aki_1977
(precursor).

Verified parts retained from the earlier slowness-FFT draft:

* `ricker_spectrum` (closed-form Ricker FT).
* `acoustic_acoustic_reflection_pdomain` (R(p) — slowness form).
* `acoustic_reflection_from_qw` (R(q_w, q_z_lower) — avoids
  overflow near $\\omega \\to 0$).
"""
from __future__ import annotations

import numpy as np


# ─── Frequency-domain source wavelets ─────────────────────────────


def ricker_spectrum(omega: np.ndarray, f0: float,
                    t_delay: float | None = None) -> np.ndarray:
    """Fourier transform of the Ricker wavelet
    $s(t) = (1 - 2\\pi^2 f_0^2 (t - t_d)^2) \\exp(-\\pi^2 f_0^2 (t - t_d)^2)$
    under Convention A ($\\hat s(\\omega) = \\int s(t) e^{-i\\omega t} dt$,
    inverse $s(t) = (1/(2\\pi)) \\int \\hat s e^{+i\\omega t} d\\omega$).

    $$\\hat s(\\omega) = \\frac{\\omega^2}{2 \\sqrt{\\pi}\\, \\pi^2 f_0^3}
        \\exp\\!\\Big(-\\frac{\\omega^2}{(2\\pi f_0)^2}\\Big)
        e^{-i \\omega t_d}.$$

    Verified by round-trip against the time-domain Ricker: discrete
    inverse FT (``np.fft.ifft(spec)*N/T_max``) recovers the wavelet
    peak to machine precision.
    """
    if t_delay is None:
        t_delay = 1.5 / f0
    w = omega
    amp = (w ** 2 / (2.0 * np.sqrt(np.pi) * (np.pi * f0) ** 2 * f0)
           * np.exp(-(w / (2.0 * np.pi * f0)) ** 2))
    return amp * np.exp(-1j * w * t_delay)


# ─── Reflection coefficient (two equivalent forms) ────────────────


def acoustic_acoustic_reflection_pdomain(p: np.ndarray,
                                          rho_upper: float, V_upper: float,
                                          rho_lower: float, V_lower: float
                                          ) -> np.ndarray:
    """Angle-dependent acoustic-acoustic R(p) using horizontal
    slowness $p$. Use this for direct R(p) verification or for
    ray-theory-style geometric reflection — see also
    ``00_common/analytical_acoustic_2d.py:angle_dependent_acoustic_R``
    which is the same formula.

    Returns complex R; $|R| = 1$ for $p$ post-critical from the
    lower side ($p > 1/V_\\text{lower}$).
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    arg1 = 1.0 - (V_upper * p_arr) ** 2
    arg2 = 1.0 - (V_lower * p_arr) ** 2
    cos_t1 = np.sqrt(arg1.astype(np.complex128))
    cos_t2 = np.sqrt(arg2.astype(np.complex128))
    Z1 = rho_upper * V_upper
    Z2 = rho_lower * V_lower
    return (Z2 * cos_t1 - Z1 * cos_t2) / (Z2 * cos_t1 + Z1 * cos_t2)


def acoustic_reflection_from_qw(qw: np.ndarray, qz_lower: np.ndarray,
                                 rho_upper: float, V_upper: float,
                                 rho_lower: float, V_lower: float
                                 ) -> np.ndarray:
    """Angle-dependent R expressed via vertical wavenumbers
    $q_w, q_z^\\text{lower}$ instead of horizontal slowness
    $p = k_x / \\omega$. Mathematically equivalent to
    ``acoustic_acoustic_reflection_pdomain`` (the $|\\omega|$
    factors cancel) but avoids the kx/ω quotient that overflows
    near $\\omega \\to 0$ — essential for the DWN inner loop.
    """
    Z1 = rho_upper * V_upper
    Z2 = rho_lower * V_lower
    num = Z2 * V_upper * qw - Z1 * V_lower * qz_lower
    den = Z2 * V_upper * qw + Z1 * V_lower * qz_lower
    return num / den


# ─── DWN engine (Phase 2c-DWN) ────────────────────────────────────


def dwn_wavefield_acoustic_acoustic_horizontal(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    z_interface: float,
    rho_upper: float, V_upper: float,
    rho_lower: float, V_lower: float,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
) -> np.ndarray:
    """Discrete Wavenumber Method snapshot of the upper-layer 2D
    acoustic wavefield over a planar horizontal interface
    (rotated-frame geometry). @bouchon_1981.

    Computes $u(x, z, t = t_\\text{target})$ in the upper layer
    above $z = z_\\text{interface}$. Source is at
    $(x_\\text{source}, z_\\text{source})$ with $z_\\text{source}
    > z_\\text{interface}$.

    The wavefield is the sum of a direct wave (source-to-receiver
    Green's function) and a reflected wave (with angle-dependent
    $R(\\omega, k_x)$). For receivers below the interface the
    function returns zero (the transmitted-wave Green's function
    requires lower-layer parameters, deferred to Phase 2d).

    Parameters
    ----------
    x_grid, z_grid : 1D arrays — receiver coordinates (km).
        Uniformly spaced is preferred but not required.
    t_target : snapshot time (s).
    x_source, z_source : source coords (km).
    z_interface : interface depth (km). Must satisfy
        $z_\\text{source} > z_\\text{interface}$.
    rho_upper, V_upper, rho_lower, V_lower : layer parameters.
    f0 : Ricker peak frequency (Hz).
    T_max : maximum time window for which the DWN result is
        accurate (default $1.5 \\cdot t_\\text{target}$, gives
        ~33% safety margin past the snapshot time).
    L_periodicity : DWN periodicity length (km). If None, set to
        $1.3 \\cdot (V_\\text{lower} T_\\text{max} +
        x_\\text{max})$ where $x_\\text{max}$ is the largest
        source-receiver horizontal distance.
    n_omega : number of frequency samples (power of 2 recommended).
    M_wavenumbers : half-range of discrete wavenumbers
        ($k_m \\in \\{-M, ..., +M\\} \\cdot 2\\pi/L$).
    a_damping : complex-frequency damping (rad/s). If None, set to
        $\\pi / T_\\text{max}$ (Bouchon-1981 standard).
    t_delay : Ricker time-delay (default $1.5/f_0$).
    include_direct, include_reflected : toggles for diagnostic
        purposes. Default: include both.

    Returns
    -------
    u : 2D array shape ``(len(x_grid), len(z_grid))`` — the
        wavefield at $t = t_\\text{target}$ in the upper layer.
        Cells below the interface are set to zero.

    Notes
    -----
    Memory scales as $O(N_\\omega \\cdot M \\cdot N_z)$; for our
    Petrobras config (256 × 1024 × 215) this is ~450 MB of
    complex doubles. The $x$-grid is decoupled from the inner
    DWN sum (we just multiply by $e^{i k_m (x - x_s)}$ per
    receiver), so it has minimal additional memory cost.
    """
    if z_source <= z_interface:
        raise ValueError(
            f"z_source ({z_source}) must be above z_interface "
            f"({z_interface}) for the upper-layer reflectivity.")

    if T_max is None:
        T_max = 1.5 * t_target

    if a_damping is None:
        a_damping = np.pi / T_max

    # Strict Bouchon-1981 periodicity condition: L > V_max T_max + x_max
    if L_periodicity is None:
        x_max = max(
            float(np.max(np.abs(x_grid - x_source))),
            1e-3,  # avoid zero
        )
        L_periodicity = 1.3 * (V_lower * T_max + x_max)

    if t_delay is None:
        t_delay = 1.5 / f0

    # ─── Frequency grid (real ω, complex shift applied to derived quantities) ─
    # Use a real DFT frequency grid: ω_n = 2π n / T_max with t-grid
    # spanning [0, T_max]. The complex shift ω → ω - ia is applied to
    # q_w, q_lower, R, exp factors; the time-domain back-correction is
    # multiplication by exp(a t_target).
    dt_grid = T_max / n_omega
    t_grid = np.arange(n_omega) * dt_grid  # [0, T_max)
    # Find target time index
    it = int(np.argmin(np.abs(t_grid - t_target)))
    # Frequency grid for FFT
    omega = 2.0 * np.pi * np.fft.fftfreq(n_omega, d=dt_grid)  # rad/s

    # Apply complex shift
    omega_c = omega - 1j * a_damping  # shape (n_omega,)

    # ─── Wavenumber grid (discrete, finite) ───
    m_arr = np.arange(-M_wavenumbers, M_wavenumbers + 1)
    k_m = 2.0 * np.pi * m_arr / L_periodicity  # rad/km, shape (2M+1,)
    n_k = len(k_m)
    dk = 2.0 * np.pi / L_periodicity  # = k_{m+1} - k_m

    # ─── Vertical wavenumbers (with complex frequency) ───
    # Broadcast: ω in axis 0, k_m in axis 1
    W = omega_c[:, None]                # (n_omega, 1)
    K = k_m[None, :]                    # (1, n_k)
    qw_sq = (W / V_upper) ** 2 - K ** 2
    qz_lower_sq = (W / V_lower) ** 2 - K ** 2
    qw = np.sqrt(qw_sq)
    qz_lower = np.sqrt(qz_lower_sq)
    # Branch convention: choose +imag-part for evanescent modes so
    # e^{i q |z|} decays. After complex sqrt of complex argument the
    # default branch usually returns positive real or positive
    # imaginary; ensure consistency by flipping where Im(qw) < 0.
    # Convention A (inverse FT $e^{+i\omega t}$, paired with the
    # spatial kernel $e^{-i q_w |z|}$ below): outgoing evanescent
    # modes require $\\mathrm{Im}(q_w) \\le 0$.
    qw = np.where(np.imag(qw) > 0, -qw, qw)
    # In this acoustic engine `qw, qz_lower` are **wavenumbers**
    # (units rad/km, from sqrt of (W/V)² − K²). The downgoing
    # propagator is exp(-i·qz·dz), so the correct decay criterion
    # is Im(qz_lower) ≤ 0 (NOT Im(W·qz_lower) ≤ 0 — that latter
    # criterion is for the anisotropic engine where qz is a
    # slowness and propagator uses W·qz, per §82 Stage E1.2).
    qz_lower = np.where(np.imag(qz_lower) > 0, -qz_lower, qz_lower)

    # Reflection coefficient (no kx/ω quotient; uses qw, qz_lower)
    R = acoustic_reflection_from_qw(qw, qz_lower,
                                     rho_upper, V_upper,
                                     rho_lower, V_lower)

    # Source spectrum at the complex frequency
    src_spec = ricker_spectrum(omega_c, f0, t_delay=t_delay)
    # (n_omega,) - broadcasts to (n_omega, n_k) below

    # ─── DWN summation over (ω, k_m) → u(x, z, t) ───
    nx = len(x_grid)
    nz = len(z_grid)
    u_out = np.zeros((nx, nz), dtype=np.float64)

    # The DWN sum's expensive axes are (ω, k). The (x, z) loop is
    # additionally vectorisable per receiver: for each (x, z), the
    # contribution is Σ_m G_tilde(k_m, z, ω_n) · e^{i k_m (x - x_s)},
    # summed over m, then inverse-FFT'd over ω.
    #
    # Memory budget: G_tilde at fixed z = (n_omega, n_k) complex →
    # 256 × 2049 × 16 B = 8 MB. Per-z slice loop avoids storing the
    # full (z, ω, k) array.

    # Plan §83: rendered quantity is **divergence of displacement**
    # ∇·u = ∂_x u_x + ∂_z u_z in both layers. Rotation-invariant
    # scalar; matches the FD benchmark's canonical `div_late`. Upper
    # acoustic: ∇·u = ∇²φ = -(ω/V_U)² φ. Lower acoustic: same form
    # with transmitted potential T_pot = (ρ_U/ρ_L)(1+R).
    upper_div_factor = -(W / V_upper) ** 2     # (n_omega, 1)
    lower_div_factor = -(W / V_lower) ** 2     # (n_omega, 1)
    # Acoustic-acoustic transmitted-potential coefficient (pressure
    # continuity): T_pot = (ρ_U/ρ_L)(1+R). Verified consistent with
    # u_z-continuity T = q_w(1-R)/q_z_lower for the R from
    # `acoustic_reflection_from_qw`.
    T_pot = (rho_upper / rho_lower) * (1.0 + R)
    qw_safe = np.where(np.abs(qw) < 1e-10, 1e-10, qw)

    for iz, z in enumerate(z_grid):
        if z > z_interface:
            # ─── Upper-layer divergence kernel ────────────────────
            abs_dz = abs(z - z_source)            # |z - z_s|
            z_refl = z + z_source - 2.0 * z_interface  # > 0
            # Convention A outgoing kernel: e^{-i q_w |z - z_s|}
            # with Im(q_w) ≤ 0 for evanescent decay.
            prop_direct = (
                np.exp(-1j * qw * abs_dz) if include_direct
                else np.zeros_like(qw))
            prop_refl = (
                R * np.exp(-1j * qw * z_refl) if include_reflected
                else np.zeros_like(qw))
            # Potential kernel from 2D Green's function (Bouchon 1981
            # normalisation 1/(2j q_w)), then convert to divergence
            # by multiplying by -(ω/V_U)² (since ∇²φ = -(ω/V)²φ for
            # the homogeneous wave equation).
            kernel = (upper_div_factor
                      * (prop_direct + prop_refl) / (2j * qw_safe))
        else:
            # ─── Lower-layer transmitted divergence kernel ──────
            # Plan §83 Stage F3 (closes Codex 2nd HIGH finding).
            # Total transmitted wave from source at z_s above iface:
            #   φ^L(z) = T_pot · G_source-side · e^{-iq_w(z_s-z_a)}
            #                                 · e^{-iq_z^L(z_a-z)}
            # where G_source-side = -i/(2 q_w) is the upper-layer
            # 2D Green's function normalisation (geometric spreading
            # from source through the interface). Divergence:
            # ∇²φ^L = -(ω/V_L)² φ^L.
            if not include_reflected:
                # Transmitted modes share the reflected-system
                # solve; for diagnostics that turn reflection off,
                # skip lower-layer too.
                continue
            src_to_iface = z_source - z_interface   # > 0 (src above)
            dz_below = z_interface - z              # > 0 (rcv below)
            # NOTE: in this acoustic engine, `qw` and `qz_lower` are
            # **wavenumbers** (rad/km) derived from `sqrt((W/V)²−K²)`.
            # The propagator is `exp(-i · qz · dz)` — DO NOT multiply
            # by W (that would mix frequency and wavenumber and give
            # a ~10² × larger phase that aliases under IFFT). The
            # anisotropic engine's `qz_qP/qSV` are slownesses; that
            # other code path correctly uses `W * qz` to convert.
            prop_src = np.exp(-1j * qw * src_to_iface)
            prop_T = np.exp(-1j * qz_lower * dz_below)
            kernel = (lower_div_factor * T_pot
                      * prop_src * prop_T
                      / (2j * qw_safe))

        # Multiply by source spectrum (broadcast over n_k)
        kernel = kernel * src_spec[:, None]

        # ─── Per-receiver: inner DWN sum over k_m ───
        # For each (x_grid[ix], z): wavefield = Σ_m kernel[ω, k_m] ·
        # e^{i k_m (x - x_s)} · (1/L), summed over m. Then inverse
        # FFT over ω → t.
        # Vectorise: do the k_m sum for all x simultaneously.
        # phase shape (n_x, n_k): exp(i k_m (x - x_s))
        dxs = x_grid - x_source  # (nx,)
        phase = np.exp(1j * np.outer(dxs, k_m))  # (nx, n_k)
        # u_omega shape (nx, n_omega): Σ_m kernel[ω, m] · phase[x, m]
        u_omega = (kernel @ phase.T).T  # (n_omega, nx)^T → (nx, n_omega)
        u_omega /= L_periodicity  # DWN normalisation 1/L

        # Inverse FT over ω → t (Convention A: $u(t) =
        # (1/(2\\pi)) \\int \\hat u(\\omega) e^{+i\\omega t} d\\omega$).
        # Discrete approximation:
        #   $u(t_n) \\approx (1/T_{max}) \\sum_k \\hat u_k
        #     e^{+i\\omega_k t_n}$
        # and ``np.fft.ifft(X)*N = \\sum_k X_k e^{+i\\omega_k t_n}``.
        # So the correct normalisation is ``ifft(u_omega)*N/T_max``.
        u_t = np.fft.ifft(u_omega, axis=-1) * n_omega / T_max
        u_t = u_t.real  # imaginary part should be ~0 at machine
        # precision because the original kernel is the FT of a real
        # signal.

        # Back-correct the complex-frequency damping: multiply by
        # exp(a t_target) at the target snapshot time.
        u_target = u_t[:, it] * np.exp(a_damping * t_grid[it])

        u_out[:, iz] = u_target

    return u_out


def dwn_time_series_acoustic_acoustic_horizontal(
    x_receivers: np.ndarray,
    z_receivers: np.ndarray,
    x_source: float,
    z_source: float,
    z_interface: float,
    rho_upper: float, V_upper: float,
    rho_lower: float, V_lower: float,
    f0: float,
    T_max: float,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan §91 Phase 2b — DWN time-series at receivers for the
    2D iso-iso horizontal-interface acoustic problem.

    Returns time series at each receiver instead of a wavefield
    snapshot. Same DWN engine as
    ``dwn_wavefield_acoustic_acoustic_horizontal`` but extracts
    the full IFFT output at each receiver (rather than discarding
    all but one time sample).

    Parameters
    ----------
    x_receivers, z_receivers : 1D arrays, same length n_rcv.
        Receiver coordinates (km).
    T_max : float
        Time window for the DWN result. The output time grid is
        ``np.arange(n_omega) * T_max / n_omega`` ∈ [0, T_max).
    Other parameters : see
        ``dwn_wavefield_acoustic_acoustic_horizontal``.

    Returns
    -------
    t_grid : (n_omega,) — time samples (s).
    u : (n_rcv, n_omega) — wavefield divergence at each receiver
        for each time sample. Receivers above the interface get
        the upper-layer kernel; below the interface get the
        transmitted-wave kernel.
    """
    x_receivers = np.asarray(x_receivers, dtype=np.float64)
    z_receivers = np.asarray(z_receivers, dtype=np.float64)
    if x_receivers.shape != z_receivers.shape:
        raise ValueError("x_receivers and z_receivers must have the same shape")
    if z_source <= z_interface:
        raise ValueError(
            f"z_source ({z_source}) must be above z_interface "
            f"({z_interface}) for the upper-layer reflectivity.")

    if a_damping is None:
        a_damping = np.pi / T_max
    if L_periodicity is None:
        x_max = max(
            float(np.max(np.abs(x_receivers - x_source))),
            1e-3,
        )
        L_periodicity = 1.3 * (V_lower * T_max + x_max)
    if t_delay is None:
        t_delay = 1.5 / f0

    dt_grid = T_max / n_omega
    t_grid = np.arange(n_omega) * dt_grid
    omega = 2.0 * np.pi * np.fft.fftfreq(n_omega, d=dt_grid)
    omega_c = omega - 1j * a_damping

    m_arr = np.arange(-M_wavenumbers, M_wavenumbers + 1)
    k_m = 2.0 * np.pi * m_arr / L_periodicity

    W = omega_c[:, None]
    K = k_m[None, :]
    qw_sq = (W / V_upper) ** 2 - K ** 2
    qz_lower_sq = (W / V_lower) ** 2 - K ** 2
    qw = np.sqrt(qw_sq)
    qz_lower = np.sqrt(qz_lower_sq)
    qw = np.where(np.imag(qw) > 0, -qw, qw)
    qz_lower = np.where(np.imag(qz_lower) > 0, -qz_lower, qz_lower)

    R = acoustic_reflection_from_qw(
        qw, qz_lower, rho_upper, V_upper, rho_lower, V_lower)
    src_spec = ricker_spectrum(omega_c, f0, t_delay=t_delay)
    upper_div_factor = -(W / V_upper) ** 2
    lower_div_factor = -(W / V_lower) ** 2
    T_pot = (rho_upper / rho_lower) * (1.0 + R)
    qw_safe = np.where(np.abs(qw) < 1e-10, 1e-10, qw)

    n_rcv = len(x_receivers)
    u_out = np.zeros((n_rcv, n_omega), dtype=np.float64)
    damping_correction = np.exp(a_damping * t_grid)   # shape (n_omega,)

    for i_rcv in range(n_rcv):
        x_r = float(x_receivers[i_rcv])
        z_r = float(z_receivers[i_rcv])

        if z_r > z_interface:
            abs_dz = abs(z_r - z_source)
            z_refl = z_r + z_source - 2.0 * z_interface
            prop_direct = (
                np.exp(-1j * qw * abs_dz) if include_direct
                else np.zeros_like(qw))
            prop_refl = (
                R * np.exp(-1j * qw * z_refl) if include_reflected
                else np.zeros_like(qw))
            kernel = (upper_div_factor
                      * (prop_direct + prop_refl) / (2j * qw_safe))
        else:
            if not include_reflected:
                continue
            src_to_iface = z_source - z_interface
            dz_below = z_interface - z_r
            prop_src = np.exp(-1j * qw * src_to_iface)
            prop_T = np.exp(-1j * qz_lower * dz_below)
            kernel = (lower_div_factor * T_pot
                      * prop_src * prop_T / (2j * qw_safe))

        kernel = kernel * src_spec[:, None]

        # x-phase for THIS receiver (1D, shape (n_k,))
        phase_x = np.exp(1j * k_m * (x_r - x_source))     # (n_k,)
        u_omega_rcv = (kernel * phase_x[None, :]).sum(axis=-1) / L_periodicity
        # (n_omega,)

        u_t = np.fft.ifft(u_omega_rcv) * n_omega / T_max
        u_t = u_t.real
        u_out[i_rcv, :] = u_t * damping_correction

    return t_grid, u_out


# ─── Dipping-interface wrapper (Stage B / plan §81b) ──────────────


def _rotate_about_anchor(x: np.ndarray, y: np.ndarray,
                          beta_rad: float,
                          x_anchor: float, y_anchor: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 2D rotation by ``+beta_rad`` about ``(x_anchor, y_anchor)``.

    Convention (matches @analytical_acoustic_2d.mirror_source_across_dipping_line):
    a dipping interface with slope $-\\tan\\beta$ passing through
    $(x_a, y_a)$ becomes horizontal at $y' = y_a$ after rotation by
    $+\\beta$. The rotation sends:
    $(x, y) \\mapsto (x', y') = (x_a + \\Delta x \\cos\\beta -
    \\Delta y \\sin\\beta, y_a + \\Delta x \\sin\\beta +
    \\Delta y \\cos\\beta)$
    where $\\Delta x = x - x_a$, $\\Delta y = y - y_a$.

    Returns arrays with the same shape as the inputs.
    """
    dx = np.asarray(x, dtype=np.float64) - x_anchor
    dy = np.asarray(y, dtype=np.float64) - y_anchor
    cb = float(np.cos(beta_rad))
    sb = float(np.sin(beta_rad))
    x_prime = x_anchor + dx * cb - dy * sb
    y_prime = y_anchor + dx * sb + dy * cb
    return x_prime, y_prime


def dwn_wavefield_acoustic_acoustic_dipping(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    x_anchor: float,
    y_anchor: float,
    dip_deg: float,
    rho_upper: float, V_upper: float,
    rho_lower: float, V_lower: float,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
    primed_grid_refinement: float = 3.0,
) -> np.ndarray:
    """DWN snapshot of the 2D acoustic wavefield with a **dipping**
    interface (planar; dip ``dip_deg``; passing through
    ``(x_anchor, y_anchor)``).

    Strategy (plan §81b Stage B):
    1. Rotate the lab source about ``(x_anchor, y_anchor)`` by
       $+\\beta = +\\mathrm{dip}$ — sends the dipping interface to
       the horizontal line $z' = y_a$ in the primed frame.
    2. Compute a primed-frame bounding box that covers all rotated
       lab receivers (the four corners of the lab grid).
    3. Build a regular primed-frame grid over the bbox at
       ``dx_lab/primed_grid_refinement`` resolution.
    4. Call ``dwn_wavefield_acoustic_acoustic_horizontal`` on the
       primed grid with the rotated source.
    5. Bilinear-interpolate the primed-frame snapshot at each
       rotated lab receiver position.

    The result is the wavefield value at the **lab-frame** receiver
    grid (returned as ``(len(x_grid), len(z_grid))`` array).

    Notes
    -----
    The wavefield is a scalar (acoustic pressure) so no
    polarisation-vector rotation is needed; only the receiver
    coordinates are rotated.

    At ``dip_deg = 0`` this function is identical to a direct call
    to ``dwn_wavefield_acoustic_acoustic_horizontal`` (modulo
    bilinear-interpolation residual; aim for < 1% by setting
    ``primed_grid_refinement >= 1.5``).
    """
    from scipy.interpolate import RegularGridInterpolator

    beta_rad = np.deg2rad(dip_deg)

    # Source in primed frame
    xs_prime, zs_prime = _rotate_about_anchor(
        np.array([x_source]), np.array([z_source]),
        beta_rad, x_anchor, y_anchor,
    )
    xs_prime = float(xs_prime[0])
    zs_prime = float(zs_prime[0])

    # Lab grid corners → primed frame; bbox over rotated corners
    x_lab_min = float(np.min(x_grid))
    x_lab_max = float(np.max(x_grid))
    z_lab_min = float(np.min(z_grid))
    z_lab_max = float(np.max(z_grid))
    corners_x = np.array([x_lab_min, x_lab_max, x_lab_min, x_lab_max])
    corners_z = np.array([z_lab_min, z_lab_min, z_lab_max, z_lab_max])
    corners_xp, corners_zp = _rotate_about_anchor(
        corners_x, corners_z, beta_rad, x_anchor, y_anchor,
    )

    # Margin so bilinear interp at lab cells near the bbox edge has
    # support; one primed cell on each side.
    dx_lab = (
        float(np.diff(x_grid).mean()) if x_grid.size > 1 else 1e-3
    )
    dz_lab = (
        float(np.diff(z_grid).mean()) if z_grid.size > 1 else 1e-3
    )
    dx_primed = min(dx_lab, dz_lab) / float(primed_grid_refinement)

    x_pmin = float(np.min(corners_xp)) - 2.0 * dx_primed
    x_pmax = float(np.max(corners_xp)) + 2.0 * dx_primed
    z_pmin = float(np.min(corners_zp)) - 2.0 * dx_primed
    z_pmax = float(np.max(corners_zp)) + 2.0 * dx_primed

    # Build primed-frame regular grid
    n_x_primed = max(int(np.ceil((x_pmax - x_pmin) / dx_primed)) + 1, 4)
    n_z_primed = max(int(np.ceil((z_pmax - z_pmin) / dx_primed)) + 1, 4)
    x_primed = np.linspace(x_pmin, x_pmax, n_x_primed)
    z_primed = np.linspace(z_pmin, z_pmax, n_z_primed)

    # Compute DWN on primed-frame grid (horizontal interface at z' = y_a)
    u_primed = dwn_wavefield_acoustic_acoustic_horizontal(
        x_grid=x_primed, z_grid=z_primed, t_target=t_target,
        x_source=xs_prime, z_source=zs_prime,
        z_interface=y_anchor,
        rho_upper=rho_upper, V_upper=V_upper,
        rho_lower=rho_lower, V_lower=V_lower,
        f0=f0, T_max=T_max,
        L_periodicity=L_periodicity,
        n_omega=n_omega, M_wavenumbers=M_wavenumbers,
        a_damping=a_damping, t_delay=t_delay,
        include_direct=include_direct,
        include_reflected=include_reflected,
    )

    # Interpolator on the primed grid: u_primed has shape
    # (n_x_primed, n_z_primed) and is indexed as (x', z').
    interp = RegularGridInterpolator(
        (x_primed, z_primed), u_primed,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    # Rotate each lab cell to its primed coordinates
    X_lab, Z_lab = np.meshgrid(x_grid, z_grid, indexing="ij")
    X_lab_prime, Z_lab_prime = _rotate_about_anchor(
        X_lab, Z_lab, beta_rad, x_anchor, y_anchor,
    )
    pts = np.stack([X_lab_prime.ravel(), Z_lab_prime.ravel()], axis=-1)
    u_lab = interp(pts).reshape(X_lab.shape)
    return u_lab


# ─── DWN with anisotropic (TTI elastic) lower layer (Phase 2d) ────


def dwn_wavefield_acoustic_anisotropic_horizontal(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    z_interface: float,
    rho_upper: float, V_upper: float,
    C_lower: dict,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
) -> np.ndarray:
    """Phase 2d DWN snapshot of the 2D acoustic-over-TTI-elastic
    wavefield. Both upper-layer (acoustic potential) and
    lower-layer (vertical displacement from transmitted qP + qSV
    modes) are computed; plan §82 Stage E1 closes the original
    lower-layer-skip gap (Codex review 2026-05-16).

    Geometry identical to ``dwn_wavefield_acoustic_acoustic_horizontal``
    but the reflection coefficient $R(k_x, \\omega)$ is computed
    from the **anisotropic acoustic-elastic 3×3 R/T system**
    (Mallick-Frazer 1990 [@mallick_frazer_1990]). The same 3×3
    solve also yields the transmission coefficients $T_{qP},
    T_{qSV}$ which propagate the wave below the interface.

    Output-quantity caveat: the upper-layer panel produces a
    quantity proportional to the acoustic **velocity potential**
    $\\phi$ via the Green's-function factor $1/(2j q_w)$, while
    the lower-layer panel produces **vertical displacement**
    $u_z(z) = \\sum_m T_m U_z^m e^{-i\\omega q_z^m (z_a - z)}$.
    These are different physical quantities and not directly
    comparable in absolute amplitude across the interface; with
    per-panel/per-row 99.5th-percentile colour normalisation the
    two layers' wavefront patterns are visually interpretable on
    a shared plot.

    Parameters
    ----------
    Same as ``dwn_wavefield_acoustic_acoustic_horizontal`` except
    ``rho_lower``/``V_lower`` are replaced by the lower-layer
    stiffness dictionary ``C_lower`` with keys C11, C13, C15, C33,
    C35, C55, rho (standard Voigt naming). Use
    ``anisotropic_rt.stiffness_from_thomsen`` to build it from
    Thomsen parameters.

    Returns
    -------
    u_out : ndarray, shape ``(len(x_grid), len(z_grid))``. Upper
        layer (z > z_interface) carries acoustic potential;
        lower layer (z <= z_interface) carries vertical
        displacement $u_z$.
    """
    from anisotropic_rt import acoustic_elastic_reflection

    if T_max is None:
        T_max = 1.2 * (t_target if t_target > 0 else 1.0)
    if a_damping is None:
        a_damping = np.pi / T_max
    # Periodicity: use the lower-layer's C11/rho velocity as a
    # rough V_max upper bound (avoids importing more from C_lower).
    V_max = max(V_upper, np.sqrt(C_lower["C11"] / C_lower["rho"]))
    if L_periodicity is None:
        x_max = max(
            float(np.max(np.abs(x_grid - x_source))),
            1e-3,
        )
        L_periodicity = 1.3 * (V_max * T_max + x_max)
    if t_delay is None:
        t_delay = 1.5 / f0

    dt_grid = T_max / n_omega
    t_grid = np.arange(n_omega) * dt_grid
    it = int(np.argmin(np.abs(t_grid - t_target)))
    omega = 2.0 * np.pi * np.fft.fftfreq(n_omega, d=dt_grid)
    omega_c = omega - 1j * a_damping

    m_arr = np.arange(-M_wavenumbers, M_wavenumbers + 1)
    k_m = 2.0 * np.pi * m_arr / L_periodicity
    n_k = len(k_m)

    W = omega_c[:, None]
    K = k_m[None, :]
    qw_sq = (W / V_upper) ** 2 - K ** 2
    qw = np.sqrt(qw_sq)
    qw = np.where(np.imag(qw) > 0, -qw, qw)

    # Horizontal slowness p_x = k_x / ω (complex)
    # Avoid 0/0 at ω=0: use a small epsilon.
    omega_safe = np.where(np.abs(W) < 1e-15, 1e-15 + 0j, W)
    p_x = K / omega_safe

    # Anisotropic R/T via 3×3 system — return ALL modes (R, T_qP,
    # T_qSV, lower-layer slownesses, polarisations) so the
    # transmitted-wave synthesis below has everything it needs.
    #
    # Unit convention (plan §83 Stage F2 bugfix 2026-05-16):
    # `acoustic_elastic_reflection` expects qw in **slowness**
    # form (1/V² − p_x²)^½. The DWN engine's `qw` is in
    # **wavenumber** form (ω·q_slow). Pass qw/W to recover the
    # expected slowness form; without this, BC2's k_w factor was
    # off by ω and T_qP came out ~5 orders of magnitude too
    # small.
    omega_safe_full = np.where(np.abs(W) < 1e-15, 1e-15 + 0j, W)
    qw_slow = qw / omega_safe_full
    rt = acoustic_elastic_reflection(p_x, W, qw_slow,
                                      rho_upper, V_upper, C_lower,
                                      return_transmitted=True)
    R = rt["R"]
    T_qP = rt["T_qP"]
    T_qSV = rt["T_qSV"]
    qz_qP = rt["qz_qP"]
    qz_qSV = rt["qz_qSV"]
    # Plan §83 Stage F2 — expose BOTH polarisation components.
    # The §82 fix used only Uz; Codex review 2026-05-16 HIGH finding
    # flagged dropping Ux. Rendering divergence ∇·u =
    # i(k_x·Ux + k_z·Uz) naturally includes both via the inner
    # product k·U.
    Ux_qP = rt["Ux_qP"]
    Uz_qP = rt["Uz_qP"]
    Ux_qSV = rt["Ux_qSV"]
    Uz_qSV = rt["Uz_qSV"]
    # Handle ω → 0 limit cleanly: R / T values may be NaN.
    R = np.where(np.isfinite(R), R, 0.0)
    T_qP = np.where(np.isfinite(T_qP), T_qP, 0.0)
    T_qSV = np.where(np.isfinite(T_qSV), T_qSV, 0.0)

    src_spec = ricker_spectrum(omega_c, f0, t_delay=t_delay)

    nx = len(x_grid)
    nz = len(z_grid)
    u_out = np.zeros((nx, nz), dtype=np.float64)

    # Pre-compute x-phase matrix; same for every z slice.
    dxs = x_grid - x_source
    phase = np.exp(1j * np.outer(dxs, k_m))

    # Plan §83 Stage F1: rendered quantity is **divergence of
    # displacement** ∇·u (a scalar, rotation-invariant; matches
    # the FD benchmark's canonical `div_late`). Upper layer:
    # ∇²φ = -(ω/V_U)² φ. Lower layer per mode m:
    # ∇·u^m = i·(k_x·U_x^m + k_z^m·U_z^m), summed weighted by T_m.
    upper_div_factor = -(W / V_upper) ** 2     # (n_omega, 1)
    qw_safe = np.where(np.abs(qw) < 1e-10, 1e-10, qw)

    for iz, z in enumerate(z_grid):
        if z > z_interface:
            # ─── Upper-layer divergence kernel ──────────────────
            abs_dz = abs(z - z_source)
            z_refl = z + z_source - 2.0 * z_interface

            prop_direct = (
                np.exp(-1j * qw * abs_dz) if include_direct
                else np.zeros_like(qw))
            prop_refl = (
                R * np.exp(-1j * qw * z_refl) if include_reflected
                else np.zeros_like(qw))
            kernel = (upper_div_factor
                      * (prop_direct + prop_refl)
                      / (2j * qw_safe))
        else:
            # ─── Lower-layer divergence (both polarisations) ────
            # Plan §83 Stage F2 (closes Codex 1st HIGH finding).
            # ∇·u^m = i·(k_x·U_x^m + k_z^m·U_z^m), where
            # k_z^m = W·q_z^m (q_z is slowness from Christoffel).
            #
            # Normalization (plan §83 derivation):
            #   1. The 2D Helmholtz Green's function in (k_x, ω)
            #      gives φ̃ = (-1/(2j q_w)) · exp(-iq_w|z−z_s|)
            #      · src_spec(ω) for the upper-layer potential.
            #   2. At the interface (z=z_a, z_s > z_a), the
            #      incident-wave vertical displacement is
            #      u_z^inc = ∂_z φ̃ |_{z_a-} = +(prop_src/2) · src_spec.
            #      (∂_z of (1/(2j q_w))·exp(-iq_w(z_s-z)) gives
            #      +iq_w/(2j q_w) = +1/2 at z<z_s.)
            #   3. The 3×3 system was derived in plan §81 Stage C2
            #      with BC1: 1 - R = U_z^qP T_qP + U_z^qSV T_qSV,
            #      i.e. **R, T_qP, T_qSV are referenced to unit
            #      incident u_z at the interface**.
            #   4. Therefore the actual lower-layer displacement of
            #      mode m is T_m·(Ux^m, Uz^m)·u_z^inc·prop_T.
            #   5. Divergence: ∇·u^m = i·(k_x·Ux^m + k_z^m·Uz^m).
            #
            # Combining: lower kernel = -(prop_src/2) · Σ_m T_m·div_m·prop_T
            # (with src_spec applied uniformly later).
            if not include_reflected:
                continue
            src_to_iface = z_source - z_interface   # > 0 (src above)
            dz_below = z_interface - z              # > 0 (rcv below)
            prop_src = np.exp(-1j * qw * src_to_iface)
            prop_qP = np.exp(-1j * W * qz_qP * dz_below)
            prop_qSV = np.exp(-1j * W * qz_qSV * dz_below)
            # Per-mode divergence amplitude (wavenumber form).
            # K is k_x in this engine; W*qz is k_z.
            div_qP = 1j * (K * Ux_qP + W * qz_qP * Uz_qP)
            div_qSV = 1j * (K * Ux_qSV + W * qz_qSV * Uz_qSV)
            kernel = (0.5 * prop_src
                      * (T_qP * div_qP * prop_qP
                         + T_qSV * div_qSV * prop_qSV))

        kernel = kernel * src_spec[:, None]

        u_omega = (kernel @ phase.T).T
        u_omega /= L_periodicity

        u_t = np.fft.ifft(u_omega, axis=-1) * n_omega / T_max
        u_t = u_t.real

        u_target = u_t[:, it] * np.exp(a_damping * t_grid[it])
        u_out[:, iz] = u_target

    return u_out


def dwn_wavefield_acoustic_anisotropic_dipping(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    x_anchor: float,
    y_anchor: float,
    dip_deg: float,
    rho_upper: float, V_upper: float,
    C_lower_lab: dict,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
    primed_grid_refinement: float = 3.0,
) -> np.ndarray:
    """Phase 2d DWN with a dipping interface.

    Strategy (plan §81b Stage B + Phase 2d):
    1. Rotate source about ``(x_anchor, y_anchor)`` by $+\\beta$
       — interface becomes horizontal at $z' = y_a$ in primed frame.
    2. Bond-rotate ``C_lower_lab`` by ``-dip_deg`` so the rotated
       stiffness reflects the effective tilt $\\theta - \\beta$ in
       the rotated frame.
    3. Compute DWN with anisotropic R on a refined primed-frame
       grid.
    4. Bilinear-interpolate back to lab-frame receivers.

    Parameters
    ----------
    Same as ``dwn_wavefield_acoustic_acoustic_dipping`` but
    ``C_lower_lab`` is a stiffness dict in the **lab frame**
    (must include the layer's intrinsic Thomsen tilt $\\theta$
    via standard Cij components). The function applies an
    additional Bond rotation by $-\\beta$ internally so the
    primed-frame Cij are correct for the horizontal-interface
    inner DWN call.
    """
    from scipy.interpolate import RegularGridInterpolator
    from bond_rotation import compute_tti_stiffness_2d

    beta_rad = np.deg2rad(dip_deg)

    # Source in primed frame
    xs_prime, zs_prime = _rotate_about_anchor(
        np.array([x_source]), np.array([z_source]),
        beta_rad, x_anchor, y_anchor,
    )
    xs_prime = float(xs_prime[0])
    zs_prime = float(zs_prime[0])

    # Lab grid corners → primed frame
    x_lab_min = float(np.min(x_grid))
    x_lab_max = float(np.max(x_grid))
    z_lab_min = float(np.min(z_grid))
    z_lab_max = float(np.max(z_grid))
    corners_x = np.array([x_lab_min, x_lab_max, x_lab_min, x_lab_max])
    corners_z = np.array([z_lab_min, z_lab_min, z_lab_max, z_lab_max])
    corners_xp, corners_zp = _rotate_about_anchor(
        corners_x, corners_z, beta_rad, x_anchor, y_anchor,
    )

    dx_lab = (
        float(np.diff(x_grid).mean()) if x_grid.size > 1 else 1e-3
    )
    dz_lab = (
        float(np.diff(z_grid).mean()) if z_grid.size > 1 else 1e-3
    )
    dx_primed = min(dx_lab, dz_lab) / float(primed_grid_refinement)

    x_pmin = float(np.min(corners_xp)) - 2.0 * dx_primed
    x_pmax = float(np.max(corners_xp)) + 2.0 * dx_primed
    z_pmin = float(np.min(corners_zp)) - 2.0 * dx_primed
    z_pmax = float(np.max(corners_zp)) + 2.0 * dx_primed

    n_x_primed = max(int(np.ceil((x_pmax - x_pmin) / dx_primed)) + 1, 4)
    n_z_primed = max(int(np.ceil((z_pmax - z_pmin) / dx_primed)) + 1, 4)
    x_primed = np.linspace(x_pmin, x_pmax, n_x_primed)
    z_primed = np.linspace(z_pmin, z_pmax, n_z_primed)

    # Rotate the lower-layer Cij to the primed frame.
    # The lab-frame stiffness `C_lower_lab` already has the layer's
    # intrinsic tilt baked in. We need to apply an ADDITIONAL Bond
    # rotation by `-beta` to express it in the primed frame.
    # For simplicity, recompute from Thomsen if available; otherwise
    # apply a general 2D rotation here. We use the original Thomsen
    # parameters stored in C_lower_lab._thomsen if present.
    if "_thomsen" in C_lower_lab:
        Vp, Vs, rho_l, eps, delta, theta_lab = C_lower_lab["_thomsen"]
        theta_primed = theta_lab - beta_rad
        C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
            Vp, Vs, rho_l, eps, delta, theta_primed,
        )
        C_lower_primed = {
            "C11": float(C11), "C33": float(C22),
            "C13": float(C12), "C15": float(C16),
            "C35": float(C26), "C55": float(C66),
            "rho": float(rho_l),
        }
    else:
        # Fallback: assume C_lower_lab is the primed-frame
        # stiffness already (caller has handled rotation).
        C_lower_primed = C_lower_lab

    u_primed = dwn_wavefield_acoustic_anisotropic_horizontal(
        x_grid=x_primed, z_grid=z_primed, t_target=t_target,
        x_source=xs_prime, z_source=zs_prime,
        z_interface=y_anchor,
        rho_upper=rho_upper, V_upper=V_upper,
        C_lower=C_lower_primed,
        f0=f0, T_max=T_max,
        L_periodicity=L_periodicity,
        n_omega=n_omega, M_wavenumbers=M_wavenumbers,
        a_damping=a_damping, t_delay=t_delay,
        include_direct=include_direct,
        include_reflected=include_reflected,
    )

    interp = RegularGridInterpolator(
        (x_primed, z_primed), u_primed,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    X_lab, Z_lab = np.meshgrid(x_grid, z_grid, indexing="ij")
    X_lab_prime, Z_lab_prime = _rotate_about_anchor(
        X_lab, Z_lab, beta_rad, x_anchor, y_anchor,
    )
    pts = np.stack([X_lab_prime.ravel(), Z_lab_prime.ravel()], axis=-1)
    u_lab = interp(pts).reshape(X_lab.shape)
    return u_lab


def dwn_wavefield_acoustic_elastic_elastic_stack_horizontal(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    z_interface_upper: float,
    h_middle: float,
    rho_upper: float, V_upper: float,
    C_middle: dict,
    C_lower: dict,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
) -> np.ndarray:
    """Plan §90 Stage 2 — DWN snapshot of the 3-region
    acoustic-elastic(VTI)-elastic(TTI) stack with a horizontal
    interface.

    Mirrors :func:`dwn_wavefield_acoustic_anisotropic_horizontal`
    but uses the 7×7 stack R/T solver
    (:func:`anisotropic_rt.acoustic_elastic_elastic_stack_reflection`)
    and renders all three regions:

    - **Upper acoustic** (``z > z_interface_upper``): incident +
      reflected acoustic potential → divergence
      ``-(ω/V_U)² · φ`` (matches the Phase 2d sign convention).
    - **Middle VTI** (``z_interface_lower < z < z_interface_upper``
      with ``z_interface_lower = z_interface_upper - h_middle``):
      4 elastic modes (qP↓, qSV↓, qP↑, qSV↑) summed with
      divergence amplitudes. Down-mode reference at the bottom
      of the middle layer; up-mode reference at the top.
    - **Lower TTI** (``z ≤ z_interface_lower``): 2 transmitted
      modes (qP, qSV) — same form as Phase 2d's lower kernel.

    Quantity rendered: **divergence of displacement** in all three
    regions, matching the FD benchmark's canonical ``div_late``
    quantity (Stage F1 of plan §83 — rotation-invariant scalar).

    Parameters
    ----------
    z_interface_upper : float
        Upper interface position (acoustic ↔ middle VTI).
        ``z_source > z_interface_upper`` (source above
        the upper interface).
    h_middle : float
        Thickness of the middle VTI layer. The lower interface
        is implicitly at ``z_interface_upper - h_middle``.
    C_middle : dict
        Middle-layer stiffness with keys C11, C13, C15, C33,
        C35, C55, rho. Standard Voigt naming.
    C_lower : dict
        Lower-layer (TTI) stiffness, same schema.
    """
    from anisotropic_rt import acoustic_elastic_elastic_stack_reflection

    z_interface_lower = z_interface_upper - h_middle

    if T_max is None:
        T_max = 1.2 * (t_target if t_target > 0 else 1.0)
    if a_damping is None:
        a_damping = np.pi / T_max
    V_max = max(
        V_upper,
        float(np.sqrt(C_middle["C11"] / C_middle["rho"])),
        float(np.sqrt(C_lower["C11"] / C_lower["rho"])),
    )
    if L_periodicity is None:
        x_max = max(
            float(np.max(np.abs(x_grid - x_source))),
            1e-3,
        )
        L_periodicity = 1.3 * (V_max * T_max + x_max)
    if t_delay is None:
        t_delay = 1.5 / f0

    dt_grid = T_max / n_omega
    t_grid = np.arange(n_omega) * dt_grid
    it = int(np.argmin(np.abs(t_grid - t_target)))
    omega = 2.0 * np.pi * np.fft.fftfreq(n_omega, d=dt_grid)
    omega_c = omega - 1j * a_damping

    m_arr = np.arange(-M_wavenumbers, M_wavenumbers + 1)
    k_m = 2.0 * np.pi * m_arr / L_periodicity
    n_k = len(k_m)

    W = omega_c[:, None]
    K = k_m[None, :]
    qw_sq = (W / V_upper) ** 2 - K ** 2
    qw = np.sqrt(qw_sq)
    qw = np.where(np.imag(qw) > 0, -qw, qw)

    omega_safe = np.where(np.abs(W) < 1e-15, 1e-15 + 0j, W)
    p_x = K / omega_safe
    qw_slow = qw / omega_safe

    # 7×7 stack R/T — returns R + 4 middle modes + 2 transmitted
    # modes + all qz/polarisation data needed for the three
    # rendering regions below.
    rt = acoustic_elastic_elastic_stack_reflection(
        p_x, W, qw_slow,
        rho_upper, V_upper,
        C_middle, h_middle,
        C_lower,
        return_full=True,
    )
    R = np.where(np.isfinite(rt["R"]), rt["R"], 0.0)
    A_pd = np.where(np.isfinite(rt["A_pd"]), rt["A_pd"], 0.0)
    A_sd = np.where(np.isfinite(rt["A_sd"]), rt["A_sd"], 0.0)
    A_pu = np.where(np.isfinite(rt["A_pu"]), rt["A_pu"], 0.0)
    A_su = np.where(np.isfinite(rt["A_su"]), rt["A_su"], 0.0)
    T_qP = np.where(np.isfinite(rt["T_qP"]), rt["T_qP"], 0.0)
    T_qSV = np.where(np.isfinite(rt["T_qSV"]), rt["T_qSV"], 0.0)
    qz_p_d = rt["qz_p_d"]; qz_s_d = rt["qz_s_d"]
    qz_p_u = rt["qz_p_u"]; qz_s_u = rt["qz_s_u"]
    qz_qP_l = rt["qz_qP_lower"]; qz_qSV_l = rt["qz_qSV_lower"]
    Ux_p_d = rt["Ux_p_d"]; Uz_p_d = rt["Uz_p_d"]
    Ux_s_d = rt["Ux_s_d"]; Uz_s_d = rt["Uz_s_d"]
    Ux_p_u = rt["Ux_p_u"]; Uz_p_u = rt["Uz_p_u"]
    Ux_s_u = rt["Ux_s_u"]; Uz_s_u = rt["Uz_s_u"]
    Ux_qP_l = rt["Ux_qP_lower"]; Uz_qP_l = rt["Uz_qP_lower"]
    Ux_qSV_l = rt["Ux_qSV_lower"]; Uz_qSV_l = rt["Uz_qSV_lower"]

    src_spec = ricker_spectrum(omega_c, f0, t_delay=t_delay)

    nx = len(x_grid)
    nz = len(z_grid)
    u_out = np.zeros((nx, nz), dtype=np.float64)

    dxs = x_grid - x_source
    phase = np.exp(1j * np.outer(dxs, k_m))

    upper_div_factor = -(W / V_upper) ** 2
    qw_safe = np.where(np.abs(qw) < 1e-10, 1e-10, qw)

    # Per-mode divergence amplitudes in the middle layer.
    # Both down and up modes use Phase 2d's lab-frame convention:
    # profile ∝ e^{+iω q z'} → ∂_z = +iω q → div = i(k_x U_x + ω q U_z).
    # The down/up distinction is in the q value (different roots).
    div_pd = 1j * (K * Ux_p_d + W * qz_p_d * Uz_p_d)
    div_sd = 1j * (K * Ux_s_d + W * qz_s_d * Uz_s_d)
    div_pu = 1j * (K * Ux_p_u + W * qz_p_u * Uz_p_u)
    div_su = 1j * (K * Ux_s_u + W * qz_s_u * Uz_s_u)

    # Per-mode divergence in the lower layer — Phase 2d convention
    # (transmitted-mode reference is at z_interface_lower, receiver
    # below → ∂_z = +iω q^↓).
    div_qP_l = 1j * (K * Ux_qP_l + W * qz_qP_l * Uz_qP_l)
    div_qSV_l = 1j * (K * Ux_qSV_l + W * qz_qSV_l * Uz_qSV_l)

    # The 7×7 system's R, A_*, T_* are referenced to **incident
    # u_z = 1 at the upper interface** (plan §90.1 stack BC1
    # derivation: "1 - R = sum"). The DWN engine's incident-acoustic
    # potential at the upper interface contributes
    # u_z^inc = +(prop_src/2) · src_spec where
    # prop_src = exp(-iω q_w · (z_source - z_interface_upper)).
    # Same factor as Phase 2d's transmitted-wave synthesis.
    src_to_iface_upper = z_source - z_interface_upper

    for iz, z in enumerate(z_grid):
        if z > z_interface_upper:
            # ─── Upper acoustic layer: divergence ───────────────
            abs_dz = abs(z - z_source)
            z_refl = z + z_source - 2.0 * z_interface_upper
            prop_direct = (
                np.exp(-1j * qw * abs_dz) if include_direct
                else np.zeros_like(qw))
            prop_refl = (
                R * np.exp(-1j * qw * z_refl) if include_reflected
                else np.zeros_like(qw))
            kernel = (upper_div_factor
                      * (prop_direct + prop_refl)
                      / (2j * qw_safe))
        elif z >= z_interface_lower:
            # ─── Middle VTI layer: 4 modes ──────────────────────
            if not include_reflected:
                continue
            z_prime = z - z_interface_lower    # in [0, h_middle]
            # Down-mode reference at z'=h_middle (top): profile
            # value at z' is e^{+iω q^↓ (z' - h)}; at ref (z'=h): 1.
            prop_pd = np.exp(+1j * W * qz_p_d * (z_prime - h_middle))
            prop_sd = np.exp(+1j * W * qz_s_d * (z_prime - h_middle))
            # Up-mode reference at z'=0 (bottom): profile value at
            # z' is e^{+iω q^↑ z'}; at ref (z'=0): 1.
            prop_pu = np.exp(+1j * W * qz_p_u * z_prime)
            prop_su = np.exp(+1j * W * qz_s_u * z_prime)
            # Incident-displacement factor at upper interface
            prop_src = np.exp(-1j * qw * src_to_iface_upper)
            kernel = (0.5 * prop_src * (
                A_pd * div_pd * prop_pd
                + A_sd * div_sd * prop_sd
                + A_pu * div_pu * prop_pu
                + A_su * div_su * prop_su
            ))
        else:
            # ─── Lower TTI layer: 2 transmitted modes ───────────
            if not include_reflected:
                continue
            dz_below = z_interface_lower - z   # > 0 (rcv below)
            prop_src = np.exp(-1j * qw * src_to_iface_upper)
            prop_qP = np.exp(-1j * W * qz_qP_l * dz_below)
            prop_qSV = np.exp(-1j * W * qz_qSV_l * dz_below)
            kernel = (0.5 * prop_src
                      * (T_qP * div_qP_l * prop_qP
                         + T_qSV * div_qSV_l * prop_qSV))

        kernel = kernel * src_spec[:, None]

        u_omega = (kernel @ phase.T).T
        u_omega /= L_periodicity

        u_t = np.fft.ifft(u_omega, axis=-1) * n_omega / T_max
        u_t = u_t.real

        u_target = u_t[:, it] * np.exp(a_damping * t_grid[it])
        u_out[:, iz] = u_target

    return u_out


def dwn_wavefield_acoustic_elastic_elastic_stack_dipping(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    t_target: float,
    x_source: float,
    z_source: float,
    x_anchor: float,
    y_anchor: float,
    dip_deg: float,
    h_middle: float,
    rho_upper: float, V_upper: float,
    C_middle_lab: dict,
    C_lower_lab: dict,
    f0: float,
    T_max: float | None = None,
    L_periodicity: float | None = None,
    n_omega: int = 256,
    M_wavenumbers: int = 1024,
    a_damping: float | None = None,
    t_delay: float | None = None,
    include_direct: bool = True,
    include_reflected: bool = True,
    primed_grid_refinement: float = 3.0,
) -> np.ndarray:
    """Plan §90 Stage 3 — DWN with a 3-region stack at a dipping
    interface.

    Same rotation-then-interpolate strategy as
    :func:`dwn_wavefield_acoustic_anisotropic_dipping` (plan
    §81b). The interface anchor at ``(x_anchor, y_anchor)`` lies
    on the **upper** interface (between acoustic upper and middle
    VTI); the lower interface is at depth ``h_middle`` below it
    in the primed frame.

    Both middle and lower layer Cij are passed in the **lab
    frame**; the wrapper applies the additional Bond rotation by
    ``-beta`` internally so the primed-frame solver sees correctly-
    rotated stiffness for the horizontal-stack inner call.
    """
    from scipy.interpolate import RegularGridInterpolator
    from bond_rotation import compute_tti_stiffness_2d

    beta_rad = np.deg2rad(dip_deg)

    xs_prime, zs_prime = _rotate_about_anchor(
        np.array([x_source]), np.array([z_source]),
        beta_rad, x_anchor, y_anchor,
    )
    xs_prime = float(xs_prime[0])
    zs_prime = float(zs_prime[0])

    x_lab_min = float(np.min(x_grid))
    x_lab_max = float(np.max(x_grid))
    z_lab_min = float(np.min(z_grid))
    z_lab_max = float(np.max(z_grid))
    corners_x = np.array([x_lab_min, x_lab_max, x_lab_min, x_lab_max])
    corners_z = np.array([z_lab_min, z_lab_min, z_lab_max, z_lab_max])
    corners_xp, corners_zp = _rotate_about_anchor(
        corners_x, corners_z, beta_rad, x_anchor, y_anchor,
    )

    dx_lab = (
        float(np.diff(x_grid).mean()) if x_grid.size > 1 else 1e-3
    )
    dz_lab = (
        float(np.diff(z_grid).mean()) if z_grid.size > 1 else 1e-3
    )
    dx_primed = min(dx_lab, dz_lab) / float(primed_grid_refinement)

    x_pmin = float(np.min(corners_xp)) - 2.0 * dx_primed
    x_pmax = float(np.max(corners_xp)) + 2.0 * dx_primed
    z_pmin = float(np.min(corners_zp)) - 2.0 * dx_primed
    z_pmax = float(np.max(corners_zp)) + 2.0 * dx_primed

    n_x_primed = max(int(np.ceil((x_pmax - x_pmin) / dx_primed)) + 1, 4)
    n_z_primed = max(int(np.ceil((z_pmax - z_pmin) / dx_primed)) + 1, 4)
    x_primed = np.linspace(x_pmin, x_pmax, n_x_primed)
    z_primed = np.linspace(z_pmin, z_pmax, n_z_primed)

    def _rotate_cij_to_primed(C_lab: dict) -> dict:
        """Bond-rotate the lab-frame stiffness by -beta to the
        primed frame. Requires `_thomsen` metadata on the input
        (Vp, Vs, rho, eps, delta, theta_lab); else assumes the
        input is already primed-frame Cij."""
        if "_thomsen" not in C_lab:
            return C_lab
        Vp, Vs, rho_l, eps, delta, theta_lab = C_lab["_thomsen"]
        theta_primed = theta_lab - beta_rad
        C11, C22, C12, C16, C26, C66 = compute_tti_stiffness_2d(
            Vp, Vs, rho_l, eps, delta, theta_primed,
        )
        return {
            "C11": float(C11), "C33": float(C22),
            "C13": float(C12), "C15": float(C16),
            "C35": float(C26), "C55": float(C66),
            "rho": float(rho_l),
        }

    C_middle_primed = _rotate_cij_to_primed(C_middle_lab)
    C_lower_primed = _rotate_cij_to_primed(C_lower_lab)

    u_primed = dwn_wavefield_acoustic_elastic_elastic_stack_horizontal(
        x_grid=x_primed, z_grid=z_primed, t_target=t_target,
        x_source=xs_prime, z_source=zs_prime,
        z_interface_upper=y_anchor,
        h_middle=h_middle,
        rho_upper=rho_upper, V_upper=V_upper,
        C_middle=C_middle_primed,
        C_lower=C_lower_primed,
        f0=f0, T_max=T_max,
        L_periodicity=L_periodicity,
        n_omega=n_omega, M_wavenumbers=M_wavenumbers,
        a_damping=a_damping, t_delay=t_delay,
        include_direct=include_direct,
        include_reflected=include_reflected,
    )

    interp = RegularGridInterpolator(
        (x_primed, z_primed), u_primed,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    X_lab, Z_lab = np.meshgrid(x_grid, z_grid, indexing="ij")
    X_lab_prime, Z_lab_prime = _rotate_about_anchor(
        X_lab, Z_lab, beta_rad, x_anchor, y_anchor,
    )
    pts = np.stack([X_lab_prime.ravel(), Z_lab_prime.ravel()], axis=-1)
    u_lab = interp(pts).reshape(X_lab.shape)
    return u_lab


__all__ = [
    "ricker_spectrum",
    "acoustic_acoustic_reflection_pdomain",
    "acoustic_reflection_from_qw",
    "dwn_wavefield_acoustic_acoustic_horizontal",
    "dwn_time_series_acoustic_acoustic_horizontal",
    "dwn_wavefield_acoustic_acoustic_dipping",
    "dwn_wavefield_acoustic_anisotropic_horizontal",
    "dwn_wavefield_acoustic_anisotropic_dipping",
    "dwn_wavefield_acoustic_elastic_elastic_stack_horizontal",
    "dwn_wavefield_acoustic_elastic_elastic_stack_dipping",
]
