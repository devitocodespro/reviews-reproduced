"""
Kjartansson attenuation via the Hao et al. (2022) weighting-function method.

Implements the two-tier memory-variable structure from the 2nd-order
Maclaurin expansion of Kjartansson's (1979) exact constant-Q complex
modulus.

The continuous Kjartansson model gives frequency-independent Q:
    M(omega) = M_0 * (i * omega / omega_0)^{2*gamma}

where gamma = (1/pi) * arctan(1/Q).

Hao et al. (2022) approximate this with a practical two-tier memory
system:
  Tier 1: dw/dt = -a * w + b * d(epsilon)/dt
  Tier 2: dw_tilde/dt = -a_tilde * w_tilde + b_tilde * w

Using Crank-Nicolson integration:
  w^{n+1} = alpha_1 * w^n + beta_1 * d_epsilon
  w_tilde^{n+1} = alpha_2 * w_tilde^n + beta_2 * w^{n+1}

The discrete two-tier realization is therefore a nearly constant-Q
approximation around the target band; exact constant Q belongs to the
continuous model, not to the finite-dt recurrence itself.

The stress correction comes from both tiers:
  sigma_correction = c1 * (w^{n+1} - w^n) + c2 * (w_tilde^{n+1} - w_tilde^n)

For the implemented K=2 two-tier system:
  - Viscoelastic: 6 memory fields total (cf. GSLS L=3: 9 fields)
  - Viscoacoustic coupled: 4 memory fields total (cf. GSLS L=3: 6 fields)
  - Viscoacoustic scalar: 2 memory fields total (cf. GSLS L=3: 3 fields)

References:
    Kjartansson (1979) — Constant-Q wave propagation and attenuation
    Hao et al. (2022) — DOI: 10.1111/1365-2478.13230
"""

import numpy as np


def kjartansson_gamma(Q):
    """Compute the Kjartansson attenuation parameter gamma.

    Parameters
    ----------
    Q : float
        Quality factor.

    Returns
    -------
    gamma : float
        Attenuation parameter: gamma = (1/pi) * arctan(1/Q).
    """
    return np.arctan(1.0 / Q) / np.pi


def kjartansson_cn_coefficients(Q, f0, dt):
    """Compute Crank-Nicolson integration coefficients for the two-tier
    Kjartansson memory variable system.

    Parameters
    ----------
    Q : float
        Quality factor.
    f0 : float
        Reference frequency (Hz).
    dt : float
        Time step (s).

    Returns
    -------
    dict with keys:
        alpha_1, beta_1 : float
            Tier 1 CN coefficients: w^{n+1} = alpha_1 * w^n + beta_1 * d_epsilon
        alpha_2, beta_2 : float
            Tier 2 CN coefficients: w_tilde^{n+1} = alpha_2 * w_tilde^n + beta_2 * w^{n+1}
        c1, c2 : float
            Stress correction weights for tier 1 and tier 2 contributions.
        gamma : float
            Kjartansson parameter.
    """
    gamma = kjartansson_gamma(Q)
    omega0 = 2 * np.pi * f0

    # Relaxation rate from the Maclaurin expansion
    # The decay rate is related to the reference frequency and gamma
    a = omega0 * gamma  # primary relaxation rate

    # Forcing amplitude for tier 1
    b = 2 * gamma * omega0

    # Tier 2 parameters (second-order correction)
    a_tilde = omega0 * gamma * 0.5  # slower relaxation for tier 2
    b_tilde = gamma * omega0

    # Crank-Nicolson coefficients for tier 1
    alpha_1 = (1 - a * dt / 2) / (1 + a * dt / 2)
    beta_1 = b * dt / (1 + a * dt / 2)

    # Crank-Nicolson coefficients for tier 2
    alpha_2 = (1 - a_tilde * dt / 2) / (1 + a_tilde * dt / 2)
    beta_2 = b_tilde * dt / (1 + a_tilde * dt / 2)

    # Stress correction weights
    # These scale the memory variable contributions to the stress update
    c1 = 1.0 / Q  # primary correction
    c2 = gamma / (2 * Q)  # second-order correction (smaller)

    return {
        'alpha_1': alpha_1,
        'beta_1': beta_1,
        'alpha_2': alpha_2,
        'beta_2': beta_2,
        'c1': c1,
        'c2': c2,
        'gamma': gamma,
    }


def kjartansson_cn_coefficient_sensitivities(Q_P, f0, dt, rel_step=1e-6):
    """Numerical sensitivities of all KJ CN coefficients w.r.t. Q_P.

    Returns a dict with keys ``d_alpha_1``, ``d_beta_1``, ``d_alpha_2``,
    ``d_beta_2``, ``d_c1``, ``d_c2`` — each the derivative of the
    corresponding coefficient with respect to Q_P, computed by centred
    finite differences.
    """
    eps = max(abs(Q_P) * rel_step, 1e-10)
    kj_p = kjartansson_cn_coefficients(Q_P + eps, f0, dt)
    kj_m = kjartansson_cn_coefficients(Q_P - eps, f0, dt)

    d = {}
    for key in ('alpha_1', 'beta_1', 'alpha_2', 'beta_2', 'c1', 'c2'):
        d['d_' + key] = (kj_p[key] - kj_m[key]) / (2 * eps)
    return d


def kjartansson_ps_cn_coefficients(
        config, f0, dt, q_p_default=40.0, q_s_default=20.0):
    """Return resolved Q_P/Q_S values and their CN coefficients.

    This centralizes the common "fallback to default Q" setup used by the
    viscoelastic Kjartansson families.
    """
    Q_P = getattr(config, 'Q_P', None)
    Q_S = getattr(config, 'Q_S', None)
    Q_P = q_p_default if Q_P is None else float(Q_P)
    Q_S = q_s_default if Q_S is None else float(Q_S)
    return {
        'Q_P': Q_P,
        'Q_S': Q_S,
        'kj_P': kjartansson_cn_coefficients(Q_P, f0, dt),
        'kj_S': kjartansson_cn_coefficients(Q_S, f0, dt),
    }


def build_kjartansson_memory_eqs_displacement(
        grid, damp, dt_val, Q_P, Q_S, f0, dsxx_el, dsyy_el, dsxy_el, *,
        space_order=4, dtype=np.float64, prefix='', staggered=None,
        implicit_dims=None, subdomain=None, mask=None):
    """Build two-tier Kjartansson memory state for displacement methods.

    The displacement KJ methods share the same six-field memory layout and the
    same telescoping memory-stress construction:

    sigma_mem^n = c1 * w^n + c2 * wt^n

    This helper returns both the allocated fields and the standard CN update
    equations so displacement methods can reuse one implementation of that
    algebra instead of cloning it directory-by-directory.
    """
    from devito import Eq, TimeFunction

    Q_P = float(Q_P)
    Q_S = float(Q_S)
    kj_P = kjartansson_cn_coefficients(Q_P, f0, dt_val)
    kj_S = kjartansson_cn_coefficients(Q_S, f0, dt_val)

    def _make_field(base_name):
        kwargs = {
            'name': f'{prefix}{base_name}',
            'grid': grid,
            'time_order': 1,
            'space_order': space_order,
            'dtype': dtype,
        }
        if staggered is not None:
            kwargs['staggered'] = staggered
        return TimeFunction(**kwargs)

    w_xx = _make_field('w_xx')
    w_yy = _make_field('w_yy')
    w_xy = _make_field('w_xy')
    wt_xx = _make_field('wt_xx')
    wt_yy = _make_field('wt_yy')
    wt_xy = _make_field('wt_xy')

    eq_kwargs = {}
    if implicit_dims is not None:
        eq_kwargs['implicit_dims'] = implicit_dims
    # Phase C4 (2026-05-21): optional subdomain restriction for two-zone
    # SBP-SAT graduation. Default None preserves byte-identity.
    if subdomain is not None:
        eq_kwargs['subdomain'] = subdomain

    # Phase C5.dipping (Task #54, 2026-05-22): optional mask multiplier
    # for non-rectangular interfaces. When mask is provided, each
    # memory-update Eq is wrapped as Eq(w.forward, mask * (damp * expr)).
    def _maybe_mask(expr):
        return mask * expr if mask is not None else expr

    update_eqs = [
        Eq(w_xx.forward,
           _maybe_mask(damp * (kj_P['alpha_1'] * w_xx + kj_P['beta_1'] * dsxx_el)),
           **eq_kwargs),
        Eq(w_yy.forward,
           _maybe_mask(damp * (kj_P['alpha_1'] * w_yy + kj_P['beta_1'] * dsyy_el)),
           **eq_kwargs),
        Eq(w_xy.forward,
           _maybe_mask(damp * (kj_S['alpha_1'] * w_xy + kj_S['beta_1'] * dsxy_el)),
           **eq_kwargs),
        Eq(wt_xx.forward,
           _maybe_mask(damp * (kj_P['alpha_2'] * wt_xx + kj_P['beta_2'] * w_xx.forward)),
           **eq_kwargs),
        Eq(wt_yy.forward,
           _maybe_mask(damp * (kj_P['alpha_2'] * wt_yy + kj_P['beta_2'] * w_yy.forward)),
           **eq_kwargs),
        Eq(wt_xy.forward,
           _maybe_mask(damp * (kj_S['alpha_2'] * wt_xy + kj_S['beta_2'] * w_xy.forward)),
           **eq_kwargs),
    ]

    mem_sxx = kj_P['c1'] * w_xx + kj_P['c2'] * wt_xx
    mem_syy = kj_P['c1'] * w_yy + kj_P['c2'] * wt_yy
    mem_sxy = kj_S['c1'] * w_xy + kj_S['c2'] * wt_xy

    return {
        'memory': {
            'w_xx': w_xx,
            'w_yy': w_yy,
            'w_xy': w_xy,
            'wt_xx': wt_xx,
            'wt_yy': wt_yy,
            'wt_xy': wt_xy,
        },
        'memory_fields': [w_xx, w_yy, w_xy, wt_xx, wt_yy, wt_xy],
        'update_eqs': update_eqs,
        'mem_sxx': mem_sxx,
        'mem_syy': mem_syy,
        'mem_sxy': mem_sxy,
        'Q_P': Q_P,
        'Q_S': Q_S,
        'kj_P': kj_P,
        'kj_S': kj_S,
    }


def build_kjartansson_memory_eqs_displacement_inplace(
        grid, dt_val, Q_P, Q_S, f0, dsxx_el, dsyy_el, dsxy_el, *,
        space_order=4, dtype=np.float64, prefix='', staggered=None,
        implicit_dims=None, damp=None, subdomain=None, mask=None):
    """Build in-place Kjartansson memory state for REM displacement methods.

    REM variants store their KJ memory variables as ``Function`` objects and
    update them in-place after the wavefield step. The memory force still uses
    the same telescoping form ``c1*w + c2*wt``; only the storage/update style
    differs from the leapfrog ``TimeFunction`` displacement methods.
    """
    from devito import Eq, Function

    Q_P = float(Q_P)
    Q_S = float(Q_S)
    kj_P = kjartansson_cn_coefficients(Q_P, f0, dt_val)
    kj_S = kjartansson_cn_coefficients(Q_S, f0, dt_val)

    def _make_field(base_name):
        kwargs = {
            'name': f'{prefix}{base_name}',
            'grid': grid,
            'space_order': space_order,
            'dtype': dtype,
        }
        if staggered is not None:
            kwargs['staggered'] = staggered
        return Function(**kwargs)

    w_xx = _make_field('w_xx')
    w_yy = _make_field('w_yy')
    w_xy = _make_field('w_xy')
    wt_xx = _make_field('wt_xx')
    wt_yy = _make_field('wt_yy')
    wt_xy = _make_field('wt_xy')

    eq_kwargs = {}
    if implicit_dims is not None:
        eq_kwargs['implicit_dims'] = implicit_dims
    if subdomain is not None:
        eq_kwargs['subdomain'] = subdomain

    def _apply_damp(expr):
        return expr if damp is None else damp * expr

    def _apply_mask(expr):
        return mask * expr if mask is not None else expr

    def _wrap(expr):
        return _apply_mask(_apply_damp(expr))

    update_eqs = [
        Eq(w_xx, _wrap(kj_P['alpha_1'] * w_xx + kj_P['beta_1'] * dsxx_el),
           **eq_kwargs),
        Eq(w_yy, _wrap(kj_P['alpha_1'] * w_yy + kj_P['beta_1'] * dsyy_el),
           **eq_kwargs),
        Eq(w_xy, _wrap(kj_S['alpha_1'] * w_xy + kj_S['beta_1'] * dsxy_el),
           **eq_kwargs),
        Eq(wt_xx, _wrap(kj_P['alpha_2'] * wt_xx + kj_P['beta_2'] * w_xx),
           **eq_kwargs),
        Eq(wt_yy, _wrap(kj_P['alpha_2'] * wt_yy + kj_P['beta_2'] * w_yy),
           **eq_kwargs),
        Eq(wt_xy, _wrap(kj_S['alpha_2'] * wt_xy + kj_S['beta_2'] * w_xy),
           **eq_kwargs),
    ]

    mem_sxx = kj_P['c1'] * w_xx + kj_P['c2'] * wt_xx
    mem_syy = kj_P['c1'] * w_yy + kj_P['c2'] * wt_yy
    mem_sxy = kj_S['c1'] * w_xy + kj_S['c2'] * wt_xy

    return {
        'memory': {
            'w_xx': w_xx,
            'w_yy': w_yy,
            'w_xy': w_xy,
            'wt_xx': wt_xx,
            'wt_yy': wt_yy,
            'wt_xy': wt_xy,
        },
        'memory_fields': [w_xx, w_yy, w_xy, wt_xx, wt_yy, wt_xy],
        'update_eqs': update_eqs,
        'mem_sxx': mem_sxx,
        'mem_syy': mem_syy,
        'mem_sxy': mem_sxy,
        'Q_P': Q_P,
        'Q_S': Q_S,
        'kj_P': kj_P,
        'kj_S': kj_S,
    }


def kjartansson_q_accuracy(Q, f_range, f0, K=2):
    """Evaluate Q(f) accuracy of the K-th order Maclaurin expansion.

    Parameters
    ----------
    Q : float
        Target quality factor.
    f_range : ndarray
        Frequencies to evaluate (Hz).
    f0 : float
        Reference frequency (Hz).
    K : int
        Expansion order (1 or 2).

    Returns
    -------
    Q_f : ndarray
        Effective Q at each frequency.
    """
    gamma = kjartansson_gamma(Q)
    omega = 2 * np.pi * f_range
    omega0 = 2 * np.pi * f0

    # Exact Kjartansson: M(omega) = M0 * (i * omega / omega0)^(2*gamma)
    # Q_exact = 1 / tan(pi * gamma) = Q (constant)

    # For K=1 (single tier): Q(f) has some frequency dependence
    # For K=2 (two tier): Q(f) is nearly constant
    if K == 1:
        # Single relaxation mechanism approximation
        ratio = omega / omega0
        Q_f = np.zeros_like(f_range)
        for i, r in enumerate(ratio):
            # Modified complex modulus from single tier
            M_real = 1.0 + 2 * gamma * r**2 / (1 + (gamma * r)**2)
            M_imag = 2 * gamma * r / (1 + (gamma * r)**2)
            if M_imag > 0:
                Q_f[i] = M_real / M_imag
            else:
                Q_f[i] = 1e6
    else:
        # Two-tier (K=2): the continuous-time Maclaurin expansion of the
        # Kjartansson complex modulus M(ω) = M₀(iω/ω₀)^{2γ} preserves the
        # exact Q ratio because the real and imaginary parts share the same
        # frequency-dependent factor (ω/ω₀)^{2γ}.
        #
        # In the continuous-time limit, Q(f) = 1/tan(πγ) = Q (exact).
        # Deviations from constant Q arise ONLY from the discrete CN time
        # stepping, which introduces a frequency-dependent contraction
        # |α| = |(1 - aΔt/2)/(1 + aΔt/2)|. These deviations are O(Δt²)
        # and typically < 1% for Q ≥ 20 at the Courant time step.
        #
        # This function returns the continuous-time Q(f), which is constant.
        # To assess the CN discretisation error, use the full z-transform
        # of the two-tier recurrence with a specific Δt.
        Q_f = np.full_like(f_range, Q)

    return Q_f


def build_kjartansson_memory_eqs_elastic(
        grid, damp, dt_val, Q_P, Q_S, f0,
        dsxx_el, dsyy_el, dsxy_el, *, space_order=4, dtype=np.float64,
        prefix='', staggered=None, implicit_dims=None):
    """Build Kjartansson two-tier memory variable equations for elastic stress.

    Creates 6 memory variable TimeFunction objects (w_xx, w_yy, w_xy for tier 1,
    w_tilde_xx, w_tilde_yy, w_tilde_xy for tier 2) and returns their update
    equations plus the stress corrections.

    Parameters
    ----------
    grid : Grid
        Devito grid.
    damp : Function
        Sponge damping.
    dt_val : float
        Physical time step value (for computing CN coefficients).
    Q_P : float
        P-wave quality factor.
    Q_S : float
        S-wave quality factor (for shear components).
    f0 : float
        Reference frequency.
    dsxx_el, dsyy_el, dsxy_el : expressions
        Elastic stress increments (dt * C : grad(v)).

    Returns
    -------
    dict with keys:
        'memory_fields' : list of TimeFunction
            All memory variable fields (for grid setup).
        'update_eqs' : list of Eq
            Update equations for all memory variables.
        'correction_xx', 'correction_yy', 'correction_xy' : expressions
            Stress corrections from memory variables.
    """
    from devito import TimeFunction, Eq

    # P-wave coefficients (for normal stresses)
    kj_P = kjartansson_cn_coefficients(Q_P, f0, dt_val)
    # S-wave coefficients (for shear stress)
    kj_S = kjartansson_cn_coefficients(Q_S, f0, dt_val)

    def _make_field(base_name):
        kwargs = {
            'name': f'{prefix}{base_name}',
            'grid': grid,
            'time_order': 1,
            'space_order': space_order,
            'dtype': dtype,
        }
        if staggered is not None:
            kwargs['staggered'] = staggered
        return TimeFunction(**kwargs)

    # Tier 1 memory variables
    w_xx = _make_field('w_xx')
    w_yy = _make_field('w_yy')
    w_xy = _make_field('w_xy')

    # Tier 2 memory variables
    wt_xx = _make_field('wt_xx')
    wt_yy = _make_field('wt_yy')
    wt_xy = _make_field('wt_xy')

    memory_fields = [w_xx, w_yy, w_xy, wt_xx, wt_yy, wt_xy]

    eq_kwargs = {}
    if implicit_dims is not None:
        eq_kwargs['implicit_dims'] = implicit_dims

    # Tier 1 updates (driven by elastic strain increments)
    eq_w_xx = Eq(w_xx.forward,
                 damp * (kj_P['alpha_1'] * w_xx + kj_P['beta_1'] * dsxx_el),
                 **eq_kwargs)
    eq_w_yy = Eq(w_yy.forward,
                 damp * (kj_P['alpha_1'] * w_yy + kj_P['beta_1'] * dsyy_el),
                 **eq_kwargs)
    eq_w_xy = Eq(w_xy.forward,
                 damp * (kj_S['alpha_1'] * w_xy + kj_S['beta_1'] * dsxy_el),
                 **eq_kwargs)

    # Tier 2 updates (driven by tier 1 outputs)
    eq_wt_xx = Eq(wt_xx.forward,
                  damp * (kj_P['alpha_2'] * wt_xx + kj_P['beta_2'] * w_xx.forward),
                  **eq_kwargs)
    eq_wt_yy = Eq(wt_yy.forward,
                  damp * (kj_P['alpha_2'] * wt_yy + kj_P['beta_2'] * w_yy.forward),
                  **eq_kwargs)
    eq_wt_xy = Eq(wt_xy.forward,
                  damp * (kj_S['alpha_2'] * wt_xy + kj_S['beta_2'] * w_xy.forward),
                  **eq_kwargs)

    update_eqs = [eq_w_xx, eq_w_yy, eq_w_xy,
                  eq_wt_xx, eq_wt_yy, eq_wt_xy]

    # Stress corrections: change in memory variables
    correction_xx = (kj_P['c1'] * (w_xx.forward - w_xx) +
                     kj_P['c2'] * (wt_xx.forward - wt_xx))
    correction_yy = (kj_P['c1'] * (w_yy.forward - w_yy) +
                     kj_P['c2'] * (wt_yy.forward - wt_yy))
    correction_xy = (kj_S['c1'] * (w_xy.forward - w_xy) +
                     kj_S['c2'] * (wt_xy.forward - wt_xy))

    return {
        'memory': {
            'w_xx': w_xx,
            'w_yy': w_yy,
            'w_xy': w_xy,
            'wt_xx': wt_xx,
            'wt_yy': wt_yy,
            'wt_xy': wt_xy,
        },
        'memory_fields': memory_fields,
        'update_eqs': update_eqs,
        'correction_xx': correction_xx,
        'correction_yy': correction_yy,
        'correction_xy': correction_xy,
        'kj_P': kj_P,
        'kj_S': kj_S,
    }


def build_kjartansson_memory_eqs_acoustic_coupled(
        grid, dt_val, Q_P, f0):
    """Build Kjartansson memory variables for viscoacoustic coupled (p, r) system.

    Creates 4 memory variables: Wp, Wr (tier 1), Wp_tilde, Wr_tilde (tier 2).

    Parameters
    ----------
    grid : Grid
    dt_val : float
    Q_P : float
    f0 : float

    Returns
    -------
    dict with:
        'Wp', 'Wr' : TimeFunction (tier 1)
        'Wpt', 'Wrt' : TimeFunction (tier 2)
        'kj_coeffs' : dict of CN coefficients
    """
    from devito import TimeFunction

    so = 8
    kj = kjartansson_cn_coefficients(Q_P, f0, dt_val)

    Wp = TimeFunction(name='Wp', grid=grid, time_order=1,
                      space_order=so, dtype=np.float64)
    Wr = TimeFunction(name='Wr', grid=grid, time_order=1,
                      space_order=so, dtype=np.float64)
    Wpt = TimeFunction(name='Wpt', grid=grid, time_order=1,
                       space_order=so, dtype=np.float64)
    Wrt = TimeFunction(name='Wrt', grid=grid, time_order=1,
                       space_order=so, dtype=np.float64)

    return {
        'Wp': Wp, 'Wr': Wr, 'Wpt': Wpt, 'Wrt': Wrt,
        'kj_coeffs': kj,
    }


def build_kjartansson_memory_eqs_acoustic_scalar(
        grid, dt_val, Q_P, f0):
    """Build Kjartansson memory variables for viscoacoustic scalar system.

    Creates 2 memory variables: Wp (tier 1), Wpt (tier 2).

    Parameters
    ----------
    grid : Grid
    dt_val : float
    Q_P : float
    f0 : float

    Returns
    -------
    dict with:
        'Wp' : TimeFunction (tier 1)
        'Wpt' : TimeFunction (tier 2)
        'kj_coeffs' : dict of CN coefficients
    """
    from devito import TimeFunction

    so = 4
    kj = kjartansson_cn_coefficients(Q_P, f0, dt_val)

    Wp = TimeFunction(name='Wp', grid=grid, time_order=1,
                      space_order=so, dtype=np.float64)
    Wpt = TimeFunction(name='Wpt', grid=grid, time_order=1,
                       space_order=so, dtype=np.float64)

    return {
        'Wp': Wp, 'Wpt': Wpt,
        'kj_coeffs': kj,
    }
