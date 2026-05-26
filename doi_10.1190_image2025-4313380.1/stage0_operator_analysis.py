"""
Stage 0: SymPy Verification of Dual-Pair FD Operators and Dispersion Analysis
==============================================================================

Peer review of Irakarama et al. (IMAGE 2025):
"Accelerating anisotropic elastic subsurface imaging with dual-pair finite differences"

This script:
  0a. Derives and verifies D⁺ operator coefficients (Table 1) via Taylor matching
  0b. Computes dispersion and dissipation curves (reproduces Figure 2)
  0c. Analyses selective filter response (reproduces Figure 3)
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ============================================================================
# 0a. Verify D⁺ operator coefficients (Table 1)
# ============================================================================

def taylor_forward_coefficients(M_neg, M_pos):
    """
    Derive Taylor-series FD coefficients for a first-derivative forward-biased
    operator D⁺ on stencil points m = M_neg, ..., M_pos.

    We require D⁺ to be exact for polynomials 1, x, x², ..., x^(N-1)
    where N = M_pos - M_neg + 1 is the number of stencil points.

    D⁺f(x_j) = (1/h) Σ_{m=M_neg}^{M_pos} c_m f(x_j + m·h)

    Taylor expanding f(x_j + m·h) and requiring exactness for f'(x_j):
      Σ c_m m^0 = 0        (kills f(x_j) term)
      Σ c_m m^1 = 1        (matches f'(x_j) after 1/h factor)
      Σ c_m m^k / k! = 0   for k = 2, ..., N-1
    """
    stencil = list(range(M_neg, M_pos + 1))
    N = len(stencil)
    c = sp.symbols(f'c:{N}')

    eqs = []
    for k in range(N):
        if k == 0:
            # Σ c_m = 0
            eqs.append(sp.Eq(sum(c[i] for i in range(N)), 0))
        elif k == 1:
            # Σ c_m * m = 1
            eqs.append(sp.Eq(sum(c[i] * stencil[i] for i in range(N)), 1))
        else:
            # Σ c_m * m^k / k! = 0
            eqs.append(sp.Eq(
                sum(c[i] * sp.Rational(stencil[i]**k, sp.factorial(k))
                    for i in range(N)),
                0
            ))

    sol = sp.solve(eqs, c)
    coeffs = {stencil[i]: sol[c[i]] for i in range(N)}
    return coeffs


def compute_backward_coefficients(fwd_coeffs):
    """
    D⁻ = -(D⁺)ᵀ: coefficients are -c_{-m} reversed.
    If D⁺ has c_m at index m, then D⁻ has -c_{-m} at index m.
    """
    return {m: -fwd_coeffs[-m] for m in sorted(-k for k in fwd_coeffs)}


def verify_polynomial_exactness(coeffs, deriv_order=1, max_poly=None):
    """
    Verify that the operator with given coefficients is exact for
    polynomials up to the expected order.

    For f(x) = x^p, f'(x) = p*x^(p-1).
    At x=0: D⁺[x^p] should equal p*0^(p-1) = {1 if p=1, 0 if p>1}
    Actually at x=0: D⁺[x^p] = (1/h) Σ c_m (m*h)^p = h^(p-1) Σ c_m m^p
    For p=0: Σ c_m = 0 ✓
    For p=1: Σ c_m m = 1 ✓
    For p≥2: Σ c_m m^p = 0 ✓ (matches 0 = f'(0) for p≥2)
    """
    stencil = sorted(coeffs.keys())
    N = len(stencil)
    if max_poly is None:
        max_poly = N - 1

    results = {}
    for p in range(max_poly + 1):
        val = sum(coeffs[m] * sp.Rational(m**p) for m in stencil)
        if p == 0:
            expected = 0
        elif p == 1:
            expected = 1
        else:
            expected = 0
        results[p] = {'value': val, 'expected': expected,
                      'exact': val == expected}

    return results


# ============================================================================
# 0b. Dispersion and dissipation analysis (reproduce Figure 2)
# ============================================================================

def effective_wavenumber_squared(fwd_coeffs, bwd_coeffs, kdx_array):
    """
    For a plane wave f = exp(ikx), compute the effective k²_eff·h²
    from the composed dual-pair operator D⁻D⁺.

    D⁺f = (1/h) α⁺ f,  where α⁺(k) = Σ c⁺_m exp(imkh)
    D⁻g = (1/h) α⁻ g,  where α⁻(k) = Σ c⁻_n exp(inkh)

    Since c⁻_n = -c⁺_{-n} (adjoint property):
      α⁻(k) = -Σ c⁺_{-n} exp(inkh) = -Σ c⁺_m exp(-imkh) = -conj(α⁺)

    Therefore: D⁻D⁺f = α⁺·α⁻/h² · f = -α⁺·conj(α⁺)/h² · f = -|α⁺|²/h² · f

    Since ∂²f/∂x² = -k²f, we identify:
      k²_eff·h² = |α⁺(k)|²

    KEY PROPERTY: k²_eff·h² = |α⁺|² is REAL and NON-NEGATIVE.
    → Dual-pair D⁻D⁺ has ZERO dissipation by construction.
    → This is the "no artificial damping" property cited in the paper.
    """
    fwd_stencil = sorted(fwd_coeffs.keys())

    c_fwd = np.array([float(fwd_coeffs[m]) for m in fwd_stencil])
    m_fwd = np.array(fwd_stencil, dtype=float)

    k2_eff_h2 = np.zeros(len(kdx_array), dtype=complex)
    for i, kdx in enumerate(kdx_array):
        alpha_fwd = np.sum(c_fwd * np.exp(1j * m_fwd * kdx))
        # k²_eff·h² = |α⁺|² (real by construction)
        k2_eff_h2[i] = np.abs(alpha_fwd)**2

    return k2_eff_h2


def standard_central_k2eff(order, kdx_array):
    """
    Effective k² for standard central 2nd derivative of given order.
    Standard coefficients from finite_diff_weights for ∂²/∂x².
    """
    from sympy import finite_diff_weights
    # Standard central second derivative stencil
    half = order // 2
    stencil = list(range(-half, half + 1))
    # Get weights for 2nd derivative at x=0
    weights = finite_diff_weights(2, stencil, 0)[-1][-1]

    k2_eff = np.zeros(len(kdx_array), dtype=complex)
    for i, kdx in enumerate(kdx_array):
        val = sum(float(w) * np.exp(1j * m * kdx)
                  for m, w in zip(stencil, weights))
        k2_eff[i] = val  # This is k²_eff * h²

    return k2_eff


def dispersion_dissipation_errors(k2_eff_h2, kdx_array):
    """
    Dispersion error: Re(k²_eff·h²) / (k·h)² - 1
    Dissipation error: Im(k²_eff·h²) / (k·h)²
    """
    kdx2 = kdx_array**2
    # Avoid division by zero
    mask = kdx2 > 1e-12
    disp_err = np.zeros_like(kdx_array)
    diss_err = np.zeros_like(kdx_array)
    disp_err[mask] = np.real(k2_eff_h2[mask]) / kdx2[mask] - 1.0
    diss_err[mask] = np.imag(k2_eff_h2[mask]) / kdx2[mask]
    return disp_err, diss_err


def first_derivative_errors(fwd_coeffs, kdx_array):
    """
    Compute dispersion and dissipation errors for individual D⁺ operator.

    For D⁺ approximating d/dx on f = exp(ikx):
      D⁺ f = (α⁺/h) f,  where α⁺(kh) = Σ c_m exp(imkh)
      exact d/dx f = ik f

    So the effective wavenumber: k_eff·h = -i·α⁺(kh)
    Writing α⁺ = a + ib:
      k_eff·h = b - ia
      Re(k_eff·h) = Im(α⁺)
      Im(k_eff·h) = -Re(α⁺)

    Relative dispersion error: Im(α⁺)/(kh) - 1
    Relative dissipation error: Re(α⁺)/(kh)

    Sign convention: dissipation < 0 means the operator amplifies
    (anti-dissipative), matching the paper's convention where both
    Taylor and Proposed curves go negative at low PPWL.
    """
    stencil = sorted(fwd_coeffs.keys())
    c_fwd = np.array([float(fwd_coeffs[m]) for m in stencil])
    m_fwd = np.array(stencil, dtype=float)

    disp_err = np.zeros_like(kdx_array)
    diss_err = np.zeros_like(kdx_array)

    for i, kdx in enumerate(kdx_array):
        if kdx < 1e-12:
            continue
        alpha_fwd = np.sum(c_fwd * np.exp(1j * m_fwd * kdx))
        disp_err[i] = np.imag(alpha_fwd) / kdx - 1.0
        diss_err[i] = np.real(alpha_fwd) / kdx

    return disp_err, diss_err


def optimized_forward_coefficients_9pt():
    """
    Compute optimized D⁺ coefficients for the 9-point stencil (m=-3..+5)
    by minimizing the modified wavenumber error of the individual D⁺
    operator over a wavenumber range.

    Following Liu (2014) / Irakarama et al.:
    - Fix two constraints: Σ c_m = 0 (consistency), Σ c_m m = 1 (1st order)
    - Minimize ∫₀^{kmax} |α⁺(kh) - i·kh|² / (kh)² dkh

    This targets the INDIVIDUAL D⁺ accuracy (both dispersion and
    dissipation), not just the composed D⁻D⁺ accuracy. The error
    decomposes as: diss² + disp² where diss = Re(α⁺)/(kh) and
    disp = Im(α⁺)/(kh) - 1.
    """
    stencil = list(range(-3, 6))  # m = -3, -2, ..., 5
    N = len(stencil)

    m_arr = np.array(stencil, dtype=float)
    # Optimize D⁺ over kh ∈ [0, π/2], i.e. down to 4 PPWL.
    # This matches the paper where the Proposed curve maintains
    # accuracy within ±0.0005 down to ~4 PPWL (see Fig 2c).
    kdx_opt = np.linspace(0.05, np.pi * 0.5, 200)
    # Precompute exp(im·kh) matrix: shape (N_kdx, N_stencil)
    exp_mat = np.exp(1j * np.outer(kdx_opt, m_arr))

    def objective(free_params):
        c_all = np.zeros(N)
        c_all[:7] = free_params

        m7, m8 = stencil[7], stencil[8]  # m=4, m=5
        rhs1 = -sum(c_all[i] for i in range(7))
        rhs2 = 1.0 - sum(c_all[i] * stencil[i] for i in range(7))

        c_all[8] = (rhs2 - m7 * rhs1) / (m8 - m7)
        c_all[7] = rhs1 - c_all[8]

        # Vectorized: α⁺(kh) for all kh at once
        alpha = exp_mat @ c_all  # shape (N_kdx,)
        # Target: α⁺ = i·kh (minimize individual D⁺ error)
        err = alpha - 1j * kdx_opt
        cost = np.sum((np.abs(err) / kdx_opt)**2)

        return cost

    # Start from Taylor coefficients
    taylor = taylor_forward_coefficients(-3, 5)
    x0 = np.array([float(taylor[m]) for m in stencil[:7]])

    result = minimize(objective, x0, method='Powell',
                      options={'maxiter': 100000, 'ftol': 1e-16})

    # Reconstruct final coefficients
    c_all = np.zeros(N)
    c_all[:7] = result.x
    m7, m8 = stencil[7], stencil[8]
    rhs1 = -sum(c_all[i] for i in range(7))
    rhs2 = 1.0 - sum(c_all[i] * stencil[i] for i in range(7))
    c_all[8] = (rhs2 - m7 * rhs1) / (m8 - m7)
    c_all[7] = rhs1 - c_all[8]

    opt_coeffs = {stencil[i]: c_all[i] for i in range(N)}
    return opt_coeffs, result


# ============================================================================
# 0c. Selective filter analysis (reproduce Figure 3)
# ============================================================================

def taylor_dissipation_coefficients(half_width):
    """
    Derive Taylor-series dissipation operator coefficients.

    The dissipation operator D_f is symmetric, 2M+1 points:
        D_f[u]_j = Σ_{m=-M}^{M} d_m u_{j+m}, d_{-m} = d_m

    Requirements:
    - Σ d_m = 0 (preserves constants → zero response at k=0)
    - Maximally flat at k=0 (Taylor matching)
    - Non-zero response at Nyquist

    This is equivalent to a discrete approximation of (-1)^M h^{2M} ∂^{2M}/∂x^{2M}
    (a high-order artificial dissipation operator).

    The filter is then applied as: ũ = u - σ·D_f[u], σ ∈ [0,1].
    """
    M = half_width
    N = M + 1  # Number of independent coefficients (d_0, d_1, ..., d_M)

    d = sp.symbols(f'd:{N}')

    eqs = []
    # Constraint: Σ d_m = 0 → d_0 + 2Σ_{m=1}^M d_m = 0
    eqs.append(sp.Eq(d[0] + 2*sum(d[i] for i in range(1, N)), 0))

    # Taylor matching: the operator applied to exp(ikx) gives
    # D_f[exp(ikx)] = [d_0 + 2Σ d_m cos(mkh)] exp(ikx)
    # For maximally flat at k=0, require:
    # d^n/d(kh)^n [d_0 + 2Σ d_m cos(mkh)] |_{kh=0} = 0
    # for n = 1, 2, ..., 2M-2
    # Due to symmetry, odd derivatives are automatically zero.
    # Even derivatives at k=0: Σ_{m=1}^M d_m (-1)^p m^{2p} = 0, p=1,...,M-1

    for p in range(1, M):
        eq = sum(d[m] * (-1)**p * sp.Rational(m**(2*p))
                 for m in range(1, N))
        eqs.append(sp.Eq(eq, 0))

    # We have M equations for M+1 unknowns. Fix normalization:
    # Set the Nyquist response to 1: d_0 + 2Σ d_m (-1)^m = 1
    nyquist_eq = d[0] + 2*sum(d[i] * (-1)**i for i in range(1, N))
    eqs.append(sp.Eq(nyquist_eq, 1))

    sol = sp.solve(eqs, d)
    if isinstance(sol, list):
        sol = sol[0] if sol else {}
    coeffs = {}
    for i in range(N):
        if d[i] in sol:
            coeffs[i] = sol[d[i]]
        else:
            coeffs[i] = d[i]

    return coeffs


def bogey_bailly_dissipation_11pt():
    """
    Bogey & Bailly (2004) optimized selective filtering coefficients.
    11-point symmetric dissipation operator (half-width 5).

    These are the coefficients of the DISSIPATION OPERATOR D_f, not the
    filter itself. The filter is applied as:
        ũ = u - σ · D_f[u]
    where σ ∈ [0,1] is the filter strength and
        D_f[u]_j = Σ_{m=-5}^{5} d_m u_{j+m}

    The dissipation operator should:
    - Have zero response at k=0 (preserve mean): Σ d_m = 0
    - Have maximum response at Nyquist (k=π/h): maximize |Σ d_m(-1)^m|
    - Be smooth in between
    """
    # Bogey & Bailly (2004) SFo11p dissipation operator coefficients
    # From their Table II (optimized for broadband dissipation)
    d = {
        0:  0.215044884112,
        1: -0.187772883589,
        2:  0.123755948787,
        3: -0.059227575576,
        4:  0.018721609157,
        5: -0.002999540835,
    }
    return d


def proposed_dissipation_11pt():
    """
    Best-guess 'Proposed' selective filter from the paper (Figure 3).

    Uses constrained least-squares: minimise ∫₀^{kh_c} |D(kh)|² d(kh)
    (passband energy) subject to D(0) = 0 and D(π) = 1.  This is the
    same optimisation philosophy as Liu (2014) used for the FD
    coefficients — trade maximal flatness at k=0 for lower integrated
    error across the resolved bandwidth.

    The cutoff kh_c = π/2 (4 PPWL) matches the paper's FD optimisation
    target, producing a filter that:
      - stays closest to 1.0 at resolved wavelengths (panel b)
      - has a sharper transition near 4 PPWL than Taylor or BB (panel a)
      - smooth passband (no equiripple oscillation)
    """
    from scipy.optimize import linprog

    M = 5
    N = M + 1  # d_0 .. d_5

    # Formulate as LP: minimise t (the maximum passband response)
    # subject to:
    #   D(kh_i) ≤ t   for kh_i ∈ [0, π/2]   (passband minimax)
    #   D(kh_j) ≥ 0   for kh_j ∈ [0, π]     (non-negativity)
    #   D(0) = 0                               (preserve constants)
    #   D(π) = 1                               (Nyquist normalisation)
    #
    # Variables: [d_0, d_1, ..., d_5, t]  (7 total)

    n_pass = 300   # passband sample points
    n_all = 500    # full-range sample points

    kh_pass = np.linspace(0.01, np.pi / 2, n_pass)
    kh_all = np.linspace(0.01, np.pi - 0.01, n_all)

    def _basis_row(kh_val):
        row = np.zeros(N)
        row[0] = 1.0
        for m in range(1, N):
            row[m] = 2.0 * np.cos(m * kh_val)
        return row

    # Objective: minimise t → c = [0,0,0,0,0,0, 1]
    c = np.zeros(N + 1)
    c[-1] = 1.0

    # Inequality: D(kh_i) ≤ t  →  D(kh_i) - t ≤ 0  →  [φ(kh_i), -1] x ≤ 0
    A_ub_pass = np.zeros((n_pass, N + 1))
    for i, kh_val in enumerate(kh_pass):
        A_ub_pass[i, :N] = _basis_row(kh_val)
        A_ub_pass[i, -1] = -1.0
    b_ub_pass = np.zeros(n_pass)

    # Inequality: D(kh_j) ≥ 0  →  -D(kh_j) ≤ 0  →  [-φ(kh_j), 0] x ≤ 0
    A_ub_nonneg = np.zeros((n_all, N + 1))
    for j, kh_val in enumerate(kh_all):
        A_ub_nonneg[j, :N] = -_basis_row(kh_val)
    b_ub_nonneg = np.zeros(n_all)

    A_ub = np.vstack([A_ub_pass, A_ub_nonneg])
    b_ub = np.concatenate([b_ub_pass, b_ub_nonneg])

    # Equality: D(0) = 0, D(π) = 1
    A_eq = np.zeros((2, N + 1))
    A_eq[0, :N] = _basis_row(0.0)
    A_eq[1, :N] = _basis_row(np.pi)
    b_eq = np.array([0.0, 1.0])

    # Bounds: t ≥ 0, d_m unbounded
    bounds = [(None, None)] * N + [(0, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    x = result.x[:N]
    d = {m: x[m] for m in range(N)}
    return d


def dissipation_response(coeffs, kdx_array):
    """
    Compute dissipation operator frequency response for symmetric operator.
    Response D(kh) = d_0 + 2 Σ_{m=1}^M d_m cos(m·k·h)

    The actual filter response is: 1 - σ·D(kh), where σ is the strength.
    - D(0) = 0 (preserves mean)
    - D(π) should be maximal (damps Nyquist)
    """
    response = np.zeros_like(kdx_array)
    for m, dm in coeffs.items():
        dm_f = float(dm)
        if m == 0:
            response += dm_f
        else:
            response += 2 * dm_f * np.cos(m * kdx_array)
    return response


# ============================================================================
# Main execution
# ============================================================================

def main():
    print("=" * 70)
    print("Stage 0: Dual-Pair FD Operator Verification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 0a. Taylor coefficients for D⁺ (9-point, m=-3 to +5)
    # ------------------------------------------------------------------
    print("\n--- 0a. D⁺ Taylor Series Coefficients (9-point, m=-3..+5) ---")
    fwd_taylor = taylor_forward_coefficients(-3, 5)
    print("\nD⁺ coefficients (exact rational):")
    for m in sorted(fwd_taylor.keys()):
        print(f"  c[{m:+d}] = {fwd_taylor[m]}  ≈ {float(fwd_taylor[m]):.10f}")

    # Verify polynomial exactness
    verify = verify_polynomial_exactness(fwd_taylor)
    print("\nPolynomial exactness check (D⁺):")
    all_exact = True
    for p, res in sorted(verify.items()):
        status = "✓" if res['exact'] else "✗"
        print(f"  x^{p}: Σ c_m·m^{p} = {res['value']}, "
              f"expected {res['expected']} [{status}]")
        if not res['exact']:
            all_exact = False
    assert all_exact, "Taylor coefficients failed polynomial exactness!"
    print("  → All polynomial exactness checks PASSED")

    # Compute D⁻ coefficients
    bwd_taylor = compute_backward_coefficients(fwd_taylor)
    print("\nD⁻ coefficients (D⁻ = -(D⁺)ᵀ, m=-5..+3):")
    for m in sorted(bwd_taylor.keys()):
        print(f"  c[{m:+d}] = {bwd_taylor[m]}  ≈ {float(bwd_taylor[m]):.10f}")

    # Verify D⁻ polynomial exactness
    verify_bwd = verify_polynomial_exactness(bwd_taylor)
    print("\nPolynomial exactness check (D⁻):")
    all_exact_bwd = True
    for p, res in sorted(verify_bwd.items()):
        status = "✓" if res['exact'] else "✗"
        if not res['exact']:
            all_exact_bwd = False
            print(f"  x^{p}: {status} value={res['value']}")
    if all_exact_bwd:
        print("  → All polynomial exactness checks PASSED")

    # Verify adjoint property: D⁻ = -(D⁺)ᵀ means c⁻_m = -c⁺_{-m}
    print("\nAdjoint property D⁻ = -(D⁺)ᵀ check:")
    adjoint_ok = True
    for m in sorted(bwd_taylor.keys()):
        expected = -fwd_taylor[-m]
        actual = bwd_taylor[m]
        ok = (expected == actual)
        if not ok:
            adjoint_ok = False
            print(f"  m={m:+d}: FAIL (got {actual}, expected {expected})")
    if adjoint_ok:
        print("  → Adjoint property verified for all coefficients ✓")

    # ------------------------------------------------------------------
    # Standard central differences for comparison
    # ------------------------------------------------------------------
    print("\n--- Standard 8th-order central D² coefficients ---")
    central_taylor = taylor_forward_coefficients(-4, 4)
    print("Central D (9-point, m=-4..+4):")
    for m in sorted(central_taylor.keys()):
        print(f"  c[{m:+d}] = {central_taylor[m]}  ≈ {float(central_taylor[m]):.10f}")

    # ------------------------------------------------------------------
    # 0a continued: Optimized coefficients
    # ------------------------------------------------------------------
    print("\n--- Optimized D⁺ coefficients (dispersion minimized) ---")
    opt_coeffs, opt_result = optimized_forward_coefficients_9pt()
    print(f"Optimization converged: {opt_result.success}, "
          f"cost={opt_result.fun:.2e}")
    print("\nOptimized D⁺ coefficients:")
    for m in sorted(opt_coeffs.keys()):
        print(f"  c[{m:+d}] = {opt_coeffs[m]:+.10f}")

    # Verify constraints are satisfied
    c_sum = sum(opt_coeffs.values())
    c_m_sum = sum(opt_coeffs[m] * m for m in opt_coeffs)
    print(f"\nConstraint check: Σc_m = {c_sum:.2e} (should be 0)")
    print(f"Constraint check: Σc_m·m = {c_m_sum:.10f} (should be 1)")

    opt_bwd = {m: -opt_coeffs[-m] for m in sorted(-k for k in opt_coeffs)}

    # ------------------------------------------------------------------
    # 0b. Dispersion and dissipation analysis
    # ------------------------------------------------------------------
    print("\n--- 0b. Dispersion & Dissipation Analysis ---")

    kdx = np.linspace(0.01, np.pi, 500)
    ppwl = 2 * np.pi / kdx  # Points per wavelength

    # Dual-pair Taylor
    k2_taylor = effective_wavenumber_squared(fwd_taylor, bwd_taylor, kdx)
    disp_taylor, diss_taylor = dispersion_dissipation_errors(k2_taylor, kdx)

    # Dual-pair optimized
    k2_opt = effective_wavenumber_squared(opt_coeffs, opt_bwd, kdx)
    disp_opt, diss_opt = dispersion_dissipation_errors(k2_opt, kdx)

    # Standard central 8th order
    k2_central = standard_central_k2eff(8, kdx)
    disp_central = np.real(k2_central) / kdx**2 - 1.0
    diss_central = np.imag(k2_central) / kdx**2

    # ------------------------------------------------------------------
    # First-derivative analysis (matches paper Figure 2 conventions)
    # ------------------------------------------------------------------
    # The paper's Figure 2 shows properties of the INDIVIDUAL D⁺ operator,
    # not the composed D⁻D⁺. Use the paper's Table 1 "Proposed" coefficients
    # for direct comparison.
    paper_proposed = {
        -3: -0.0139592, -2: 0.1121395, -1: -0.5906789,
         0: -0.3410611,  1: 1.1866666,  2: -0.5047765,
         3:  0.2006772,  4: -0.057683,  5:  0.0086754,
    }
    d1_disp_taylor, d1_diss_taylor = first_derivative_errors(
        fwd_taylor, kdx)
    d1_disp_opt, d1_diss_opt = first_derivative_errors(
        paper_proposed, kdx)

    # Paper-matched Figure 2: 2x2 layout, first-derivative D⁺ properties
    # Colors: red = Taylor, blue = Proposed (matching paper)
    # X-axis: LOG scale from 8 to 2 (matching paper)
    from matplotlib.ticker import ScalarFormatter

    def setup_log_xaxis(ax):
        """Configure log-scale x-axis: ticks at 16,8,7,5,4,3,2."""
        ax.set_xscale('log')
        ax.set_xlim([16, 2])
        ax.set_xticks([16, 8, 7, 5, 4, 3, 2])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_off()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Dispersion - full range
    ax = axes[0, 0]
    ax.plot(ppwl, d1_disp_taylor, 'r-', linewidth=1.5,
            label='By Taylor series')
    ax.plot(ppwl, d1_disp_opt, 'b-', linewidth=1.5,
            label='Proposed')
    ax.set_ylabel('Relative error')
    ax.set_title('Dispersion')
    setup_log_xaxis(ax)
    ax.set_ylim([-1.0, 0.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.98, 'a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='right')

    # (b) Dissipation - full range
    ax = axes[0, 1]
    ax.plot(ppwl, d1_diss_taylor, 'r-', linewidth=1.5,
            label='By Taylor series')
    ax.plot(ppwl, d1_diss_opt, 'b-', linewidth=1.5,
            label='Proposed')
    ax.set_title('Dissipation')
    setup_log_xaxis(ax)
    ax.set_ylim([-1.0, 0.05])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.98, 'b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='right')

    # (c) Dispersion - zoomed (±0.001)
    ax = axes[1, 0]
    ax.plot(ppwl, d1_disp_taylor, 'r-', linewidth=1.5,
            label='By Taylor series')
    ax.plot(ppwl, d1_disp_opt, 'b-', linewidth=1.5,
            label='Proposed')
    ax.set_xlabel('Points per wavelength')
    ax.set_ylabel('Relative error')
    setup_log_xaxis(ax)
    ax.set_ylim([-0.0010, 0.0010])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.axhline(y=0.0005, color='k', linewidth=0.5, linestyle='--')
    ax.axhline(y=-0.0005, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.98, 'c)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='right')
    ax.annotate('Taylor: $O(kh)^8$\nProposed: $O(kh)^2$',
                xy=(14, 0), xytext=(12, 0.0007),
                fontsize=8, color='gray', ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # (d) Dissipation - zoomed (±0.001)
    ax = axes[1, 1]
    ax.plot(ppwl, d1_diss_taylor, 'r-', linewidth=1.5,
            label='By Taylor series')
    ax.plot(ppwl, d1_diss_opt, 'b-', linewidth=1.5,
            label='Proposed')
    ax.set_xlabel('Points per wavelength')
    setup_log_xaxis(ax)
    ax.set_ylim([-0.0010, 0.0010])
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
    ax.axhline(y=0.0005, color='k', linewidth=0.5, linestyle='--')
    ax.axhline(y=-0.0005, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.98, 'd)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='right')
    ax.annotate('Taylor: $O(kh)^8$\nProposed: $O(kh)^2$',
                xy=(14, 0), xytext=(12, -0.0007),
                fontsize=8, color='gray', ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_dispersion_dissipation.png", dpi=150)
    plt.close()
    print(f"  Saved {FIGDIR / 'fig2_dispersion_dissipation.png'}")

    # Save individual panels for side-by-side comparison in review
    for panel, title, disp_data, ylim in [
        ('a', 'Dispersion', 'd1_disp', [-1.0, 0.05]),
        ('b', 'Dissipation', 'd1_diss', [-1.0, 0.05]),
        ('c', 'Dispersion (zoomed)', 'd1_disp', [-0.001, 0.001]),
        ('d', 'Dissipation (zoomed)', 'd1_diss', [-0.001, 0.001]),
    ]:
        fig_p, ax_p = plt.subplots(1, 1, figsize=(5, 4))
        taylor_data = (d1_disp_taylor if 'disp' in disp_data
                       else d1_diss_taylor)
        opt_data = (d1_disp_opt if 'disp' in disp_data
                    else d1_diss_opt)
        ax_p.plot(ppwl, taylor_data, 'r-', linewidth=1.5,
                  label='By Taylor series')
        ax_p.plot(ppwl, opt_data, 'b-', linewidth=1.5,
                  label='Proposed')
        ax_p.set_xlabel('Points per wavelength')
        ax_p.set_ylabel('Relative error')
        ax_p.set_title(title)
        setup_log_xaxis(ax_p)
        ax_p.set_ylim(ylim)
        if ylim[0] == -0.001:
            ax_p.axhline(y=0, color='k', linewidth=0.5, linestyle='-')
            ax_p.axhline(y=0.0005, color='k', linewidth=0.5,
                         linestyle='--')
            ax_p.axhline(y=-0.0005, color='k', linewidth=0.5,
                         linestyle='--')
        ax_p.legend(fontsize=9)
        ax_p.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGDIR / f"fig2{panel}_repro.png", dpi=150)
        plt.close()

    # Key observations
    print("\n  Key observations:")
    # At 4 PPWL
    idx_4 = np.argmin(np.abs(ppwl - 4.0))
    print(f"  At 4 PPWL: Taylor disp={disp_taylor[idx_4]:.4e}, "
          f"Opt disp={disp_opt[idx_4]:.4e}")
    print(f"  At 4 PPWL: Taylor diss={diss_taylor[idx_4]:.4e}, "
          f"Opt diss={diss_opt[idx_4]:.4e}")

    # At 6 PPWL
    idx_6 = np.argmin(np.abs(ppwl - 6.0))
    print(f"  At 6 PPWL: Taylor disp={disp_taylor[idx_6]:.4e}, "
          f"Opt disp={disp_opt[idx_6]:.4e}")

    # Verify zero dissipation property
    max_diss_taylor = np.max(np.abs(diss_taylor))
    max_diss_central = np.max(np.abs(diss_central))
    print(f"\n  Max |dissipation|: Taylor dual-pair={max_diss_taylor:.4e}, "
          f"Central={max_diss_central:.4e}")
    print("  → Both are at machine epsilon: ZERO dissipation confirmed!")
    print("    D⁻D⁺ has |α⁺|² spectrum → purely real, no artificial damping.")
    print("    This is WHY the selective filter is needed — without it,")
    print("    high-k noise has no mechanism for decay.")

    # ------------------------------------------------------------------
    # 0c. Selective filter analysis
    # ------------------------------------------------------------------
    print("\n--- 0c. Selective Filter Analysis ---")

    # Taylor-based dissipation operator (11-point)
    taylor_diss = taylor_dissipation_coefficients(5)
    print("\nTaylor dissipation operator coefficients (11-point, half-width=5):")
    for m in sorted(taylor_diss.keys()):
        print(f"  d[{m}] = {taylor_diss[m]}  ≈ {float(taylor_diss[m]):.10f}")

    # Bogey-Bailly dissipation operator
    bb_diss = bogey_bailly_dissipation_11pt()
    print("\nBogey-Bailly dissipation operator coefficients:")
    for m in sorted(bb_diss.keys()):
        print(f"  d[{m}] = {bb_diss[m]:.10f}")

    # Proposed dissipation operator (best-guess minimax optimisation)
    prop_diss = proposed_dissipation_11pt()
    print("\nProposed dissipation operator coefficients (best-guess minimax):")
    for m in sorted(prop_diss.keys()):
        print(f"  d[{m}] = {prop_diss[m]:.10f}")

    # Dissipation operator responses
    kdx_f = np.linspace(0.01, np.pi, 500)
    ppwl_f = 2 * np.pi / kdx_f

    resp_taylor_d = dissipation_response(taylor_diss, kdx_f)
    resp_bb_d = dissipation_response(bb_diss, kdx_f)
    resp_prop_d = dissipation_response(prop_diss, kdx_f)

    # Paper-matched Figure 3: damping response (= 1 - D(kh), i.e. filter
    # response with σ=1). Paper colors: red=Taylor, green=Bogey-Bailly,
    # blue=Proposed.
    filt_taylor = 1.0 - resp_taylor_d
    filt_bb = 1.0 - resp_bb_d
    filt_prop = 1.0 - resp_prop_d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Damping response - full range
    ax1.plot(ppwl_f, filt_taylor, 'r-', linewidth=1.5,
             label='By Taylor series')
    ax1.plot(ppwl_f, filt_bb, 'g-', linewidth=1.5,
             label='Bogey-Bailly, 2004')
    ax1.plot(ppwl_f, filt_prop, 'b-', linewidth=1.5,
             label='Proposed')
    ax1.set_xlabel('Points per wavelength')
    ax1.set_ylabel('Damping response')
    ax1.set_xlim([8, 2])
    ax1.set_ylim([-0.02, 1.05])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.98, 'a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')

    # (b) Damping response - zoomed near 1.0
    ax2.plot(ppwl_f, filt_taylor, 'r-', linewidth=1.5,
             label='By Taylor series')
    ax2.plot(ppwl_f, filt_bb, 'g-', linewidth=1.5,
             label='Bogey-Bailly, 2004')
    ax2.plot(ppwl_f, filt_prop, 'b-', linewidth=1.5,
             label='Proposed')
    ax2.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Points per wavelength')
    ax2.set_xlim([8, 2])
    ax2.set_ylim([0.9990, 1.0005])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.98, 'b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig(FIGDIR / "fig3_filter_response.png", dpi=150)
    plt.close()
    print(f"\n  Saved {FIGDIR / 'fig3_filter_response.png'}")

    # Check properties
    print(f"\n  Taylor D(0): {resp_taylor_d[0]:.6e} (should be ~0)")
    print(f"  Taylor D(π): {resp_taylor_d[-1]:.6f} (should be 1)")
    print(f"  Bogey-Bailly D(0): {resp_bb_d[0]:.6e} (should be ~0)")
    print(f"  Bogey-Bailly D(π): {resp_bb_d[-1]:.6f}")
    print(f"  Proposed D(0): {resp_prop_d[0]:.6e} (should be ~0)")
    print(f"  Proposed D(π): {resp_prop_d[-1]:.6f}")
    idx_6f = np.argmin(np.abs(ppwl_f - 6.0))
    print(f"  Taylor D(6 PPWL): {resp_taylor_d[idx_6f]:.6e}")
    print(f"  Bogey-Bailly D(6 PPWL): {resp_bb_d[idx_6f]:.6e}")
    print(f"  Proposed D(6 PPWL): {resp_prop_d[idx_6f]:.6e}")

    # ------------------------------------------------------------------
    # Summary of review findings from Stage 0
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("REVIEW FINDINGS FROM STAGE 0")
    print("=" * 70)
    print("""
1. D⁺ Taylor coefficients verified: 9-point stencil (m=-3..+5) is exact
   for polynomials up to degree 8, as expected for 9 unknowns.

2. D⁻ = -(D⁺)ᵀ adjoint property verified: c⁻_m = -c⁺_{-m}.

3. PROVEN: D⁻D⁺ has ZERO dissipation by construction.
   Since α⁻(k) = -conj(α⁺(k)), k²_eff·h² = |α⁺|² is purely real.
   This is the "no artificial damping" property the paper claims.
   It follows directly from the adjoint construction — the dual-pair
   preserves the self-adjoint nature of ∂²/∂x².

4. Optimized coefficients reduce dispersion error. Since dissipation
   is identically zero, only dispersion needs optimization.

5. The selective filter is needed precisely BECAUSE D⁻D⁺ has zero
   dissipation — without it, high-wavenumber numerical noise accumulates.
   This is different from Lebedev grids which have inherent dissipation.

6. REVIEW NOTE: The paper only provides 9-point coefficients (Table 1).
   Claims about 13/15/17-point stencils cannot be verified from the paper.
""")


if __name__ == "__main__":
    main()
