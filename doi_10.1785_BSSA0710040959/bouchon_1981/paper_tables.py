"""Hand-transcribed paper anchors for Bouchon (1981) DWN.

Bouchon, M. (1981). "A simple method to calculate Green's
functions for elastic layered media." *Bulletin of the
Seismological Society of America* 71(4): 959-971.
DOI: 10.1785/BSSA0710040959.

This module pins the load-bearing prescriptions from the paper
that the `dwn_solver` module implements, so the test suite can
regression-guard against silent drift in those constants.

Anchors transcribed here:

1. **Complex-frequency regularisation prescription** (paper
   page 962, equation discussion around the periodicity).
   Bouchon prescribes shifting `ω → ω − iα` where `α = π/T_max`,
   which gives ~25 dB damping at `t = T_max` (recovered by the
   time-domain back-correction `× exp(α·t)`).

2. **Periodicity-length condition** (paper page 962, the
   non-overlap condition for spatial periodic source copies).
   The strict condition is `L > V_max · T_max + x_max`. Our
   implementation uses a 30% safety margin: `L = 1.3 · (V_max ·
   T_max + x_max)`.

3. **Critical-angle singularity policy** (paper page 961
   discussion of `1/q_w` near `q_w → 0`). The complex shift in
   item 1 moves all `q_w` values off zero, eliminating the
   divide-by-zero singularity at the critical angle. The
   2003 review (Bouchon 2003, page 449) discusses this
   explicitly: "the complex-frequency shift… removes the
   singularity of the integrand on the real axis."
"""
from __future__ import annotations

import numpy as np


# ─── (1) Complex-frequency regularisation α prescription ───────────


# Bouchon 1981 prescribes α = π / T_max. This is the canonical
# damping rate for the complex-frequency shift ω → ω − iα.
# Implementation detail: `dwn_solver.dwn_wavefield_*` reads this
# from the function argument `a_damping=None` and defaults to
# this prescription.
BOUCHON_1981_ALPHA_PRESCRIPTION = "pi/T_max"


def alpha_damping(T_max: float) -> float:
    """Bouchon (1981) canonical damping rate.

    Parameters
    ----------
    T_max : float
        The simulation time window, in seconds.

    Returns
    -------
    alpha : float
        The damping rate `α = π / T_max`, in 1/s. The
        time-domain wavefield is corrected by multiplying by
        `exp(α·t)` after the inverse FFT.
    """
    if T_max <= 0:
        raise ValueError(f"T_max must be positive (got {T_max!r})")
    return float(np.pi / T_max)


# ─── (2) Periodicity-length safety margin ──────────────────────────


# Strict Bouchon-1981 periodicity condition: L > V_max · T_max + x_max.
# We use a 30% safety margin past the strict bound.
BOUCHON_1981_PERIODICITY_SAFETY_MARGIN = 1.3


def periodicity_length(V_max: float, T_max: float, x_max: float) -> float:
    """Periodicity length L per Bouchon 1981 + 30% safety margin.

    Parameters
    ----------
    V_max : float
        Maximum P-wave velocity across the layered medium (m/s
        or km/s, consistent with `x_max`).
    T_max : float
        Simulation time window in s.
    x_max : float
        Maximum source-receiver horizontal offset.

    Returns
    -------
    L : float
        Periodicity length used by the DWN sum, in the same
        units as `x_max`. Strict Bouchon-1981 requires
        ``L > V_max · T_max + x_max``; this returns
        ``BOUCHON_1981_PERIODICITY_SAFETY_MARGIN × (V_max ·
        T_max + x_max)`` ≈ 1.3 × (strict).
    """
    return float(BOUCHON_1981_PERIODICITY_SAFETY_MARGIN *
                 (V_max * T_max + x_max))


# ─── (3) Critical-angle singularity policy ─────────────────────────


# Sentinel: the DWN implementation must use the complex-frequency
# shift (item 1) to regularise the 1/q_w singularity at q_w = 0.
# A direct real-axis evaluation would produce the 10^301 amplitude
# blowup that motivated Bouchon's method in the first place
# (relative to the slowness-FFT predecessor approach). The parent
# repo's slowness-FFT draft hit exactly that failure mode at
# plan §80 Phase 2c-FFT before switching to DWN at Phase 2c-DWN.
DWN_SOLVES_CRITICAL_ANGLE_SINGULARITY = True


# ─── (4) Recommended sampling parameter ranges ─────────────────────


# Bouchon-1981 + Bouchon-2003 review give practical guidance on
# (n_omega, M_wavenumbers) tuples that achieve typical accuracy
# targets. These bounds are not byte-anchored to specific paper
# equations but reflect community practice (Bouchon-2003 §3).

# n_omega: positive frequency samples. Must be ≥ 64 for any
# reasonable Ricker wavelet; ≥ 256 for the parent repo's
# Petrobras configuration.
RECOMMENDED_N_OMEGA_MIN = 64

# M_wavenumbers: the half-width of the symmetric k_m sum,
# m ∈ {-M, ..., +M}. Must be large enough that periodic
# copies don't appear before t = T_max at any receiver.
RECOMMENDED_M_WAVENUMBERS_MIN = 256
