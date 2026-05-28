"""Yang, Yan & Liu (2015) — byte-transcribed paper constants.

DOI 10.1016/j.jappgeo.2015.08.007 · *J. Appl. Geophys.* 122:40-52

All entries below come from the paper PDF via the side-by-side
user-confirmed transcription protocol (see
`feedback_transcription_workflow_arxiv_then_side_by_side` memory
entry). No arXiv preprint exists, so PDF-side-by-side review is
mandatory for every transcribed entry.

Transcription provenance log:

| Date       | Entry                          | Source      | Status |
|------------|--------------------------------|-------------|--------|
| 2026-05-27 | Eq 1 (PDE structural)          | paper p. 41 | ✓ user-confirmed (transcription_review/yang2015_eq1_pde.pdf) |
| 2026-05-27 | Eq 2 (D = R·C·Rᵀ)              | paper p. 41 | textbook (Winterstein 1990 Bond identity) |
| 2026-05-27 | Eq 3 (C VTI form)              | paper p. 41 | ✓ user-confirmed (transcription_review/yang2015_eq3_C_matrix.pdf) |
| 2026-05-27 | Eq 4 (R Bond rotation 6×6)     | paper p. 41 | ✓ user-confirmed (transcription_review/yang2015_eq4_R_bond.pdf) |
| (pending)  | TE / SA / LS RSG coefficients  | paper §3-4  | not yet transcribed |

Sign + ordering conventions
---------------------------
Velocity-stress state per Eq 1 (2D RSG-TTI):
    (v_x, v_y, v_z; σ_xx, σ_zz, σ_yz, σ_xz, σ_xy)
Note σ_yy is ABSENT from the state because the system propagates
in the (x, z) plane only (y-derivatives all zero); v_y carries
the SH out-of-plane motion.

Voigt index convention: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.
The d_ij entries appearing in Eq 1 use rows/cols {1, 3, 4, 5, 6}
of the 6×6 D matrix (skipping the y-axis-derivative row/col 2).

C and R conventions
-------------------
- C is the natural-coordinate-system Voigt stiffness with VTI
  symmetry (5 indep + c66 = ½(c11-c12)).
- R is the Bond rotation matrix per Eq 4, parametrised by
  azimuth ϕ and tilt θ. Liu et al. (1998) form.
- D = R · C · Rᵀ is the observation-coordinate stiffness used
  in the PDE.
"""
from __future__ import annotations

import sympy as sp

# Symbolic parameters
phi, theta = sp.symbols("varphi theta", real=True)
c11, c12, c13, c33, c44 = sp.symbols("c_11 c_12 c_13 c_33 c_44",
                                       real=True, positive=True)


# ─── Eq 3 — C matrix VTI form (paper p. 41) ─────────────────────────────


def C_matrix_VTI(c11_val=c11, c12_val=c12, c13_val=c13,
                  c33_val=c33, c44_val=c44):
    """Build the 6×6 VTI Voigt stiffness matrix per Yang 2015 Eq 3.

    Five independent parameters (c11, c12, c13, c33, c44) +
    derived c66 = ½(c11 - c12) per the VTI constraint.

    Returns
    -------
    sympy.Matrix of shape (6, 6) — symmetric, with the standard
    VTI block structure (upper 3×3 block has c11/c12/c13/c33;
    lower 3×3 block is diag(c44, c44, c66)).
    """
    c66_val = sp.Rational(1, 2) * (c11_val - c12_val)
    return sp.Matrix([
        [c11_val, c12_val, c13_val, 0,       0,       0      ],
        [c12_val, c11_val, c13_val, 0,       0,       0      ],
        [c13_val, c13_val, c33_val, 0,       0,       0      ],
        [0,       0,       0,       c44_val, 0,       0      ],
        [0,       0,       0,       0,       c44_val, 0      ],
        [0,       0,       0,       0,       0,       c66_val],
    ])


# ─── Eq 4 — Bond rotation matrix R (paper p. 41) ────────────────────────


def bond_rotation_R(phi_val=phi, theta_val=theta):
    """Build the 6×6 Bond rotation matrix R per Yang 2015 Eq 4.

    Parameters
    ----------
    phi_val   : azimuth angle (radians) — rotation about vertical axis
    theta_val : tilt angle (radians) — TTI symmetry-axis tilt

    Returns
    -------
    sympy.Matrix of shape (6, 6).

    Smoke checks
    ------------
    R(0, 0) = I_6 (no rotation → identity); see test_paper_tables.
    """
    cp = sp.cos(phi_val)
    sp_ = sp.sin(phi_val)
    ct = sp.cos(theta_val)
    st = sp.sin(theta_val)
    s2p = sp.sin(2 * phi_val)
    c2p = sp.cos(2 * phi_val)
    s2t = sp.sin(2 * theta_val)
    c2t = sp.cos(2 * theta_val)
    half = sp.Rational(1, 2)
    return sp.Matrix([
        # Row 1: x-stress (σ_xx) Voigt row
        [cp**2,             sp_**2 * ct**2,         sp_**2 * st**2,
         -sp_**2 * s2t,     s2p * st,                -s2p * ct],
        # Row 2: y-stress (σ_yy) Voigt row
        [sp_**2,            cp**2 * ct**2,           cp**2 * st**2,
         -cp**2 * s2t,      -s2p * st,                s2p * ct],
        # Row 3: z-stress (σ_zz) Voigt row
        [sp.Integer(0),     st**2,                   ct**2,
         s2t,                sp.Integer(0),            sp.Integer(0)],
        # Row 4: yz shear (σ_yz) Voigt row
        [sp.Integer(0),     half * cp * s2t,        -half * cp * s2t,
         cp * c2t,           sp_ * ct,                sp_ * st],
        # Row 5: xz shear (σ_xz) Voigt row
        [sp.Integer(0),     -half * sp_ * s2t,       half * sp_ * s2t,
         -sp_ * c2t,         cp * ct,                 cp * st],
        # Row 6: xy shear (σ_xy) Voigt row
        [half * s2p,        -half * s2p * ct**2,    -half * s2p * st**2,
         half * s2p * s2t,  -c2p * st,                c2p * ct],
    ])


# ─── Eq 2 — Bond-rotated stiffness D = R · C · Rᵀ ───────────────────────


def D_TTI(c11_val=c11, c12_val=c12, c13_val=c13, c33_val=c33,
          c44_val=c44, phi_val=phi, theta_val=theta):
    """Apply Bond rotation D = R · C · Rᵀ per Yang 2015 Eq 2.

    Per Winterstein (1990): R is the 6×6 Bond rotation matrix
    transforming Voigt stiffness from the natural to the
    observation coordinate system.

    Returns
    -------
    sympy.Matrix of shape (6, 6) — symmetric.
    """
    C = C_matrix_VTI(c11_val, c12_val, c13_val, c33_val, c44_val)
    R = bond_rotation_R(phi_val, theta_val)
    return R * C * R.T


# ─── Eq 1 — PDE structural (d_ij dependency tracker) ────────────────────

# Maps each stress-equation row (1d-1h) to the d_ij subset it uses:
#   key: stress component receiving the time-derivative
#   value: sorted tuple of (i, j) tuples appearing in that equation
# Per Yang 2015 Eq 1 (paper p. 41) — used by tests to verify the
# transcription's structural pattern + by run_reproduction.py to
# assemble the Devito-compatible PDE expressions.
EQ1_STRESS_DIJ_DEPENDENCIES: dict[str, frozenset[tuple[int, int]]] = {
    "sigma_xx": frozenset({(1, 1), (1, 3), (1, 4), (1, 5), (1, 6)}),  # Eq 1d
    "sigma_zz": frozenset({(1, 3), (3, 3), (3, 4), (3, 5), (3, 6)}),  # Eq 1e
    "sigma_yz": frozenset({(1, 4), (3, 4), (4, 4), (4, 5), (4, 6)}),  # Eq 1f
    "sigma_xz": frozenset({(1, 5), (3, 5), (4, 5), (5, 5), (5, 6)}),  # Eq 1g
    "sigma_xy": frozenset({(1, 6), (3, 6), (4, 6), (5, 6), (6, 6)}),  # Eq 1h
}

# Union over all stress equations — these are the d_ij entries Yang 2015
# actually uses in the 2D RSG-TTI PDE.
EQ1_DIJ_USED: frozenset[tuple[int, int]] = frozenset().union(
    *EQ1_STRESS_DIJ_DEPENDENCIES.values()
)

# d_ij entries CONSPICUOUSLY ABSENT from Eq 1 — the σ_yy stress equation
# is omitted because 2D propagation in the (x, z) plane has no
# y-derivatives. So column/row 2 of D never appears in Eq 1.
EQ1_DIJ_ABSENT_ROWS_COLS: frozenset[int] = frozenset({2})


__all__ = [
    "phi", "theta",
    "c11", "c12", "c13", "c33", "c44",
    "C_matrix_VTI",
    "bond_rotation_R",
    "D_TTI",
    "EQ1_STRESS_DIJ_DEPENDENCIES",
    "EQ1_DIJ_USED",
    "EQ1_DIJ_ABSENT_ROWS_COLS",
]
