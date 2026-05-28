"""Summation-by-Parts (SBP) first-derivative operator weights.

Phase 5b implementation — plan §73 / §68g.

Provides the canonical (4, 2) diagonal-norm SBP D1 boundary stencils
and norm matrix used by the Petersson-Sjögreen 2015 elastic-TTI
scheme (@petersson_sjogreen_2015). "(4, 2)" denotes 4th-order
interior accuracy + 2nd-order boundary accuracy, with a 4-row
diagonal-norm boundary modification.

Primary source
--------------
Strand (1994) "Summation by parts for finite difference
approximations for d/dx", *J. Comp. Phys.* 110:47-67 — the original
derivation. Tabulated in PS 2015 reference [20]; not in our Zotero.

The exact rational values committed here are extracted byte-for-byte
from **sw4** (the @petersson_sjogreen_2015 reference implementation,
`geodynamics/sw4` on GitHub, file ``src/boundaryOpc.C``). Using sw4
as the primary source makes Phase 5 a faithful reproduction of PS
2015's published code by construction.

The same (4, 2) tabulation appears in Mattsson & Nordström (2004)
*J. Comp. Phys.* 199:503-540 §3 and Mattsson & Carpenter (2010)
*SIAM J. Sci. Comput.* 32(4):2298-2320. Neither is in our Zotero
corpus; sw4 source is the authoritative primary anchor for our
purposes.

Conventions
-----------
- Norm matrix ``H`` is diagonal: ``H = h * diag(w_0, w_1, ..., w_{n-1})``
  where the first 4 and last 4 weights are boundary-modified
  (Strand 1994 §4 + sw4 ``normwgh``):

      H_corner = (17/48, 59/48, 43/48, 49/48, 1, ..., 1,
                  49/48, 43/48, 59/48, 17/48) · h

- D1 boundary stencil (4 rows × 6 cols on the left/bottom side,
  symmetric mirror on the right/top side). Row ``i`` of the matrix
  encodes ``(D_1 u)_i = (1/h) * Σ_j bop[i,j] * u_j``. From
  sw4 ``src/boundaryOpc.C``:

      row 0: ( -24/17,   59/34,   -4/17,   -3/34,   0,    0   )
      row 1: ( -1/2,      0,       1/2,     0,      0,    0   )
      row 2: (  4/43,   -59/86,    0,      59/86,  -4/43, 0   )
      row 3: (  3/98,    0,      -59/98,   0,      32/49, -4/49)

  Interior rows (4 .. N-5) use the standard 4th-order centred FD
  weights ``(1/12, -8/12, 0, 8/12, -1/12)``.

- SBP property: the operator decomposes as ``D_1 = H^{-1} (Q + B/2)``
  where ``Q + Q^T = diag(-1, 0, ..., 0, +1)`` and ``B = diag(-1, 0,
  ..., 0, +1)``. Equivalently:

      H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)

  This is the mathematical guarantee of energy-stability (Strand
  1994 Theorem 1, PS 2015 §3.2). The unit tests in
  ``tests/test_sbp_d1_coefficients.py`` verify it symbolically via
  rational arithmetic.

Out of scope for Phase 5b
-------------------------
- Higher-order SBP D1 (6, 3) or (8, 4): deferred. PS 2015 uses
  (4, 2) per their §3.2 and the sw4 reference implementation.
- SAT penalty terms: covered by Phase 5d. The PS 2015 paper uses
  ghost-point BC imposition (their §3.1) rather than SAT-penalty;
  this module exposes the SBP D1 weights only.

Phase 5g additions (2026-05-15) — variable-coefficient SBP D2
--------------------------------------------------------------
The variable-coefficient SBP D2 operator (the sw4 ``acof`` arrays
for $\\nabla \\cdot (a \\nabla u)$) and the FS BC ghost-point
coefficient ``ghcof`` are now also exposed. See
:func:`sbp_d2_coeffs`, :func:`build_sbp_d2_matrix`, and
:func:`sbp_d2_boundary_exprs`. Together with the existing D1
infrastructure, this completes the (4, 2) diagonal-norm SBP-SAT
operator family used by @petersson_sjogreen_2015.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple


# Canonical (4, 2) diagonal-norm SBP D1 boundary stencil — Strand 1994
# tabulation as in @petersson_sjogreen_2015's sw4 ``src/boundaryOpc.C``.

_BOP_4_2: Tuple[Tuple[Fraction, ...], ...] = (
    # Row 0: one-sided 2nd-order accurate at the boundary.
    (Fraction(-24, 17), Fraction(59, 34), Fraction(-4, 17),
     Fraction(-3, 34),  Fraction(0),      Fraction(0)),
    # Row 1: centred 2nd-order (forced by the SBP constraint).
    (Fraction(-1, 2), Fraction(0), Fraction(1, 2),
     Fraction(0),     Fraction(0), Fraction(0)),
    # Row 2: 4-point antisymmetric, 2nd-order at the boundary.
    (Fraction(4, 43),  Fraction(-59, 86), Fraction(0),
     Fraction(59, 86), Fraction(-4, 43),  Fraction(0)),
    # Row 3: 5-point asymmetric; transitions to interior 4th-order.
    (Fraction(3, 98), Fraction(0),       Fraction(-59, 98),
     Fraction(0),     Fraction(32, 49),  Fraction(-4, 49)),
)

# Diagonal norm matrix H = h * diag(...).
# First 4 weights on each side are boundary-modified; interior is 1.
_NORM_WEIGHTS_4_2: Tuple[Fraction, ...] = (
    Fraction(17, 48),
    Fraction(59, 48),
    Fraction(43, 48),
    Fraction(49, 48),
)

# Interior 4th-order centred FD weights at offsets (-2, -1, 0, 1, 2).
_INTERIOR_4: Tuple[Fraction, ...] = (
    Fraction(1, 12), Fraction(-8, 12), Fraction(0),
    Fraction(8, 12), Fraction(-1, 12),
)


def sbp_d1_coeffs(order: int = 4) -> dict:
    """Return the (4, 2) diagonal-norm SBP D1 boundary stencil + norm
    weights.

    Parameters
    ----------
    order : int
        Interior order of accuracy. Only ``order=4`` is currently
        tabulated (the most common SBP-SAT variant; matches PS 2015).
        Higher orders (6, 3) / (8, 4) raise ``NotImplementedError``.

    Returns
    -------
    dict with keys:
        ``"boundary_rows"`` : list of 4 lists of ``Fraction``
            Each row of the (4, 6) boundary stencil matrix. Apply as
            ``(D_1 u)_i = (1/h) * Σ_j boundary_rows[i][j] * u_j`` for
            ``i ∈ {0, 1, 2, 3}``.

        ``"interior"`` : list of 5 ``Fraction``
            Standard 4th-order centred FD weights at offsets
            ``(-2, -1, 0, 1, 2)``. Used for rows ``i ∈ {4, ..., N-5}``.

        ``"norm_weights"`` : list of 4 ``Fraction``
            Diagonal norm matrix weights ``H_{ii}/h`` for the first
            4 rows. Interior rows have weight 1; the top 4 rows
            mirror these (i.e. ``H_{N-1, N-1} = w_0``, etc.).

    Notes
    -----
    The norm matrix and boundary stencil together satisfy the SBP
    property: ``H D_1 + D_1^T H = diag(-1, 0, ..., 0, +1)``. This is
    verified symbolically in
    ``tests/test_sbp_d1_coefficients.py``.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D1 (the (4, 2) diagonal-norm variant "
            f"used by @petersson_sjogreen_2015) is currently "
            f"implemented; got order={order}."
        )

    return {
        "boundary_rows": [list(row) for row in _BOP_4_2],
        "interior": list(_INTERIOR_4),
        "norm_weights": list(_NORM_WEIGHTS_4_2),
    }


def build_sbp_d1_matrix(N: int, order: int = 4) -> List[List[Fraction]]:
    """Construct the full N × N SBP D1 matrix as a list of lists of
    ``Fraction``.

    Useful for symbolic verification of the SBP property (see the
    unit tests). For Devito operator construction, callers should use
    ``sbp_d1_coeffs(order)`` directly and apply the row-wise stencils
    inside a SubDomain.

    Parameters
    ----------
    N : int
        Number of grid points. Must satisfy ``N >= 8`` so that the
        boundary stencils (4 rows on each side, 6 cols wide) don't
        overlap.
    order : int
        Interior order. Only ``order=4`` supported.

    Returns
    -------
    N × N matrix as a list of N rows, each a list of N ``Fraction``s.
    Multiplying by 1/h gives the actual D1 operator: ``D = (1/h) * M``.
    """
    if N < 8:
        raise ValueError(
            f"build_sbp_d1_matrix requires N >= 8 (boundary stencils "
            f"don't overlap); got N={N}."
        )
    coeffs = sbp_d1_coeffs(order=order)
    rows = coeffs["boundary_rows"]
    interior = coeffs["interior"]

    M = [[Fraction(0)] * N for _ in range(N)]

    # Left boundary rows 0..3 — write rows[i] at columns 0..5.
    for i in range(4):
        for j, w in enumerate(rows[i]):
            M[i][j] = w

    # Interior rows 4..N-5 — centred FD at offsets (-2, -1, 0, 1, 2).
    for i in range(4, N - 4):
        for k, w in enumerate(interior):
            j = i + (k - 2)
            M[i][j] = w

    # Right boundary rows N-4..N-1 — mirror of left rows with sign flip
    # (since D1 is anti-symmetric in the SBP property sense:
    # M^T = -M relative to the centre of the domain).
    # Specifically: M[N-1-i][N-1-j] = -M[i][j].
    for i in range(4):
        ri = N - 1 - i
        for j, w in enumerate(rows[i]):
            rj = N - 1 - j
            M[ri][rj] = -w

    return M


def build_sbp_norm_diag(N: int, order: int = 4) -> List[Fraction]:
    """Construct the diagonal of the SBP norm matrix H/h as a list of
    ``Fraction``s. ``H = h * diag(result)``.

    First and last 4 entries are boundary-modified per
    ``_NORM_WEIGHTS_4_2``; interior entries are 1.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP norm matrix is implemented; got "
            f"order={order}."
        )
    if N < 8:
        raise ValueError(
            f"build_sbp_norm_diag requires N >= 8; got N={N}."
        )
    weights = _NORM_WEIGHTS_4_2
    h_diag = [Fraction(1)] * N
    for i, w in enumerate(weights):
        h_diag[i] = w
        h_diag[N - 1 - i] = w
    return h_diag


# ---------------------------------------------------------------------------
# Devito integration helpers (Phase 5d)
# ---------------------------------------------------------------------------

def sbp_d1_boundary_exprs(field, dim, order=4, fs_location='bottom',
                          free_dim=None, base_idx=None):
    """Build symbolic Devito expressions for the SBP D1 boundary rows.

    Phase 5d helper (plan §73 / §68g) + plan §104 Phase E.1 extension
    (added ``fs_location='top'`` + 2D ``free_dim`` support per global
    ~/CLAUDE.md rule 11: framework-audit before declaring "missing").

    Returns a list of 4 sympy expressions, one per boundary row,
    each computing ``(D_1 field)_i = (1/h) * Σ_j bop[i, j] *
    field[t, j]`` (for ``fs_location='bottom'``) or the
    mirror-symmetric stencil with sign flip (for
    ``fs_location='top'``).

    **1D vs 2D modes** (plan §104 Phase E.1; mirrors the §102a /
    Phase 5h.0 ``sbp_d2_boundary_exprs`` overload pattern):

    - **1D mode** (``free_dim=None`` AND the field has exactly one
      spatial dim): backward-compatible with Phase 5d signature.
      Indexing is ``field[t, j]``.

    - **2D mode** (``free_dim`` supplied, or the field is detected
      to have 2 spatial dims): the FS boundary runs along the
      ``free_dim`` direction; ``dim`` is the FS-perpendicular
      coordinate. Indexing becomes ``field[t, free_dim, idx_j]`` —
      the free dim's symbolic index propagates so the resulting Eq
      can be used inside a per-row ``SubDomain`` that restricts
      ``dim`` to a single row while leaving ``free_dim`` open.

    Caller is responsible for:

    1. Defining ``SubDomain``s for the boundary rows
       (``{dim: ('left', 1)}`` / ``{dim: ('middle', i, N-i-1)}`` etc).
    2. Restricting the bulk ``.dx`` / ``.dy`` to the interior rows
       via a separate SubDomain (rows ``4 .. N-5``).
    3. Applying these boundary expressions inside ``Eq(..., expr,
       subdomain=...)`` for the corresponding rows.

    Parameters
    ----------
    field : Devito TimeFunction or Function
        The field to differentiate.
    dim : Devito Dimension
        Dimension along which to take the derivative.
    order : int
        Interior order. Only ``order=4`` is supported (the (4, 2)
        diagonal-norm SBP D1).
    fs_location : str
        ``'bottom'`` writes boundary rows at indices 0..3 reading
        cells 0..5 (SBP D1 left-boundary stencil);
        ``'top'`` writes boundary rows at indices ``N-1`` down to
        ``N-4`` reading cells ``N-1`` down to ``N-6`` with sign
        flip (SBP D1 right-boundary mirror).
    free_dim : Devito Dimension or None
        For 2D fields, the other spatial dimension (the one NOT
        being differentiated). If provided, expressions index
        ``field[t, free_dim, idx_j]``. If ``None`` and the field
        has 2 spatial dims, the helper auto-infers ``free_dim``
        as the spatial dim ≠ ``dim``.

    Returns
    -------
    list of 4 sympy expressions, each evaluating to the SBP D1
    boundary-row derivative at the corresponding row. Index 0 of
    the returned list is the row CLOSEST to the boundary; index 3
    is 3 rows in from the boundary.

    Notes
    -----
    See ``tests/test_sbp_d1_devito.py`` for the 1D-bottom worked
    example. The 2D and 'top' branches mirror the ``sbp_d2_boundary_exprs``
    pattern validated by ``tests/test_sbp_d2_devito_2d.py``.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D1 is supported; got order={order}."
        )
    if fs_location not in ('bottom', 'top'):
        raise ValueError(
            f"fs_location must be 'bottom' or 'top'; got {fs_location!r}"
        )

    # Detect TimeFunction vs Function. Handles both:
    #  - regular TimeFunction (dims include grid.stepping_dim);
    #  - save=N TimeFunction (dims include grid.time_dim instead, e.g.
    #    when storing a recorded history).
    has_grid = hasattr(field, 'grid')
    if has_grid and field.grid.stepping_dim in field.dimensions:
        t = field.grid.stepping_dim
        is_time = True
    elif has_grid and field.grid.time_dim in field.dimensions:
        t = field.grid.time_dim
        is_time = True
    else:
        t = None
        is_time = False

    # Detect spatial dimensionality + auto-infer free_dim if needed
    spatial_dims = [d for d in field.dimensions if d is not t]
    is_2d = len(spatial_dims) == 2
    if free_dim is not None and not is_2d:
        raise ValueError(
            f"free_dim={free_dim!r} supplied but field is 1D in space. "
            f"Either drop free_dim or use a 2D field."
        )
    if is_2d and free_dim is None:
        # Auto-infer: free_dim is the spatial dim != dim
        for d in spatial_dims:
            if d is not dim:
                free_dim = d
                break

    coeffs = sbp_d1_coeffs(order=order)
    boundary_rows = coeffs["boundary_rows"]

    # For 'top', the boundary rows are read with reversed indexing
    # and sign-flipped (SBP D1 right-boundary is the mirror of left).
    # Index into field at (N-1-j) — sympy expression-compatible via
    # dim.symbolic_max - j.

    exprs = []
    for i in range(4):
        row = boundary_rows[i]
        expr = 0
        for j, w in enumerate(row):
            if w == 0:
                continue
            # Compute the index along `dim` for this boundary stencil tap.
            # Phase C5 (2026-05-21): ``base_idx`` allows the boundary "row 0"
            # to sit at a dim-internal index (e.g., a seafloor row at z=solid_cells
            # rather than the literal grid edge). Default None preserves
            # byte-identical legacy behaviour.
            if fs_location == 'bottom':
                if base_idx is None:
                    idx_j = j
                else:
                    idx_j = base_idx + j
            else:  # 'top' — mirror, reading from (N-1-j) with sign flip
                if base_idx is None:
                    idx_j = dim.symbolic_max - j
                else:
                    idx_j = base_idx - j

            # Build the field sample with appropriate indexing
            if is_2d:
                if is_time:
                    sample = field[t, free_dim, idx_j] if dim is spatial_dims[-1] \
                        else field[t, idx_j, free_dim]
                else:
                    sample = field[free_dim, idx_j] if dim is spatial_dims[-1] \
                        else field[idx_j, free_dim]
            else:
                if is_time:
                    sample = field[t, idx_j]
                else:
                    sample = field[idx_j]

            # SBP D1 right-boundary applies sign flip to mirror-symmetric stencil
            w_signed = (-float(w)) if fs_location == 'top' else float(w)
            expr += w_signed * sample
        expr /= dim.spacing
        exprs.append(expr)

    return exprs


def sbp_boundary_normal_derivative(field, dim, fs_location='bottom',
                                    free_dim=None, order=4, base_idx=None):
    """Return the SBP boundary normal-derivative ``S_3`` expression at
    the boundary row.

    Plan §108.followup-SBP-SAT-graduation Phase C2.4 helper for the
    Bader-Almquist-Dunham 2023 framework. Equivalent to
    ``sbp_d1_boundary_exprs(...)[0]`` — the first (boundary-closest) row
    of the SBP D1 boundary stencil. Caller binds the returned
    expression inside an Eq with ``subdomain=`` restricted to the
    boundary row.

    Bader 2023 uses ``S_3`` in the SAT penalty terms for boundary and
    interface conditions (eqs 9-10). It approximates :math:`\\partial_n
    \\phi` at the boundary using the SBP one-sided stencil (which does
    NOT read across the boundary).

    Parameters
    ----------
    field : Devito TimeFunction or Function
    dim : Devito Dimension
        The normal-to-boundary direction.
    fs_location : str
        ``'bottom'`` (boundary at z=0) or ``'top'`` (boundary at z=Nz-1).
    free_dim : Devito Dimension or None
        For 2D fields, the dimension parallel to the boundary.
    order : int
        SBP order. Currently only ``order=4`` is supported (matches
        Mattsson 2004 / PS-2015 / Bader 2023).

    Returns
    -------
    sympy.Expr
        The SBP boundary normal-derivative of ``field`` at the boundary
        row. Multiply by ``1/h_dim`` is already baked in (S_3 has
        units of inverse length).
    """
    exprs = sbp_d1_boundary_exprs(field, dim, order=order,
                                   fs_location=fs_location,
                                   free_dim=free_dim,
                                   base_idx=base_idx)
    return exprs[0]


def sbp_boundary_norm_weight(order: int = 4) -> Fraction:
    """Return the SBP diagonal-norm weight at the boundary row.

    The full norm matrix is ``H = h * diag(H_diag)`` where ``H_diag[0]
    = H_diag[N-1] = sbp_boundary_norm_weight(order)``. For order 4
    (Mattsson 2004 / PS-2015 / Bader 2023), the boundary norm weight
    is ``17/48``, so :math:`\\mathbf H_3^{-1}` at the boundary row
    is ``48 / (17 * h)``.

    Bader 2023 uses ``H_3^{-1}`` in every SAT penalty term (eqs 9-10).

    Parameters
    ----------
    order : int
        SBP order. Currently only ``order=4`` is supported.

    Returns
    -------
    fractions.Fraction
        The norm weight as a Fraction. Cast to float for use in
        Devito expressions.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP norm weights are tabulated; got order={order}."
        )
    return _NORM_WEIGHTS_4_2[0]


# ===========================================================================
# Phase 5g — variable-coefficient SBP D2 boundary stencils
# ===========================================================================
#
# The narrow SBP D2(a) variable-coefficient second-derivative operator
# from sw4 src/boundaryOpc.C (extracted byte-for-byte, 2026-05-15).
#
# Convention: ``(D2(a) u)_q = (1/h^2) * Σ_{k, m} acof(q, k, m) * a_m * u_k``
# where the indices are 1-based per the sw4 macro
# ``acof(q,k,m) = _acof[q-1+6*(k-1)+48*(m-1)]``.
#
# Boundary support: 6 rows × 8 cols × 8 coeff-indices = 384 dense entries
# (129 non-zero). Beyond the 6 boundary rows, the operator transitions to
# the standard 4th-order centred narrow D2(a) stencil documented in
# Mattsson & Nordström 2004 §2.

# Ghost-point coefficient (sw4 ``ghcof``) for the strong FS BC at the
# bottom row. Used in conjunction with acof row 1 to enforce
# σ_yy = σ_xy = 0 via image extension into a ghost row at index -1.
# Sw4 source: src/boundaryOpc.C lines 13-18.
_GHCOF_4_2: Tuple[Fraction, ...] = (
    Fraction(12, 17),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)


# Variable-coefficient SBP D2 boundary stencil table.
# Only non-zero entries are stored; lookups return Fraction(0) by default.
# Sw4 source: src/boundaryOpc.C lines 20-403 (acof block).
_ACOF_4_2: dict = {
    (1, 1, 1): Fraction(104, 289),
    (1, 1, 2): Fraction(-2476335, 2435692),
    (1, 1, 3): Fraction(-16189, 84966),
    (1, 1, 4): Fraction(-9, 3332),
    (1, 2, 1): Fraction(-516, 289),
    (1, 2, 2): Fraction(544521, 1217846),
    (1, 2, 3): Fraction(2509879, 3653538),
    (1, 3, 1): Fraction(312, 289),
    (1, 3, 2): Fraction(1024279, 2435692),
    (1, 3, 3): Fraction(-687797, 1217846),
    (1, 3, 4): Fraction(177, 3332),
    (1, 4, 1): Fraction(-104, 289),
    (1, 4, 2): Fraction(181507, 1217846),
    (1, 4, 3): Fraction(241309, 3653538),
    (1, 5, 3): Fraction(5, 2193),
    (1, 5, 4): Fraction(-48, 833),
    (1, 6, 4): Fraction(6, 833),
    (2, 1, 1): Fraction(12, 17),
    (2, 1, 2): Fraction(544521, 4226642),
    (2, 1, 3): Fraction(2509879, 12679926),
    (2, 2, 1): Fraction(-59, 68),
    (2, 2, 2): Fraction(-1633563, 4226642),
    (2, 2, 3): Fraction(-21510077, 25359852),
    (2, 2, 4): Fraction(-12655, 372939),
    (2, 3, 1): Fraction(2, 17),
    (2, 3, 2): Fraction(1633563, 4226642),
    (2, 3, 3): Fraction(2565299, 4226642),
    (2, 3, 4): Fraction(40072, 372939),
    (2, 4, 1): Fraction(3, 68),
    (2, 4, 2): Fraction(-544521, 4226642),
    (2, 4, 3): Fraction(987685, 25359852),
    (2, 4, 4): Fraction(-14762, 124313),
    (2, 5, 3): Fraction(1630, 372939),
    (2, 5, 4): Fraction(18976, 372939),
    (2, 6, 4): Fraction(-1, 177),
    (3, 1, 1): Fraction(-96, 731),
    (3, 1, 2): Fraction(1024279, 6160868),
    (3, 1, 3): Fraction(-687797, 3080434),
    (3, 1, 4): Fraction(177, 8428),
    (3, 2, 1): Fraction(118, 731),
    (3, 2, 2): Fraction(1633563, 3080434),
    (3, 2, 3): Fraction(2565299, 3080434),
    (3, 2, 4): Fraction(40072, 271803),
    (3, 3, 1): Fraction(-16, 731),
    (3, 3, 2): Fraction(-5380447, 6160868),
    (3, 3, 3): Fraction(-3569115, 3080434),
    (3, 3, 4): Fraction(-331815, 362404),
    (3, 3, 5): Fraction(-283, 6321),
    (3, 4, 1): Fraction(-6, 731),
    (3, 4, 2): Fraction(544521, 3080434),
    (3, 4, 3): Fraction(2193521, 3080434),
    (3, 4, 4): Fraction(8065, 12943),
    (3, 4, 5): Fraction(381, 2107),
    (3, 5, 3): Fraction(-14762, 90601),
    (3, 5, 4): Fraction(32555, 271803),
    (3, 5, 5): Fraction(-283, 2107),
    (3, 6, 4): Fraction(9, 2107),
    (3, 6, 5): Fraction(-11, 6321),
    (4, 1, 1): Fraction(-36, 833),
    (4, 1, 2): Fraction(181507, 3510262),
    (4, 1, 3): Fraction(241309, 10530786),
    (4, 2, 1): Fraction(177, 3332),
    (4, 2, 2): Fraction(-544521, 3510262),
    (4, 2, 3): Fraction(987685, 21061572),
    (4, 2, 4): Fraction(-14762, 103243),
    (4, 3, 1): Fraction(-6, 833),
    (4, 3, 2): Fraction(544521, 3510262),
    (4, 3, 3): Fraction(2193521, 3510262),
    (4, 3, 4): Fraction(8065, 14749),
    (4, 3, 5): Fraction(381, 2401),
    (4, 4, 1): Fraction(-9, 3332),
    (4, 4, 2): Fraction(-181507, 3510262),
    (4, 4, 3): Fraction(-2647979, 3008796),
    (4, 4, 4): Fraction(-80793, 103243),
    (4, 4, 5): Fraction(-1927, 2401),
    (4, 4, 6): Fraction(-2, 49),
    (4, 5, 3): Fraction(57418, 309729),
    (4, 5, 4): Fraction(51269, 103243),
    (4, 5, 5): Fraction(1143, 2401),
    (4, 5, 6): Fraction(8, 49),
    (4, 6, 4): Fraction(-283, 2401),
    (4, 6, 5): Fraction(403, 2401),
    (4, 6, 6): Fraction(-6, 49),
    (5, 1, 3): Fraction(5, 6192),
    (5, 1, 4): Fraction(-1, 49),
    (5, 2, 3): Fraction(815, 151704),
    (5, 2, 4): Fraction(1186, 18963),
    (5, 3, 3): Fraction(-7381, 50568),
    (5, 3, 4): Fraction(32555, 303408),
    (5, 3, 5): Fraction(-283, 2352),
    (5, 4, 3): Fraction(28709, 151704),
    (5, 4, 4): Fraction(51269, 101136),
    (5, 4, 5): Fraction(381, 784),
    (5, 4, 6): Fraction(1, 6),
    (5, 5, 3): Fraction(-349, 7056),
    (5, 5, 4): Fraction(-247951, 303408),
    (5, 5, 5): Fraction(-577, 784),
    (5, 5, 6): Fraction(-5, 6),
    (5, 5, 7): Fraction(-1, 24),
    (5, 6, 4): Fraction(1135, 7056),
    (5, 6, 5): Fraction(1165, 2352),
    (5, 6, 6): Fraction(1, 2),
    (5, 6, 7): Fraction(1, 6),
    (5, 7, 5): Fraction(-1, 8),
    (5, 7, 6): Fraction(1, 6),
    (5, 7, 7): Fraction(-1, 8),
    (6, 1, 4): Fraction(1, 392),
    (6, 2, 4): Fraction(-1, 144),
    (6, 3, 4): Fraction(3, 784),
    (6, 3, 5): Fraction(-11, 7056),
    (6, 4, 4): Fraction(-283, 2352),
    (6, 4, 5): Fraction(403, 2352),
    (6, 4, 6): Fraction(-1, 8),
    (6, 5, 4): Fraction(1135, 7056),
    (6, 5, 5): Fraction(1165, 2352),
    (6, 5, 6): Fraction(1, 2),
    (6, 5, 7): Fraction(1, 6),
    (6, 6, 4): Fraction(-47, 1176),
    (6, 6, 5): Fraction(-5869, 7056),
    (6, 6, 6): Fraction(-3, 4),
    (6, 6, 7): Fraction(-5, 6),
    (6, 6, 8): Fraction(-1, 24),
    (6, 7, 5): Fraction(1, 6),
    (6, 7, 6): Fraction(1, 2),
    (6, 7, 7): Fraction(1, 2),
    (6, 7, 8): Fraction(1, 6),
    (6, 8, 6): Fraction(-1, 8),
    (6, 8, 7): Fraction(1, 6),
    (6, 8, 8): Fraction(-1, 8),
}


# Standard 4th-order narrow centred D2(a) interior stencil at offsets
# (-2, -1, 0, 1, 2). For constant a this reduces to the well-known
# centred D2 weights (-1/12, 4/3, -5/2, 4/3, -1/12). For variable a,
# the narrow form per Mattsson & Nordström 2004 eq 2.5 is:
#
#   (D2(a) u)_i = (1/h^2) * [
#       (1/2)*(-a_{i-1} - a_{i})*(u_{i-1} - u_i) ... (compatible form)
#   ]
#
# Per sw4 convention (boundaryOpc.C uses acof for the first 6 boundary
# rows and the standard narrow centred stencil for the interior), the
# interior weights for constant-coefficient D2 are:
_INTERIOR_D2_4: Tuple[Fraction, ...] = (
    Fraction(-1, 12), Fraction(4, 3), Fraction(-5, 2),
    Fraction(4, 3),   Fraction(-1, 12),
)


def sbp_d2_coeffs(order: int = 4) -> dict:
    """Return the (4, 2) diagonal-norm SBP D2 boundary stencil + ghost
    coefficient + interior stencil + norm weights.

    Parameters
    ----------
    order : int
        Interior order of accuracy. Only ``order=4`` is currently
        tabulated (matches PS 2015 / sw4).

    Returns
    -------
    dict with keys:
        ``"acof"`` : dict[tuple[int, int, int], Fraction]
            Sparse representation of the variable-coefficient D2
            boundary stencils. Keys are ``(q, k, m)`` 1-indexed per the
            sw4 macro convention; missing keys are zero. Apply as
            ``(D_2(a) u)_q = (1/h^2) * Σ_{k, m} acof[(q, k, m)] * a_m * u_k``
            for boundary rows ``q ∈ {1, 2, 3, 4, 5, 6}``.

        ``"ghcof"`` : list of 6 ``Fraction``
            Ghost-point coefficient for strong-FS BC enforcement via
            image extension. Only ``ghcof[0] = 12/17`` is non-zero;
            this is the coefficient multiplying the ghost row in
            acof(1, ...) for the boundary D2 stencil at the FS.

        ``"interior"`` : list of 5 ``Fraction``
            Standard 4th-order centred D2 weights at offsets
            ``(-2, -1, 0, 1, 2)``. For constant coefficient ``a``,
            the interior D2 reads ``(D_2(a) u)_i ≈ (a/h^2) * Σ
            interior[k] * u_{i+k-2}``. For variable ``a``, the
            compatible narrow form (Mattsson & Nordström 2004 eq 2.5)
            applies — see ``build_sbp_d2_matrix``.

        ``"norm_weights"`` : list of 4 ``Fraction``
            Same diagonal norm matrix as D1 (the SBP property couples
            D1, D2, and H via a single norm matrix). Reused from
            :data:`_NORM_WEIGHTS_4_2`.

        ``"max_boundary_row"`` : int = 6
            Number of boundary rows on each side of the domain. Rows
            ``q ∈ {1..6}`` use ``acof``; rows beyond use the interior
            stencil.

        ``"stencil_width"`` : int = 8
            Maximum ``k`` and ``m`` indices for the boundary stencils
            (i.e. the boundary stencil reads up to 8 cells into the
            interior).

    Notes
    -----
    The SBP property for variable-coefficient D2 is more subtle than
    for D1: per Mattsson 2012 Definition 2.3,
    $D_2(b) = H^{-1}(-M(b) + \\tilde{B} S)$ with $M(b)$ symmetric and
    positive semi-definite. The exact $M(b)$ for sw4's specific
    variant is not in our literature corpus (the canonical sw4
    reference is Sjögreen-Petersson 2012, JSC 49(1):129-149, DOI
    10.1007/s10915-011-9501-7). Empirical verification (interior
    rate-4 + boundary rate-2 convergence on smooth functions) is
    provided in ``tests/test_sbp_d2_convergence.py``.

    Sw4 reference: ``src/boundaryOpc.C`` lines 13-403 (ghcof + acof).

    Related literature: Mattsson & Nordström 2004 (constant-coef
    SBP D2); Mattsson, Ham, Iaccarino 2008 §2.1 + Appendix I.2
    (4th-order variant); Mattsson 2012 §2.2 + Appendix A.2
    (compatible variable-coef variant — *different* from sw4's
    frozen-coefficient narrow scheme).
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D2 (the (4, 2) diagonal-norm variant "
            f"used by @petersson_sjogreen_2015) is currently "
            f"implemented; got order={order}."
        )

    return {
        "acof": dict(_ACOF_4_2),
        "ghcof": list(_GHCOF_4_2),
        "interior": list(_INTERIOR_D2_4),
        "norm_weights": list(_NORM_WEIGHTS_4_2),
        "max_boundary_row": 6,
        "stencil_width": 8,
    }


def acof_value(q: int, k: int, m: int, order: int = 4) -> Fraction:
    """Lookup helper: return ``acof[(q, k, m)]`` for the (4, 2) SBP D2
    boundary stencil. Returns ``Fraction(0)`` for any (q, k, m) not in
    the sw4 source table. Indices are 1-based per the sw4 macro
    convention; valid ranges are ``q ∈ [1, 6]``, ``k ∈ [1, 8]``,
    ``m ∈ [1, 8]``.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D2 is supported; got order={order}."
        )
    return _ACOF_4_2.get((q, k, m), Fraction(0))


def build_sbp_d2_matrix(N: int, b: List[Fraction],
                        order: int = 4) -> List[List[Fraction]]:
    """Construct the ``N × N`` sw4-faithful SBP D2(b) operator (the
    variable-coefficient second-derivative discretisation as
    implemented in @petersson_sjogreen_2015's reference code).

    **Convention** (verified empirically against sw4 ``src/boundaryOpc.C``
    and against analytic test functions):

    Multiplying by $1/h^2$ gives the actual D2(b) operator:
        $(D_2(b) \\, u)_i \\approx \\frac{1}{h^2} \\sum_j (\\text{D2}_{\\text{matrix}})_{ij} u_j$

    Boundary rows 0..5 use the sw4 ``acof`` table:
        $(\\text{D2}_{\\text{matrix}})_{q-1, k-1}
         = \\sum_m \\text{acof}(q, k, m) \\, b_{m-1}$
    for $q \\in \\{1..6\\}$, $k \\in \\{1..8\\}$, $m \\in \\{1..8\\}$.

    Interior rows 6..N-7 use the **frozen-coefficient** narrow centred
    5-point stencil scaled by $b$ at the row centre:
        $(\\text{D2}_{\\text{matrix}})_{i, i+s} = b[i] \\cdot
            \\text{interior}[s+2]$
    where ``interior = (-1/12, 4/3, -5/2, 4/3, -1/12)``.

    Right boundary rows N-1..N-6 mirror left boundary rows (D2 is
    symmetric at the boundary level — no sign flip in mirror).

    **What this is NOT**: this is NOT Mattsson 2012's "compatible
    narrow-diagonal D2" with variable-coefficient interior (Mattsson
    2012 page 668). sw4 uses a simpler frozen-coefficient narrow
    scheme. For our 4th-order-accurate elastic-TTI use case the
    difference is sub-leading (both schemes are 4th-order in smooth
    media); the formal SBP property of sw4's variant follows from
    Sjögreen-Petersson 2012 (the actual sw4 reference, JSC 49(1):
    129-149, DOI 10.1007/s10915-011-9501-7) rather than Mattsson 2012.

    **Row 0 and row N-1** specifically: these rows return partial
    results that require completion via the ``ghcof`` ghost-point
    coefficient when a free-surface BC is imposed (see
    :func:`sbp_d2_boundary_exprs` and Phase 5e in plan §73). Without
    BC enforcement, applying this matrix at rows 0/N-1 gives the
    "free / unspecified" wavefield boundary — useful for interior
    diagnostics but not for FS enforcement.

    Parameters
    ----------
    N : int
        Grid size. Must satisfy ``N >= 14`` (left + right 6-row
        boundary on each side + 2-row interior gap).
    b : list of ``Fraction``
        Variable coefficient values at each of the ``N`` grid points.
        Must have length ``N``. Pass ``[Fraction(1)] * N`` for
        constant-coefficient testing.
    order : int
        Only ``order=4`` is supported.

    Returns
    -------
    ``N × N`` matrix as list of lists of ``Fraction``.

    Notes
    -----
    Empirical verification (``tests/test_sbp_d2_convergence.py``)
    confirms 4th-order interior accuracy and 2nd-order boundary
    accuracy on smooth test functions, consistent with the SBP-(4,2)
    declaration in the sw4 source comments.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D2 is supported; got order={order}."
        )
    if N < 14:
        raise ValueError(
            f"build_sbp_d2_matrix requires N >= 14 (left + right "
            f"6-row boundary stencils on each side + 2-row interior "
            f"gap); got N={N}."
        )
    if len(b) != N:
        raise ValueError(
            f"Coefficient array `b` must have length N={N}; got "
            f"len(b)={len(b)}."
        )

    M: List[List[Fraction]] = [[Fraction(0)] * N for _ in range(N)]

    # Left boundary rows q=1..6 (0-indexed rows 0..5) from acof table.
    for q in range(1, 7):
        for k in range(1, 9):
            for m in range(1, 9):
                w = _ACOF_4_2.get((q, k, m), Fraction(0))
                if w == 0:
                    continue
                M[q - 1][k - 1] += w * b[m - 1]

    # Interior rows 6..N-7 use the frozen-coefficient 4th-order narrow
    # centred D2 stencil (-1/12, 4/3, -5/2, 4/3, -1/12) scaled by b[i].
    for i in range(6, N - 6):
        for k, w in enumerate(_INTERIOR_D2_4):
            j = i + (k - 2)
            M[i][j] += b[i] * w

    # Right boundary rows N-1..N-6 mirror left boundary rows.
    # D2 boundary is symmetric in structure (same mirror as M is
    # symmetric in Mattsson 2012's narrow scheme).
    for q in range(1, 7):
        for k in range(1, 9):
            for m in range(1, 9):
                w = _ACOF_4_2.get((q, k, m), Fraction(0))
                if w == 0:
                    continue
                M[N - q][N - k] += w * b[N - m]

    return M


def sbp_d2_boundary_exprs(field, coeff_field, dim, order=4,
                          fs_location='bottom', free_dim=None,
                          base_idx=None):
    """Build symbolic Devito expressions for the SBP D2(a) boundary rows.

    Phase 5g helper. Returns a list of 6 sympy expressions, one per
    boundary row, each computing
    ``(D_2(a) field)_q = (1/h^2) * Σ_{k, m} acof(q, k, m) * a[m-1] * field[t, k-1]``
    (for ``fs_location='bottom'``) using 1-indexed sw4 macro
    convention.

    **1D vs 2D modes** (plan §100 Phase 5h.0):

    - **1D mode** (default; ``free_dim=None`` AND the field has
      exactly one spatial dim): backward-compatible with the §77
      Phase 5g signature. Indexing is ``field[t, idx_k]``.

    - **2D mode** (``free_dim`` supplied, or the field is detected
      to have 2 spatial dims): the FS boundary runs along the
      ``free_dim`` direction; ``dim`` is the FS-perpendicular
      coordinate. Indexing becomes ``field[t, free_dim, idx_k]`` —
      the free dim's symbolic index propagates so the resulting Eq
      can be used inside a per-row ``SubDomain`` that restricts
      ``dim`` to a single row while leaving ``free_dim`` open.

    Caller is responsible for:

    1. Defining ``SubDomain``s for the 6 boundary rows
       (``{dim: ('left', 1)}``, ``{dim: ('middle', 1, ...)}``, etc).
       In 2D, the SubDomain also covers the bulk ``free_dim``.
    2. Restricting the bulk centred-D2 stencil to the interior rows
       via a separate SubDomain (rows ``6 .. N-7``).
    3. Applying these boundary expressions inside ``Eq(..., expr,
       subdomain=...)`` for the corresponding rows.

    Parameters
    ----------
    field : Devito TimeFunction or Function
        The field to differentiate twice.
    coeff_field : Devito Function or TimeFunction
        The variable coefficient ``a(z)``. Indexed via ``coeff_field[m]``
        for plain Functions (analogous indexing to ``field``).
    dim : Devito Dimension
        Dimension along which to take the second derivative.
    order : int
        Interior order. Only ``order=4`` is supported.
    fs_location : str
        ``'bottom'`` writes boundary rows at indices 0..5;
        ``'top'`` writes mirror-symmetric rows at indices N-1..N-6.
    free_dim : Devito Dimension or None
        For 2D fields, the other spatial dimension (the one that
        is NOT being differentiated). If provided, the returned
        expressions index ``field`` and ``coeff_field`` at
        ``[t, free_dim, idx_k]``. If ``None`` (default) and the
        field has 2 spatial dims, the helper auto-infers ``free_dim``
        as the spatial dim ≠ ``dim``.

    Returns
    -------
    list of 6 sympy expressions, each evaluating to the SBP D2
    boundary-row second derivative of ``field`` at the corresponding
    row.

    Notes
    -----
    The Devito convention: for a TimeFunction ``field``, ``field[t, k]``
    reads the current time slot at position ``k``. Variable coefficient
    is read as ``coeff_field[m]``.

    See ``tests/test_sbp_d2_devito.py`` for a 1D worked example that
    validates these expressions produce machine-precision match
    against the pure-NumPy reference in
    ``tests/test_sbp_d2_convergence.py``. See
    ``tests/test_sbp_d2_devito_2d.py`` (plan §100 Phase 5h.0) for
    the 2D extension.
    """
    if order != 4:
        raise NotImplementedError(
            f"Only order=4 SBP D2 is supported; got order={order}."
        )
    if fs_location not in ('bottom', 'top'):
        raise ValueError(
            f"fs_location must be 'bottom' or 'top'; got {fs_location!r}"
        )

    # Detect TimeFunction vs Function
    is_time = hasattr(field, 'grid') and (field.grid.stepping_dim
                                          in field.dimensions)
    t = field.grid.stepping_dim if is_time else None

    # Detect coefficient time-indexing
    coeff_is_time = (hasattr(coeff_field, 'grid')
                     and coeff_field.grid.stepping_dim
                     in coeff_field.dimensions)
    t_c = coeff_field.grid.stepping_dim if coeff_is_time else None

    # Plan §100 Phase 5h.0: 2D-aware free_dim detection.
    # If the field has 2 spatial dims (= 1 time dim + 2 space dims,
    # OR 2 space dims if Function) and free_dim is not explicitly
    # provided, auto-infer.
    spatial_dims = [d for d in field.dimensions
                    if d != field.grid.stepping_dim] if is_time \
        else list(field.dimensions)
    is_2d = len(spatial_dims) == 2
    if is_2d and free_dim is None:
        # The non-FS-perpendicular dim is the one ≠ `dim`.
        other = [d for d in spatial_dims if d != dim]
        if len(other) != 1:
            raise ValueError(
                f"Cannot auto-infer free_dim: field.dimensions = "
                f"{field.dimensions}, dim = {dim}. Pass free_dim "
                f"explicitly.")
        free_dim = other[0]
    elif not is_2d and free_dim is not None:
        raise ValueError(
            f"free_dim={free_dim} supplied but field is 1D in "
            f"space (spatial dims = {spatial_dims}). Either pass "
            f"a 2D field or omit free_dim.")

    if fs_location == 'top':
        # Boundary indices are dim.symbolic_max - (q - 1) for q ∈ {1..6}
        # but the relative offsets within each row remain k=1..8 reading
        # outward from the boundary toward the interior.
        # Plan §108.followup-SBP-SAT-graduation Phase C2.2 (2026-05-21):
        # ``base_idx`` allows the boundary "row 0" to be moved away from
        # the literal grid top. Default (None) preserves the legacy
        # ``dim.symbolic_max`` behaviour. Use the parameter when the
        # boundary lives at a fluid-internal row (e.g., the seafloor row
        # for an upper-fluid SBP-SAT operator).
        zmax = dim.symbolic_max if base_idx is None else base_idx
        boundary_idx = lambda q: zmax - (q - 1)
        # Reading k cells inward from the top: index = zmax - (k - 1).
        read_idx = lambda k: zmax - (k - 1)
    else:
        # fs_location == 'bottom'
        zmin = 0 if base_idx is None else base_idx
        boundary_idx = lambda q: zmin + (q - 1)
        read_idx = lambda k: zmin + (k - 1)

    exprs = []
    h_squared = dim.spacing ** 2

    for q in range(1, 7):
        expr = 0
        for k in range(1, 9):
            for m in range(1, 9):
                w = _ACOF_4_2.get((q, k, m), Fraction(0))
                if w == 0:
                    continue
                idx_k = read_idx(k)
                idx_m = read_idx(m)
                if free_dim is not None:
                    # 2D: index the free dim symbolically; restrict
                    # `dim` to read_idx(k) and read_idx(m).
                    if is_time:
                        u_sample = field[t, free_dim, idx_k]
                    else:
                        u_sample = field[free_dim, idx_k]
                    if coeff_is_time:
                        a_sample = coeff_field[t_c, free_dim, idx_m]
                    else:
                        a_sample = coeff_field[free_dim, idx_m]
                else:
                    # 1D backwards-compatible path.
                    if is_time:
                        u_sample = field[t, idx_k]
                    else:
                        u_sample = field[idx_k]
                    if coeff_is_time:
                        a_sample = coeff_field[t_c, idx_m]
                    else:
                        a_sample = coeff_field[idx_m]
                expr += float(w) * a_sample * u_sample
        expr /= h_squared
        exprs.append(expr)

    return exprs


# ===========================================================================
# Plan §104p.1 — Shared helpers for the 5-step SBP-SAT pattern.
# ===========================================================================
#
# These helpers factor the SBP-SAT bulk operator pattern out of per-method
# implementations. They were extracted after Methods 5 (SBP-elastic) and 9
# (SBP-VE) were independently graduated in §104 Phase E.6/E.7a; the
# §104p completion plan unifies Methods 5, 9, 44, 50, 73, 75, 78 around
# a single source-of-truth.
#
# Architecture (per plan §104 Phase E.1):
#
#   1. Materialised auxiliary stress Functions σ_aux (sxx_aux, sxy_aux,
#      syy_aux). Caller allocates them as plain ``Function`` (not
#      TimeFunction) with ``space_order=so`` matching the displacement
#      space_order.
#
#   2. Bulk strain → stress (whole domain, centred-FD strain). Stress
#      Eqs carry ``implicit_dims=t_dim`` per the §58 LW4-DP cascade fix
#      pattern (Function-only chain inside time loop).
#
#   3. SBP D1 boundary correction for σ_aux at top 4 y-rows (indexed-LHS
#      Eqs; SBP D1 stencil from ``sbp_d1_boundary_exprs`` for ∂u/∂y;
#      centred FD for ∂u/∂x).
#
#   4. Bulk displacement update (TimeFunction LHS; naturally in time
#      loop) via centred-FD divergence of σ_aux.
#
#   5. SBP D1 boundary correction for u at top 4 y-rows (indexed-LHS
#      Eqs; SBP D1 stencils on ∂σ_aux/∂y).
#
# Variant-axis extensions are passed through the ``extra_stress`` kwarg
# (sympy expressions summed into elastic σ inside σ_aux):
#
#   - Method 9 (GSLS-SBP-VE):  Σ_l memory[ll]
#   - Methods 50 (LW4-SBP-VE), 73/75/78 (KJ-SBP variants): KJ collapsed
#     mem_sxx/syy/sxy from ``kjartansson.build_kjartansson_memory_eqs_*``
#   - Methods 44/50/75 (LW4-SBP variants): see ``sbp_sat_lw4_stage_eqs``
#     below — LW4 needs SBP corrections at BOTH the L(u) stage and the
#     L²(u) divergence step.
# ---------------------------------------------------------------------------


def sbp_sat_eqs_for_displacement(
    grid,
    ux,
    uy,
    sxx_aux,
    sxy_aux,
    syy_aux,
    rho_f,
    damp,
    C11_f,
    C22_f,
    C12_f,
    C16_f,
    C26_f,
    C66_f,
    *,
    mms_source=None,
    extra_stress=None,
    space_order: int = 4,
    apply_top_boundary_correction: bool = True,
):
    """Build the 5-step SBP-SAT auxiliary-σ Eq list for a displacement
    formulation per Petersson-Sjögreen 2015 + Mattsson 2004 (4, 2)
    boundary stencils.

    Plan §104p.1 shared helper, extracted from
    ``05_sbp/sbp_displacement_tti.py:build_sbp_sat_operator`` and
    ``09_viscoelastic_sbp/viscoelastic_sbp_tti.py:build_viscoelastic_sbp_sat_operator``.

    Parameters
    ----------
    grid : devito.Grid
        2D grid with dims (x_dim, y_dim). y_dim is the FS-perpendicular
        direction by project convention.
    ux, uy : devito.TimeFunction
        Displacement components. For Method 78 REM k-loop usage, pass
        the Chebyshev buffer Functions (Txc, Tyc) instead; the helper
        treats them as "displacement" for the 5-step pattern. (The
        REM-specific output target — Txn / Tyn — is handled by the
        caller via a separate Eq written after this helper's output.)
    sxx_aux, sxy_aux, syy_aux : devito.Function
        Pre-allocated auxiliary stress Functions (NOT TimeFunctions),
        with ``space_order=space_order`` and the same dtype as ux.
    rho_f, C11_f .. C66_f : devito.Function
        Density and 2D stiffness components.
    damp : devito.Function
        Cerjan damping field (set to 1.0 in MMS mode).
    mms_source : dict or None
        Optional {'fx': sympy_expr, 'fy': sympy_expr} body-force
        contributions for MMS testing.
    extra_stress : dict or None
        Optional dict with optional keys 'sxx', 'sxy', 'syy'. Each
        value is a sympy expression added to the elastic stress inside
        σ_aux. Used for memory-variable contributions (GSLS / KJ).
        The boundary-correction step (top 4 rows) sums the same
        extra_stress at the row's indexed (x_dim, y_idx) location —
        the caller must provide expressions that can be evaluated at
        a per-row index (e.g., ``memory[ll][t, x_dim, y_idx]`` for
        TimeFunction memory).

        NOTE: if extra_stress depends on TimeFunctions and the helper
        is invoked with explicit ``time_idx`` reads, the caller must
        supply per-row expressions via a callable interface in a
        future extension; the current implementation passes the bulk
        extra_stress through unchanged at both bulk and boundary
        steps (relying on Devito's _eval_at to evaluate at the
        iteration point).
    space_order : int
        Spatial order; must be ≥ 4 for the SBP D1 (4, 2) stencils.
    apply_top_boundary_correction : bool
        If True (default), include SBP D1 boundary correction Eqs at
        the top 4 y-rows. If False, returns the bulk-only 5-step
        Eq list (back-compat with the pre-§104 centred-FD path).

    Returns
    -------
    list of devito.Eq
        The Eq list in the order required for correct stencil
        composition (bulk σ, then σ-boundary correction, then bulk u
        update, then u-boundary correction). Caller appends source
        injection / extra Eqs after this list.

    Notes
    -----
    The (4, 2) SBP D1 boundary stencil byte-match against the sw4
    reference (Petersson-Sjögreen 2015 [20]) is gated by
    ``tests/test_sbp_d1_coefficients.py`` (43 tests).
    """
    from devito import Eq

    t = grid.stepping_dim
    t_dim = grid.time_dim
    dt = t.spacing
    x_dim, y_dim = grid.dimensions
    so = space_order

    # Default-empty extra_stress for cleaner downstream sums.
    es_sxx = (extra_stress or {}).get('sxx', 0)
    es_sxy = (extra_stress or {}).get('sxy', 0)
    es_syy = (extra_stress or {}).get('syy', 0)

    # ----- Step 2: Bulk strain → stress (centred FD; +extra_stress) -----
    ux_x = ux.dx
    ux_y = ux.dy
    uy_x = uy.dx
    uy_y = uy.dy

    eqs = [
        Eq(
            sxx_aux,
            C11_f * ux_x + C12_f * uy_y + C16_f * (ux_y + uy_x) + es_sxx,
            implicit_dims=t_dim,
        ),
        Eq(
            sxy_aux,
            C16_f * ux_x + C26_f * uy_y + C66_f * (ux_y + uy_x) + es_sxy,
            implicit_dims=t_dim,
        ),
        Eq(
            syy_aux,
            C12_f * ux_x + C22_f * uy_y + C26_f * (ux_y + uy_x) + es_syy,
            implicit_dims=t_dim,
        ),
    ]

    fx_mms = mms_source['fx'] if mms_source else 0
    fy_mms = mms_source['fy'] if mms_source else 0

    # ----- Step 3: SBP D1 boundary correction for σ at top 4 y-rows -----
    if apply_top_boundary_correction and so >= 4:
        dux_dy_sbp = sbp_d1_boundary_exprs(
            ux, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )
        duy_dy_sbp = sbp_d1_boundary_exprs(
            uy, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )

        for q in range(4):
            y_idx = y_dim.symbolic_max - q
            dux_dy_q = dux_dy_sbp[q]
            duy_dy_q = duy_dy_sbp[q]

            # Elastic σ at row q with SBP D1 for ∂u/∂y; centred FD on
            # tangent ∂u/∂x (evaluated at the iteration point via
            # Devito's _eval_at machinery).
            sxx_q = (
                C11_f * ux.dx + C12_f * duy_dy_q
                + C16_f * (dux_dy_q + uy.dx)
                + es_sxx
            )
            sxy_q = (
                C16_f * ux.dx + C26_f * duy_dy_q
                + C66_f * (dux_dy_q + uy.dx)
                + es_sxy
            )
            syy_q = (
                C12_f * ux.dx + C22_f * duy_dy_q
                + C26_f * (dux_dy_q + uy.dx)
                + es_syy
            )

            eqs.append(Eq(
                sxx_aux[x_dim, y_idx], sxx_q,
                implicit_dims=t_dim,
            ))
            eqs.append(Eq(
                sxy_aux[x_dim, y_idx], sxy_q,
                implicit_dims=t_dim,
            ))
            eqs.append(Eq(
                syy_aux[x_dim, y_idx], syy_q,
                implicit_dims=t_dim,
            ))

    # ----- Step 4: Bulk displacement update (centred-FD divergence) -----
    rhs_x_bulk = (sxx_aux.dx + sxy_aux.dy + fx_mms) / rho_f
    rhs_y_bulk = (sxy_aux.dx + syy_aux.dy + fy_mms) / rho_f

    eqs.extend([
        Eq(ux.forward, damp * (2 * ux - ux.backward + dt**2 * rhs_x_bulk)),
        Eq(uy.forward, damp * (2 * uy - uy.backward + dt**2 * rhs_y_bulk)),
    ])

    # ----- Step 5: SBP D1 boundary correction for u at top 4 y-rows -----
    if apply_top_boundary_correction and so >= 4:
        dsxy_dy_sbp = sbp_d1_boundary_exprs(
            sxy_aux, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )
        dsyy_dy_sbp = sbp_d1_boundary_exprs(
            syy_aux, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )

        for q in range(4):
            y_idx = y_dim.symbolic_max - q
            dsxy_dy_q = dsxy_dy_sbp[q]
            dsyy_dy_q = dsyy_dy_sbp[q]

            rhs_x_q = (sxx_aux.dx + dsxy_dy_q + fx_mms) / rho_f
            rhs_y_q = (sxy_aux.dx + dsyy_dy_q + fy_mms) / rho_f

            eqs.append(Eq(
                ux[t + 1, x_dim, y_idx],
                damp[x_dim, y_idx] * (
                    2 * ux[t, x_dim, y_idx]
                    - ux[t - 1, x_dim, y_idx]
                    + dt**2 * rhs_x_q
                )
            ))
            eqs.append(Eq(
                uy[t + 1, x_dim, y_idx],
                damp[x_dim, y_idx] * (
                    2 * uy[t, x_dim, y_idx]
                    - uy[t - 1, x_dim, y_idx]
                    + dt**2 * rhs_y_q
                )
            ))

    return eqs


def sbp_sat_lw4_stage_eqs(
    grid,
    ux,
    uy,
    Lx_f,
    Ly_f,
    rho_f,
    C11_f,
    C22_f,
    C12_f,
    C16_f,
    C26_f,
    C66_f,
    *,
    mms_source=None,
    space_order: int = 4,
    apply_top_boundary_correction: bool = True,
    name_suffix: str = '',
    memory_xx_list=None,
    memory_xy_list=None,
    memory_yy_list=None,
    implicit_dims=None,
):
    """Build the LW4 stage Eqs `Lx_f = L(u) + fx_mms` and `Ly_f = L(u) + fy_mms`
    with SBP D1 boundary correction at the top 4 y-rows.

    Plan §104p.1 — for LW4-SBP methods (44, 50, 75). The LW4 scheme
    needs the stored L(u) intermediate Functions to carry SBP-faithful
    values at the top boundary so that the subsequent L²(u) divergence
    (computed via centred FD of Lx_f / Ly_f) reads consistent
    boundary-row values.

    Plan §104q.followup2 — extended for REM Chebyshev k-loop (Method 78).
    The `implicit_dims` kwarg overrides the default `t_dim` scope so the
    caller can place the Eqs inside a per-k-iteration scope, e.g.,
    ``implicit_dims=(time_dim, k)`` for Chebyshev recurrence.

    Pattern: same 5-step architecture as
    ``sbp_sat_eqs_for_displacement``, but writes the result into the
    intermediate Functions ``Lx_f`` / ``Ly_f`` instead of evolving a
    TimeFunction. Uses auxiliary σ Functions internally.

    Parameters
    ----------
    grid, ux, uy, rho_f, C11_f .. C66_f, mms_source, space_order,
    apply_top_boundary_correction
        See ``sbp_sat_eqs_for_displacement``.
    Lx_f, Ly_f : devito.Function
        LW4 stage-1 intermediate Functions (per plan §70 Phase 3b-followup
        pattern). The helper writes ``Lx_f = (1/rho) · ∂σ/∂x + ∂σ/∂y + fx_mms``
        (the L(u) expression) with SBP D1 boundary correction.
    memory_xx_list, memory_xy_list, memory_yy_list : list of TimeFunction, optional
        Plan §104q.followup: GSLS memory variables (one per L mechanism).
        When provided, σ_stage becomes σ_total = σ_elastic + Σ_l memory_l.
        Both bulk and SBP D1 boundary correction read σ_total — this is
        the σ_total approach from Method 9 (viscoelastic SBP) extended
        to LW4. Used by Method 50 (LW4-SBP-VE) stage 1. Stage 2 (L²
        applied to Lx_f) should pass these as None — the helper detects
        and treats inputs without memory contribution.

    Returns
    -------
    list of devito.Eq
        Eq list implementing the stage-1 computation. Caller must
        ensure these Eqs run BEFORE the L²(u) divergence step in the
        time-loop ordering.

    Notes
    -----
    The helper materialises its OWN auxiliary σ Functions for the
    stage-1 L(u) computation (named with a ``_stage`` suffix). These
    are independent from any displacement-step σ_aux the caller may
    allocate for the subsequent L²(u) step.
    """
    from devito import Eq, Function

    x_dim, y_dim = grid.dimensions
    so = space_order
    # Default implicit_dims to time_dim (§104p.1 LW4 pattern). Callers in REM
    # k-loop pass (time_dim, k) per §104q.followup2 so the SBP boundary
    # correction fires at every Chebyshev recurrence iteration.
    t_dim = (
        implicit_dims if implicit_dims is not None else grid.time_dim
    )

    sxx_stage = Function(
        name=f'sxx_aux_lw4_stage{name_suffix}', grid=grid, space_order=so,
        dtype=ux.dtype,
    )
    sxy_stage = Function(
        name=f'sxy_aux_lw4_stage{name_suffix}', grid=grid, space_order=so,
        dtype=ux.dtype,
    )
    syy_stage = Function(
        name=f'syy_aux_lw4_stage{name_suffix}', grid=grid, space_order=so,
        dtype=ux.dtype,
    )

    # ----- Step 2: Bulk strain → stage stress (centred FD) -----
    ux_x = ux.dx
    ux_y = ux.dy
    uy_x = uy.dx
    uy_y = uy.dy

    # Optional GSLS memory contributions to σ_total (plan §104q.followup).
    # When provided, σ_stage = σ_elastic + Σ_l memory_l (the Method 9
    # σ_total approach extended to LW4). When None, pure elastic.
    sxx_mem = (
        sum((m for m in memory_xx_list), 0) if memory_xx_list else 0
    )
    sxy_mem = (
        sum((m for m in memory_xy_list), 0) if memory_xy_list else 0
    )
    syy_mem = (
        sum((m for m in memory_yy_list), 0) if memory_yy_list else 0
    )

    eqs = [
        Eq(
            sxx_stage,
            C11_f * ux_x + C12_f * uy_y + C16_f * (ux_y + uy_x) + sxx_mem,
            implicit_dims=t_dim,
        ),
        Eq(
            sxy_stage,
            C16_f * ux_x + C26_f * uy_y + C66_f * (ux_y + uy_x) + sxy_mem,
            implicit_dims=t_dim,
        ),
        Eq(
            syy_stage,
            C12_f * ux_x + C22_f * uy_y + C26_f * (ux_y + uy_x) + syy_mem,
            implicit_dims=t_dim,
        ),
    ]

    fx_mms = mms_source['fx'] if mms_source else 0
    fy_mms = mms_source['fy'] if mms_source else 0

    # ----- Step 3: SBP D1 boundary correction for stage σ at top 4 rows -----
    if apply_top_boundary_correction and so >= 4:
        t = grid.stepping_dim
        dux_dy_sbp = sbp_d1_boundary_exprs(
            ux, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )
        duy_dy_sbp = sbp_d1_boundary_exprs(
            uy, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )

        for q in range(4):
            y_idx = y_dim.symbolic_max - q
            dux_dy_q = dux_dy_sbp[q]
            duy_dy_q = duy_dy_sbp[q]

            # Memory contributions at boundary row q. Use explicit
            # [t, x_dim, y_idx] indexing for TimeFunctions at the
            # indexed-LHS Eq (per Method 9 pattern; Devito _eval_at
            # doesn't auto-shift TimeFunctions at indexed LHS reliably).
            # For non-TimeFunctions (sympy expressions) the same
            # expression is used as in bulk.
            def _at_row(m):
                """Index a TimeFunction-or-expression at [t, x_dim, y_idx]."""
                from devito.types import TimeFunction
                if isinstance(m, TimeFunction):
                    return m[t, x_dim, y_idx]
                return m

            sxx_q_mem = (
                sum((_at_row(m) for m in memory_xx_list), 0)
                if memory_xx_list else 0
            )
            sxy_q_mem = (
                sum((_at_row(m) for m in memory_xy_list), 0)
                if memory_xy_list else 0
            )
            syy_q_mem = (
                sum((_at_row(m) for m in memory_yy_list), 0)
                if memory_yy_list else 0
            )

            sxx_q = (
                C11_f * ux.dx + C12_f * duy_dy_q
                + C16_f * (dux_dy_q + uy.dx) + sxx_q_mem
            )
            sxy_q = (
                C16_f * ux.dx + C26_f * duy_dy_q
                + C66_f * (dux_dy_q + uy.dx) + sxy_q_mem
            )
            syy_q = (
                C12_f * ux.dx + C22_f * duy_dy_q
                + C26_f * (dux_dy_q + uy.dx) + syy_q_mem
            )

            eqs.append(Eq(
                sxx_stage[x_dim, y_idx], sxx_q,
                implicit_dims=t_dim,
            ))
            eqs.append(Eq(
                sxy_stage[x_dim, y_idx], sxy_q,
                implicit_dims=t_dim,
            ))
            eqs.append(Eq(
                syy_stage[x_dim, y_idx], syy_q,
                implicit_dims=t_dim,
            ))

    # ----- Step 4: Bulk Lx_f = L(u) + fx_mms = (1/rho) * div(σ) + fx_mms -----
    # Same divergence-of-stress structure as the displacement update,
    # but writes to Lx_f / Ly_f (Functions) instead of u.forward
    # (TimeFunction). implicit_dims=t_dim keeps it in the time loop.
    Lx_bulk = (sxx_stage.dx + sxy_stage.dy) / rho_f + fx_mms / rho_f
    Ly_bulk = (sxy_stage.dx + syy_stage.dy) / rho_f + fy_mms / rho_f

    eqs.extend([
        Eq(Lx_f, Lx_bulk, implicit_dims=t_dim),
        Eq(Ly_f, Ly_bulk, implicit_dims=t_dim),
    ])

    # ----- Step 5: SBP D1 boundary correction for Lx_f, Ly_f at top 4 rows -----
    if apply_top_boundary_correction and so >= 4:
        dsxy_dy_sbp = sbp_d1_boundary_exprs(
            sxy_stage, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )
        dsyy_dy_sbp = sbp_d1_boundary_exprs(
            syy_stage, y_dim, order=4, fs_location='top', free_dim=x_dim,
        )

        for q in range(4):
            y_idx = y_dim.symbolic_max - q
            dsxy_dy_q = dsxy_dy_sbp[q]
            dsyy_dy_q = dsyy_dy_sbp[q]

            Lx_q = (sxx_stage.dx + dsxy_dy_q) / rho_f + fx_mms / rho_f
            Ly_q = (sxy_stage.dx + dsyy_dy_q) / rho_f + fy_mms / rho_f

            eqs.append(Eq(Lx_f[x_dim, y_idx], Lx_q, implicit_dims=t_dim))
            eqs.append(Eq(Ly_f[x_dim, y_idx], Ly_q, implicit_dims=t_dim))

    return eqs
