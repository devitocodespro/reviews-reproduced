"""Seafloor-coupling SAT helpers for the Bader-Almquist-Dunham 2023
two-zone framework.

Plan §108.followup-SBP-SAT-graduation Phase C2.4 (2026-05-21).
Implements the weak SAT penalty terms that couple the fluid momentum
potential ``phi`` (upper domain, see :mod:`acoustic_sbp_sat`) with the
elastic displacement ``u_i`` (lower domain) across the seafloor
interface.

Reference
---------
Bader M., Almquist M., Dunham E.M. (2023). "Modeling and inversion in
acoustic-elastic coupled media using energy-stable summation-by-parts
operators." Geophysics 88(3) T137-T150.
DOI: 10.1190/geo2022-0195.1. See eqs 9-10 for the exact SAT forms.

Interface conditions (Bader 2023 eq 3, 2D)
------------------------------------------
At the fluid-solid interface (seafloor), the continuous theory imposes
3 conditions in 2D (the σ_23 condition vanishes; we drop the i=2
direction):

1. :math:`\\sigma_{13} = 0` — no shear traction
2. :math:`-\\sigma_{33} + \\dot\\phi = 0` — normal stress continuity
   (:math:`p = -\\dot\\phi` matches solid normal stress :math:`\\sigma_{33}`)
3. :math:`\\rho_f^{-1}\\,\\partial_3\\phi - \\dot u_3 = 0` — normal
   velocity continuity

Weak imposition via SAT penalties (Bader 2023 eqs 9-10, 2D)
-----------------------------------------------------------
Each condition contributes SAT terms to the discrete equations. In 2D
(i ∈ {1, 3}), there are 5 SAT contributions:

**Fluid side** (added as Inc to the phi equation at the seafloor row):

- :math:`\\mathbf A_f^{(\\text{bot})}\\phi = -\\mathbf H_3^{-1}(\\rho_f^{-1}\\mathbf S_3)\\phi`
- :math:`\\mathbf C_f\\,\\mathbf u = \\mathbf H_3^{-1}\\,\\dot u_3`

**Solid side** (added as Inc to the u_1, u_3 equations at the seafloor row):

- :math:`\\mathbf A_{s,1}^{(\\text{top})}\\mathbf u = -\\mathbf H_3^{-1}\\,\\mu\\,(-\\partial_1 u_3 + \\mathbf S_3 u_1)`
- :math:`\\mathbf A_{s,3}^{(\\text{top})}\\mathbf u = -\\mathbf H_3^{-1}\\,(-\\lambda\\,(\\partial_i u_i) + 2\\mu\\,\\mathbf S_3 u_3)`
- :math:`\\mathbf C_s\\,\\phi = -\\mathbf H_3^{-1}\\,\\dot\\phi`

where :math:`\\mathbf H_3^{-1} = 48 / (17\\,h_z)` is the inverse of
the SBP diagonal-norm weight at the boundary row (order 4, Mattsson
2004); :math:`\\mathbf S_3` is the SBP one-sided boundary derivative.

Implementation milestones
-------------------------
- **M1 (this module, draft)**: ``S_3`` placeholder uses Devito's
  built-in ``.dy`` (centred FD). Coupling structure is correct;
  boundary-row stencil is approximate. Suitable for end-to-end
  framework verification.
- **M2 (Phase C2.2)**: replace ``S_3`` with the proper SBP D1 one-sided
  boundary stencil from :func:`sbp_sat.sbp_boundary_normal_derivative`.
  Recovers O(dz²) boundary accuracy per Bader 2023.

2D-to-3D mapping
----------------
- Bader 2023's :math:`x_1, x_3` (horizontal-1, vertical) ↦ our ``x,
  y`` (Devito grid dimensions).
- Bader 2023's :math:`u_1, u_3` ↦ our ``ux, uy`` (uy is the
  depth-axis displacement).
- Bader 2023's :math:`\\partial_3` ↦ our ``.dy``.
- The i=2 direction is dropped in 2D; :math:`\\sigma_{23} = 0` is
  trivial.

Convention: in our Devito grid, the fluid occupies the TOP of the
domain (z = z_max region) and the solid occupies the BOTTOM. The
seafloor is the single row separating them. This is opposite to
Bader 2023's z-downward convention but the equations are identical
up to a sign on the normal direction.
"""
from fractions import Fraction

from deprecated.classic import deprecated
from devito import Eq, Function, Inc


def _h3_inv_factor(order=4):
    """Return the inverse-norm-weight factor (unit-less, gets divided
    by h_dim at call site).

    For order 4 (Mattsson 2004): norm weight at boundary = 17/48,
    so H_3^{-1} = 48/17 / h_dim.
    """
    from sbp_sat import sbp_boundary_norm_weight
    weight = sbp_boundary_norm_weight(order=order)
    return float(Fraction(1) / weight)


def seafloor_coupling_eqs_2d_bader_implicit(phi, uy, rho_f, kappa_f,
                                              seafloor_subdomain=None,
                                              order=4, *,
                                              seafloor_mask=None,
                                              ux=None, mu_s=None, rho_s=None):
    """Bader 2023 reference-faithful seafloor coupling via implicit 2x2 solve.

    Plan §108.followup-SBP-SAT-graduation Phase C5.SAT-stability finding
    (2026-05-22): the original ``seafloor_coupling_eqs_2d`` Inc-based SAT
    is structurally wrong. Bader 2023's actual implementation
    (``fwi2d/src/cpp/we_op.cpp`` lines 2487-2512) couples the fluid φ
    and solid u_y at the seafloor row via an IMPLICIT solve:

        h     = dz * (17/48)          # Mattsson 2004 boundary norm × dz
        val0  = dt / (2*h)             # = (24/17) * dt/dz
        val1  = K·val0·(uy_fwd - uy_bwd + (val0/ρ_f)·φ_bwd)
        val2  = 1 + (val0² · K / ρ_f)
        new_φ_sf = (φ_fwd + val1) / val2          ← IMPLICIT
        new_uy_sf = uy_fwd - (val0/ρ_f) · (new_φ_sf - φ_bwd)

    Where K = κ_f (fluid bulk modulus) and ρ_f is fluid density at
    seafloor.

    The bulk Eq's compute φ_fwd and uy_fwd as if the interface were a
    free surface (Neumann BC); the implicit step then OVERWRITES φ_fwd
    and uy_fwd at the seafloor row to enforce continuity of normal
    velocity (φ_z/ρ = u_y.dt) and normal traction (σ_yy = -φ.dt).

    This formulation is energy-stable BY CONSTRUCTION (the implicit
    solve denominator val2 > 0 for any CFL-stable (dt, dz)) and does
    NOT require alpha tuning.

    Phase A.3 / Codex Round 2 objection #4 fix (2026-05-23): when
    ``ux``, ``mu_s``, and ``rho_s`` are all provided, ALSO emit a
    σ_xy=0 SAT penalty on ``ux`` at the seafloor row. This mirrors
    Bader's reference `esat_neumann_top<expr2, expr2>` call at
    ``fwi2d/src/cpp/we_op.cpp:754`` which enforces zero
    tangential traction on the solid side via SBP D1 boundary
    stencil. Without these kwargs, the helper falls back to
    architectural σ_xy=0 enforcement (ux/uy never written in
    fluid zone, so centred-FD bulk at seafloor reads zero from
    above) — that is a valid but weaker enforcement.

    Returns
    -------
    list of devito.Eq
        - First Eq: solves for new φ at seafloor
        - Second Eq: updates u_y at seafloor using new φ
        - (Optional third Eq) σ_xy=0 SAT penalty on u_x at seafloor
          if ``ux``, ``mu_s``, ``rho_s`` provided.
        Order matters: Devito schedules sequentially.
    """
    from fractions import Fraction
    from sbp_sat import sbp_boundary_norm_weight

    if order != 4:
        raise NotImplementedError(
            f"Only order=4 supported; got order={order}.")

    z_dim = phi.grid.dimensions[-1]
    h_z = z_dim.spacing
    dt_sym = phi.grid.stepping_dim.spacing
    t_dim = phi.grid.time_dim

    # Boundary norm × dz: h = dz * d_3 = dz * 17/48 (Mattsson 2004).
    weight = sbp_boundary_norm_weight(order=order)   # = 17/48
    h_boundary = h_z * float(weight)                  # = h_z * 17/48
    val0 = dt_sym / (2 * h_boundary)                  # = (24/17) * dt/dz

    # val1 = K · val0 · (uy.forward - uy.backward + val0/ρ_f · φ.backward)
    val1 = kappa_f * val0 * (uy.forward - uy.backward
                              + val0 / rho_f * phi.backward)

    # val2 = 1 + val0² · K / ρ_f
    val2 = 1 + val0**2 * kappa_f / rho_f

    # Step 1: implicit solve for new φ at the seafloor row.
    # (Note: the bulk-Eq value of φ.forward at the seafloor enters as
    # the first additive term in the numerator.)
    new_phi = (phi.forward + val1) / val2

    # Step 2: update uy.forward at the seafloor row using the new φ.
    # We MUST schedule this AFTER the phi update so phi.forward reads
    # the just-updated value.
    new_uy = uy.forward - val0 / rho_f * (phi.forward - phi.backward)

    # Phase A.3 (2026-05-23): σ_xy=0 SAT penalty on ux at the seafloor
    # row. Bader's reference applies this via `esat_neumann_top<expr2,
    # expr2>` at `fwi2d/src/cpp/we_op.cpp:754`. The formula is:
    #
    #   ux.forward(seafloor) += dt² / rho_s · (-1/(h0·dz)) ·
    #                          μ_s · (∂_y u_y + S_3 u_x)
    #
    # where ∂_y u_y is computed via centred FD (already in the bulk) and
    # S_3 u_x is the SBP boundary derivative operator on u_x reading
    # downward into the solid. Bader uses scoef = [11/6, -3, 3/2, -1/3]
    # — a 4-tap one-sided D1 stencil at the boundary row.
    #
    # The penalty is *added* to the centred-FD bulk update at the
    # seafloor row, not an overwrite — so its effect is to drive
    # σ_xy → 0 at the boundary without disturbing the bulk solution.
    sat_xy_zero_enabled = (ux is not None and mu_s is not None
                            and rho_s is not None)
    sat_xy_eq = None
    if sat_xy_zero_enabled:
        # Mattsson 2004 boundary derivative coefficients (S_3 stencil),
        # one-sided 4-tap reading downward from the boundary row.
        scoef = (Fraction(11, 6), Fraction(-3), Fraction(3, 2),
                 Fraction(-1, 3))
        # x_dim is the free dimension; z_dim is the boundary normal.
        # The boundary "row" sits at the top of the elastic grid (=
        # seafloor row, from the elastic perspective). For the mask
        # path we use the seafloor_mask; for the subdomain path the
        # seafloor_subdomain handles row restriction.
        # S_3 u_x at boundary = (1/dz) · Σ scoef[k] · u_x[boundary - k]
        # (reading DOWNWARD; k=0 at boundary, k=1 one row deeper).
        # In Devito the boundary row index is `z_dim` symbolic; we
        # express the one-sided stencil via field shifts.
        x_dim_free = ux.grid.dimensions[0]
        # Devito symbolic shifts: ux.subs({z_dim: z_dim - k}) for the
        # k-th tap below the boundary row.
        s3_ux = sum(
            float(scoef[k]) * ux.subs({z_dim: z_dim - k})
            for k in range(4)
        ) / h_z
        # σ_xy in 2D = μ · (∂_y u_x + ∂_x u_y). The boundary-normal
        # component is ∂_y u_x (captured by s3_ux via the one-sided
        # SBP boundary stencil reading downward into the solid); the
        # tangential-derivative component is ∂_x u_y (a regular
        # interior derivative).
        #
        # Codex Round 3 fix (2026-05-23): pre-fix this code used
        # ``uy.dy`` (= ∂_y u_y) which is NOT part of σ_xy — it's the
        # diagonal stress σ_yy strain component. The correct
        # tangential term is ``uy.dx`` = ∂_x u_y.
        duy_dx = uy.dx
        # σ_xy SAT penalty: σ_xy = μ · (∂_y u_x + ∂_x u_y); the SAT
        # closure enforces σ_xy → 0 via a penalty proportional to
        # σ_xy_computed.
        sigma_xy_at_boundary = mu_s * (s3_ux + duy_dx)
        # Penalty correction to ux.forward (Bader's `esat_neumann_top`
        # formula adapted to our leapfrog displacement timestep):
        #   ux.forward += dt² · (- (1/(h0·dz))) · σ_xy / rho_s
        sat_xy_correction = (
            -(dt_sym ** 2) / (h_boundary * rho_s) * sigma_xy_at_boundary
        )

    if seafloor_mask is not None:
        # Mask-based formulation (Phase C5.dipping, Task #53, 2026-05-22).
        # Supports non-rectangular interfaces (e.g. dipping at non-zero
        # angles). At cells where seafloor_mask = 1, the implicit-solve
        # value is written; elsewhere phi.forward / uy.forward retain
        # their just-bulk-computed values.
        if seafloor_subdomain is not None:
            raise ValueError(
                "Pass either seafloor_subdomain (rectangular interface) "
                "or seafloor_mask (non-rectangular interface), not both."
            )
        out_eqs = [
            Eq(phi.forward,
                seafloor_mask * new_phi
                + (1 - seafloor_mask) * phi.forward,
                implicit_dims=t_dim),
            Eq(uy.forward,
                seafloor_mask * new_uy
                + (1 - seafloor_mask) * uy.forward,
                implicit_dims=t_dim),
        ]
        if sat_xy_zero_enabled:
            out_eqs.append(Eq(
                ux.forward,
                seafloor_mask * (ux.forward + sat_xy_correction)
                + (1 - seafloor_mask) * ux.forward,
                implicit_dims=t_dim,
            ))
        return out_eqs

    if seafloor_subdomain is None:
        raise ValueError(
            "Either seafloor_subdomain or seafloor_mask must be provided."
        )
    out_eqs = [
        Eq(phi.forward, new_phi, subdomain=seafloor_subdomain,
            implicit_dims=t_dim),
        Eq(uy.forward, new_uy, subdomain=seafloor_subdomain,
            implicit_dims=t_dim),
    ]
    if sat_xy_zero_enabled:
        out_eqs.append(Eq(
            ux.forward, ux.forward + sat_xy_correction,
            subdomain=seafloor_subdomain, implicit_dims=t_dim,
        ))
    return out_eqs


@deprecated(
    reason=(
        "Wrong-reading of Bader 2023 eqs 9-10. The formal SAT terms "
        "must be discretised as a 2x2 implicit solve at the seafloor "
        "row (the time-derivative cross-couplings C_f·u and C_s·φ "
        "create an implicit algebraic system), NOT as Inc-add SAT "
        "contributions. This Inc-add form is unconditionally unstable "
        "at production resolution. Use "
        "seafloor_coupling_eqs_2d_bader_implicit instead. Retained "
        "for pedagogical comparison only — see "
        "notes/petrobras_bulcao_figure_analysis.qmd Caveat 1 + "
        "fwi2d/src/cpp/we_op.cpp lines 2487-2512 (github.com/nmbader/fwi2d) "
        "for the correct reference implementation. Phase D close-out, "
        "2026-05-22."
    ),
    version="0.2.0",
    category=DeprecationWarning,
)
def seafloor_coupling_eqs_2d(phi, ux, uy, rho_f, lambda_s, mu_s, rho_s,
                              kappa_f, seafloor_subdomain,
                              order=4):
    """Implement Bader 2023 eqs 9-10 (2D) as SAT-corrected Inc Eq's at
    the seafloor row.

    .. deprecated:: 0.2.0
       Wrong-reading of Bader 2023. Use
       :func:`seafloor_coupling_eqs_2d_bader_implicit` instead.
       Retained for pedagogical comparison.

    Returns 5 Inc Eq's, one per SAT contribution, all bound to
    ``seafloor_subdomain`` (the single z-row at the fluid-solid
    interface). Each Inc fires AFTER the bulk Eq's update phi.forward /
    ux.forward / uy.forward, ADDING the SAT correction at the seafloor
    row only.

    Parameters
    ----------
    phi : TimeFunction (time_order=2)
        Fluid momentum potential.
    ux, uy : TimeFunction (time_order=2)
        Solid displacement components. ``uy`` is the vertical (depth)
        direction in our 2D Devito convention.
    rho_f : Function
        Fluid density at the seafloor row (read at z=seafloor).
    lambda_s, mu_s : Function
        Solid Lamé parameters at the seafloor row.
    rho_s : Function
        Solid density at the seafloor row.
    kappa_f : Function
        Fluid bulk modulus at the seafloor row.
    seafloor_subdomain : SubDomain
        The single z-row at the interface
        (:class:`free_surface.SeafloorInterface`).
    order : int
        SBP order. Currently only ``order=4`` is supported.
    use_sbp_S3 : bool, default False
        When True (Phase C2.2 milestone), uses
        :func:`sbp_sat.sbp_boundary_normal_derivative` for the
        boundary normal-derivative S_3. When False (M1 default), uses
        Devito's built-in ``.dy`` (interior centred FD) as a placeholder.
        The M1 path is faster to wire but boundary accuracy degrades
        to O(dz) near the seafloor.

    Returns
    -------
    list of devito.Inc
        5 SAT Inc Eq's targeting (phi, phi, ux, uy, uy).
    """
    h3_inv = _h3_inv_factor(order=order)
    t_dim = phi.grid.time_dim
    x_dim, z_dim = phi.grid.dimensions
    h_z = z_dim.spacing
    dt_sym = phi.grid.stepping_dim.spacing

    # M1 placeholder: Devito centred FD .dy (NEGATED for fluid-side
    # outward-normal convention in our y-up Devito frame). The
    # exploratory use_sbp_S3=True / sat_alpha kwargs were deleted in
    # Phase D close-out (2026-05-22) — neither shifted the empirical
    # stability boundary. See plan §108 Phase D + qmd Caveat 1.
    S_3_phi = -phi.dy
    S_3_ux = ux.dy
    S_3_uy = uy.dy

    eqs = []

    # ===== Fluid side =====
    # phi equation form (Bader 2023 eq 4 fluid):
    #   k_f^{-1} phi.dt2 = L_f phi + A_f phi + C_f u + g_f
    # In Devito leapfrog second-order time:
    #   phi.forward = 2*phi - phi.backward + dt^2 * kappa_f * RHS
    # The SAT contributions to RHS are A_f^bot phi and C_f u.
    # Each SAT contribution is added as Inc(phi.forward, dt^2 * kappa_f * sat).

    # A_f^bot phi = -H_3^{-1} (rho_f^{-1} S_3 phi)
    a_f_bot = -(h3_inv / h_z) / rho_f * S_3_phi
    eqs.append(Inc(phi.forward, dt_sym**2 * kappa_f * a_f_bot,
                   subdomain=seafloor_subdomain,
                   implicit_dims=t_dim))

    # C_f u = H_3^{-1} D_t u_3 (the solid normal-velocity at seafloor
    # forces the fluid). uy.dt = centred (uy.forward - uy.backward) / (2 dt).
    c_f_u = (h3_inv / h_z) * uy.dt
    eqs.append(Inc(phi.forward, dt_sym**2 * kappa_f * c_f_u,
                   subdomain=seafloor_subdomain,
                   implicit_dims=t_dim))

    # ===== Solid side =====
    # u_i equation form (Bader 2023 eq 4 solid):
    #   rho_s u_i.dt2 = L_s u_i + A_s^top u + C_s phi + g_s,i
    # In Devito: u_i.forward = 2*u_i - u_i.backward + dt^2 / rho_s * RHS

    # A_{s,1}^top u = -H_3^{-1} mu (-d_1 u_3 + S_3 u_1)
    # In 2D: d_1 u_3 = uy.dx; S_3 u_1 = (S_3 evaluated on ux) ≈ ux.dy at seafloor.
    a_s_1 = -(h3_inv / h_z) * mu_s * (-uy.dx + S_3_ux)
    eqs.append(Inc(ux.forward, dt_sym**2 / rho_s * a_s_1,
                   subdomain=seafloor_subdomain,
                   implicit_dims=t_dim))

    # A_{s,3}^top u = -H_3^{-1} (-lambda d_i u_i + 2 mu S_3 u_3)
    # In 2D: d_i u_i = ux.dx + uy.dy; S_3 u_3 ≈ uy.dy at seafloor.
    div_u = ux.dx + uy.dy
    a_s_3 = -(h3_inv / h_z) * (-lambda_s * div_u + 2 * mu_s * S_3_uy)
    eqs.append(Inc(uy.forward, dt_sym**2 / rho_s * a_s_3,
                   subdomain=seafloor_subdomain,
                   implicit_dims=t_dim))

    # C_s phi = -H_3^{-1} D_t phi.
    # phi.dt = -p (negative pressure); adding it to uy.forward via Inc
    # implements sigma_33 = dotPhi = -p ≡ pressure pushes solid down at the
    # seafloor when pressure is positive (compressive). The sign convention
    # follows Bader 2023 eq 3 (-sigma_33 + dotPhi = 0).
    c_s_phi = -(h3_inv / h_z) * phi.dt
    eqs.append(Inc(uy.forward, dt_sym**2 / rho_s * c_s_phi,
                   subdomain=seafloor_subdomain,
                   implicit_dims=t_dim))

    return eqs


# (Phase C2.3 placeholder stubs ``seafloor_coupling_strong_eqs`` and
# ``seafloor_coupling_sat_eqs`` deleted in Phase D close-out, 2026-05-22.
# Both raised NotImplementedError and were never called.)
