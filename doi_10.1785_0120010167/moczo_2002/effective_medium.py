"""Effective-medium averaging for the dipping water-TTI
interface.

Plan §85 Phase 3b — transcribes the sympy-verified formulas
from `scripts/derive_effective_medium.py` (and the derivation
memo `notes/effective_medium_derivation.md`) into a NumPy
production implementation.

Three averaging schemes:

- ``moczo2002_average(...)`` — Moczo et al. 2002 BSSA
  92:3042-3066 (DOI 10.1785/0120010167). Isotropic-effective
  averaging for an axis-aligned interface. Stepping-stone.

- ``schoenberg_muir_average(...)`` — Schoenberg-Muir calculus
  (Muir et al. 1992; Koene-Wittsten-Robertsson 2022 Eqs 23-27).
  General-orientation effective-medium averaging via
  rotate-partition-average-rotate-back. Production target for
  the dipping Petrobras interface; reduces to Kristek 2017
  axis-aligned orthorhombic in the β=0 limit.

- ``cell_volume_fractions(...)`` — per-cell upper-layer volume
  fraction f1 and interface normal direction for a dipping
  interface line, computed by sub-pixel area sampling.

All routines operate on vectorised NumPy arrays for efficiency
in per-cell evaluation across a (Nx, Ny) grid.

Plan §85n MVP scope: `interface_treatment` kwarg in
`tti_params.build_layered_cij_arrays` dispatches to either
``moczo2002_average`` or ``schoenberg_muir_average`` per cell;
default ``'sharp'`` preserves the pre-§85 per-pixel sharp
classification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── 1. Volume-fraction computation ─────────────────────────────


@dataclass(frozen=True)
class CellFractionResult:
    """Per-cell volume-fraction + interface-normal info.

    Attributes
    ----------
    f_upper : np.ndarray, shape ``(Nx, Ny)``
        Volume fraction of the upper layer in each cell. 1.0
        when fully upper; 0.0 when fully lower; intermediate
        when the interface crosses the cell.
    is_mixed : np.ndarray, shape ``(Nx, Ny)``, dtype bool
        True if the cell straddles the interface (i.e.
        ``0 < f_upper < 1``).
    normal_x, normal_z : float
        Components of the interface unit normal in lab
        (x, z) coordinates. Constant across cells for a planar
        dipping line: $\\mathbf{n} = (\\sin\\beta, \\cos\\beta)$.
    """
    f_upper: np.ndarray
    is_mixed: np.ndarray
    normal_x: float
    normal_z: float


def cell_volume_fractions(
    grid_shape: tuple[int, int],
    extent: tuple[float, float],
    dip_deg: float,
    anchor: tuple[float, float],
    sub_resolution: int = 16,
) -> CellFractionResult:
    """Per-cell upper-layer volume fraction for a dipping
    interface line.

    The interface is the line $z = z_a - \\tan(\\text{dip}) (x - x_a)$
    passing through the anchor point. The upper layer is the
    half-plane $z > z_\\text{interface}(x)$.

    Implementation: sub-pixel sampling. Each cell of size
    ``Δx × Δz`` is sub-divided into ``sub_resolution²``
    sub-cells; ``f_upper`` is the fraction of sub-cell
    centres lying above the interface line. For
    sub_resolution=16, area precision is ≈ 1/256 = 0.4%; good
    enough for SM averaging (the underlying SM error is
    O(h²); cell-internal area precision below that is
    immaterial).

    Parameters
    ----------
    grid_shape : (Nx, Ny)
    extent : (Lx, Lz) in km
    dip_deg : float, interface dip from horizontal, in degrees
    anchor : (fx, fz) in [0, 1] — **fractional** (x, z)
        coordinates of the interface anchor point (matches
        the project-wide convention used by
        `LayeredBenchmarkConfig.anchor` and
        `dipping_layer_mask`). Pre-§87 code treated this as
        absolute km, which produced the wrong-location bug
        documented in plan §87. Converted to km internally
        via ``x_a = anchor[0] * Lx``.
    sub_resolution : int, default 16 — sub-grid for area
        integration in mixed cells.

    Returns
    -------
    CellFractionResult
    """
    Nx, Ny = grid_shape
    Lx, Lz = extent
    dx = Lx / (Nx - 1) if Nx > 1 else Lx
    dz = Lz / (Ny - 1) if Ny > 1 else Lz
    # Plan §87.F1: anchor is **fractional** (x, y) in [0, 1] per the
    # project-wide convention (LayeredBenchmarkConfig.anchor doc;
    # tti_params.dipping_layer_mask lines 126-127). The pre-§87 code
    # treated anchor as absolute km, placing the mixed-cell band at
    # the wrong location and producing the §85 / §86 "no interface
    # reflection" symptom. Convert to km here.
    x_a = anchor[0] * Lx
    z_a = anchor[1] * Lz
    tan_dip = float(np.tan(np.deg2rad(dip_deg)))

    # Cell centres
    x_centres = np.arange(Nx) * dx           # (Nx,)
    z_centres = np.arange(Ny) * dz           # (Ny,)

    # For each cell, compute the interface z-position at the cell's x
    # range. For axis-aligned interface (dip=0), z_iface is constant.
    # Sub-sample over cell area.
    # Sub-cell offsets in (-0.5, 0.5) × Δ:
    half = 0.5 - 1.0 / (2 * sub_resolution)
    offsets = np.linspace(-half, half, sub_resolution)
    dx_offs = offsets * dx                     # (sub,)
    dz_offs = offsets * dz                     # (sub,)

    # Build (Nx, sub) array of sub-x positions
    X_sub = x_centres[:, None] + dx_offs[None, :]  # (Nx, sub)
    Z_sub = z_centres[:, None] + dz_offs[None, :]  # (Ny, sub)

    # Interface z at each sub-x (independent of sub-z):
    # z_iface(x) = z_a - tan(dip) * (x - x_a)
    z_iface_at_sub_x = z_a - tan_dip * (X_sub - x_a)  # (Nx, sub)

    # For each (cell_x, cell_y, sub_x, sub_y) → is sub-cell upper?
    # upper if z_sub > z_iface_at_x
    # Broadcasting: shape (Nx, Ny, sub_x, sub_y)
    # Z_sub[None, :, None, :] (Ny, sub_y) → (1, Ny, 1, sub_y)
    # z_iface_at_sub_x[:, None, :, None] → (Nx, 1, sub_x, 1)
    is_upper_sub = (
        Z_sub[None, :, None, :] > z_iface_at_sub_x[:, None, :, None]
    )   # shape (Nx, Ny, sub_x, sub_y)

    # Average over sub-cells in each direction
    f_upper = is_upper_sub.mean(axis=(2, 3))   # (Nx, Ny)
    is_mixed = (f_upper > 0.0) & (f_upper < 1.0)

    # Interface normal in lab frame: line is z = z_a - tan(β)(x-x_a),
    # so the line direction is (1, -tan(β)); the upward normal
    # is (sin β, cos β).
    normal_x = float(np.sin(np.deg2rad(dip_deg)))
    normal_z = float(np.cos(np.deg2rad(dip_deg)))

    return CellFractionResult(
        f_upper=f_upper,
        is_mixed=is_mixed,
        normal_x=normal_x,
        normal_z=normal_z,
    )


# ─── 2. Moczo 2002 — isotropic effective at axis-aligned ────────


def moczo2002_average(
    lambda1: float, mu1: float, rho1: float,
    lambda2: float, mu2: float, rho2: float,
    f1: float,
) -> dict:
    """Moczo et al. 2002 isotropic-effective averaging for two
    isotropic layers with volume fractions ``f1, f2 = 1-f1``.

    Produces an isotropic effective medium (rotationally invariant
    in 2D). For an axis-aligned interface this is exact to
    O(h²); for dipping interfaces it underestimates the
    anisotropic content the Schoenberg-Muir form captures.

    All inputs can be scalars or NumPy arrays of the same shape.

    Returns a dict with keys ``rho, c11, c22, c12, c66, c16, c26``
    matching the `build_layered_cij_arrays` 2D in-plane Voigt
    convention.

    See `notes/effective_medium_derivation.md` §4.
    """
    f2 = 1.0 - f1
    rho_eff = f1 * rho1 + f2 * rho2

    K1 = lambda1 + mu1
    K2 = lambda2 + mu2

    # Harmonic averages
    inv_K_avg = f1 / K1 + f2 / K2
    K_eff = 1.0 / inv_K_avg
    # μ harmonic: regularise the fluid limit
    mu_eps = 1e-30
    inv_mu_avg = f1 / np.maximum(mu1, mu_eps) + f2 / np.maximum(mu2, mu_eps)
    mu_eff = 1.0 / inv_mu_avg
    # If both μ are zero, mu_eff via the regulariser will be huge
    # — but the convention here is that 1/(1/0 + 1/0) = 0 (both
    # contribute infinite compliance ⇒ zero stiffness). Correct by
    # checking the mask:
    both_fluid = (mu1 <= 0) & (mu2 <= 0)
    mu_eff = np.where(both_fluid, 0.0, mu_eff)

    c11 = K_eff + mu_eff
    c22 = K_eff + mu_eff
    c12 = K_eff - mu_eff
    c66 = mu_eff
    zero = np.zeros_like(c11) if hasattr(c11, 'shape') else 0.0
    return {
        'rho': rho_eff,
        'c11': c11, 'c22': c22, 'c12': c12,
        'c66': c66, 'c16': zero, 'c26': zero,
    }


# ─── 3. Schoenberg-Muir 2D effective at general orientation ─────


def _bond_rotation_2d(theta_rad):
    """2D in-plane Bond rotation matrix R(θ) for Voigt stress
    transformation in (xx, zz, xz) basis. C' = R · C · R^T.
    Carcione 2014 Eq. 1.59. Returns 3×3 NumPy matrix.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c * c, s * s, 2 * c * s],
        [s * s, c * c, -2 * c * s],
        [-c * s, c * s, c * c - s * s],
    ])


def _stiffness_from_thomsen_2d(lambda_, mu_):
    """2D in-plane isotropic stiffness 3×3 in Voigt (xx, zz, xz).

    Helper retained for callers that have isotropic Lamé scalars
    and want an isotropic 3×3. **Not** used by the SM averager
    itself any more (plan §86.G1 fix); the SM averager takes
    full Voigt input directly to preserve TTI anisotropy across
    mixed cells.
    """
    return np.array([
        [lambda_ + 2 * mu_, lambda_, 0.0],
        [lambda_, lambda_ + 2 * mu_, 0.0],
        [0.0, 0.0, mu_],
    ])


def _voigt_3x3_from_cij(cij: dict) -> np.ndarray:
    """Assemble in-plane 2D Voigt 3×3 stiffness from a dict
    ``{c11, c22, c12, c16, c26, c66}``. Voigt index ordering
    is (xx, zz, xz). Symmetric by construction.

    Plan §86.G1: this replaces the prior `_stiffness_from_thomsen_2d`
    path inside the SM averager so the full anisotropic tensor
    of each layer is preserved through the
    rotate-partition-average-rotate-back pipeline.
    """
    c11 = float(cij['c11'])
    c22 = float(cij['c22'])
    c12 = float(cij['c12'])
    c16 = float(cij['c16'])
    c26 = float(cij['c26'])
    c66 = float(cij['c66'])
    return np.array([
        [c11, c12, c16],
        [c12, c22, c26],
        [c16, c26, c66],
    ])


def schoenberg_muir_average_scalar(
    cij1: dict, rho1: float,
    cij2: dict, rho2: float,
    f1: float,
    interface_normal_angle_rad: float,
    fluid_regulariser: float = 1e-10,
) -> dict:
    """Schoenberg-Muir effective-medium averaging at general
    interface orientation, for two layers given as **full 2D
    in-plane Voigt stiffness tensors** (six components each
    plus density).

    Plan §86.G1 / §86.G2 fix: the prior signature took scalar
    Lamé parameters and built an isotropic 3×3 stiffness inside
    the function, silently discarding the lower-layer TTI
    anisotropy (c12, c16, c22, c26). That bug produced flat-
    across-dips hf_ratio in §85 MVP. This signature accepts the
    full Voigt 6-tuple per layer so SM operates on the actual
    anisotropic tensors.

    Returns 2D in-plane Voigt-form dict with keys:
    rho, c11, c22, c12, c66, c16, c26 (the full anisotropic
    six-component form).

    Algorithm (Koene-Wittsten-Robertsson 2022 Eqs. 23-27;
    see `notes/effective_medium_derivation.md` §5):

    1. Rotate each layer's stiffness to interface-aligned frame
       by ``-θ_n``.
    2. Partition into T (tangential, index 0) and N (normal,
       indices 1-2) blocks.
    3. Compute compliance partition S_NN = C_NN⁻¹, S_TN.
    4. Arithmetic-average S_NN and S_TN over volume fractions.
    5. Recover effective stiffness from averaged compliance.
    6. Rotate back to lab frame by ``+θ_n``.

    Fluid-degenerate handling (plan §86.G3): regularise
    c66_i → max(c66_i, ε × max(|c11_i|, |c22_i|)) so the
    bond-rotated C_NN block remains invertible. For water
    (c66 = 0) this floors c66 to ε·κ ≈ 2.25e-10 in the
    Petrobras config — negligible amplitude but keeps the
    matrix inverse well-defined. For solid layers the floor
    is a no-op (their c66 already exceeds ε × c11).

    Parameters
    ----------
    cij1, cij2 : dict
        Per-layer 2D Voigt Cij dicts each with keys
        ``c11, c22, c12, c16, c26, c66``. Layer 1 is the
        upper layer.
    rho1, rho2 : float
        Densities of layers 1 and 2.
    f1 : float in [0, 1]
        Volume fraction of layer 1.
    interface_normal_angle_rad : float
        Angle of the interface normal from the lab +z axis, in
        radians. For a horizontal interface, angle = 0. For
        an interface dipping at β degrees, the normal is at
        angle β from +z.
    fluid_regulariser : float, default 1e-10
        Used to floor c66 against max(|c11|, |c22|). Has
        negligible effect on the result for typical Petrobras
        parameters; keeps the matrix inverse well-conditioned
        when one layer is a fluid (c66 = 0).
    """
    # Regularise c66 per layer so the rotated C_NN block stays
    # invertible when one layer is a fluid (c66 = 0). Floor is
    # ε × max(|c11|, |c22|). For solid layers this is a no-op.
    cij1_reg = dict(cij1)
    cij2_reg = dict(cij2)
    scale1 = max(abs(float(cij1['c11'])), abs(float(cij1['c22'])))
    scale2 = max(abs(float(cij2['c11'])), abs(float(cij2['c22'])))
    cij1_reg['c66'] = max(float(cij1['c66']), fluid_regulariser * scale1)
    cij2_reg['c66'] = max(float(cij2['c66']), fluid_regulariser * scale2)

    # Build full 3×3 Voigt stiffness for each layer (lab frame).
    # Now uses the full Cij directly — no isotropic surrogate.
    C1 = _voigt_3x3_from_cij(cij1_reg)
    C2 = _voigt_3x3_from_cij(cij2_reg)

    # Rotate to interface-aligned frame: lab → normal frame
    R_lab2norm = _bond_rotation_2d(-interface_normal_angle_rad)
    R_norm2lab = _bond_rotation_2d(+interface_normal_angle_rad)

    C1_rot = R_lab2norm @ C1 @ R_lab2norm.T
    C2_rot = R_lab2norm @ C2 @ R_lab2norm.T

    # Partition (T = index 0; N = indices 1, 2)
    def partition(C):
        return C[0:1, 0:1], C[0:1, 1:3], C[1:3, 1:3]

    C1_TT, C1_TN, C1_NN = partition(C1_rot)
    C2_TT, C2_TN, C2_NN = partition(C2_rot)

    # Compliance partition
    S1_NN = np.linalg.inv(C1_NN)
    S2_NN = np.linalg.inv(C2_NN)
    S1_TN = -S1_NN @ C1_TN.T
    S2_TN = -S2_NN @ C2_TN.T

    # Volume-weighted arithmetic averaging in compliance domain
    f2 = 1.0 - f1
    S_NN_bar = f1 * S1_NN + f2 * S2_NN
    S_TN_bar = f1 * S1_TN + f2 * S2_TN

    # Effective stiffness blocks
    C_NN_bar = np.linalg.inv(S_NN_bar)
    C_TN_bar = -(S_TN_bar.T @ C_NN_bar)   # 1×2

    # Effective TT
    inner1 = C1_TT - C1_TN @ S1_NN @ C1_TN.T
    inner2 = C2_TT - C2_TN @ S2_NN @ C2_TN.T
    C_TT_bar = (
        f1 * inner1 + f2 * inner2
        + C_TN_bar @ S_NN_bar @ C_TN_bar.T
    )

    # Assemble in interface-normal frame
    C_eff_rot = np.zeros((3, 3))
    C_eff_rot[0:1, 0:1] = C_TT_bar
    C_eff_rot[0:1, 1:3] = C_TN_bar
    C_eff_rot[1:3, 0:1] = C_TN_bar.T
    C_eff_rot[1:3, 1:3] = C_NN_bar

    # Rotate back to lab frame
    C_eff = R_norm2lab @ C_eff_rot @ R_norm2lab.T

    rho_eff = f1 * rho1 + f2 * rho2

    return {
        'rho': float(rho_eff),
        'c11': float(C_eff[0, 0]),
        'c22': float(C_eff[1, 1]),
        'c12': float(C_eff[0, 1]),
        'c66': float(C_eff[2, 2]),
        'c16': float(C_eff[0, 2]),
        'c26': float(C_eff[1, 2]),
    }


def schoenberg_muir_average_grid(
    f1_grid: np.ndarray,
    interface_normal_angle_rad: float,
    bulk_arrs_layer1: dict,
    bulk_arrs_layer2: dict,
    fluid_regulariser: float = 1e-10,
) -> dict:
    """Vectorised SM averaging over a (Nx, Ny) grid.

    Where ``f1_grid == 1`` (fully layer-1 cells): return layer-1
    bulk Cij. Where ``f1_grid == 0``: layer-2 bulk Cij. For mixed
    cells, call the scalar SM averaging routine per cell.

    Parameters
    ----------
    f1_grid : (Nx, Ny) ndarray of volume fractions.
    interface_normal_angle_rad : float (constant across grid
        for a planar interface).
    bulk_arrs_layer1, bulk_arrs_layer2 : dict
        Per-cell Cij + rho arrays for each layer's bulk
        properties. Layer 1 is the upper layer.
    fluid_regulariser : float, default 1e-10.

    Returns
    -------
    dict with keys ``rho, c11, c22, c12, c66, c16, c26`` —
    each a (Nx, Ny) ndarray of effective Cij.
    """
    Nx, Ny = f1_grid.shape

    # Initialise output arrays. For fully-layer-1 and
    # fully-layer-2 cells, the result is just the bulk.
    keys = ['rho', 'c11', 'c22', 'c12', 'c66', 'c16', 'c26']
    out = {}
    for k in keys:
        if k in bulk_arrs_layer1 and k in bulk_arrs_layer2:
            out[k] = np.where(
                f1_grid >= 0.999,
                bulk_arrs_layer1[k],
                np.where(
                    f1_grid <= 0.001,
                    bulk_arrs_layer2[k],
                    np.zeros_like(f1_grid),
                ),
            )
        else:
            out[k] = np.zeros_like(f1_grid)

    # Mixed cells: SM averaging per cell
    mixed = (f1_grid > 0.001) & (f1_grid < 0.999)
    ix, iy = np.where(mixed)
    if len(ix) == 0:
        return out

    # For each mixed cell, extract per-cell **full Voigt Cij**
    # for each layer (plan §86.G2 fix — was Lamé-only, dropping
    # c12/c16/c22/c26 silently). Pass the full Cij dict to
    # schoenberg_muir_average_scalar so TTI tilt is preserved
    # across mixed cells.
    cij_keys = ('c11', 'c22', 'c12', 'c16', 'c26', 'c66')
    for k in range(len(ix)):
        i, j = ix[k], iy[k]
        f1v = float(f1_grid[i, j])

        cij_1 = {key: float(bulk_arrs_layer1[key][i, j])
                  for key in cij_keys
                  if key in bulk_arrs_layer1}
        # Default-fill missing components (shouldn't happen in
        # production but defensive).
        for key in cij_keys:
            cij_1.setdefault(key, 0.0)
        rho_1 = float(bulk_arrs_layer1['rho'][i, j])

        cij_2 = {key: float(bulk_arrs_layer2[key][i, j])
                  for key in cij_keys
                  if key in bulk_arrs_layer2}
        for key in cij_keys:
            cij_2.setdefault(key, 0.0)
        rho_2 = float(bulk_arrs_layer2['rho'][i, j])

        eff = schoenberg_muir_average_scalar(
            cij_1, rho_1,
            cij_2, rho_2,
            f1v, interface_normal_angle_rad,
            fluid_regulariser=fluid_regulariser,
        )
        for key in keys:
            out[key][i, j] = eff[key]

    return out


__all__ = [
    'CellFractionResult',
    'cell_volume_fractions',
    'moczo2002_average',
    'schoenberg_muir_average_scalar',
    'schoenberg_muir_average_grid',
]
