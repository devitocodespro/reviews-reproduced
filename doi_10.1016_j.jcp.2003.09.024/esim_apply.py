"""LP04 ESIM substitution layer for the Lax-Wendroff bulk solver.

Implements the modified-value Ũ field per LP04 Eq 43-44: at each
irregular cell M, scatter the projector output U*_M into U_tilde
at M's position. The LW step then reads U_tilde uniformly — neighbour
reads automatically pick up U* at cross-Γ targets, the per-leg
substitution rule of Eq 43.

Conventions match `lax_wendroff.py`:
  • Single 5-component field everywhere: (v_x, v_y, σ_xx, σ_xy, σ_yy)
  • Fluid as degenerate-solid surrogate (μ=0 → σ_xx = σ_yy = -p, σ_xy = 0)
  • Cell-centred grid: y_j = (j + 0.5) · dx

The projector itself (`esim_projector.build_projector`) is paper-faithful
(3-fluid + 5-solid). Per-cell N matrices are built ONCE during setup;
the per-timestep cost is just the matvec U*_M = N · (U^n)_B and the
scatter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import esim_projector as ep


@dataclass
class IrregularCellData:
    """One irregular cell's precomputed ESIM data."""
    target_ij: tuple[int, int]    # grid indices
    target_side: str              # 'fluid' or 'solid'
    N: np.ndarray                 # (5, sum_B_dims) projector row matrix
    B1_ij: list[tuple[int, int]]  # disk-B cells on fluid side
    B2_ij: list[tuple[int, int]]  # disk-B cells on solid side
    cond_M: float


def find_irregular_cells_horizontal(Nx: int, Ny: int, dx: float,
                                       y_interface: float
                                       ) -> list[IrregularCellData]:
    """Identify irregular cells for a horizontal interface at y_interface.

    For a 2nd-order LW stencil (radius 1 in x and y), irregular cells
    are those whose y-neighbours cross Γ — i.e., one row above and
    one row below the interface.

    Returns a list of (target_ij, target_side) — projector data filled
    by `build_projectors_for_cells`.
    """
    y_cells = (np.arange(Ny) + 0.5) * dx
    cells: list[IrregularCellData] = []
    for j in range(Ny):
        y = y_cells[j]
        # Is this cell on the fluid side but its y-1 neighbour crosses Γ?
        if y > y_interface and (y - dx) < y_interface:
            # Fluid-side irregular: all x columns
            for i in range(Nx):
                cells.append(IrregularCellData(
                    target_ij=(i, j), target_side='fluid',
                    N=None, B1_ij=[], B2_ij=[], cond_M=float('nan')))
        elif y < y_interface and (y + dx) > y_interface:
            # Solid-side irregular
            for i in range(Nx):
                cells.append(IrregularCellData(
                    target_ij=(i, j), target_side='solid',
                    N=None, B1_ij=[], B2_ij=[], cond_M=float('nan')))
    return cells


def build_projectors_for_cells(cells: list[IrregularCellData],
                                  Nx: int, Ny: int, dx: float,
                                  y_interface: float,
                                  c_p_solid: float, c_s_solid: float,
                                  q_factor: float = 3.5,
                                  rho_fluid: float = 1.0,
                                  c_fluid: float = 1.5,
                                  rho_solid: float = 2.6,
                                  ) -> None:
    """Populate each cell's `N` matrix + disk-B index lists in-place.

    For a horizontal interface, the foot-of-perp P_M for cell M at
    position (x_i, y_j) is (x_i, y_interface) — directly above/below
    M. The disk B is centred on P_M with radius q_factor · dx. Cell
    membership in B_1 (fluid) or B_2 (solid) is determined by y-sign
    relative to y_interface.
    """
    tangent = (1.0, 0.0)
    q = q_factor * dx
    n_search = int(np.ceil(q_factor))
    for cell in cells:
        i_t, j_t = cell.target_ij
        x_t = (i_t + 0.5) * dx
        y_t = (j_t + 0.5) * dx
        P_xy = (x_t, y_interface)
        # Build disk B around P_xy
        B1_phys: list[tuple[float, float]] = []
        B2_phys: list[tuple[float, float]] = []
        B1_ij: list[tuple[int, int]] = []
        B2_ij: list[tuple[int, int]] = []
        # Search integer offsets around the cell whose center is closest to P_xy
        i_P = int(np.floor(x_t / dx))   # i is same column as target
        j_P_below = int(np.floor(y_interface / dx))
        j_P_above = j_P_below + 1
        # candidate cells: those with x_index near i_P, y near interface row
        for di in range(-n_search, n_search + 1):
            for dj in range(-n_search - 1, n_search + 2):
                i = i_P + di
                j = j_P_below + dj
                if not (0 <= i < Nx and 0 <= j < Ny):
                    continue
                x_c = (i + 0.5) * dx
                y_c = (j + 0.5) * dx
                # distance from P to cell centre
                r2 = (x_c - P_xy[0]) ** 2 + (y_c - P_xy[1]) ** 2
                if r2 > q ** 2:
                    continue
                if y_c > y_interface:
                    B1_phys.append((x_c, y_c))
                    B1_ij.append((i, j))
                elif y_c < y_interface:
                    B2_phys.append((x_c, y_c))
                    B2_ij.append((i, j))
        cell.B1_ij = B1_ij
        cell.B2_ij = B2_ij
        # Build the projector
        N, info = ep.build_projector(
            target_side=cell.target_side,
            target_xy=(x_t, y_t),
            P_xy=P_xy,
            tangent=tangent,
            B1_coords=B1_phys, B2_coords=B2_phys,
            k=2, c_p_solid=c_p_solid, c_s_solid=c_s_solid,
            rho_fluid=rho_fluid, c_fluid=c_fluid,
            rho_solid=rho_solid,
        )
        cell.N = N
        cell.cond_M = info.cond_M


def compute_U_tilde(U_5comp: np.ndarray,
                      cells: list[IrregularCellData],
                      ) -> np.ndarray:
    """Compute U_tilde = U^n with U*_M scattered at irregular cells.

    For each irregular cell M, gathers (U^n)_B from the disk-B cells:
      - B_1 (fluid) cells contribute (v_x, v_y, p) = (U[0], U[1], -U[2])
        where p = -σ_xx (=-σ_yy too, since fluid has σ_xx=σ_yy=-p).
      - B_2 (solid) cells contribute all 5 components.
    Then U*_M = N_M @ U_B (shape (5,)), scattered into U_tilde at M.

    Returns (5, Nx, Ny) U_tilde array.
    """
    U_tilde = U_5comp.copy()
    for cell in cells:
        i, j = cell.target_ij
        # Gather (U^n)_B
        U_B_parts: list[np.ndarray] = []
        for (i_b, j_b) in cell.B1_ij:
            v_x = U_5comp[0, i_b, j_b]
            v_y = U_5comp[1, i_b, j_b]
            # LP04 sign convention (2026-05-27): paper_tables.C1_zero
            # uses extensional-positive "p"; in physical compressive
            # convention p_LP04 = σ_xx_phys (fluid) = -p_physical.
            # So pass σ_xx (= U_5comp[2]) directly, no sign flip.
            p_LP04 = U_5comp[2, i_b, j_b]
            U_B_parts.append(np.array([v_x, v_y, p_LP04]))
        for (i_b, j_b) in cell.B2_ij:
            U_B_parts.append(U_5comp[:, i_b, j_b].copy())
        U_B_vec = np.concatenate(U_B_parts)
        U_star = cell.N @ U_B_vec
        # Per LP04 Eq 41-42 (swap-fix 2026-05-27):
        # SOLID target → U_star is 3-comp FLUID extension at solid pos
        #   in LP04 convention; embed into 5-comp degenerate-solid layout
        #   via σ_xx = σ_yy = p_LP04 = -p_physical (compressive σ).
        # FLUID target → U_star is 5-comp SOLID extension at fluid pos
        #   in physical convention; use directly.
        if cell.target_side == 'solid':
            v_x, v_y, p_LP04 = (float(U_star[0]), float(U_star[1]),
                                  float(U_star[2]))
            U_tilde[0, i, j] = v_x
            U_tilde[1, i, j] = v_y
            U_tilde[2, i, j] = p_LP04
            U_tilde[3, i, j] = 0.0
            U_tilde[4, i, j] = p_LP04
        else:
            U_tilde[:, i, j] = U_star
    return U_tilde


def lp04_step(U_5comp: np.ndarray,
                cells: list[IrregularCellData],
                mat_fluid_uniform,    # homog fluid material across grid
                mat_solid_uniform,    # homog solid material across grid
                side_mask_fluid: np.ndarray,    # (Nx, Ny) bool
                dx: float, dt: float,
                periodic: bool = True,
                ) -> np.ndarray:
    """One LP04 timestep — per-side stencil with substitution per Eq 43-44.

    Implementation:
      1. Build U_tilde_for_fluid_targets: U_n with U*_at_solid_irregular_cells
         (which IS the extrapolation of fluid → solid positions, since
         the solid-target N extends "opposite side = fluid" to solid).
      2. Build U_tilde_for_solid_targets: U_n with U*_at_fluid_irregular_cells.
      3. Run LW twice: once with homog FLUID material on U_tilde_fluid,
         once with homog SOLID material on U_tilde_solid.
      4. Combine: U^{n+1}[i,j] = V_fluid[i,j] on fluid side, V_solid[i,j]
         on solid side.

    Why per-side passes:
      LP04 Eq 43-44 substitutes Ũ at NEIGHBOUR positions based on the
      target's side. The discrete operator H at target M uses M's own
      material (LP04 §3.1: H is the bulk operator on M's homogeneous
      side, with substituted Ũ filling cross-Γ neighbour reads).
    """
    from lax_wendroff import lw_step
    # Build U* values at every irregular cell (5-component)
    U_star_per_cell: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    for cell in cells:
        i, j = cell.target_ij
        U_B_parts: list[np.ndarray] = []
        for (i_b, j_b) in cell.B1_ij:
            v_x = U_5comp[0, i_b, j_b]
            v_y = U_5comp[1, i_b, j_b]
            # LP04 sign convention (2026-05-27): pass σ_xx (= U_5comp[2])
            # directly as LP04 "p" (extensional-positive = -p_physical).
            p_LP04 = U_5comp[2, i_b, j_b]
            U_B_parts.append(np.array([v_x, v_y, p_LP04]))
        for (i_b, j_b) in cell.B2_ij:
            U_B_parts.append(U_5comp[:, i_b, j_b].copy())
        U_B_vec = np.concatenate(U_B_parts)
        U_star = cell.N @ U_B_vec
        # Per LP04 Eq 41-42 (swap-fix + sign-fix 2026-05-27):
        # SOLID target → U_star is 3-comp FLUID extension in LP04
        #   convention; embed via σ_xx = σ_yy = p_LP04 (no negation;
        #   p_LP04 = -p_physical = σ_xx_physical in compressive
        #   convention).
        # FLUID target → U_star is 5-comp SOLID extension in physical
        #   convention; use directly.
        if cell.target_side == 'solid':
            v_x, v_y, p_LP04 = (float(U_star[0]), float(U_star[1]),
                                  float(U_star[2]))
            U_5 = np.array([v_x, v_y, p_LP04, 0.0, p_LP04])
        else:
            U_5 = U_star
        U_star_per_cell[(i, j)] = (cell.target_side, U_5)

    # Build U_tilde fields for the two passes
    U_tilde_for_fluid = U_5comp.copy()
    U_tilde_for_solid = U_5comp.copy()
    for (i, j), (side, U_5) in U_star_per_cell.items():
        if side == 'solid':
            # solid-side irregular cell: U* = extrapolation of FLUID onto
            # that solid position. Used by FLUID targets reading across Γ.
            U_tilde_for_fluid[:, i, j] = U_5
        else:
            # fluid-side irregular cell: U* = extrapolation of SOLID onto
            # that fluid position. Used by SOLID targets reading across Γ.
            U_tilde_for_solid[:, i, j] = U_5

    # Run LW twice — once with homog FLUID material on U_tilde_for_fluid,
    # once with homog SOLID material on U_tilde_for_solid. This gives
    # each side its proper target-side stencil per LP04 §3.1.
    V_fluid = lw_step(U_tilde_for_fluid, mat_fluid_uniform, dx, dt,
                       periodic=periodic)
    V_solid = lw_step(U_tilde_for_solid, mat_solid_uniform, dx, dt,
                       periodic=periodic)

    # Combine: per-cell side selection
    U_next = np.where(side_mask_fluid[None, :, :], V_fluid, V_solid)
    return U_next


if __name__ == '__main__':
    # Smoke: build projectors for a small horizontal-interface config
    from lax_wendroff import LWMaterial, make_material_layered
    Nx, Ny = 60, 60
    dx = 1.0e-3   # 1 mm
    y_int = 0.5 * Ny * dx
    cells = find_irregular_cells_horizontal(Nx, Ny, dx, y_int)
    print(f"Found {len(cells)} irregular cells")
    fluid_cells = [c for c in cells if c.target_side == 'fluid']
    solid_cells = [c for c in cells if c.target_side == 'solid']
    print(f"  fluid-side: {len(fluid_cells)}, solid-side: {len(solid_cells)}")
    build_projectors_for_cells(cells, Nx, Ny, dx, y_int,
                                  c_p_solid=4000.0, c_s_solid=2000.0)
    cond_max = max(c.cond_M for c in cells)
    cond_med = np.median([c.cond_M for c in cells])
    print(f"  cond(M): median={cond_med:.3e}  max={cond_max:.3e}")
    # Apply ESIM to a uniform field
    U = np.zeros((5, Nx, Ny))
    U[0] = 1.0; U[1] = 2.0; U[2] = -3.0; U[3] = 0.0; U[4] = -3.0
    # Note: per LP04 sign convention, p = -σ_xx in fluid layout =
    # +3, but BC2 enforces p_fluid = σ_yy_solid, so for jump
    # consistency on homog the σ_yy should be +3 (not -3). Let's set
    # solid σ_yy properly.
    Y = (np.arange(Ny) + 0.5) * dx
    in_solid = Y < y_int
    for j in range(Ny):
        if in_solid[j]:
            U[4, :, j] = 3.0   # σ_yy_solid = p_fluid (jump-consistent)
    U_tilde = compute_U_tilde(U, cells)
    diff = U_tilde - U
    print(f"  max|U_tilde - U| (uniform input): {np.max(np.abs(diff)):.3e}")
    print("  (should be ~fp64 noise — projector reproduces uniform input)")
