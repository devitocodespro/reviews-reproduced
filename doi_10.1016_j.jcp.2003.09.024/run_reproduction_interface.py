"""LP04 plane-interface convergence sweep — full LW + ESIM pipeline.

SURROGATE STATUS — dual-reviewer DISAGREE 2026-05-27.
See transcription_review/lp04_close_out_codex.txt for Codex's
review of the close-out attempt.

Current state: paper-faithful at the COMPONENT level but NOT at
the integrated Table-2 level. Integrated L^1 order = 1.52 ≥ 1.5,
but L^∞ order = 0.73 — fails the LP04 Table 2 trend toward 2.0.

Codex DISAGREE diagnosis (load-bearing finding): the order-k
jump-condition matrices C_i^k / L_i^k are currently built as
block-diagonal copies of the order-0 matrices C_i^0 / L_i^0
(esim_projector.py:block_diag_C_k). LP04 §3.1 instead derives
C_i^k / L_i^k by DIFFERENTIATING C_i^0 in time + along the
interface and replacing time derivatives with the PDE — even
for a straight interface, the differentiated higher-order
conditions are NOT equivalent to imposing C_i^0 independently
on every Taylor monomial. The result: our projector achieves
slope ~1.81 instead of LP04's claimed O(dx³), and the
integrated L^∞ falls short by ~1 order.

Fix path (Codex's ranked recommendation):
  1. Implement real LP04 Eq 13-19 recursive C_i^k/L_i^k for
     k=2 (horizontal first: 6-10 h; oblique: 1-2 days)
  2. One-step local truncation diagnostic (2-4 h)
  3. Re-run projector slope expecting O(dx³) (1 h iteration)
  4. Run exact LP04 §4.2 80°-inclined, 21°-incident, Nx ∈
     {100, 200, 400, ...} geometry (6-12 h)

Honest provenance per ~/CLAUDE.md Rules 1-13: this reproduction
is `surrogate` with documented gaps — NOT `published`. The
static components (paper-tables, LW bulk, R/T at Γ, sign-fixed
projector diagnostics) are paper-faithful; the integrated
reproduction is partial pending the recursive C/L fix.

Reference for next-session implementer: the parent repository's
`scripts/tier3_esim_recursion.py` already implements the LP04
Eq 17-18 flux-Jacobian recursion (`flux_jacobians_isotropic`,
`recurse_to_order`). That code is paper-faithful and could be
ported to the reproduction folder as a standalone module.

PARTIAL-COMPLETION STATUS — 2026-05-27 (post BC + label-swap fixes):

All static components are byte-correct in isolation:
  • LW bulk: 2.000 order on homog plane wave (run_reproduction.py)
  • ESIM projector: paper-faithful per LP04 Eq 41-42 with corrected
    target-side semantics (SOLID target → fluid extension at solid
    position, FLUID target → solid extension at fluid position)
  • R/T analytical reference: textbook R_pp at normal incidence to
    fp64; energy flux conserved oblique to fp64; v_y + σ_yy
    continuity at Γ to 1.9e-14 relative (BC verification this
    session via interface-continuity probe)

The integrated convergence test still fails to reach order 2.
Error is now in the correct absolute σ-scale (~Z_p · A = 1e7) but
diverges with refinement (L^∞ order ≈ -1.4). The peak error
remains at the irregular cells.

Working diagnosis: the per-side LW pass approach (run LW once
with homog FLUID material on U_tilde_for_fluid; once with homog
SOLID material on U_tilde_for_solid; combine by side mask) is
correct in principle for 2nd-order LW (stencil radius 1) since
only distance-1-from-Γ cells need substitution. But the per-step
substitution may be amplifying a small spectral instability in
the LW operator that the static-isolation tests miss. Static
projector + R/T BCs are correct; the runtime LW × ESIM
integration coupling needs a separate diagnostic pass.

Fix path for a future session:
  1. At t=0 with analytical IC, verify U* at fluid-irregular cells
     matches analytical-solid-extended-to-fluid-position to
     O(dx³) (the LP04 truncation order claim for k=2).
  2. Run ONE LP04 step and inspect σ_yy at irregular cells: is
     the per-step error O(dx²) as expected, or does it accumulate
     to O(1) immediately?
  3. If step-1 fails: bug in projector / coordinate convention.
  4. If step-2 fails: bug in LW + U_tilde coupling — likely the
     stencil radius interaction with side mask.

----

Drives the complete reproduction of LP04 §4.2's order-2 convergence
claim (Table 2) at NORMAL INCIDENCE on a horizontal fluid-solid
interface.

Configuration (matches LP04 §4.2 Eq 48 materials, simplified to
normal incidence + horizontal interface for cleaner convergence
measurement; paper's 80°-from-horizontal + 21°-incident geometry
requires more delicate boundary handling):

  Fluid: ρ_f=1000 kg/m³, c_f=1500 m/s     (above interface)
  Solid: ρ_s=2600 kg/m³, c_p=4000 m/s, c_s=2000 m/s (below interface)
  Interface: horizontal at y = L/2
  Incident: plane wave from above, θ_inc=0 (downward)

The steady-state field is initialised with `PlaneWaveAcousticElastic`
(incident + reflected in fluid; transmitted P in solid; T_S = 0 at
normal incidence). Propagation by 1 period; compare to analytical at
t=T_end in interior window (exclude y-edges).

Outputs convergence rate to `reference_outputs/convergence_interface.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lax_wendroff import (make_material_layered, make_material_layered,
                            cfl_dt, LWMaterial)
from analytical_reference import PlaneWaveAcousticElastic
from esim_apply import (find_irregular_cells_horizontal,
                          build_projectors_for_cells, lp04_step)


def run_one_resolution(Nx: int, T_end_periods: float,
                         medium: dict,
                         interface_y: float,
                         L_x: float,
                         border_cells: int = 8) -> dict:
    """Run one convergence-sweep point.

    Initialise full steady-state at t=0, step for T_end_periods periods,
    compare to analytical in interior window.
    """
    Ny = Nx
    dx = L_x / Nx
    # Per-cell side mask (True = fluid, False = solid)
    Y = (np.arange(Ny) + 0.5) * dx
    side_mask_fluid = np.zeros((Nx, Ny), dtype=bool)
    for j in range(Ny):
        side_mask_fluid[:, j] = (Y[j] > interface_y)

    # Per-side HOMOGENEOUS materials (for LP04's target-side LW pass)
    rho_f = np.full((Nx, Ny), medium['rho_f'])
    lam_f = np.full((Nx, Ny), medium['rho_f'] * medium['c_f'] ** 2)
    mu_f = np.zeros((Nx, Ny))
    mat_fluid = LWMaterial(rho=rho_f, lam=lam_f, mu=mu_f)

    rho_s = np.full((Nx, Ny), medium['rho_s'])
    mu_s_val = medium['rho_s'] * medium['c_s'] ** 2
    lam_s_val = (medium['rho_s'] * medium['c_p'] ** 2 - 2 * mu_s_val)
    lam_s = np.full((Nx, Ny), lam_s_val)
    mu_s = np.full((Nx, Ny), mu_s_val)
    mat_solid = LWMaterial(rho=rho_s, lam=lam_s, mu=mu_s)

    # CFL based on the FASTER medium (solid c_p)
    dt = cfl_dt(mat_solid, dx, cfl=0.4)
    T_end = T_end_periods * 2 * np.pi / medium['omega']
    n_steps = max(2, int(np.ceil(T_end / dt)))
    dt = T_end / n_steps

    # Find irregular cells + build projectors
    t_setup_0 = time.time()
    cells = find_irregular_cells_horizontal(Nx, Ny, dx, interface_y)
    build_projectors_for_cells(
        cells, Nx, Ny, dx, interface_y,
        c_p_solid=medium['c_p'], c_s_solid=medium['c_s'])
    setup_s = time.time() - t_setup_0

    # Analytical reference
    pw = PlaneWaveAcousticElastic(
        A=1.0, omega=medium['omega'], theta_inc=0.0,
        c_f=medium['c_f'], rho_f=medium['rho_f'],
        c_p=medium['c_p'], c_s=medium['c_s'], rho_s=medium['rho_s'],
        y_interface=interface_y,
    )
    # Cell-centred grid
    X = (np.arange(Nx) + 0.5) * dx
    Y = (np.arange(Ny) + 0.5) * dx
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    U = pw.evaluate(XX, YY, t=0.0)

    # Time-step
    t_step_0 = time.time()
    for n in range(n_steps):
        U = lp04_step(U, cells, mat_fluid, mat_solid,
                       side_mask_fluid, dx, dt, periodic=True)
    step_s = time.time() - t_step_0

    # Analytical at T_end
    U_exact = pw.evaluate(XX, YY, t=T_end)

    # Compute error in interior window (exclude border_cells from
    # each y-edge; x is periodic so x-edges OK)
    sl = (slice(None), slice(None),
          slice(border_cells, Ny - border_cells))
    err = U[sl] - U_exact[sl]

    L_inf = float(np.max(np.abs(err)))
    L_1 = float(np.mean(np.abs(err)))

    # Diagnostic: where does L^∞ peak?
    abs_err = np.abs(err)
    flat_idx = int(np.argmax(abs_err))
    c_max, i_max, j_max_rel = np.unravel_index(flat_idx, abs_err.shape)
    j_max = j_max_rel + border_cells
    j_interface = int(round(interface_y / dx))
    dist_to_interface = abs(j_max - j_interface)
    return dict(
        Nx=Nx, dx=dx, dt=dt, n_steps=n_steps, T_end=T_end,
        n_irregular_cells=len(cells),
        L_inf=L_inf, L_1=L_1,
        setup_s=setup_s, step_s=step_s,
        border_cells=border_cells,
        L_inf_at_component=int(c_max),
        L_inf_at_ij=(int(i_max), int(j_max)),
        L_inf_dist_to_interface_cells=int(dist_to_interface),
    )


def fit_order(dx_list: list[float], err_list: list[float]) -> float:
    if len(dx_list) < 2:
        return float('nan')
    log_dx = np.log(np.asarray(dx_list))
    log_err = np.log(np.asarray(err_list))
    slope, _ = np.polyfit(log_dx, log_err, 1)
    return float(slope)


def main() -> int:
    print("=" * 70)
    print("LP04 §4.2 plane-interface convergence — LW + ESIM full pipeline")
    print("=" * 70)

    L_x = 0.1
    interface_y = 0.5 * L_x
    medium = dict(
        c_f=1500.0, rho_f=1000.0,
        c_p=4000.0, c_s=2000.0, rho_s=2600.0,
        omega=2 * np.pi * 5e4,
    )

    # Pick T_end short enough that the y-boundaries don't pollute
    # the interior. 0.4 of a period gives ~0.4·λ/c_p · c_p = 0.4·λ
    # of propagation. With λ ~ 80 mm in solid (c_p=4000, ω/2π=50kHz)
    # and L_x = 100 mm, the wave moves ~32 mm in fluid (λ_f = 30 mm).
    # Boundary perturbation reaches ~32 mm into interior → exclude
    # border_cells = 0.4 · Ny rows from each side. At Nx=60, that's
    # 24 cells = 40% — leaves only 12-cell middle band. Tight.
    # Use shorter T_end to leave more interior.
    T_end_periods = 0.05    # short propagation so y-edge effects don't penetrate
    border_cells = 18       # exclude ~T_end·c_p / dx cells from each y-edge
    print(f"Medium: fluid (ρ={medium['rho_f']}, c={medium['c_f']}) "
          f"above; solid (ρ={medium['rho_s']}, c_p={medium['c_p']}, "
          f"c_s={medium['c_s']}) below")
    print(f"Interface at y={interface_y*1000:.1f} mm; ω = {medium['omega']:.3e}")
    print(f"T_end = {T_end_periods:.2f} period; border_cells = {border_cells}")
    print()

    Nx_list = [60, 100, 150]
    results: list[dict] = []
    comp_names = ['v_x', 'v_y', 'σ_xx', 'σ_xy', 'σ_yy']
    for Nx in Nx_list:
        r = run_one_resolution(
            Nx, T_end_periods, medium, interface_y, L_x,
            border_cells=border_cells)
        print(f"  Nx={Nx:3d}  dx={r['dx']*1e6:6.1f}µm  "
              f"L^∞={r['L_inf']:.4e}  L^1={r['L_1']:.4e}  "
              f"nt={r['n_steps']:4d}  cells={r['n_irregular_cells']:3d}")
        print(f"        L^∞ peak: comp={comp_names[r['L_inf_at_component']]} "
              f"at (i={r['L_inf_at_ij'][0]}, j={r['L_inf_at_ij'][1]}); "
              f"distance to interface = {r['L_inf_dist_to_interface_cells']} cells")
        results.append(r)

    dx_arr = [r['dx'] for r in results]
    L_inf_arr = [r['L_inf'] for r in results]
    L_1_arr = [r['L_1'] for r in results]
    order_inf = fit_order(dx_arr, L_inf_arr)
    order_1 = fit_order(dx_arr, L_1_arr)

    print()
    print(f"Fitted convergence orders (log-log slope over {len(Nx_list)} grids):")
    print(f"  L^∞ order = {order_inf:.3f}")
    print(f"  L^1 order = {order_1:.3f}")
    print()
    pass_inf = order_inf >= 1.5
    pass_1 = order_1 >= 1.5
    print("Acceptance: order ≥ 1.5 in both norms")
    print(f"  L^∞ {'PASS ✓' if pass_inf else 'FAIL ✗'}: order={order_inf:.3f}")
    print(f"  L^1 {'PASS ✓' if pass_1 else 'FAIL ✗'}: order={order_1:.3f}")
    print()
    print("(LP04 Table 2 reports L^∞ orders converging to ≈ 2.0 at finer")
    print("grids. The slope at coarse grids can underestimate due to")
    print("pre-asymptotic behavior; a ≥1.5 floor is enough to demonstrate")
    print("the order-2 trend.)")

    out = {
        'config': dict(L_x=L_x, interface_y=interface_y,
                        T_end_periods=T_end_periods,
                        border_cells=border_cells,
                        medium=medium),
        'sweep': results,
        'order_L_inf': order_inf,
        'order_L_1': order_1,
        'paper_reference': 'Lombard & Piraux 2004 JCP 195 Table 2',
    }
    out_path = HERE / 'reference_outputs' / 'convergence_interface.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    return 0 if (pass_inf and pass_1) else 1


if __name__ == '__main__':
    sys.exit(main())
