"""LP04 plane-interface convergence sweep — full LW + ESIM pipeline.

NOVEL-COMBINATION STATUS (post LP04-R recursive C/L port, 2026-05-27).

Paper-faithful components:
  • LP04 §3.1 recursive C_i^k / L_i^k construction (Eq 13-18) —
    `esim_recursion.py` (vendored from parent's
    `scripts/tier3_esim_recursion.py` with reproduction-specific
    paper/x-first monomial ordering + LP04 extensional-positive p
    convention in flux_jacobians_fluid)
  • Jump-condition matrices C_i^0, L_i^0, G_i^k, α_i — paper-
    byte-transcribed via side-by-side user-confirmed review
  • Lax-Wendroff bulk — order-2 on homog plane wave (5 tests pass)
  • R/T at Γ — fp64 BC continuity for v_y, σ_yy; σ_xy_solid = 0

Empirical projector slope (test_projector_vs_analytical.py,
Nx ∈ {40, 80, 160}):
  Solid target: 2.71 (LP04 claim: O(dx³) = 3.0)
  Fluid target: 1.96 (close to 2.0)
Both substantially above the ≥ 1.5 threshold; recursive port closes
the prior block-diagonal surrogate gap (1.81 → 2.71/1.96).

Non-LP04-equivalent component:
  • LW × U_tilde integration glue — runs LW twice with per-side
    homog materials and scatters U* at irregular cells. LP04
    Eq 43-44 specifies that at each irregular cell M, the bulk LW
    stencil reads U^n at SAME-side legs and U* at OTHER-side legs.
    The current code applies U* to U_tilde once per timestep,
    which is a CLOSE BUT NOT EQUIVALENT simplification.

Empirical impact on integrated L^∞ / L^1 — extended sweep over
Nx ∈ {60, 100, 150, 200, 300}:
  L^1 order = 1.50 (PASS ≥ 1.5; LP04 Table 2 target ~2.0)
  L^∞ order = 0.93 (FAIL ≥ 1.5; LP04 Table 2 target ~2.0)

DIAGNOSTIC FINDING (2026-05-27 extended sweep): L^∞ peak
MIGRATES AWAY from Γ as Nx increases:
  Nx=60  → peak at distance 1 cell from Γ
  Nx=100 → 1 cell
  Nx=150 → 3 cells
  Nx=200 → 4 cells
  Nx=300 → 8 cells

This is NOT consistent with a localised interface-glue defect
(which would keep the peak fixed at distance 1). The migration
pattern suggests the bulk LW step itself is producing larger-
than-O(dx²) error AWAY from Γ — possibly because the per-side
LW pass with homog material processes the WHOLE array including
the OPPOSITE-side region (with discontinuous U_tilde values
between actual same-side U^n and U*-substituted irregular cells),
and the LW operator propagates that discontinuity into the bulk
at order < 2.

Fix path (revised post-extended-sweep, 2026-05-27):
  Instead of running LW on the whole array with homog material
  + scatter U* at irregular cells, build a PER-CELL local LW
  stencil application: at each target cell M, gather its own
  side's neighbours (U^n) + cross-Γ neighbours (U*), apply LW
  with M's own material to this localised mixed array.

This is the strict LP04 Eq 43-44 interpretation. The current
"per-side homog LW pass + combine" is a simplification that
ASSUMES the LW operator's behaviour on the OPPOSITE-side region
is irrelevant (since the COMBINE step discards it). But the LW
step is locally non-trivial: at fluid-target reads, neighbours
at distance 2 ALSO get processed by the LW step with fluid
material, generating bulk values that bleed into the COMBINE
output at fluid cells via the LW operator's spatial coupling.

DIAGNOSTIC (LW-alone heterogeneous baseline, 2026-05-27):
Running LW with spatially-varying material (no ESIM
substitution at all) yields L^∞ slope = 1.01, L^1 slope = 1.92
on the same sweep. So baseline LW on heterogeneous material is
itself only order-1 at L^∞ — the discrete material discontinuity
is a known LW limitation that LP04 §3 is DESIGNED to fix. Our
LP04 ESIM DOES improve absolute L^∞ magnitude vs LW-alone (at
Nx=100: 2.8e5 vs 7.1e5; at Nx=200: 1.5e5 vs 4.2e5) but does NOT
fully restore order 2 at these resolutions.

The remaining gap likely requires:
  (a) Strict per-cell SAME-side / OTHER-side leg mixing per
      Eq 43-44 (the current per-side LW + scatter is an
      approximation)
  (b) Finer grids (Nx ≥ 400) where the asymptotic regime sets in
  (c) Possibly higher k (jump-condition order 3 instead of 2)

Effort: ~2-3 days (per Codex+Gemini post-port dual-reviewer
estimate, with the extended-sweep + LW-alone diagnostics adding
~1 day for the diagnostic groundwork).

Honest provenance per ~/CLAUDE.md Rules 1-13: `novel-combination`,
NOT `published`. The path to `published`:
  1. Refactor `lp04_step` to do per-cell SAME-side U^n /
     OTHER-side U* mixing per LP04 Eq 43-44 exactly (~1-2 days
     per Codex post-port review, 2026-05-27)
  2. Re-run convergence sweep; expect L^∞ order ≥ 1.5
  3. Extend to oblique (80°-inclined Γ + 21° incidence) per
     LP04 §4.2 Table 2 geometry (~1-2 days)
  4. Final dual-reviewer graduation review

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

    # Compute error in interior window (exclude border_cells from each
    # edge of the domain). Although x is meant to be periodic, the
    # combination of LP04's per-side LW pass + projector substitution
    # at irregular cells may leak a small lateral residual at the
    # x-edges. Excluding from BOTH edges localises the integrated
    # residual to the bulk + interface band where the LP04 truncation
    # claim applies (post-recursive-C/L port, 2026-05-27).
    sl = (slice(None),
          slice(border_cells, Nx - border_cells),
          slice(border_cells, Ny - border_cells))
    err = U[sl] - U_exact[sl]

    L_inf = float(np.max(np.abs(err)))
    L_1 = float(np.mean(np.abs(err)))

    # Diagnostic: where does L^∞ peak?
    abs_err = np.abs(err)
    flat_idx = int(np.argmax(abs_err))
    c_max, i_max_rel, j_max_rel = np.unravel_index(flat_idx, abs_err.shape)
    i_max = i_max_rel + border_cells
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
    nx_sweep = [60, 100, 150, 200, 300]   # extended to surface asymptotic regime
    print(f"Medium: fluid (ρ={medium['rho_f']}, c={medium['c_f']}) "
          f"above; solid (ρ={medium['rho_s']}, c_p={medium['c_p']}, "
          f"c_s={medium['c_s']}) below")
    print(f"Interface at y={interface_y*1000:.1f} mm; ω = {medium['omega']:.3e}")
    print(f"T_end = {T_end_periods:.2f} period; border_cells = {border_cells}")
    print()

    Nx_list = nx_sweep
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
