"""LP04 reproduction driver — Lax-Wendroff bulk convergence sweep.

Validates the standalone LW bulk solver against the analytical
travelling-plane-wave solution in a HOMOGENEOUS fluid. Sweeps
N_x ∈ {50, 100, 200, 400} and fits L^∞ + L^1 convergence orders.

The full LP04 §4.2 plane-interface convergence test (Table 2)
additionally requires the analytical R/T reference for fluid-solid
coupling — see `analytical_reference.py:PlaneWaveAcousticElastic`
docstring KNOWN-INCOMPLETE caveat. The HOMOG bulk convergence here
is the bridge that validates the LW machinery before that
extension lands; it locks in the foundational 2nd-order property
of the LW step that LP04 §4.2's Table 2 leans on.

Outputs convergence rates to `reference_outputs/convergence_homog.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lax_wendroff import (LWMaterial, lw_step, cfl_dt,
                            make_material_layered)
from analytical_reference import PlaneWaveFluid


def run_one_resolution(Nx: int, T_end: float, plane_wave: PlaneWaveFluid,
                         L_x: float = 0.1) -> dict:
    """Run one convergence-sweep point.

    Initialise the field with the analytical plane wave at t=0, step
    for T_end with periodic boundaries, compare against the analytical
    at t=T_end. The analytical plane wave is constructed to be
    periodic on [0, L_x] × [0, L_x].

    Returns dict with L^∞, L^1 errors, dt, n_steps, wall time.
    """
    Ny = Nx
    # Use cell-spaced (NOT node-spaced) discretisation so the periodic
    # wrap-around is exact: cells at x_i = (i + 0.5) dx with dx = L_x/Nx.
    dx = L_x / Nx
    # Material: homogeneous fluid (μ=0 → degenerate solid)
    mat = make_material_layered(
        Nx, Ny, dx,
        interface_y=2 * L_x,    # interface outside domain → all-fluid
        tan_dip=0.0,
        anchor_x=0.5 * L_x,
        rho_fluid=plane_wave.rho, c_fluid=plane_wave.c,
        rho_solid=plane_wave.rho,
        c_p_solid=plane_wave.c, c_s_solid=0.0,
    )
    dt = cfl_dt(mat, dx, cfl=0.4)
    n_steps = max(2, int(np.ceil(T_end / dt)))
    dt = T_end / n_steps    # adjust dt to land exactly on T_end

    # Cell-centred grid → x_i = (i + 0.5) dx
    X = (np.arange(Nx) + 0.5) * dx
    Y = (np.arange(Ny) + 0.5) * dx
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    U = plane_wave.evaluate(XX, YY, t=0.0)

    # Time-step (PERIODIC)
    t0 = time.time()
    for n in range(n_steps):
        U = lw_step(U, mat, dx, dt, periodic=True)
    elapsed = time.time() - t0

    # Compare to analytical at T_end (periodic — no border exclusion)
    U_exact = plane_wave.evaluate(XX, YY, t=T_end)
    err = U - U_exact

    L_inf = float(np.max(np.abs(err)))
    L_1 = float(np.mean(np.abs(err)))

    return dict(
        Nx=Nx, dx=dx, dt=dt, n_steps=n_steps,
        L_inf=L_inf, L_1=L_1, wall_s=elapsed,
    )


def fit_order(dx_list: list[float], err_list: list[float]) -> float:
    """Fit a log-log slope to (dx, err) data."""
    if len(dx_list) < 2:
        return float('nan')
    log_dx = np.log(np.asarray(dx_list))
    log_err = np.log(np.asarray(err_list))
    slope, _ = np.polyfit(log_dx, log_err, 1)
    return float(slope)


def main() -> int:
    print("=" * 70)
    print("LP04-R.1.4 — LW bulk convergence on homog fluid plane wave")
    print("=" * 70)

    L_x = 0.1   # 10 cm domain
    rho = 1000.0
    c = 1500.0
    # For periodicity on [0, L_x]², propagation direction must align
    # with rational wavenumber: n̂ = (m, n) / √(m² + n²). Choose m=2,
    # n=1 (5 = m² + n²). The wave repeats over an integer number of
    # cycles in the box.
    m, n = 2, 1
    norm = np.sqrt(m * m + n * n)
    alpha = np.arctan2(n, m)
    # |k| chosen so that k_x = 2π · m / L_x is an integer multiple of
    # the fundamental wavenumber 2π/L_x → automatically periodic.
    k_mag = (2 * np.pi / L_x) * norm
    omega = c * k_mag
    plane_wave = PlaneWaveFluid(
        A=1.0, omega=omega, alpha=alpha,
        phase=0.0, c=c, rho=rho,
    )

    # T_end: time for wave to travel ~half a wavelength
    T_end = 0.5 / (omega / (2 * np.pi))   # half a period
    print(f"Plane wave: c={c} m/s, ω={omega:.3e} rad/s, "
          f"λ={2 * np.pi * c / omega * 1000:.2f} mm")
    print(f"Domain: {L_x*1000} × {L_x*1000} mm, T_end={T_end*1e6:.1f} µs")
    print(f"Reference: PlaneWaveFluid (homog acoustic)")
    print()

    Nx_list = [50, 100, 200, 400]
    results: list[dict] = []
    for Nx in Nx_list:
        r = run_one_resolution(Nx, T_end, plane_wave, L_x=L_x)
        print(f"  Nx={Nx:4d}  dx={r['dx']*1e6:6.1f}µm  "
              f"L^∞={r['L_inf']:.4e}  L^1={r['L_1']:.4e}  "
              f"nt={r['n_steps']:4d}  wall={r['wall_s']:.2f}s")
        results.append(r)

    # Fit convergence orders
    dx_arr = [r['dx'] for r in results]
    L_inf_arr = [r['L_inf'] for r in results]
    L_1_arr = [r['L_1'] for r in results]
    order_inf = fit_order(dx_arr, L_inf_arr)
    order_1 = fit_order(dx_arr, L_1_arr)

    print()
    print(f"Fitted convergence orders (log-log slope over {len(Nx_list)} grids):")
    print(f"  L^∞ order = {order_inf:.3f}")
    print(f"  L^1 order = {order_1:.3f}")

    # LP04 Table 2 reports L^∞ orders converging to ≈ 2.0 for LW+ESIM
    # on the plane-interface case. On homog fluid (no ESIM coupling),
    # the LW scheme should match 2.0 even more cleanly because it
    # doesn't have to absorb the interface truncation error.
    print()
    pass_inf = order_inf >= 1.8
    pass_1 = order_1 >= 1.8
    print("Acceptance: order ≥ 1.8 in both norms (LW is formally 2nd-order)")
    print(f"  L^∞ {'PASS ✓' if pass_inf else 'FAIL ✗'}: order={order_inf:.3f}")
    print(f"  L^1 {'PASS ✓' if pass_1 else 'FAIL ✗'}: order={order_1:.3f}")

    out = {
        'config': dict(L_x=L_x, c=c, rho=rho, omega=omega, T_end=T_end,
                        alpha_propagation_rad=plane_wave.alpha),
        'sweep': results,
        'order_L_inf': order_inf,
        'order_L_1': order_1,
        'paper_reference': 'Lombard & Piraux 2004, JCP 195, Table 2',
        'note': ('Homogeneous fluid bulk only; the full LP04 §4.2 plane-'
                  'interface convergence requires the fluid-solid R/T '
                  'analytical reference. KNOWN-INCOMPLETE per '
                  'analytical_reference.py:PlaneWaveAcousticElastic.'),
    }
    out_path = HERE / 'reference_outputs' / 'convergence_homog.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    return 0 if (pass_inf and pass_1) else 1


if __name__ == '__main__':
    sys.exit(main())
