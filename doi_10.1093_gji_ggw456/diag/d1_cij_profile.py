"""§87.D1 diagnostic: visualise effective Cij in the mixed-cell band.

Compares per-cell Cij from `build_layered_cij_arrays` under
`interface_treatment='sharp'` vs `'kristek2017'` along a
vertical profile through the dipping interface at the Petrobras
config (dx=5m, dip=30°, SO=8). Renders 7-panel figure (rho, c11,
c22, c12, c16, c26, c66) showing both treatments overlaid.

Output: figures/diag/d1_cij_profile_dip30.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "00_common"))
sys.path.insert(0, str(REPO))

from petrobras_config import (   # noqa: E402
    LX, LY, INTERFACE_ANCHOR_X, INTERFACE_ANCHOR_Y,
    RHO_WATER, VP_WATER,
    RHO_TTI, VP_TTI, VS_TTI,
    EPSILON_TTI, DELTA_TTI, THETA_TTI_DEG,
)
from tti_common.tti_params import (   # noqa: E402
    BenchmarkConfig, LayeredBenchmarkConfig, build_layered_cij_arrays,
)


DX = 0.005   # km
DIP = 30.0
PROFILE_X_FRAC = 0.5   # column at x = LX/2 (where interface anchor sits)
ZOOM_RADIUS_KM = 0.10   # ±100 m around the interface (5m cells × 20)

OUT_PNG = REPO / "figures" / "diag" / "d1_cij_profile_dip30.png"


def build_arrays(treatment: str):
    water = BenchmarkConfig(name='water', Vp=VP_WATER, Vs=0.0,
                             rho=RHO_WATER, epsilon=0.0, delta=0.0,
                             theta=0.0)
    tti = BenchmarkConfig(name='tti', Vp=VP_TTI, Vs=VS_TTI,
                           rho=RHO_TTI, epsilon=EPSILON_TTI,
                           delta=DELTA_TTI, theta=THETA_TTI_DEG)
    cfg = LayeredBenchmarkConfig(
        name='petrobras_dip30', layers=(water, tti), dip_deg=DIP,
        anchor=(INTERFACE_ANCHOR_X / LX, INTERFACE_ANCHOR_Y / LY))
    nx = int(LX / DX) + 1
    ny = int(LY / DX) + 1
    out = build_layered_cij_arrays(
        (nx, ny), (LX, LY), cfg, interface_treatment=treatment)
    return out, nx, ny


def main():
    print(f"Building sharp arrays at dx={DX*1000:.0f}m dip={DIP}°...")
    sharp, nx, ny = build_arrays('sharp')
    print(f"Building kristek2017 arrays...")
    kristek, _, _ = build_arrays('kristek2017')

    # Profile column index
    ix = int(PROFILE_X_FRAC * (nx - 1))
    x_at_col = ix * DX
    y_iface = INTERFACE_ANCHOR_Y - np.tan(np.deg2rad(DIP)) * (
        x_at_col - INTERFACE_ANCHOR_X)
    y_axis = np.linspace(0, LY, ny)
    iy_iface = int(round(y_iface / DX))
    print(f"Profile at x={x_at_col:.3f} km (col {ix}); "
          f"interface at y={y_iface:.3f} km (row {iy_iface})")

    # Zoom window around the interface
    n_zoom = int(ZOOM_RADIUS_KM / DX)
    j_lo = max(0, iy_iface - n_zoom)
    j_hi = min(ny, iy_iface + n_zoom + 1)
    y_zoom = y_axis[j_lo:j_hi]

    # Count mixed-cell band width in Kristek arrays
    # A cell is "mixed" if its rho differs from BOTH pure-water and
    # pure-TTI by more than 1 part in 1e6 (handles fp64 noise).
    rho_col = kristek['rho'][ix, :]
    mixed_mask = (np.abs(rho_col - RHO_WATER) > 1e-6) & \
                 (np.abs(rho_col - RHO_TTI) > 1e-6)
    mixed_rows = np.where(mixed_mask)[0]
    if len(mixed_rows) > 0:
        print(f"Mixed-cell band: {len(mixed_rows)} cells "
              f"(rows {mixed_rows[0]}..{mixed_rows[-1]}, "
              f"y={y_axis[mixed_rows[0]]:.4f}..{y_axis[mixed_rows[-1]]:.4f} km)")
    else:
        print("⚠ NO MIXED CELLS DETECTED in Kristek column!")

    # 7-panel figure
    components = [('rho', 'ρ (g/cm³)'),
                   ('C11', 'C11'), ('C22', 'C22'), ('C12', 'C12'),
                   ('C16', 'C16'), ('C26', 'C26'), ('C66', 'C66')]
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for k, (key, label) in enumerate(components):
        ax = axes[k]
        s = sharp[key][ix, j_lo:j_hi]
        kk = kristek[key][ix, j_lo:j_hi]
        ax.plot(s, y_zoom, 'b-', lw=1.5, label='sharp', alpha=0.7)
        ax.plot(kk, y_zoom, 'r--', lw=1.5, label='kristek2017',
                alpha=0.9)
        ax.axhline(y_iface, color='cyan', linestyle=':', lw=1.0,
                    label='interface')
        ax.set_xlabel(label)
        ax.set_title(label)
        if k == 0:
            ax.set_ylabel(f'y (km), profile at x={x_at_col:.2f} km')
            ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_visible(False)

    fig.suptitle(f"§87.D1 — Effective Cij profile at dip={DIP}°, "
                  f"dx={DX*1000:.0f}m, x={x_at_col:.2f} km\n"
                  f"Mixed-cell band width: {len(mixed_rows)} cells",
                  fontsize=12)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\nWrote {OUT_PNG}")

    # Print key numbers for the diagnostic finding
    print("\n=== Key numbers (interface row, profile column) ===")
    print(f"  Row iy_iface = {iy_iface}, y = {y_axis[iy_iface]:.4f} km")
    print(f"  Sharp  rho  = {sharp['rho'][ix, iy_iface]:.4f}  "
          f"Kristek rho  = {kristek['rho'][ix, iy_iface]:.4f}")
    for key in ('C11', 'C22', 'C12', 'C16', 'C26', 'C66'):
        s = sharp[key][ix, iy_iface]
        k = kristek[key][ix, iy_iface]
        d = k - s
        print(f"  Sharp  {key:<3} = {s:+.4f}   "
              f"Kristek {key:<3} = {k:+.4f}   diff = {d:+.4f}")


if __name__ == "__main__":
    main()
