"""§87.D2 diagnostic: side-by-side FD-sharp / FD-Kristek / DWN
at dip=30°.

Renders a 3-column figure of divergence(v) snapshots:
- col 1: FD-with-sharp (post-§86 dx020 pickle)
- col 2: FD-with-Kristek (post-§87.F1 dx020 pickle)
- col 3: Phase 2d DWN reference at the same grid

Shared colour scale across all three. Identifies whether
post-F1 Kristek wavefield shows interface scattering (the
user's symptom).

Output: figures/diag/d2_side_by_side_dip30.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "00_common"))

from bouchon_1981.dwn_solver import (   # noqa: E402
    dwn_wavefield_acoustic_anisotropic_dipping,
)
from mallick_frazer_1990.anisotropic_rt import stiffness_from_thomsen   # noqa: E402
from petrobras_config import (   # noqa: E402
    LX, LY, SRC_X, SRC_Y,
    INTERFACE_ANCHOR_X, INTERFACE_ANCHOR_Y,
    RHO_WATER, VP_WATER,
    RHO_TTI, VP_TTI, VS_TTI,
    EPSILON_TTI, DELTA_TTI, THETA_TTI_DEG,
    F0_HZ, T_LATE_S,
)


SHARP_PKL = REPO / "figures" / "dip_layered" / "dip_layered_tier4_sharp_dx020.pkl"
KRISTEK_PKL = REPO / "figures" / "dip_layered" / "dip_layered_tier4_kristek_dx020.pkl"
DIP_DEG = 30.0
T_MAX_DWN = 1.2
N_OMEGA = 256
M_WAVENUMBERS = 1024

OUT_PNG = REPO / "figures" / "diag" / "d2_side_by_side_dip30.png"


def main():
    print(f"Loading FD pickles...")
    with open(SHARP_PKL, "rb") as f:
        sharp_data = pickle.load(f)
    with open(KRISTEK_PKL, "rb") as f:
        kristek_data = pickle.load(f)
    env = sharp_data['env']
    dx_km = env['dx_km']
    nx = int(LX / dx_km) + 1
    ny = int(LY / dx_km) + 1

    # Extract SSG (method 1) div at dip=30°
    key = (1, DIP_DEG)
    sharp_div = sharp_data['results'][key]['div_late']
    kristek_div = kristek_data['results'][key]['div_late']

    # DWN reference at the same grid
    print(f"Computing DWN reference at dip={DIP_DEG}°...")
    x_grid = np.linspace(0, LX, nx)
    z_grid = np.linspace(0, LY, ny)
    C_lower = stiffness_from_thomsen(
        Vp=VP_TTI, Vs=VS_TTI, rho=RHO_TTI,
        epsilon=EPSILON_TTI, delta=DELTA_TTI,
        theta_rad=np.deg2rad(THETA_TTI_DEG))
    dwn_div = dwn_wavefield_acoustic_anisotropic_dipping(
        x_grid=x_grid, z_grid=z_grid, t_target=T_LATE_S,
        x_source=SRC_X, z_source=SRC_Y,
        x_anchor=INTERFACE_ANCHOR_X, y_anchor=INTERFACE_ANCHOR_Y,
        dip_deg=DIP_DEG,
        rho_upper=RHO_WATER, V_upper=VP_WATER,
        C_lower_lab=C_lower,
        f0=F0_HZ, T_max=T_MAX_DWN,
        n_omega=N_OMEGA, M_wavenumbers=M_WAVENUMBERS)

    # Shared scale: peak-normalise each then plot at common vmax
    sharp_norm = sharp_div / max(np.max(np.abs(sharp_div)), 1e-30)
    kristek_norm = kristek_div / max(np.max(np.abs(kristek_div)), 1e-30)
    dwn_norm = dwn_div / max(np.max(np.abs(dwn_div)), 1e-30)

    vmax = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    titles = ['FD sharp', 'FD Kristek (post §87.F1)',
              'Phase 2d DWN reference']
    for ax, fld, ttl in zip(axes, [sharp_norm, kristek_norm, dwn_norm],
                              titles):
        im = ax.imshow(fld.T, origin='lower', cmap='seismic',
                        vmin=-vmax, vmax=vmax,
                        extent=[0, LX*1000, 0, LY*1000],
                        aspect='equal',
                        interpolation='nearest')
        # Interface line
        x_km = np.array([0, LX])
        y_iface = INTERFACE_ANCHOR_Y - np.tan(np.deg2rad(DIP_DEG)) * (
            x_km - INTERFACE_ANCHOR_X)
        ax.plot(x_km*1000, y_iface*1000, color='cyan',
                linestyle=':', linewidth=1.0)
        # Source
        ax.plot(SRC_X*1000, SRC_Y*1000, marker='*', color='gold',
                markersize=12, markeredgecolor='k')
        ax.set_xlim(0, LX*1000)
        ax.set_ylim(0, LY*1000)
        ax.set_xlabel('x (m)')
        ax.set_title(ttl)
    axes[0].set_ylabel('z (m)')
    fig.suptitle(f"§87.D2 — SSG at dip {DIP_DEG}°, dx = {dx_km*1000:.0f} m, "
                  f"t = {T_LATE_S} s, peak-normalised divergence",
                  fontsize=12)
    cbar = fig.colorbar(im, ax=axes, location='right',
                         shrink=0.85, pad=0.02)
    cbar.set_label('div(v) / max|div(v)|')
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
