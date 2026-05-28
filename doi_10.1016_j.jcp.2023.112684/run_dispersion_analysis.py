"""Reproduce Xie & He 2024 §4 dispersion analysis as a regression
figure + scalar L1-error metric.

This script produces three artifacts under `reference_outputs/`:

1. `dispersion_curve.png` — modified-wavenumber error |k̃h − kh|
   vs kh for Taylor (Fornberg 1988) and STO (TE+LS Xie-He 2024)
   at P ∈ {4, 5, 6}, cfl=0.4. Side-by-side panels make the
   tradeoff visually obvious: Taylor wins at low kh, STO wins
   at moderate-to-high kh.

2. `dispersion_l1_errors.json` — scalar L1 error integral of the
   dispersion error across two bands:
   - LOW band [0.1, 0.7]: ∫|k̃h − kh| dkh — Taylor expected < STO
   - HIGH band [1.2, 2.0]: ∫|k̃h − kh| dkh — STO expected < Taylor

   These integrals are pinned in
   `tests/test_dispersion_l1_regression.py` for stable comparison
   against single-point sampling.

3. The figure also annotates the crossover wavenumber kh* where
   Taylor and STO curves intersect (per-P, per-CFL).

Per Xie & He §3.2 + §4: the paper's dispersion analysis
demonstrates the tradeoff at wavenumbers approaching Nyquist
(kh = π/2 per axis, ~2.22 rad along the diagonal). The HIGH
band chosen here sits inside that region.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "00_common"))
sys.path.insert(0, str(HERE))

from spacetime_coefficients import (  # noqa: E402
    compute_sto_coefficients,
    sto_modified_wavenumber,
)
from drp_coefficients import centered_fd_coefficients  # noqa: E402
import paper_tables as pt  # noqa: E402


def _taylor_kh(taylor: dict[int, float], P: int, kh: float | np.ndarray):
    """Modified wavenumber for antisymmetric centered Taylor FD."""
    return 2 * sum(taylor[m] * np.sin(m * kh) for m in range(1, P + 1))


def _l1_integral(err_curve: np.ndarray, kh_grid: np.ndarray) -> float:
    """Trapezoidal L1 integral of |error| over a kh band."""
    return float(np.trapezoid(np.abs(err_curve), kh_grid))


def main():
    out_dir = HERE / "reference_outputs"
    out_dir.mkdir(exist_ok=True)

    cfl = 0.4
    Ps = (4, 5, 6)
    kh_dense = np.linspace(0.05, np.pi / 1.05, 400)  # avoid the Nyquist edge

    # Compute per-P dispersion curves
    curves = {}
    for P in Ps:
        sto = compute_sto_coefficients(P, cfl)
        taylor = centered_fd_coefficients(2 * P)
        sto_err = np.array([
            abs(sto_modified_wavenumber(sto, kh) - kh) for kh in kh_dense
        ])
        taylor_err = np.array([
            abs(_taylor_kh(taylor, P, kh) - kh) for kh in kh_dense
        ])
        curves[P] = {"sto": sto_err, "taylor": taylor_err}

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, P in zip(axes, Ps):
        ax.semilogy(kh_dense, curves[P]["taylor"], label="Taylor",
                    color="steelblue", lw=1.6)
        ax.semilogy(kh_dense, curves[P]["sto"], label="STO (TE+LS)",
                    color="orangered", lw=1.6)

        # Highlight band regions
        lo_lo, lo_hi = pt.XIE_HE_2024_DISPERSION_LOW_KH_BAND
        hi_lo, hi_hi = pt.XIE_HE_2024_DISPERSION_HIGH_KH_BAND
        ax.axvspan(lo_lo, lo_hi, alpha=0.10, color="steelblue",
                   label="Taylor wins band")
        ax.axvspan(hi_lo, hi_hi, alpha=0.10, color="orangered",
                   label="STO wins band")

        ax.set_title(f"P={P}  (so={2 * P})  cfl={cfl}")
        ax.set_xlabel("kh [rad]")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(0, np.pi / 1.05)
        ax.set_ylim(1e-12, 10)
    axes[0].set_ylabel(r"$|\tilde k h - k h|$  (log scale)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Xie & He 2024 §4 dispersion analysis — STO (TE+LS) vs Taylor\n"
        "STO trades low-kh accuracy for high-kh accuracy (LS tradeoff)",
        fontsize=11
    )
    fig.tight_layout()
    fig_path = out_dir / "dispersion_curve.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    # ── L1 error integrals per band ──────────────────────────────────
    l1_results = {
        "_provenance": {
            "generated": "2026-05-28",
            "cfl": cfl,
            "low_band": list(pt.XIE_HE_2024_DISPERSION_LOW_KH_BAND),
            "high_band": list(pt.XIE_HE_2024_DISPERSION_HIGH_KH_BAND),
            "note": (
                "L1 integrals of |modified_wavenumber_error| over "
                "low and high kh bands. Used by "
                "tests/test_dispersion_l1_regression.py as a "
                "robust replacement for single-point dispersion "
                "tests. Re-pin after intentional optimizer/objective "
                "changes by re-running run_dispersion_analysis.py."
            ),
        },
    }
    for P in Ps:
        lo_lo, lo_hi = pt.XIE_HE_2024_DISPERSION_LOW_KH_BAND
        hi_lo, hi_hi = pt.XIE_HE_2024_DISPERSION_HIGH_KH_BAND
        # Restrict the dense grid to each band
        mask_lo = (kh_dense >= lo_lo) & (kh_dense <= lo_hi)
        mask_hi = (kh_dense >= hi_lo) & (kh_dense <= hi_hi)

        l1_results[f"P{P}"] = {
            "low_band": {
                "taylor_l1": _l1_integral(curves[P]["taylor"][mask_lo],
                                          kh_dense[mask_lo]),
                "sto_l1": _l1_integral(curves[P]["sto"][mask_lo],
                                       kh_dense[mask_lo]),
            },
            "high_band": {
                "taylor_l1": _l1_integral(curves[P]["taylor"][mask_hi],
                                          kh_dense[mask_hi]),
                "sto_l1": _l1_integral(curves[P]["sto"][mask_hi],
                                       kh_dense[mask_hi]),
            },
        }

    json_path = out_dir / "dispersion_l1_errors.json"
    json_path.write_text(json.dumps(l1_results, indent=2))
    print(f"Wrote {json_path}")

    # ── Console summary ──────────────────────────────────────────────
    print(f"\nDispersion L1 error summary (cfl={cfl}):")
    print(f"{'P':>3}  {'band':>5}  {'taylor_L1':>12}  {'STO_L1':>12}  "
          f"{'STO/Taylor':>12}")
    for P in Ps:
        for band_name in ("low_band", "high_band"):
            t = l1_results[f"P{P}"][band_name]["taylor_l1"]
            s = l1_results[f"P{P}"][band_name]["sto_l1"]
            band_label = "LOW" if band_name == "low_band" else "HIGH"
            print(f"{P:>3}  {band_label:>5}  {t:12.4e}  {s:12.4e}  "
                  f"{s / t:12.4f}")


if __name__ == "__main__":
    main()
