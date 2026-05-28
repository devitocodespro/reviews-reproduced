"""Yang 2015 RSG-TTI top-level reproduction driver.

Regenerates the canonical reference outputs:
  reference_outputs/yang2015_table_1_te.json — Yang 2015 §3.1 TE-RSFD
    coefficients M=2..11 (sympy-rational solve)
  reference_outputs/yang2015_table_2_sa_u1p10.json — §3.2 SA-RSFD
    coefficients at u=1.10 (closed-form scipy solve)
  reference_outputs/yang2015_table_3_ls_u1p10.json — §3.3 LS-RSFD
    coefficients at u=1.10 (scipy quad + linear solve)
  reference_outputs/yang2015_table_4_dispersion_u.json — §4 RMS
    dispersion error u-values for ε ∈ {1e-6, 1e-5, 1e-4, 1e-3} ×
    M=2..11 × {TE, SA, LS} (120-cell byte-match anchor)
  reference_outputs/yang2015_dispersion_curves_m10_u1p10.json —
    dispersion curve δ_M(β) data for M=10 at u=1.10, replicating
    Yang 2015 Fig 7 sampling.

Usage:
    cd reproduced/doi_10.1016_j.jappgeo.2015.08.007/
    uv sync
    uv run python run_reproduction.py

All outputs are deterministic; the same inputs (paper-cited M, u, ε
values) reproduce byte-identical outputs across runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yang2015_rsfd_solvers import (
    solve_te_rsfd, solve_sa_rsfd, solve_ls_rsfd,
)
from yang2015_dispersion import (
    YANG_2015_TABLE_4, dispersion_error_rms, find_u_for_target_error,
)


HERE = Path(__file__).resolve().parent
REF = HERE / "reference_outputs"


def main() -> int:
    REF.mkdir(parents=True, exist_ok=True)

    # ─── Table 1 — TE-RSFD coefficients (M=2..11, no u dependence) ─────
    print("Generating Table 1 (TE-RSFD coefficients M=2..11)...")
    table_1 = {str(M): [float(v) for v in solve_te_rsfd(M)]
               for M in range(2, 12)}
    out = REF / "yang2015_table_1_te.json"
    out.write_text(json.dumps(table_1, indent=2))
    print(f"  → {out}")

    # ─── Table 2 — SA-RSFD coefficients at u=1.10 (M=2..11) ─────────────
    print("Generating Table 2 (SA-RSFD coefficients at u=1.10)...")
    table_2 = {str(M): [float(v) for v in solve_sa_rsfd(M, u=1.10)]
               for M in range(2, 12)}
    out = REF / "yang2015_table_2_sa_u1p10.json"
    out.write_text(json.dumps(table_2, indent=2))
    print(f"  → {out}")

    # ─── Table 3 — LS-RSFD coefficients at u=1.10 (M=2..11) ─────────────
    print("Generating Table 3 (LS-RSFD coefficients at u=1.10)...")
    table_3 = {str(M): [float(v) for v in solve_ls_rsfd(M, u=1.10)]
               for M in range(2, 12)}
    out = REF / "yang2015_table_3_ls_u1p10.json"
    out.write_text(json.dumps(table_3, indent=2))
    print(f"  → {out}")

    # ─── Table 4 — Dispersion u-values (4 ε × 10 M × 3 schemes) ────────
    print("Generating Table 4 (dispersion u-values for 120 cells)...")
    table_4: dict = {}
    for eps in sorted(YANG_2015_TABLE_4.keys()):
        eps_key = f"{eps:.0e}"
        table_4[eps_key] = {}
        for M in sorted(YANG_2015_TABLE_4[eps].keys()):
            row: dict = {}
            for scheme in ("TE", "SA", "LS"):
                u = find_u_for_target_error(M, eps, scheme)
                row[scheme] = float(u)
            table_4[eps_key][str(M)] = row
    out = REF / "yang2015_table_4_dispersion_u.json"
    out.write_text(json.dumps(table_4, indent=2))
    print(f"  → {out}")

    # ─── Fig 7 — dispersion curves at M=10, u=1.10 for TE/SA/LS ────────
    print("Generating Fig 7 dispersion curves (M=10, u=1.10)...")
    n_beta = 200
    betas = np.linspace(1e-4, 0.5 * np.pi, n_beta)
    M = 10
    u = 1.10
    a_te = solve_te_rsfd(M)
    a_sa = solve_sa_rsfd(M, u)
    a_ls = solve_ls_rsfd(M, u)
    delta_curves: dict[str, list[float]] = {"beta": betas.tolist()}
    for label, a in (("TE", a_te), ("SA", a_sa), ("LS", a_ls)):
        delta = []
        for beta in betas:
            total = sum(a[m] * np.sin((2 * (m + 1) - 1) * beta)
                        for m in range(M))
            delta.append(total / beta - 1.0)
        delta_curves[label] = delta
    out = REF / "yang2015_dispersion_curves_m10_u1p10.json"
    out.write_text(json.dumps(delta_curves, indent=2))
    print(f"  → {out}")

    # ─── Diagnostic: print RMS errors at u=1.10 for all three schemes ──
    print()
    print("RMS dispersion error ε at M=10, u=1.10:")
    for label, a in (("TE", a_te), ("SA", a_sa), ("LS", a_ls)):
        eps = dispersion_error_rms(a, u)
        print(f"  {label}: ε = {eps:.3e}")

    print()
    print("Reproduction complete. See reference_outputs/ for pinned outputs.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
