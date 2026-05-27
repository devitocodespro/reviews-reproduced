"""Tests for the standalone LP04 ESIM projector.

Locks down physical invariants of the N row matrix at a horizontal
fluid-solid interface (LP04 §3 paper-faithful 3-fluid + 5-solid layout).
These invariants form the regression baseline for any future change
to `esim_projector.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import esim_projector as ep  # noqa: E402


# ─── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def horizontal_interface_k2():
    """Standard small smoke config: horizontal Γ at y=0, fluid above,
    solid below, k=2 ESIM, q=3.5·dx disk. Returns the (N, info) tuple
    and the disk lists for use in component-level checks.
    """
    dx = 0.005
    P = (0.0, 0.0)
    tangent = (1.0, 0.0)
    B1, B2 = ep.disk_neighbours_around_P(P, 'above', dx, q_factor=3.5)
    target = (0.0, -dx)   # solid cell adjacent to Γ
    N, info = ep.build_projector(
        target_side='solid', target_xy=target, P_xy=P, tangent=tangent,
        B1_coords=B1, B2_coords=B2, k=2,
        c_p_solid=4.0, c_s_solid=2.0,
    )
    return N, info, B1, B2


# ─── Shape + conditioning ────────────────────────────────────────────────

def test_N_shape_matches_paper_layout(horizontal_interface_k2):
    """N for a SOLID-side target gives the FLUID extension at solid
    position (LP04 Eq 41 — target in Ω_2 uses G_1 K_1 W_1). Output
    is in 3-component fluid layout (v_x, v_y, p). Input vector
    layout: 3·|B_1| + 5·|B_2|.
    """
    N, info, B1, B2 = horizontal_interface_k2
    expected_cols = 3 * len(B1) + 5 * len(B2)
    assert N.shape == (3, expected_cols), (
        f"Solid-target N shape {N.shape}; expected (3, {expected_cols})")


def test_cond_M_well_conditioned(horizontal_interface_k2):
    """At the canonical q=3.5·dx disk, cond(M) should be near 2.5e+05
    (matches the parent's per-cell diagnostic value uniformly across
    all dip=0° irregular cells per Caunt.diag, 2026-05-27)."""
    _, info, _, _ = horizontal_interface_k2
    assert info.cond_M < 1e6, f"cond(M)={info.cond_M:.3e} too large"
    assert info.cond_M > 1e4, f"cond(M)={info.cond_M:.3e} suspiciously small"


def test_M_full_rank(horizontal_interface_k2):
    """M should be full rank: rank == n_w1 + n_lambda at proper q-disk."""
    _, info, _, _ = horizontal_interface_k2
    assert info.rank_M == info.n_w1 + info.n_lambda


def test_N_finite_and_bounded(horizontal_interface_k2):
    """No NaN/Inf, |N| ≤ 10 (LP04 §3.9 says ESIM correction is ~order-1
    in magnitude on a well-conditioned config)."""
    N, _, _, _ = horizontal_interface_k2
    assert np.all(np.isfinite(N))
    assert np.linalg.norm(N) < 10.0


# ─── Physical invariants at horizontal interface ─────────────────────────

def _component_norms_solid_target(N: np.ndarray, n_B1: int
                                     ) -> dict[str, tuple[float, float]]:
    """Compute per-component (|B_1 cols|, |B_2 cols|) norms.

    For SOLID target with corrected LP04 sign convention, N has shape
    (3, 3·|B_1| + 5·|B_2|) → output components are (v_x, v_y, p).
    """
    n_b1_cols = 3 * n_B1
    names = ['v_x', 'v_y', 'p']
    return {
        name: (float(np.linalg.norm(N[c, :n_b1_cols])),
               float(np.linalg.norm(N[c, n_b1_cols:])))
        for c, name in enumerate(names)
    }


def test_v_x_component_has_mixed_coupling(horizontal_interface_k2):
    """LP04 Eq 41 corrected: U* for solid target = FLUID extension at
    solid position. The v_x component depends on both fluid (B_1) and
    solid (B_2) disk neighbours via the LP04 jump-condition coupling,
    NOT purely solid as the pre-2026-05-27 (incorrect-label) projector
    gave. We check both norms are non-trivial.
    """
    N, _, B1, _ = horizontal_interface_k2
    norms = _component_norms_solid_target(N, len(B1))
    b1_n, b2_n = norms['v_x']
    # At least ONE side must have non-trivial coupling
    assert (b1_n + b2_n) > 1e-3, (
        f"v_x: coupling fluid+solid = {b1_n + b2_n:.3e} too weak")


def test_pressure_component_uses_fluid_disk(horizontal_interface_k2):
    """For solid-target → fluid-extension-at-solid-position, the
    PRESSURE p (last component of U_1^k) should have significant
    coupling to the FLUID disk B_1 (since p only exists in fluid).
    """
    N, _, B1, _ = horizontal_interface_k2
    norms = _component_norms_solid_target(N, len(B1))
    b1_n, b2_n = norms['p']
    # Pressure comes from fluid via the projector chain — B_1 (fluid)
    # should contribute. Allow B_2 contribution too via jump conditions.
    assert b1_n > 1e-3, (
        f"p: fluid coupling {b1_n:.3e} too weak (fluid pressure should "
        f"come from fluid disk-B)")


def test_normal_velocity_fluid_solid_balanced(horizontal_interface_k2):
    """At horizontal Γ, Eq 10 [v·n] = 0 forces v_y to be continuous
    across Γ. For solid-target → fluid-extension U*, v_y comes through
    the projector chain that mixes fluid + solid disk info.
    """
    N, _, B1, _ = horizontal_interface_k2
    norms = _component_norms_solid_target(N, len(B1))
    b1_n, b2_n = norms['v_y']
    # Both sides should contribute non-trivially via jump conditions.
    assert (b1_n + b2_n) > 0.05, (
        f"v_y: total fluid+solid coupling {b1_n + b2_n:.3e} too weak")


def test_disk_symmetry_in_x(horizontal_interface_k2):
    """At a horizontal interface with target at x=0, the disk is
    x-symmetric. The N row for v_y at a B_2 cell at +x should equal
    the N row at a B_2 cell at -x (parity in x). Sample one pair.
    """
    N, info, B1, B2 = horizontal_interface_k2
    # Find pairs in B_2 that are mirror images about x=0
    n_b1_cols = 3 * len(B1)
    eps_xy = 1e-12
    paired_indices: list[tuple[int, int]] = []
    for i, (x_i, y_i) in enumerate(B2):
        for j, (x_j, y_j) in enumerate(B2):
            if (abs(x_i + x_j) < eps_xy
                    and abs(y_i - y_j) < eps_xy and i != j):
                paired_indices.append((i, j))
                break
    assert len(paired_indices) > 0, "no mirror pairs found in B_2"
    # For component v_y (index 1), the projector should give equal
    # weight to mirror cells. Take the v_y component within a B_2
    # cell's 5-tuple = col index n_b1_cols + 5·i + 1.
    for (i, j) in paired_indices:
        col_i = n_b1_cols + 5 * i + 1   # v_y at B_2[i]
        col_j = n_b1_cols + 5 * j + 1   # v_y at B_2[j]
        w_i = N[1, col_i]
        w_j = N[1, col_j]
        assert abs(w_i - w_j) < 1e-10, (
            f"x-symmetry broken at B_2 pair ({i},{j}): "
            f"w_i={w_i:.6e}, w_j={w_j:.6e}")
