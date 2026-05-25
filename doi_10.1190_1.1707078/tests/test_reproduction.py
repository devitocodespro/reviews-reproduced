"""Standalone reproduction tests for Bohlen & Saenger (2004).

Pinned quantitative anchors:

  1. Homogeneous-field invariance.
  2. Sharp y-contrast: averaged value at the interface = arithmetic
     mean of the two sides.
  3. Sharp x-contrast: same recipe, different axis.
  4. Diagonal contrast: each averaged cell is the 2×2 mean of its
     four surrounding ρ values.
  5. Reference output regression: pinned `.npz` outputs match a
     fresh driver run to fp64.
  6. Cross-validation against parent repo helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
PARENT = THIS_DIR.parent
sys.path.insert(0, str(PARENT))

from run_reproduction import (  # noqa: E402
    bs04_corner_average,
    run_corner_average_sweep,
)


def test_homogeneous_field_invariant():
    """A constant ρ field is unchanged by 2×2 averaging."""
    rho = np.full((8, 12), 2.5, dtype=np.float64)
    out = bs04_corner_average(rho)
    np.testing.assert_array_equal(out, rho)


def test_vertical_contrast_interface_is_arithmetic_mean():
    """Step ρ=1 → ρ=3 at j=6: averaged cell (i, 5) sees
    ρ[i,5]=1, ρ[i+1,5]=1, ρ[i,6]=3, ρ[i+1,6]=3 → mean = 2.0."""
    rho = np.ones((8, 12), dtype=np.float64)
    rho[:, 6:] = 3.0
    out = bs04_corner_average(rho)
    for i in range(7):  # exclude last-row replicate
        assert abs(out[i, 5] - 2.0) < 1e-14, (
            f"out[{i}, 5] = {out[i, 5]}, expected 2.0"
        )


def test_horizontal_contrast_interface_is_arithmetic_mean():
    """Step ρ=1 → ρ=3 at i=4: averaged cell (3, j) sees the same
    pattern transposed."""
    rho = np.ones((8, 12), dtype=np.float64)
    rho[4:, :] = 3.0
    out = bs04_corner_average(rho)
    for j in range(11):
        assert abs(out[3, j] - 2.0) < 1e-14


def test_paper_recipe_byte_match():
    """Each interior cell holds the literal 2×2 arithmetic corner
    average. This is the BS04 paper prescription."""
    rng = np.random.default_rng(0)
    rho = rng.uniform(1.0, 3.0, size=(8, 12)).astype(np.float64)
    out = bs04_corner_average(rho)
    for i in range(7):
        for j in range(11):
            expected = 0.25 * (rho[i, j] + rho[i + 1, j]
                               + rho[i, j + 1] + rho[i + 1, j + 1])
            assert abs(out[i, j] - expected) < 1e-14


def test_boundary_replicate_extend_preserves_shape():
    """Last row/column replicated from nearest interior."""
    rho = np.arange(96, dtype=np.float64).reshape(8, 12)
    out = bs04_corner_average(rho)
    assert out.shape == rho.shape
    assert np.array_equal(out[-1, :-1], out[-2, :-1])
    assert np.array_equal(out[:-1, -1], out[:-1, -2])


def test_reference_outputs_match_driver():
    """Pinned `corner_average_battery.npz` matches a freshly-run
    driver to fp64."""
    ref_path = PARENT / 'reference_outputs' / 'corner_average_battery.npz'
    assert ref_path.exists()
    ref = np.load(ref_path)
    current = run_corner_average_sweep()
    for key in ['rho_homog', 'avg_homog', 'rho_vert', 'avg_vert',
                'rho_horz', 'avg_horz', 'rho_diag', 'avg_diag',
                'rho_rand', 'avg_rand']:
        np.testing.assert_array_equal(
            ref[key], current[key], err_msg=f"key {key!r} drift"
        )


def test_cross_validation_against_parent_repo_helper():
    """The parent repo's `apply_bohlen_saenger_density_averaging`
    must produce byte-identical output to the standalone driver —
    i.e. the helper implements the BS04 paper recipe exactly."""
    parent_root = THIS_DIR.parents[2]
    parent_common = parent_root / '00_common'
    if not parent_common.exists():
        pytest.skip(
            f"Parent repo 00_common not available at {parent_common}; "
            f"cross-validation skipped (standalone tests still pass)."
        )
    sys.path.insert(0, str(parent_common))
    try:
        from material_averaging import (
            apply_bohlen_saenger_density_averaging as parent_helper,
        )
    except ImportError as exc:
        pytest.skip(f"Cannot import parent helper: {exc}")

    rng = np.random.default_rng(1)
    rho = rng.uniform(1.0, 3.0, size=(8, 12)).astype(np.float64)
    standalone = bs04_corner_average(rho)
    via_parent = parent_helper(rho)
    np.testing.assert_array_equal(standalone, via_parent)
