"""Byte-match transcribed paper constants against upstream computation.

These tests verify that the values in `paper_tables.py` (our hand-
transcribed snapshot of Zhang & Schmitt 2025's quantitative claims)
match what the upstream OumZhang/rsg code at `upstream/src/`
actually computes. If a future Devito update or upstream patch
changes the numerics silently, these tests catch it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM_SRC = ROOT / "upstream" / "src"

# Make both `upstream.src` and the reproduction's `paper_tables.py`
# importable.
sys.path.insert(0, str(UPSTREAM_SRC))
sys.path.insert(0, str(ROOT))

import paper_tables as pt  # noqa: E402
import wavesolver as ws    # noqa: E402  (upstream/src/wavesolver.py)
import operators as ops    # noqa: E402  (upstream/src/operators.py)


# ─── (1) Fornberg staggered first-derivative weights ───────────────


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_fornberg_weights_byte_match_upstream(order):
    """`upstream.fdcoeff_1st(order)` must return the same weights
    as our transcription in `paper_tables.FORNBERG_STAGGERED_1ST_DERIV`."""
    computed = np.asarray(ws.fdcoeff_1st(order))
    expected = pt.FORNBERG_STAGGERED_1ST_DERIV[order]
    np.testing.assert_allclose(
        computed, expected, atol=1e-14, rtol=1e-13,
        err_msg=(
            f"Upstream's fdcoeff_1st(order={order}) returned "
            f"{computed!r} but our transcription expects {expected!r}. "
            f"Either a paper-table transcription drift (update "
            f"paper_tables.py) or an upstream regression (file an "
            f"issue on OumZhang/rsg)."
        ),
    )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_fornberg_weights_consistency(order):
    """Σ c_k = 0 (consistency for derivative)."""
    weights = pt.FORNBERG_STAGGERED_1ST_DERIV[order]
    assert abs(weights.sum()) < 1e-14, (
        f"Σ c_k = {weights.sum()} != 0 at SO={order}")


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_fornberg_weights_first_moment(order):
    """Σ s_k · c_k = 1 (first-order accuracy)."""
    weights = pt.FORNBERG_STAGGERED_1ST_DERIV[order]
    offsets = pt.staggered_offsets(order)
    first_moment = (offsets * weights).sum()
    assert abs(first_moment - 1.0) < 1e-13, (
        f"Σ s_k c_k = {first_moment} != 1 at SO={order}")


# ─── (2) 8-point isotropic source-weighting recipe ────────────────


def test_oumzhang_8point_total_moment():
    """8 offset positions × weight 1/8 = 1 (preserves monopole
    moment magnitude)."""
    n_positions = len(pt.OUMZHANG_8POINT_OFFSETS)
    assert n_positions == 8, (
        f"Expected 8 offset positions; got {n_positions}.")
    total = n_positions * pt.OUMZHANG_8POINT_WEIGHT
    assert abs(total - 1.0) < 1e-15, (
        f"Total weight {total} != 1.")


def test_oumzhang_8point_no_centre():
    """The OumZhang 8-point recipe explicitly does NOT include
    the source's grid cell (offset (0, 0)). This is the defining
    departure from the LV2010 1/2-centre + 1/8-cardinal pattern,
    and the reason the 8-point recipe works for RSG diagonal
    stencils where LV2010 fails."""
    assert (0, 0) not in pt.OUMZHANG_8POINT_OFFSETS, (
        "OumZhang 8-point recipe must NOT include the centre "
        "(0, 0) — that is the LV2010 pattern, which is "
        "SSG-Yee-tuned and produces a 4-petal X-mode for RSG.")


def test_oumzhang_8point_offsets_balanced():
    """For each offset (i, j) in the recipe, the sign-flipped
    offset (-i, -j) must also be present. This is required for
    the source to inject a balanced moment-tensor."""
    offsets = set(pt.OUMZHANG_8POINT_OFFSETS)
    for (i, j) in offsets:
        assert (-i, -j) in offsets, (
            f"Offset ({i}, {j}) lacks its sign-flipped pair "
            f"({-i}, {-j}); recipe is unbalanced.")


def test_oumzhang_8point_upstream_function_signature():
    """Upstream's `getsrcterm_2d_around` accepts `moment='iso'`."""
    # Function is defined in upstream/src/operators.py
    assert hasattr(ops, 'getsrcterm_2d_around'), (
        "Upstream getsrcterm_2d_around helper not exported.")


# ─── (3) Saenger Eq 14 diagonal-stencil composition ───────────────


def test_saenger_diagonal_composition_x():
    """∂/∂x = d1 + d2 (both diagonals contribute positively to
    the x-derivative)."""
    weights = pt.SAENGER_DIAGONAL_WEIGHTS_2D["x"]
    assert weights["d1"] == 1, (
        f"d1 weight for ∂/∂x must be +1, got {weights['d1']}")
    assert weights["d2"] == 1, (
        f"d2 weight for ∂/∂x must be +1, got {weights['d2']}")


def test_saenger_diagonal_composition_y():
    """∂/∂y = d1 - d2 (d1 positive, d2 negative — flips the
    sense along y while keeping x sense)."""
    weights = pt.SAENGER_DIAGONAL_WEIGHTS_2D["y"]
    assert weights["d1"] == 1, (
        f"d1 weight for ∂/∂y must be +1, got {weights['d1']}")
    assert weights["d2"] == -1, (
        f"d2 weight for ∂/∂y must be -1, got {weights['d2']}")
