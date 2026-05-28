"""Bader 2023 canonical Fig 4 / Fig 7 runners.

Plan §108.followup-SBP-SAT-graduation Tasks #38, #50, #51, #52
(2026-05-22).

Provides reference-implementation runners on the canonical Bader 2023
geometry (see :mod:`bader_2023_canonical`):

- :func:`run_sbp_sat_bader_canonical` — Method 5 two-zone SBP-SAT
  with Bader-implicit seafloor coupling. Gold-standard reference for
  the Fig 7 comparison.
- :func:`run_sds_bader_canonical` — single-domain SSG with three
  material-averaging variants at the seafloor row (Moczo / fluid /
  solid) — Bader Fig 7 SDS-1/2/3 reproduction.
- :func:`run_sbp_sat_bader_high_resolution_reference` — generates +
  saves a high-resolution proxy SPECFEM2D reference for Bader Fig 4
  self-consistency gate (Task #52).
- :func:`compare_to_high_resolution_reference` — L² comparison
  helper used by the self-consistency gate.

Receiver-trace convention (returned dict keys):

- ``rcv_phi`` : ndarray (nt,) — fluid momentum-potential trace at
  the seafloor receiver (SBP-SAT only).
- ``rcv_ux``, ``rcv_uy`` : ndarray (nt,) — solid-displacement traces
  (SBP-SAT only).
- ``rcv_vx``, ``rcv_vy`` : ndarray (nt,) — solid-velocity traces
  (SDS only).
- ``rcv_p`` : ndarray (nt,) — pressure trace (SDS only).
- ``stable`` : bool
- ``dt`` : float
- ``nt`` : int
- ``geom`` : :class:`BaderCanonicalGeometry`

The ux-at-seafloor gap (previously documented here as Task #38b
known limitation) was fixed by Task #51 (2026-05-22) — all seven
two-zone factories now impose σ_xy = 0 BC weakly via a separate
``seafloor_solid_ux_bulk`` Eq mirroring the existing
``seafloor_solid_uy_bulk`` pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from devito import (TimeFunction, SparseTimeFunction)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / '05_sbp'))


def run_sbp_sat_bader_canonical(
        geom,
        *,
        coupling_form: str = 'bader-implicit',
        space_order: int = 4,
        rcv_y_offset: int = -1,
):
    """Run Method 5 two-zone SBP-SAT on the canonical Bader geometry.

    Parameters
    ----------
    geom : BaderCanonicalGeometry
        From :func:`bader_2023_canonical.setup_bader_geometry`.
    coupling_form : {'bader-implicit', 'inc'}
        Forwarded to ``build_sbp_sat_two_zone_operator``. Default
        ``'bader-implicit'`` (Bader 2023 reference-faithful 2×2
        implicit solve).
    space_order : int
        Spatial order. Default 4 for smoke runs.
    rcv_y_offset : int
        Vertical offset from the seafloor row at which to place the
        ux/uy receivers (in grid cells; negative = below seafloor).
        Workaround for the "ux at seafloor row = 0" gap (Task #38a).
        Default ``-1`` (one cell below seafloor); set to 0 to record
        exactly on the seafloor row.

    Returns
    -------
    dict
    """
    from bader_2023_canonical import setup_bader_geometry  # noqa: F401
    from sbp_displacement_tti import build_sbp_sat_two_zone_operator

    grid = geom.grid
    so = space_order
    phi = TimeFunction(name='phi_bcr', grid=grid, time_order=2,
                        space_order=so, dtype=grid.dtype)
    ux = TimeFunction(name='ux_bcr', grid=grid, time_order=2,
                      space_order=so, dtype=grid.dtype)
    uy = TimeFunction(name='uy_bcr', grid=grid, time_order=2,
                      space_order=so, dtype=grid.dtype)

    # Source: inject Ricker into phi (fluid momentum potential).
    src_coords = np.array([[
        geom.src_x_idx * geom.dx,
        geom.src_y_idx * geom.dy,
    ]])
    src = SparseTimeFunction(
        name='src_bcr', grid=grid, npoint=1, nt=geom.nt,
        coordinates=src_coords, dtype=grid.dtype,
    )
    src.data[:, 0] = geom.src_data
    dt_sym = grid.stepping_dim.spacing
    src_inj = src.inject(field=phi.forward,
                          expr=src * dt_sym**2 / geom.rho_fluid)

    # Receivers: 3C at (rcv_x_idx, rcv_y_idx + rcv_y_offset).
    rcv_y_phi = geom.rcv_y_idx
    rcv_y_disp = geom.rcv_y_idx + rcv_y_offset
    rcv_coords_phi = np.array([[
        geom.rcv_x_idx * geom.dx,
        rcv_y_phi * geom.dy,
    ]])
    rcv_coords_disp = np.array([[
        geom.rcv_x_idx * geom.dx,
        rcv_y_disp * geom.dy,
    ]])
    rcv_phi = SparseTimeFunction(
        name='rcv_phi_bcr', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords_phi, dtype=grid.dtype,
    )
    rcv_ux = SparseTimeFunction(
        name='rcv_ux_bcr', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords_disp, dtype=grid.dtype,
    )
    rcv_uy = SparseTimeFunction(
        name='rcv_uy_bcr', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords_disp, dtype=grid.dtype,
    )
    rec_eqs = [
        rcv_phi.interpolate(expr=phi),
        rcv_ux.interpolate(expr=ux),
        rcv_uy.interpolate(expr=uy),
    ]

    op = build_sbp_sat_two_zone_operator(
        grid, ux, uy, phi,
        geom.rho_fluid, geom.kappa_fluid, geom.rho_s,
        geom.lambda_s, geom.mu_s,
        geom.C11_f, geom.C22_f, geom.C12_f,
        geom.C16_f, geom.C26_f, geom.C66_f,
        geom.damp,
        fluid_subdomain=grid.subdomains['fluid_upper'],
        solid_subdomain=grid.subdomains['solid_lower'],
        seafloor_subdomain=grid.subdomains['seafloor_interface'],
        fs_top_subdomain=grid.subdomains['fs_row_top'],
        name='bader_canonical_sbp_sat',
        space_order=so,
        coupling_form=coupling_form,
        src_inject=[src_inj] + rec_eqs,
    )
    op.apply(time_m=1, time_M=geom.nt - 1, dt=geom.dt_val)

    rcv_phi_arr = np.asarray(rcv_phi.data).squeeze().copy()
    rcv_ux_arr = np.asarray(rcv_ux.data).squeeze().copy()
    rcv_uy_arr = np.asarray(rcv_uy.data).squeeze().copy()
    stable = (bool(np.all(np.isfinite(rcv_phi_arr)))
              and bool(np.all(np.isfinite(rcv_ux_arr)))
              and bool(np.all(np.isfinite(rcv_uy_arr))))
    return {
        'rcv_phi': rcv_phi_arr,
        'rcv_ux': rcv_ux_arr,
        'rcv_uy': rcv_uy_arr,
        'stable': stable,
        'dt': geom.dt_val,
        'nt': geom.nt,
        'geom': geom,
    }


def run_sds_bader_canonical(
        geom,
        *,
        sds_variant: str = 'moczo',
        space_order: int = 4,
        rcv_y_offset: int = -1,
):
    """SDS-1/2/3 runner on the canonical Bader geometry (Task #50 / #38a).

    Plan §108.followup-SBP-SAT-graduation Task #50 (2026-05-22).
    Single-domain staggered-grid (SSG, Virieux 1986 / Levander 1988)
    velocity-stress with three variants of material averaging at the
    seafloor row:

    - ``'moczo'`` (SDS-1) — Moczo 2002 arithmetic-mean ρ + harmonic-mean
      Lamé parameters (λ, μ) at the seafloor row. Harmonic mean of
      (μ_s, 0) is 0; harmonic mean of (λ_s, κ_f) gives the seafloor-row
      λ.
    - ``'fluid'`` (SDS-2) — fluid properties (ρ_f, κ_f, μ=0) at the
      seafloor row.
    - ``'solid'`` (SDS-3) — solid properties (ρ_s, λ_s, μ_s) at the
      seafloor row.

    Returns the same dict shape as
    :func:`run_sbp_sat_bader_canonical` but with velocity-stress
    receiver fields instead of displacement-based ones:

    - ``rcv_vx``, ``rcv_vy`` : particle-velocity traces at the receiver.
    - ``rcv_p``              : pressure trace ``p = -(σ_xx + σ_yy) / 2``.

    Free-surface BC at the sea surface uses Levander 1988 image method
    via ``free_surface.free_surface_eqs(formulation='velocity-stress',
    fs_location='top')``.
    """
    if sds_variant not in ('moczo', 'fluid', 'solid'):
        raise ValueError(
            f"sds_variant must be one of {{'moczo', 'fluid', 'solid'}}; "
            f"got {sds_variant!r}."
        )

    from devito import Function
    from bader_2023_canonical import (
        BADER_RHO_F, BADER_V_F, BADER_RHO_S, BADER_V_P, BADER_V_S,
    )
    sys.path.insert(0, str(_REPO / '01_ssg'))
    from ssg_elastic_tti import build_ssg_operator
    from free_surface import free_surface_eqs

    tag = sds_variant.replace('-', '_')
    grid = geom.grid
    so = space_order

    vx = TimeFunction(name=f'vx_sds_{tag}', grid=grid, time_order=1,
                       space_order=so, dtype=grid.dtype)
    vy = TimeFunction(name=f'vy_sds_{tag}', grid=grid, time_order=1,
                       space_order=so, dtype=grid.dtype)
    sxx = TimeFunction(name=f'sxx_sds_{tag}', grid=grid, time_order=1,
                        space_order=so, dtype=grid.dtype)
    syy = TimeFunction(name=f'syy_sds_{tag}', grid=grid, time_order=1,
                        space_order=so, dtype=grid.dtype)
    sxy = TimeFunction(name=f'sxy_sds_{tag}', grid=grid, time_order=1,
                        space_order=so, dtype=grid.dtype)

    def _mat_fn(name):
        return Function(name=name, grid=grid, space_order=so,
                         dtype=grid.dtype)

    rho_sds = _mat_fn(f'rho_sds_{tag}')
    C11 = _mat_fn(f'C11_sds_{tag}')
    C22 = _mat_fn(f'C22_sds_{tag}')
    C12 = _mat_fn(f'C12_sds_{tag}')
    C16 = _mat_fn(f'C16_sds_{tag}')
    C26 = _mat_fn(f'C26_sds_{tag}')
    C66 = _mat_fn(f'C66_sds_{tag}')

    kappa_f = BADER_RHO_F * BADER_V_F ** 2
    mu_s = BADER_RHO_S * BADER_V_S ** 2
    lam_s = BADER_RHO_S * BADER_V_P ** 2 - 2 * mu_s
    sfy = geom.seafloor_y_idx

    # Solid (y < sfy).
    rho_sds.data[:, :sfy] = BADER_RHO_S
    C11.data[:, :sfy] = lam_s + 2 * mu_s
    C22.data[:, :sfy] = lam_s + 2 * mu_s
    C12.data[:, :sfy] = lam_s
    C66.data[:, :sfy] = mu_s
    # Fluid (y > sfy).
    rho_sds.data[:, sfy + 1:] = BADER_RHO_F
    C11.data[:, sfy + 1:] = kappa_f
    C22.data[:, sfy + 1:] = kappa_f
    C12.data[:, sfy + 1:] = kappa_f
    C66.data[:, sfy + 1:] = 0.0

    # Seafloor row (sfy) — variant-dependent.
    if sds_variant == 'moczo':
        harm_lam = 2 * lam_s * kappa_f / (lam_s + kappa_f)
        rho_sds.data[:, sfy] = 0.5 * (BADER_RHO_F + BADER_RHO_S)
        C66.data[:, sfy] = 0.0  # harmonic of (μ_s, 0)
        C11.data[:, sfy] = harm_lam
        C22.data[:, sfy] = harm_lam
        C12.data[:, sfy] = harm_lam
    elif sds_variant == 'fluid':
        rho_sds.data[:, sfy] = BADER_RHO_F
        C11.data[:, sfy] = kappa_f
        C22.data[:, sfy] = kappa_f
        C12.data[:, sfy] = kappa_f
        C66.data[:, sfy] = 0.0
    else:  # 'solid'
        rho_sds.data[:, sfy] = BADER_RHO_S
        C11.data[:, sfy] = lam_s + 2 * mu_s
        C22.data[:, sfy] = lam_s + 2 * mu_s
        C12.data[:, sfy] = lam_s
        C66.data[:, sfy] = mu_s

    C16.data[:] = 0.0
    C26.data[:] = 0.0

    # Source: Ricker injected into σ_xx + σ_yy (isotropic pressure-like).
    src_coords = np.array([[
        geom.src_x_idx * geom.dx, geom.src_y_idx * geom.dy,
    ]])
    src = SparseTimeFunction(
        name=f'src_sds_{tag}', grid=grid, npoint=1, nt=geom.nt,
        coordinates=src_coords, dtype=grid.dtype,
    )
    src.data[:, 0] = geom.src_data
    dt_sym = grid.stepping_dim.spacing
    src_inj_xx = src.inject(field=sxx.forward, expr=src * dt_sym)
    src_inj_yy = src.inject(field=syy.forward, expr=src * dt_sym)

    # Receivers: 3C (vx, vy, p=-(σxx+σyy)/2) at (rcv_x, rcv_y+offset).
    rcv_y = geom.rcv_y_idx + rcv_y_offset
    rcv_coords = np.array([[
        geom.rcv_x_idx * geom.dx, rcv_y * geom.dy,
    ]])
    rcv_vx = SparseTimeFunction(
        name=f'rcv_vx_sds_{tag}', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords, dtype=grid.dtype,
    )
    rcv_vy = SparseTimeFunction(
        name=f'rcv_vy_sds_{tag}', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords, dtype=grid.dtype,
    )
    rcv_p = SparseTimeFunction(
        name=f'rcv_p_sds_{tag}', grid=grid, npoint=1, nt=geom.nt,
        coordinates=rcv_coords, dtype=grid.dtype,
    )
    p_expr = -0.5 * (sxx + syy)
    rec_eqs = [
        rcv_vx.interpolate(expr=vx),
        rcv_vy.interpolate(expr=vy),
        rcv_p.interpolate(expr=p_expr),
    ]

    fs_eqs = free_surface_eqs(
        {'vx': vx, 'vz': vy, 'sxx': sxx, 'szz': syy, 'sxz': sxy},
        grid, so, formulation='velocity-stress', fs_location='top',
    )

    op = build_ssg_operator(
        grid, vx, vy, sxx, syy, sxy,
        rho_sds, C11, C22, C12, C16, C26, C66,
        geom.damp, name=f'sds_bader_{tag}',
        src_inject=[src_inj_xx, src_inj_yy] + rec_eqs,
        extra_eqs=fs_eqs,
    )
    op.apply(time_m=0, time_M=geom.nt - 1, dt=geom.dt_val)

    rcv_vx_arr = np.asarray(rcv_vx.data).squeeze().copy()
    rcv_vy_arr = np.asarray(rcv_vy.data).squeeze().copy()
    rcv_p_arr = np.asarray(rcv_p.data).squeeze().copy()
    stable = (bool(np.all(np.isfinite(rcv_vx_arr)))
              and bool(np.all(np.isfinite(rcv_vy_arr)))
              and bool(np.all(np.isfinite(rcv_p_arr))))
    return {
        'rcv_vx': rcv_vx_arr,
        'rcv_vy': rcv_vy_arr,
        'rcv_p': rcv_p_arr,
        'stable': stable,
        'dt': geom.dt_val,
        'nt': geom.nt,
        'geom': geom,
        'sds_variant': sds_variant,
    }


def velocity_from_displacement(disp_trace: np.ndarray, dt: float) -> np.ndarray:
    """Centred-difference velocity trace from a displacement trace.

    Returns an array of the same length as ``disp_trace``; the first and
    last samples use forward / backward differences.
    """
    v = np.empty_like(disp_trace)
    v[1:-1] = (disp_trace[2:] - disp_trace[:-2]) / (2.0 * dt)
    v[0] = (disp_trace[1] - disp_trace[0]) / dt
    v[-1] = (disp_trace[-1] - disp_trace[-2]) / dt
    return v


def rms_event_window(
        trace_a: np.ndarray,
        trace_b: np.ndarray,
        dt: float,
        *,
        t_start: float = 0.0,
        t_end: float | None = None,
) -> float:
    """L2 (RMS) error of ``trace_a`` vs ``trace_b`` on the event window.

    Parameters
    ----------
    trace_a, trace_b : ndarray, same shape (nt,)
    dt : float
        Time-step (s).
    t_start, t_end : float
        Window bounds in seconds. ``t_end=None`` means "to the end".

    Returns
    -------
    float : sqrt(mean((a - b)**2)) over the window.
    """
    nt = len(trace_a)
    assert trace_b.shape == trace_a.shape, "trace shape mismatch"
    i_start = max(0, int(np.floor(t_start / dt)))
    i_end = nt if t_end is None else min(nt, int(np.ceil(t_end / dt)) + 1)
    diff = trace_a[i_start:i_end] - trace_b[i_start:i_end]
    if diff.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(diff ** 2)))


def run_sbp_sat_bader_high_resolution_reference(
        out_path: str | Path | None = None,
        *,
        dx: float = 0.0025,
        space_order: int = 8,
        scale: float = 1.0,
        rcv_y_offset: int = -1,
        write_metadata: bool = True,
) -> dict:
    """Generate + save a high-resolution Bader-canonical reference.

    Plan §108.followup-SBP-SAT-graduation Task #52 (2026-05-22).
    Runs Method 5 two-zone SBP-SAT (Bader-implicit) at high resolution
    on the canonical Bader 2023 geometry and saves the 3C receiver
    seismograms as a numpy ``.npz`` file. The saved file serves as
    the proxy SPECFEM2D reference for the Bader Fig 4 self-consistency
    gate (`tests/test_literature_bader_2023.py::test_specfem2d_agreement_canonical_geometry`).

    Parameters
    ----------
    out_path : str or Path, optional
        Where to write the ``.npz``. If ``None``, returns the in-memory
        traces without writing.
    dx : float
        Grid spacing in km. Default 0.0025 km (2.5 m, the high-res
        target). Pair with ``scale=1.0`` for the full Bader canonical
        geometry.
    space_order : int
        Spatial order. Default 8 (production-grade).
    scale : float
        Domain scale factor; see :func:`setup_bader_geometry`. Default
        1.0 (full Bader Fig 4 geometry).
    rcv_y_offset : int
        Receiver vertical offset from the seafloor row in grid cells.
        Default ``-1`` (one cell below seafloor, matching the
        production-resolution comparison convention).
    write_metadata : bool
        If True (default), include git SHA + timestamp + run params in
        the saved ``.npz``.

    Returns
    -------
    dict
        Same structure as :func:`run_sbp_sat_bader_canonical`, plus
        ``out_path`` if a file was written.

    Notes
    -----
    Wall time at production defaults (dx=2.5 m, so=8, scale=1.0) is
    ~30-60 min on Apple Silicon. Lower-fidelity smokes can pass
    ``scale=0.5`` or ``dx=0.005`` for faster iteration. The committed
    reference for the literature gate should use the production
    defaults.
    """
    from bader_2023_canonical import setup_bader_geometry

    geom = setup_bader_geometry(dx=dx, space_order=space_order,
                                  scale=scale)
    result = run_sbp_sat_bader_canonical(
        geom, space_order=space_order,
        coupling_form='bader-implicit',
        rcv_y_offset=rcv_y_offset,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'rcv_phi': result['rcv_phi'],
            'rcv_ux': result['rcv_ux'],
            'rcv_uy': result['rcv_uy'],
            'dt': np.array(result['dt'], dtype=np.float64),
            'nt': np.array(result['nt'], dtype=np.int64),
            'dx': np.array(geom.dx, dtype=np.float64),
            'space_order': np.array(space_order, dtype=np.int64),
            'scale': np.array(scale, dtype=np.float64),
            'rcv_y_offset': np.array(rcv_y_offset, dtype=np.int64),
            'Nx': np.array(geom.Nx, dtype=np.int64),
            'Ny': np.array(geom.Ny, dtype=np.int64),
            'seafloor_y_idx': np.array(geom.seafloor_y_idx,
                                        dtype=np.int64),
            'src_x_idx': np.array(geom.src_x_idx, dtype=np.int64),
            'src_y_idx': np.array(geom.src_y_idx, dtype=np.int64),
            'rcv_x_idx': np.array(geom.rcv_x_idx, dtype=np.int64),
            'rcv_y_idx': np.array(geom.rcv_y_idx, dtype=np.int64),
        }
        if write_metadata:
            import subprocess
            from datetime import datetime, timezone
            try:
                git_sha = subprocess.run(
                    ['git', '-C', str(_REPO), 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
            except Exception:
                git_sha = 'unknown'
            payload['git_sha'] = np.array(git_sha)
            payload['timestamp_utc'] = np.array(
                datetime.now(timezone.utc).isoformat()
            )
        np.savez_compressed(out_path, **payload)
        result['out_path'] = str(out_path)

    return result


def load_reference_npz(path: str | Path) -> dict:
    """Load a reference ``.npz`` written by
    :func:`run_sbp_sat_bader_high_resolution_reference`.

    Returns a dict with native Python scalar values (where applicable)
    plus the trace arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference {path} not found. Generate it via "
            f"`scripts/generate_bader_self_consistency_reference.py`."
        )
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    # Unwrap zero-D scalars.
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.shape == ():
            d[k] = v.item()
    return d


def _resample_trace(trace: np.ndarray, dt_src: float,
                     dt_dst: float, nt_dst: int) -> np.ndarray:
    """Linear-interpolate ``trace`` (sampled at ``dt_src``) onto a
    time grid with ``nt_dst`` samples at spacing ``dt_dst``, starting
    at t=0. Samples past the source range are zero-padded.
    """
    t_src = np.arange(len(trace)) * dt_src
    t_dst = np.arange(nt_dst) * dt_dst
    return np.interp(t_dst, t_src, trace,
                       left=trace[0], right=0.0)


def compare_to_high_resolution_reference(
        production_result: dict,
        reference_npz_path: str | Path,
        *,
        t_start: float = 0.0,
        t_end: float | None = None,
        channels: tuple = ('rcv_phi', 'rcv_ux', 'rcv_uy'),
) -> dict:
    """Compare a production-resolution result against a cached
    high-resolution reference (Bader Fig 4 self-consistency gate).

    The reference is interpolated onto the production-resolution time
    grid before computing the L² (RMS) error on the event window.

    Parameters
    ----------
    production_result : dict
        Output of :func:`run_sbp_sat_bader_canonical` at production
        resolution.
    reference_npz_path : str or Path
        Path to the ``.npz`` written by
        :func:`run_sbp_sat_bader_high_resolution_reference`.
    t_start, t_end : float
        Event window bounds (seconds). ``t_end=None`` uses the full
        simulation time. Bader 2023 Fig 4 uses ~0.0-2.5 s; for the
        ``scale=1.0`` production run that's effectively
        ``t_end=None``.
    channels : tuple of str
        Receiver channels to compare. Defaults to the three SBP-SAT
        traces (phi, ux, uy).

    Returns
    -------
    dict
        ``{channel: {'rms_error': float, 'rms_reference': float,
        'relative_rms': float}}``. The relative-RMS metric is
        ``rms_error / rms_reference`` — the dimensionless number to
        compare against the 5 % self-consistency threshold.
    """
    ref = load_reference_npz(reference_npz_path)
    dt_prod = production_result['dt']
    nt_prod = production_result['nt']
    dt_ref = ref['dt']

    # Normalize-by-dt before comparison.
    #
    # Devito's standard injection pattern ``src.inject(expr=src*dt²/ρ)``
    # produces field values whose peak amplitude scales linearly with
    # ``dt``. The reference (high-res, small dt) and production (low-
    # res, larger dt) therefore have different amplitude scalings even
    # for the SAME physical scenario. Per CLAUDE.md §"Cross-resolution
    # comparison rules" / Marmousi notes: "Normalise shot records by
    # dt: record_normalised = record / dt".
    out = {}
    for chan in channels:
        prod = np.asarray(production_result[chan]).squeeze() / dt_prod
        ref_trace = np.asarray(ref[chan]).squeeze() / dt_ref
        # Resample reference onto production time grid.
        ref_resampled = _resample_trace(ref_trace, dt_ref, dt_prod,
                                          len(prod))
        rms_err = rms_event_window(prod, ref_resampled, dt_prod,
                                     t_start=t_start, t_end=t_end)
        # Reference RMS over same window as the denominator.
        i_start = max(0, int(np.floor(t_start / dt_prod)))
        i_end = (len(prod) if t_end is None
                  else min(len(prod), int(np.ceil(t_end / dt_prod)) + 1))
        rms_ref = float(np.sqrt(np.mean(
            ref_resampled[i_start:i_end] ** 2
        ))) if i_end > i_start else float('nan')
        rel = rms_err / rms_ref if rms_ref > 0 else float('nan')
        out[chan] = {
            'rms_error': rms_err,
            'rms_reference': rms_ref,
            'relative_rms': rel,
        }
    return out
