"""Liu (2014) GJI optimal staggered-grid FD coefficients reproduction package.

Primary paper:
    Liu, Y. (2014). "Optimal staggered-grid finite-difference schemes
    based on least-squares for wave equation modelling."
    *Geophysical Journal International* 197(2): 1033-1047.
    DOI: 10.1093/gji/ggu032.

Precursor:
    Liu, Y. (2013). "Globally optimal finite-difference schemes based
    on least squares." *Geophysics* 78(4): T113-T132.
    DOI: 10.1190/geo2012-0480.1.

Equivalent at the Taylor-truncation order:
    Fornberg, B. (1988). "Generation of finite difference formulas
    on arbitrarily spaced grids." *Math. Comp.* 51(184): 699-706.
    DOI: 10.1090/S0025-5718-1988-0935077-0.

Downstream consumer:
    Jiang, L. & Zhang, W. (2021). "TTI equivalent medium
    parametrization method for the seismic waveform modelling of
    heterogeneous media with coarse grids." *Geophysical Journal
    International* 227(3): 2016-2043. DOI: 10.1093/gji/ggab310.
    (Cites Liu 2014's optimal weights in the line-179 footnote.)

This package collects the general-purpose optimal staggered-grid FD
coefficient solver shared by the parent repo's coarse-grid TTI
methods. The Liu 2014 Eq 13 closed-form Taylor-expansion (TE)
weights are byte-equivalent to Fornberg 1988 evaluated at the
half-grid point; the LS-optimised relative and absolute weights are
the Liu 2014 contribution proper.

Modules:
- `optimal_fd_coefficients`:
    * `taylor_staggered_coeffs(order)` — Liu 2014 Eq 13 closed-form
      TE weights at orders {2, 4, 6, 8, 10, 12, …}.
    * `liu_2014_relative_coeffs(order, b)` — LS minimising relative
      dispersion error over [0, b] (Liu §2.4 Eq 12 + 17).
    * `liu_2014_absolute_coeffs(order, b)` — LS minimising absolute
      dispersion error (Liu §2.3 Eq 12).
    * `optimal_staggered_fd_coeffs(order, method, b)` — dispatcher.
    * `modified_wavenumber(c_array, beta)` — k_h evaluator.
    * `dispersion_error(c_array, beta, kind)` — relative/absolute
      dispersion error vs ideal.
- `paper_tables`: hand-transcribed Liu 2014 / Fornberg 1988 byte
  anchors as exact `fractions.Fraction` weights for orders 2-8.

Phase Y/2f 2026-05-28: relocated from
`00_common/optimal_fd_coefficients.py` into this reproduction folder
per the paper-faithful-modules curation sweep; wired to the parent
via `[tool.uv.sources]` editable-dep.
"""
