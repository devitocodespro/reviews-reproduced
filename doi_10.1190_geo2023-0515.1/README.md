# Caunt, Nelson, Luporini & Gorman (2024) — Immersed Boundary for Irregular Topography

**Status**: stub. Special-case dependency pinning required (Schism +
specific Devito branch).

## Paper

> Caunt, E., Nelson, R., Luporini, F., & Gorman, G. J. (2024). A
> novel immersed boundary approach for irregular topography with
> acoustic wave equations. *GEOPHYSICS*, 89(4), T207–T220.
> DOI: [10.1190/geo2023-0515.1](https://doi.org/10.1190/geo2023-0515.1)

## Reference implementation

- [`github.com/devitocodes/schism`](https://github.com/devitocodes/schism)
  — Schism IBM utilities (MIT, ~12 Python files, last push
  2025-08-28). Builds on Ed Caunt's Imperial PhD thesis.

## Special dependency considerations

Schism may require a specific Devito branch (mainline-compatible
commit + Schism-tested commit). This folder's `pyproject.toml`
should pin BOTH:

```toml
dependencies = [
    "devito @ git+https://github.com/devitocodes/devito.git@<sha>",
    "schism @ git+https://github.com/devitocodes/schism.git@<sha>",
]
```

The parent `devitocodes/reviews` repo will not import from this
folder (per standalone discipline), so its Devito pin is
independent.

## Key feature

Per-ghost-stencil local polynomial fit through fluid points +
boundary constraint points; small dense system solved for modified
stencil weights. Preserves high-order accuracy (4th/8th) near
irregular topography without mesh generation or curvilinear
coordinates.

## TODO

- [ ] `pyproject.toml` with Devito + Schism pins
- [ ] Identify a representative topography test case from the paper
- [ ] `run_reproduction.py` reproducing paper Figs 4-7
- [ ] `tests/test_reproduction.py` — order-of-accuracy preservation
      near boundary; Vandermonde solve byte-check on simple geometry
- [ ] Graduation review (note: difficult to graduate to `published`
      faithful without verifying against Caunt's own outputs)
