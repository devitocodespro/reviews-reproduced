# Dolci et al. (2022) — ABC Methods Comparison for Acoustic FWI

**Status**: stub. Salvage source: `legacy` branch of
`devitocodes/reviews` at `gmd-2022-48/` (already partially
reproduced).

## Paper

> Dolci, D. S., Cuesta, R. F., Igreja, I., dos Santos, R. M.,
> Mello, V. M., Roberty, N. C., & Cossich, W. (2022). Comparison
> of ABC methods for the acoustic wave equation in the context of
> FWI. *Geoscientific Model Development*, 15, 5857–5881.
> DOI: 10.5194/gmd-15-5857-2022.

(Folder uses `gmd-2022-48` rather than the DOI itself because the
paper's preprint identifier is the canonical reference in the
parent repo's history.)

## Reference implementation

`devitocodes/reviews` legacy branch `gmd-2022-48/` — contains
working Devito + NumPy implementations of 5 ABC methods (Damping /
Cerjan, PML, CPML, HABC-A1, HABC-Higdon) on the Marmousi velocity
model. Includes a `verification_report.qmd` reproducing the paper's
main comparison.

## Migration

This folder bootstraps from the salvage:

```bash
# From parent repo `legacy` branch:
git checkout legacy
cp -r gmd-2022-48/* /Users/ggorman/projects/reviews-reproduced/doi_gmd-2022-48/
```

Then refactor:
- Adopt `pyproject.toml` + `uv.lock` per the reproduction standard.
- Re-organise tests into `tests/`.
- Extract reference outputs to `reference_outputs/`.

## TODO

- [ ] Salvage from `legacy` branch
- [ ] Adopt `pyproject.toml` with pinned Devito
- [ ] Refactor `tests/` per repo convention
- [ ] Re-render `verification_report.qmd` as `report.qmd`
- [ ] Graduation review
