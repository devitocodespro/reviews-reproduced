"""Pratt (1990) 2D acoustic Green's image-method reproduction package.

Primary paper:
    Pratt, R. G. (1990). "Frequency-domain elastic wave modeling by
    finite differences; a tool for crosshole seismic imaging."
    *Geophysics* 55(5): 626-632. DOI: 10.1190/1.1442874.

The Pratt 1990 paper is the seismic-modelling community's reference
for line-source projection from 3D to 2D Green's functions — the
half-derivative-in-time relationship that turns the 3D point-source
Green's function `G_3D(r,t) = δ(t − r/c) / (4πr)` into the 2D
line-source equivalent

    G_2D(r, t) = H(t − r/c) / (2π √(t² − r²/c²)).

The H(·) is the Heaviside step function. The integrable r → r/c
square-root singularity at the wavefront is the 2D acoustic
"tail" — the wavefield rings after the wavefront passes (unlike 3D
where it's an impulsive front).

This package provides the production NumPy implementation:
- Direct-wave snapshot via convolution of G_2D with a source
  wavelet (variable change τ = t − √(u² + r²/c²) regularises the
  √-singularity);
- Image-method dipping-interface reflection (mirror source across
  the interface; acoustic-acoustic angle-dependent reflection
  coefficient R(θ) with post-critical evanescent handling);
- Normal-incidence acoustic impedance reflection.

Used as the V1 verification gate's homogeneous-Green's reference
in `tests/test_dwn_homogeneous.py` and as the image-method
reference in `scripts/run_acoustic_image_reference.py`.

Modules:
- `analytical_acoustic_2d`: NumPy production implementation
  (`acoustic_2d_green_snapshot`, `mirror_source_across_dipping_line`,
  `angle_dependent_acoustic_R`, `acoustic_image_method_wavefield`,
  `acoustic_image_method_wavefield_angle_dependent`,
  `acoustic_impedance_reflection`, `ricker_wavelet`).
- `paper_tables`: hand-transcribed paper anchors:
    * G_2D(r, t) closed-form sentinel (line-source projection
      vs Hankel-frequency-domain alternative);
    * Heaviside causality (no signal before wavefront arrival);
    * angle-dependent acoustic R(p) form
      R = (Z2 cos θ1 − Z1 cos θ2) / (Z2 cos θ1 + Z1 cos θ2);
    * Normal-incidence reflection R = (Z2 − Z1) / (Z2 + Z1);
    * Post-critical |R| = 1 sentinel for evanescent regime.

Phase Y/2g 2026-05-28: relocated from
`00_common/analytical_acoustic_2d.py` into this reproduction
folder per the paper-faithful-modules curation sweep; wired to
the parent via `[tool.uv.sources]` editable-dep.
"""
