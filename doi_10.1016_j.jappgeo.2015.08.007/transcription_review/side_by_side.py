"""Side-by-side transcription review PDF generator (v2 — LaTeX rendering).

Implements the user-mandated transcription protocol
(`feedback_transcription_workflow_arxiv_then_side_by_side`):
for each table/equation that cannot be sourced from arXiv/HAL
LaTeX, generate a 2-panel PDF showing:
  - LEFT: cut-out of the original paper region (PDF → PNG via
    pdftocairo)
  - RIGHT: my rendered transcription (proper LaTeX via pdflatex
    → PNG, NOT matplotlib mathtext, so matrices align properly)

The user reviews the PDF before the transcribed value is
committed to `paper_tables.py`.

v2 changes (2026-05-27 user feedback): use full LaTeX via
`pdflatex` to render the transcription with properly-aligned
matrices and equations. The previous matplotlib mathtext renderer
mangles bmatrix / aligned environments.

Usage:
    python side_by_side.py --paper /tmp/lombard_piraux_2004.pdf \\
        --paper-page 8 --cover-pages 1 \\
        --transcription-file lp04_eq10_11_transcription.tex \\
        --output /tmp/transcription_review_lp04_eq10_11_v3.pdf

The transcription file should be a LaTeX SNIPPET (the body of a
document), not a full document. The script wraps it in a minimal
LaTeX template before compiling.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


LATEX_TEMPLATE = r"""
\documentclass[12pt]{article}
\usepackage[a4paper, margin=1.5cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{xcolor}
\usepackage{enumitem}
\setlist[itemize]{leftmargin=*}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
%s
\end{document}
"""


def extract_pdf_region(pdf_path: Path, page: int,
                       crop_bbox: tuple[int, int, int, int] | None,
                       dpi: int = 300) -> Path:
    """Extract a region from a PDF page as a PNG.

    crop_bbox: (left, top, width, height) in points; None = full page.
    Returns the path to the rendered PNG.
    """
    out_dir = Path(tempfile.mkdtemp(prefix='side_by_side_'))
    out_prefix = out_dir / 'page'
    cmd = ['pdftocairo', '-png', '-r', str(dpi),
           '-f', str(page), '-l', str(page),
           '-singlefile']
    if crop_bbox:
        left, top, width, height = crop_bbox
        cmd.extend(['-x', str(left), '-y', str(top),
                    '-W', str(width), '-H', str(height)])
    cmd.extend([str(pdf_path), str(out_prefix)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'pdftocairo stderr: {result.stderr}', file=sys.stderr)
        raise RuntimeError(f'pdftocairo failed: {result.returncode}')
    png_path = out_prefix.with_suffix('.png')
    if not png_path.exists():
        raise RuntimeError(f'expected PNG not produced: {png_path}')
    return png_path


def extract_pdf_pages_stitched(pdf_path: Path, pages: list[int],
                                dpi: int = 300) -> Path:
    """Extract multiple PDF pages and stitch them vertically into one PNG.

    Used for transcription regions that span page breaks.
    """
    import numpy as np
    out_dir = Path(tempfile.mkdtemp(prefix='side_by_side_multi_'))
    page_pngs = []
    for p in pages:
        prefix = out_dir / f'page_{p}'
        cmd = ['pdftocairo', '-png', '-r', str(dpi),
               '-f', str(p), '-l', str(p), '-singlefile',
               str(pdf_path), str(prefix)]
        subprocess.run(cmd, check=True, capture_output=True)
        page_pngs.append(prefix.with_suffix('.png'))
    imgs = [mpimg.imread(p) for p in page_pngs]
    max_w = max(img.shape[1] for img in imgs)
    padded = []
    for img in imgs:
        if img.shape[1] < max_w:
            pad_w = max_w - img.shape[1]
            if img.ndim == 3:
                pad = np.ones((img.shape[0], pad_w, img.shape[2]),
                              dtype=img.dtype)
            else:
                pad = np.ones((img.shape[0], pad_w), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)
    stitched = np.concatenate(padded, axis=0)
    out_png = out_dir / 'stitched.png'
    mpimg.imsave(out_png, stitched)
    return out_png


def render_latex_transcription(latex_body: str, dpi: int = 300) -> Path:
    """Compile a LaTeX snippet to PNG via pdflatex + pdftocairo.

    The snippet is wrapped in a minimal article template with
    amsmath/amssymb/bm preloaded. Returns the path to the PNG.
    """
    out_dir = Path(tempfile.mkdtemp(prefix='latex_render_'))
    tex_path = out_dir / 'transcription.tex'
    tex_path.write_text(LATEX_TEMPLATE % latex_body)
    # Compile (run twice for any cross-refs, though typical
    # transcription snippets don't have any)
    for _ in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode',
             '-halt-on-error', '-output-directory', str(out_dir),
             str(tex_path)],
            capture_output=True, text=True,
        )
    pdf_path = tex_path.with_suffix('.pdf')
    if not pdf_path.exists():
        print(f'pdflatex stdout tail:', file=sys.stderr)
        print(result.stdout[-2000:], file=sys.stderr)
        raise RuntimeError('pdflatex did not produce a PDF; '
                            'check for LaTeX syntax errors')
    # Convert PDF page 1 to PNG
    png_prefix = out_dir / 'render'
    subprocess.run(
        ['pdftocairo', '-png', '-r', str(dpi),
         '-f', '1', '-l', '1', '-singlefile',
         str(pdf_path), str(png_prefix)],
        check=True, capture_output=True,
    )
    return png_prefix.with_suffix('.png')


def render_side_by_side(paper_image: Path,
                        transcription_image: Path,
                        title: str,
                        output: Path,
                        figsize: tuple[float, float] = (18, 11)) -> Path:
    """Combine the paper image + my transcription image into a 2-panel PDF."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    img_l = mpimg.imread(paper_image)
    ax_l.imshow(img_l)
    ax_l.set_title('PAPER (original)', fontsize=12, fontweight='bold')
    ax_l.axis('off')

    img_r = mpimg.imread(transcription_image)
    ax_r.imshow(img_r)
    ax_r.set_title('MY TRANSCRIPTION (LaTeX-rendered)',
                   fontsize=12, fontweight='bold')
    ax_r.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paper', type=Path, required=True)
    # New: --paper-page (printed page number) + --cover-pages
    # (front-matter offset) replace the raw --page integer.
    ap.add_argument('--paper-page', type=int,
                    help='Paper-printed page number (typically what the paper cites)')
    ap.add_argument('--cover-pages', type=int, default=0,
                    help='Number of front-matter pages before paper page 1 '
                         '(HAL cover = 1, Zotero/published PDFs typically 0)')
    ap.add_argument('--page', type=int,
                    help='[legacy] Physical PDF page (use --paper-page + '
                         '--cover-pages instead)')
    ap.add_argument('--paper-pages', type=str, default=None,
                    help='Comma-separated paper-printed page numbers (e.g. '
                         '"13,14") for transcription regions spanning a '
                         'page break. Combined with --cover-pages.')
    ap.add_argument('--crop-bbox', type=str, default=None,
                    help='left,top,width,height in points (optional)')
    ap.add_argument('--transcription-file', type=Path, required=True,
                    help='File with LaTeX snippet (body only, will be '
                         'wrapped in a minimal article template)')
    ap.add_argument('--title', type=str, default='Transcription review')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    # Resolve PDF physical page(s)
    physical_pages = None
    if args.paper_pages is not None:
        paper_pages = [int(p) for p in args.paper_pages.split(',')]
        physical_pages = [p + args.cover_pages for p in paper_pages]
        physical_page = physical_pages[0]  # for legacy single-page path
    elif args.paper_page is not None:
        physical_page = args.paper_page + args.cover_pages
    elif args.page is not None:
        physical_page = args.page
    else:
        ap.error('Must specify --paper-page (with optional --cover-pages) '
                 'OR --paper-pages OR --page')

    crop_bbox = None
    if args.crop_bbox:
        parts = [int(p) for p in args.crop_bbox.split(',')]
        if len(parts) != 4:
            ap.error('--crop-bbox needs 4 ints (left,top,width,height)')
        crop_bbox = tuple(parts)

    latex_body = args.transcription_file.read_text()

    if physical_pages is not None and len(physical_pages) > 1:
        print(f'Extracting paper pages (physical PDF pages '
              f'{physical_pages})...', file=sys.stderr)
        paper_image = extract_pdf_pages_stitched(args.paper, physical_pages)
    else:
        print(f'Extracting paper page (physical PDF page '
              f'{physical_page})...', file=sys.stderr)
        paper_image = extract_pdf_region(args.paper, physical_page, crop_bbox)
    print(f'  → {paper_image}', file=sys.stderr)

    print(f'Compiling LaTeX transcription...', file=sys.stderr)
    if not shutil.which('pdflatex'):
        raise RuntimeError(
            'pdflatex not found — install TeX Live or MacTeX. '
            'Fallback to matplotlib mathtext requires reverting to v1.')
    transcription_image = render_latex_transcription(latex_body)
    print(f'  → {transcription_image}', file=sys.stderr)

    output = render_side_by_side(
        paper_image=paper_image,
        transcription_image=transcription_image,
        title=args.title,
        output=args.output,
    )
    print(f'\nSide-by-side PDF: {output}', file=sys.stderr)

    try:
        subprocess.run(['open', str(output)], check=False)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()
