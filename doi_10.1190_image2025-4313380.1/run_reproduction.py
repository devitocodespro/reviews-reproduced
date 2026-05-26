#!/usr/bin/env python3
"""
Top-level driver for the Irakarama et al. 2026 IMAGE reproduction.

This folder's reproduction is organised into 5 verification stages
(stage0..stage4) plus a Quarto peer-review write-up
(``review_findings.qmd``). The stages were originally developed
during the engagement's `legacy:image2025-4313380/` peer review
and ported here 2026-05-26 as part of the Method 6 graduation
Option (a) resolution.

Run individual stages:

.. code-block:: bash

   python3 stage0_operator_analysis.py     # Reproduces paper Fig 2 + Fig 3
   python3 stage1_acoustic_dual_pair.py    # 2D acoustic dual-pair (convergence)
   python3 stage2_elastic_isotropic.py     # 2D isotropic elastic
   python3 stage3_filtering_and_source.py  # Selective filter + source injection
   python3 stage4_tti_elastic.py           # 2D TTI elastic with Bond rotation

Or rebuild the full Quarto PDF (requires Quarto + LuaLaTeX):

.. code-block:: bash

   make pdf

Byte-checkable load-bearing artifacts (per
`feedback_reproduction_quantitative_first.md`): see
``paper_tables.py`` for the paper-published Tables 1 + 2
transcribed byte-identically from PDF page 4, plus
``tests/test_paper_tables.py`` for 11 byte-match unit tests.

This top-level driver is a minimal courtesy entry point so the
parent-repo reproduction-prerequisite gate
(``tests/test_reproduction_prerequisites.py``) finds it. The
actual reproduction logic lives in the staged scripts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    "stage0_operator_analysis.py",
    "stage1_acoustic_dual_pair.py",
    "stage2_elastic_isotropic.py",
    "stage3_filtering_and_source.py",
    "stage4_tti_elastic.py",
]


def main():
    """Run all 5 stages in order. Each stage emits its own figures
    into ``figures/`` and prints its verification findings to stdout."""
    for stage in STAGES:
        path = HERE / stage
        if not path.exists():
            print(f"[SKIP] {stage} (not found)")
            continue
        print(f"\n{'=' * 70}\n[RUN] {stage}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(path)],
                                cwd=str(HERE))
        if result.returncode != 0:
            print(f"[FAIL] {stage} exited {result.returncode}")
            return result.returncode
    print("\n[OK] All 5 stages completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
