"""Source PDF provenance gate.

Pins the SHA256 of the committed `yang2015.pdf` so any silent
PDF replacement is detected at CI time. Per pre-flight dual-reviewer
finding YF1 (Codex Q2 + Gemini Q2 both DISAGREE on initial gate set
being insufficient).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXPECTED_SHA256 = "72eeef5f063b4e3d70b6e2446fb5ce9a64a69d1d652e99b89fe13d93038e8741"


def test_yang2015_pdf_sha256_matches_pin():
    """SHA256 of the committed yang2015.pdf must match the pinned
    value. Pinned 2026-05-27 from the user's
    `~/projects/reviews/papers/yang2015__*.pdf` copy.
    """
    pdf_path = HERE / "yang2015.pdf"
    assert pdf_path.exists(), f"yang2015.pdf missing at {pdf_path}"
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA256, (
        f"PDF SHA256 mismatch — got {h}, expected {EXPECTED_SHA256}. "
        "Either the PDF was replaced (investigate provenance) or the "
        "pin needs updating (update EXPECTED_SHA256 + note in plan)."
    )
