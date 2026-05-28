"""Source PDF provenance gate.

Pins the SHA256 of all three committed PDFs:
  - yang2015.pdf — paper being reproduced
  - antecedents_liu_2014.pdf — LS-method antecedent
  - antecedents_yang_yan_liu_2015_geophys_prospect_sa.pdf — SA-method
    antecedent

Per pre-flight dual-reviewer YF1 (Codex Q2 + Gemini Q2 DISAGREE on
initial gate set insufficient) + post-graduation Codex Q4 hygiene
recommendation (add SHA pins for antecedent PDFs).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent

EXPECTED_SHAS = {
    "yang2015.pdf":
        "72eeef5f063b4e3d70b6e2446fb5ce9a64a69d1d652e99b89fe13d93038e8741",
    "antecedents_liu_2014.pdf":
        "16506606cd827672b465f7bc988948eb2cd42cd710b2a71d37ce26926a2bfa33",
    "antecedents_yang_yan_liu_2015_geophys_prospect_sa.pdf":
        "9a87c13fa21b7ffe5792f4a24073300b1aaa975628875ce2de8759ad92cdd54b",
}


@pytest.mark.parametrize("filename", sorted(EXPECTED_SHAS.keys()))
def test_pdf_sha256_matches_pin(filename: str):
    """SHA256 of each committed PDF must match the pinned value.
    Catches silent PDF replacement at CI time.
    """
    pdf_path = HERE / filename
    expected = EXPECTED_SHAS[filename]
    assert pdf_path.exists(), f"{filename} missing at {pdf_path}"
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert h == expected, (
        f"{filename} SHA256 mismatch — got {h}, expected {expected}. "
        "Either the PDF was replaced (investigate provenance) or the "
        "pin needs updating."
    )
