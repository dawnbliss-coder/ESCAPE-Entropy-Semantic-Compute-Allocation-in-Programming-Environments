"""Check the built PDFs with official ACL pubcheck and independent PDF checks.

Requires the official aclpubcheck environment plus Poppler's pdffonts. Does not
upload the paper or disable any default pubcheck checks. Fails on nonempty JSON,
not just the official CLI exit code (which can be zero when errors are found).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
VALIDATION = HERE / "validation"


def check_pdf(path: Path, *, review: bool) -> dict:
    with pdfplumber.open(path) as pdf:
        texts = [page.extract_text() or "" for page in pdf.pages]
        assert all(abs(p.width - 595.276) < 1 and abs(p.height - 841.89) < 1 for p in pdf.pages), "Non-A4 page"
        assert len(texts) > 0
        limitations_page = next(i + 1 for i, text in enumerate(texts) if re.search(r"\bLimitations\b", text))
        # A first-line Limitations heading on page 9 also satisfies eight body pages.
        first_lines = texts[limitations_page - 1].splitlines()
        assert limitations_page <= 8 or (limitations_page == 9 and "Limitations" in first_lines[0]), "Body exceeds eight pages"
        text = "\n".join(texts)
        # pdfplumber can drop inter-word spaces; compare identities whitespace-insensitively.
        compact = re.sub(r"\s+", "", text)
        assert "References" in text
        assert not any(x in text for x in ("STUDENT TEXT REQUIRED", "TODO", "??", "[?]")), "Unresolved placeholder/reference"
        assert "SYNTHETICPIPELINETEST" not in compact, "Built from synthetic test inputs, not real results"
        assert "MISSING-RESULT" not in compact, "A result macro had no data; see generated/evidence.json missing_macros"
        if review:
            assert "AnonymousACLsubmission" in compact
            assert not any(x in compact for x in ("PriyankaAgarwal", "ShashwatMukadam", "DivyanshAtri", "2024113001")), "Review PDF contains author identities"
        else:
            assert all(x in compact for x in ("PriyankaAgarwal", "ShashwatMukadam", "DivyanshAtri")), "Missing authors"
    fonts = subprocess.check_output(["pdffonts", str(path)], text=True)
    (VALIDATION / f"fonts-{path.stem}.txt").write_text(fonts)
    rows = fonts.splitlines()[2:]
    assert rows, "No PDF fonts found"
    for row in rows:
        flags = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        assert flags and flags.group(1) == "yes", f"Unembedded/unrecognized font: {row}"
        assert "Type 3" not in row, f"Type 3 font: {row}"
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pages": len(texts), "limitations_start_page": limitations_page,
            "eight_page_body_check": "pass", "a4": True, "all_fonts_embedded": True,
            "type3_fonts": False, "anonymous_review": review}


def main() -> None:
    VALIDATION.mkdir(exist_ok=True)
    (VALIDATION / "report-checks.json").unlink(missing_ok=True)
    named = HERE / "Trimax-Final-Report.pdf"
    review = HERE / "Trimax-Final-Report-review.pdf"
    checks = [check_pdf(named, review=False), check_pdf(review, review=True)]
    report = VALIDATION / "errors-Trimax-Final-Report.json"
    # Stale output must not masquerade as a successful fresh check.
    if report.exists():
        report.unlink()
    command = [sys.executable, "-m", "aclpubcheck", "--paper_type", "long", str(named)]
    result = subprocess.run(command, cwd=VALIDATION, text=True, capture_output=True, timeout=180)
    (VALIDATION / "pubcheck.txt").write_text("Command: " + repr(command) + "\n" + result.stdout + result.stderr)
    assert result.returncode == 0, "Official pubcheck failed; see validation/pubcheck.txt"
    assert report.exists(), "Official pubcheck did not produce its JSON report"
    findings = json.loads(report.read_text())
    assert findings == {}, f"Official pubcheck findings: {findings}"
    assert "All Clear!" in result.stdout, "Missing official success message"
    result_record = {"status": "pass", "official_pubcheck": "All Clear!", "paper_type": "long",
                     "default_checks_disabled": [], "findings": findings,
                     "bibliography_service": "not invoked by upstream default CLI; records checked against primary sources",
                     "pdfs": checks}
    (VALIDATION / "report-checks.json").write_text(json.dumps(result_record, indent=2) + "\n")
    print("PASS: official ACL pubcheck All Clear!; empty findings JSON; A4, embedded fonts and eight-page body verified.")


if __name__ == "__main__":
    main()
