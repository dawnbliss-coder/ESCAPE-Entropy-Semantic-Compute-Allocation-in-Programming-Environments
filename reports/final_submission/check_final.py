"""Check the built final report with Poppler (pdftotext, pdffonts, pdfinfo).

Fails if: the body before References is longer than 8 pages; any MISSING result marker,
unresolved reference (??) or synthetic watermark is present; the page is not A4; or a font
is not embedded / is Type 3. Writes validation/report-checks.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE / "Trimax-Final.pdf"
VALIDATION = HERE / "validation"


def main() -> None:
    VALIDATION.mkdir(exist_ok=True)
    info = subprocess.check_output(["pdfinfo", str(PDF)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    size = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+)", info, re.M)
    assert abs(float(size.group(1)) - 595.276) < 1 and abs(float(size.group(2)) - 841.89) < 1, "Non-A4 page"
    texts = [subprocess.check_output(["pdftotext", "-f", str(p), "-l", str(p), "-layout", str(PDF), "-"], text=True)
             for p in range(1, pages + 1)]
    # Two-column layout: the heading may share its line with the right column's first entry.
    references_page = next(i + 1 for i, t in enumerate(texts) if re.search(r"^\s*References(\s{3,}|\s*$)|\S\s{3,}References\s*$", t, re.M))
    body_pages = references_page - 1 + (0 if texts[references_page - 1].split("References")[0].strip() == "" else 1)
    assert references_page <= 9, f"References start on page {references_page}; the body exceeds 8 pages"
    text = "\n".join(texts)
    compact = re.sub(r"\s+", "", text)
    assert "MISSING" not in compact, "A result macro had no data; see generated/final_evidence.json"
    assert "??" not in text and "[?]" not in text, "Unresolved reference or citation"
    assert "SYNTHETICPIPELINETEST" not in compact, "Built from synthetic test inputs"
    assert all(x in compact for x in ("PriyankaAgarwal", "ShashwatMukadam", "DivyanshAtri")), "Missing authors"
    fonts = subprocess.check_output(["pdffonts", str(PDF)], text=True)
    (VALIDATION / "fonts-Trimax-Final.txt").write_text(fonts)
    for row in fonts.splitlines()[2:]:
        flags = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        assert flags and flags.group(1) == "yes", f"Unembedded font: {row}"
        assert "Type 3" not in row, f"Type 3 font: {row}"
    record = {"status": "pass", "pdf": PDF.name, "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
              "pages": pages, "references_start_page": references_page, "body_pages_upper_bound": body_pages,
              "a4": True, "all_fonts_embedded": True}
    (VALIDATION / "report-checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {pages} pages, references start on page {references_page}, A4, fonts embedded, no MISSING markers.")


if __name__ == "__main__":
    main()
