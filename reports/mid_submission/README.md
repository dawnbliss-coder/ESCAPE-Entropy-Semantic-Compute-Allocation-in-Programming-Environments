# ACL mid-submission report

The submitted mid report is `reports/Trimax-Mid-Report.pdf` (8 pages, references on page 8). The named and anonymous build outputs (`Trimax-Mid-Submission.pdf`, `-review.pdf`) and the Overleaf source zip were retired from the repo in commit `bf7bbbd`; rebuild them from the sources here (`Trimax-Mid-Submission.tex`, `Trimax-Mid-Submission-review.tex`). The final report is in `reports/final_submission/`.

## Rebuild

From the repository root, regenerate tables, evidence, and the depth figure from the retained results (requires matplotlib):

```bash
python3 reports/mid_submission/build_results.py --run results/final/c-v1-2721954
bash reports/mid_submission/build.sh
```

The build uses pdfLaTeX and BibTeX, or set `TECTONIC_BIN=/path/to/tectonic`. The TeX installation needs Times fonts, lineno, microtype, and adjustbox. The script installs no packages. A first Tectonic build may fetch its standard TeX resources.

Check the resulting PDFs in an environment with `pdfplumber`, the official `aclpubcheck` package, and Poppler's `pdffonts`:

```bash
python reports/mid_submission/check_report.py
```

The checker runs official ACL pubcheck on the named PDF without disabling default margin/font checks, requires an empty findings JSON, and checks the body length, A4 pages, embedded fonts, author identities, and missing-result/placeholders. The review PDF is checked separately for anonymity and page layout. Results and PDF hashes are recorded in `validation/report-checks.json`.

## Evidence

- `generated/results.tex`: report values and tables generated from real results.
- `generated/evidence.json`: result provenance, input hashes, and missing-macro checks.
- `generated/depth_margin.pdf`: depth analysis figure.
- `evidence/acl-style-provenance.json` and `evidence/bibliography-provenance.json`: style and citation provenance, written by `build_evidence.py` into `evidence/`, which is not kept in git.
- `results/final/c-v1-2721954/SHA256SUMS` (repository root): checksums for retained experimental outputs.
- `results/ada/` (repository root): earlier synthetic software validation, kept separate from the real results.

The result generator refuses synthetic input by default. All expected report macros must be available; the PDF checker rejects synthetic watermarks and missing results. The report describes O5 review and further analyses as remaining work.

## Course requirements

This is an ACL-formatted course report (the submitted version has a 7-page body and 8 pages overall). The course-specific rubric was not supplied, so formatting validation does not establish compliance with its page limit or required sections.
