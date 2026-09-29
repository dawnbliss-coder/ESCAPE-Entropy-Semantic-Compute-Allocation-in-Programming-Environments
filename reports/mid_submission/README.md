# ACL mid-submission report

Submit **Trimax-Mid-Submission.pdf**, the named Team Trimax report. It contains the completed real-data C-v1 analysis from `results/final/c-v1-2721954/`. The anonymous `-review.pdf` is an alternative for blind review only.

`Trimax-Mid-Submission-source.zip` is a self-contained LaTeX/Overleaf package. Set `Trimax-Mid-Submission.tex` as the main document and select pdfLaTeX. The official ACL style and bibliography style are included.

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
- `evidence/acl-style-provenance.json` and `evidence/bibliography-provenance.json`: style and citation provenance.
- `results/final/c-v1-2721954/SHA256SUMS` (repository root): checksums for retained experimental outputs.
- `results/ada/` (repository root): earlier synthetic software validation, kept separate from the real results.

The result generator refuses synthetic input by default. All expected report macros must be available; the PDF checker rejects synthetic watermarks and missing results. The report describes O5 review and further analyses as remaining work.

## Course requirements

This is an ACL-formatted course report with a four-page body and five pages overall. The course-specific rubric was not supplied, so formatting validation does not establish compliance with its page limit or required sections.

The report discloses AI assistance. Earlier project instructions recorded that AI-written academic report prose was prohibited by the course. The authors must verify the applicable authorship policy and review the manuscript; if that restriction applies, it cannot be submitted as written. No upload has been performed.
