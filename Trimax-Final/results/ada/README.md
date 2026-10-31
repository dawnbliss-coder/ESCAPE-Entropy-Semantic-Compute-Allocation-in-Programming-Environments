# Retrieved Ada runs

Complete remote directory retrieved on 2026-09-28 from
`ada:/home2/divyansh.atri/escape-mid-c-20260927/results/ada`.
The laptop project remains intact. This directory contains **synthetic software
validation**, not new real BLT inference results.

| Job | Status | Meaning |
|---|---|---|
| 2719640 | COMPLETED, exit 0 | User-approved isolated pytest/SciPy setup |
| 2719643 | FAILED, exit 1 | Tests, calibration and scoring finished; O5 JSON export failed on a NumPy boolean |
| 2719654 | COMPLETED, exit 0 | Corrected final run: 170 passed, 1 skipped; scoring and O5 exports complete |

Use `synthetic-2719654/` for final results. It includes input JSONL/text fixtures,
calibration record, evaluation JSON/CSV, O5 region table, design matrix and
coefficient output. Use `c-2719654.log` for the exact test and experiment summary.

The failed run is preserved for diagnosis. Its `o5/o5.json` is partial and must
not be used as a valid result. The serialization fix and a regression test are
in Workstream C; no producer code was changed. The two scoring JSON files are
byte-identical. The skipped test requires the unavailable `transformers` package;
no GPU/model test is claimed. O5 inference is unavailable because the synthetic
design is rank deficient.

Remote job status and retrieval SHA256 inventory were recorded under
`reports/mid_submission/validation/` and `reports/mid_submission/evidence/`
during preparation; that trail was trimmed from the repo after the report was
submitted, leaving only the submitted PDFs there.

The completed real-data C-v1 run is now retained in `../final/c-v1-2721954/`
and is the source of the mid-submission report. These older synthetic outputs
remain regression fixtures, not scientific results.
