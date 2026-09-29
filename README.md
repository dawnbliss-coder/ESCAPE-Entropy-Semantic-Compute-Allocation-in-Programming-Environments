# Project ESCAPE — Team Trimax

Entropy–Semantic Compute Allocation in Programming Environments. ANLP mid-submission by Priyanka Agarwal, Shashwat Mukadam, and Divyansh Atri.

## Submission

- **Report:** [Trimax-Mid-Submission.pdf](reports/mid_submission/Trimax-Mid-Submission.pdf).
- **Editable report:** [LaTeX source ZIP](reports/mid_submission/Trimax-Mid-Submission-source.zip), including the official ACL style.
- **Report build and checks:** [instructions](reports/mid_submission/README.md).
- **Real experimental results:** `results/final/c-v1-2721954/`. The directory retains its original run name for provenance; these are the results used in the mid-submission.

The report covers 8,000 Python functions, 8,000 C++ files, and 300 English paragraphs, with a frozen 20/80 calibration/test split, 10,000 null permutations, and 2,000 bootstrap replicates. It reports P1–P4, structure-blind baselines, span IoU, and code-noise robustness. O5 outputs are retained for follow-up review; the mid-report identifies this as remaining work.

## Repository layout

| Directory | Contents |
|---|---|
| `stream_a/` | BLT patcher, checkpoint fidelity, entropy and boundary extraction |
| `stream_b/` | Corpus restoration, parsing, taxonomies, perturbations, and region labels |
| `escape_scoring/`, `schemas/v1/` | Current one-to-one scoring and frozen protocol |
| `escape_common/`, `escape_eval/` | Shared utilities and legacy exploratory scorer |
| `tests/`, `golden_fixtures/`, `examples/` | Regression tests and small reproducible fixtures |
| `corpus/manifest.parquet`, `runs/`, `runs.json` | Frozen corpus and extraction metadata |
| `results/` | Real results, checkpoint evidence, and separately labelled historical/synthetic runs |
| `reports/mid_submission/` | Submitted PDF, sources, generated tables, and formatting checks |
| `docs/`, `scripts/` | Protocols, environment records, and reproduction commands |

Historical results remain because they document checkpoint validation and support regression tests. Synthetic examples and legacy exploratory results are not the report's held-out C-v1 results.

## Run the scorer and tests

Use Python 3.10 or newer in a virtual environment:

```bash
python3 -m venv .venv_c
.venv_c/bin/pip install -r escape_scoring/requirements.txt
.venv_c/bin/python -m escape_scoring validate --data examples/scoring_v1/data
.venv_c/bin/python -m pytest tests/scoring_v1 -q
```

See [scorer commands](docs/RUN_SCORER.md) and the [frozen scoring protocol](docs/SCORING_PROTOCOL.md). Optional `numba` accelerates scoring; optional `statsmodels` enables O5 inference. Neither is required for basic scoring.

The full test suite is `python -m pytest tests -q`. Stream A tests also need its dependencies; model tests require downloaded weights. Stream B fixture validation is `python stream_b/validate_golden_fixtures.py` in its parser environment.

## Reproduce extraction and results

Follow [Ada run instructions](docs/FINAL_RUN.md), [the byte contract](docs/SCHEMA.md), and [checkpoint fidelity notes](docs/STREAM_A_FIDELITY.md). Stream A and Stream B require separate environments because their transformer dependencies differ; pinned requirements are in their respective directories.

Raw corpora, models, and per-byte extraction arrays are regenerated rather than included. Restore the frozen manifest with `scripts/restore_stream_b_artifacts.py`; `setup.sh` builds a new corpus and should not be used to reproduce the frozen run. C++ corpus restoration requires authorized Hugging Face access to `bigcode/the-stack-smol`.

Verify the retained real-result files from the repository root:

```bash
(cd results/final/c-v1-2721954 && sha256sum -c SHA256SUMS)
```

The [final proposal](docs/proposal/Trimax-Proposal-Final.pdf) is retained for project context. Course-specific submission requirements and authorship review remain the team's responsibility; see the report instructions.
