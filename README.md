# Project ESCAPE — Entropy–Semantic Compute Allocation in Programming Environments

**Team Trimax · CS7.501 Advanced NLP · IIIT Hyderabad · Mid-submission**

| Author | Roll number |
|---|---|
| Priyanka Agarwal | 2024101016 |
| Shashwat Mukadam | 2024102008 |
| Divyansh Atri | 2024113001 |

> **Submitted report: [`reports/mid_submission/Trimax-Mid-Submission.pdf`](reports/mid_submission/Trimax-Mid-Submission.pdf)**
> ACL format · passes the official [ACL pubcheck](https://github.com/acl-org/aclpubcheck) ("All Clear!") ·
> [anonymous version](reports/mid_submission/Trimax-Mid-Submission-review.pdf) ·
> [Overleaf/LaTeX source](reports/mid_submission/Trimax-Mid-Submission-source.zip) ·
> [project proposal](docs/proposal/Trimax-Proposal-Final.pdf)

---

## Research question

The [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/) (BLT) reads raw bytes and
starts a new *patch* (a unit of expensive computation) wherever a small byte-level model's
next-byte entropy exceeds a threshold. **Do these entropy-triggered patch boundaries land on
the starts of syntactic constituents?** In particular, do they do so more than a density-matched
random null, more than trivial line or word splitting, and more in code than in prose?

We test this on **Python** (CodeSearchNet), **C++** (The Stack) and **English** (WikiText-103),
against tree-sitter ASTs and benepar constituency parses.

## Headline results (held-out test split)

Every number below is generated from [`results/final/c-v1-2721954/`](results/final/c-v1-2721954)
and appears in the report. The tolerance is k = 0 in all three languages, frozen on the
calibration split before scoring. The test split is 6,400 Python, 6,400 C++ and 240 English files.
The null uses 10,000 permutations per file, and intervals are 95% paired file bootstraps
with 2,000 replicates.

| | Python | C++ | English |
|---|---|---|---|
| **P1** start F1, BLT vs. density-matched null | 0.152 vs. 0.032 | 0.165 vs. 0.036 | 0.495 vs. 0.147 |
| F1 margin over null [95% CI] | +0.120 [0.117, 0.122] | +0.129 [0.125, 0.134] | +0.348 [0.341, 0.354] |
| Precision / recall | 0.083 / 0.822 | 0.095 / 0.646 | 0.342 / 0.894 |
| Strongest structure-blind baseline (F1) | indentation (0.394) | word start (0.339) | word start (0.769) |
| BLT − best baseline F1 [95% CI] | −0.242 [−0.247, −0.237] | −0.173 [−0.181, −0.167] | −0.274 [−0.280, −0.268] |
| **P2** end-boundary margin (starts: rows above) | −0.017 | −0.035 | −0.079 |
| **P3** mean bytes per patch | 5.68 | 7.86 | 3.77 |
| **P4** recall margin, forced-opener vs. open-ended types | +0.759 vs. +0.611 | +0.691 vs. +0.451 | — |
| Span IoU (secondary): BLT / null / line segmentation | 0.351 / 0.342 / 0.546 | 0.372 / 0.356 / 0.606 | 0.377 / 0.334 / 0.049 |
| **O4** margin on clean code → worst noise (indentation) | +0.115 → +0.099 | +0.131 → +0.118 | — |

**In short:**

| Prediction | Verdict |
|---|---|
| **P1:** boundaries align with starts beyond the null | Supported in all three languages (margins far outside the null) |
| **Structure-blind baselines** | Not supported: BLT is *beaten* by indentation or word-start splitting in every language, so most of the alignment is line and word segmentation, not grammar |
| **P2:** starts align better than ends | Supported; end margins are negative |
| **P3:** alignment stronger in code than prose | Not supported; prose has the largest margin |
| **P4:** weaker for forced-opener constructs (`if`, `for`, `def`) | Reversed; forced openers align more strongly |
| **Precision > recall** (entropy bounds branching) | Not supported: precision is far below recall; BLT proposes many more boundaries than there are targets |
| **O4:** robustness to code noise | Supported: degrades gracefully; the worst drop is −0.016, and keyword typos lower BPP most (−0.23 / −0.34) |

Entropy is a surface statistic, and none of these results is evidence of semantic understanding.

## Objectives (from the proposal)

| Objective | Status in this submission |
|---|---|
| O1 Syntactic alignment (P1, P2, P4) | Done, with null, baselines, per-type and per-depth analysis |
| O2 Compute-allocation ratio (BPP, code vs. prose) | Done (P3) |
| O3 Python vs. C++ | Done: reported side by side in every table |
| O4 Robustness to code noise | Done, with the four noise types of the proposal, 1,000 held-out files per language |
| O5 Memory-unsafe compute vs. identifier confound | Pending: computed (`results/final/c-v1-2721954/o5*/`), reviewed for the end submission |

## Method in one screen

```
 corpus bytes ──► Stream A: BLT patcher (itazap/blt-1b-hf @ 91aa6b8e, 512-token sliding
 (16,300 files)   window restored, float32, τ = 1.3354 nats) ──► per-byte entropy + patch boundaries
      │
      └────────► Stream B: tree-sitter ASTs (code), benepar constituents (prose),
                  whitespace/word baselines, O4 noised variants, O5 regions
                                    │
                                    ▼
             Stream C (escape_scoring): byte-exact contract · one-to-one matching ·
             20/80 calibration/test split · k frozen on calibration only ·
             fixed-EOF density-matched null (10,000 permutations) · file bootstrap
```

- **Primary metric:** precision, recall and F1 of interior entropy-triggered boundaries against constituent start offsets, pooled across files, reported as a margin over the null.
- **Null (H0):** the file's own patches shuffled, so the length multiset and boundary count are preserved exactly.
- **Baselines:** boundaries after every newline, at indentation changes, and at every word start.

The full protocol is in [`docs/SCORING_PROTOCOL.md`](docs/SCORING_PROTOCOL.md), and checkpoint fidelity (including the 512-token window the Hugging Face port omits) is in [`docs/STREAM_A_FIDELITY.md`](docs/STREAM_A_FIDELITY.md).

## Verify and reproduce

**1. Check the retained results** (seconds):
```bash
(cd results/final/c-v1-2721954 && sha256sum -c SHA256SUMS --ignore-missing)
```
`--ignore-missing` skips the two O5 design matrices (77 MB each). They're excluded from git and regenerated with `python -m escape_scoring o5`.

**2. Rebuild the report from those results and run the ACL check** (minutes, no GPU):
```bash
python3 reports/mid_submission/build_results.py --run results/final/c-v1-2721954
TECTONIC_BIN=/path/to/tectonic bash reports/mid_submission/build.sh   # or pdfLaTeX + BibTeX
python reports/mid_submission/check_report.py                          # needs aclpubcheck, pdfplumber, pdffonts
```
Details: [`reports/mid_submission/README.md`](reports/mid_submission/README.md).

**3. Run the scorer tests** (seconds). These include an exact reproduction of a reference scoring run:
```bash
python3 -m venv .venv_c && .venv_c/bin/pip install -r escape_scoring/requirements.txt numba
.venv_c/bin/python -m pytest tests/scoring_v1 -q
```

**4. Re-run everything from scratch** on IIIT-H Ada (SLURM; a few hours with 4 × RTX 2080 Ti):
```bash
bash scripts/ada_final/sync_to_ada.sh                       # laptop → new Ada directory
ssh ada 'cd ~/escape-final-20260928 && bash scripts/ada_final/submit_all.sh'
bash scripts/ada_final/fetch_from_ada.sh                    # results back to the laptop
```
[`docs/FINAL_RUN.md`](docs/FINAL_RUN.md) walks through every job, including span IoU (`40_iou`) and O4 (`50–52_o4_*`). Restoring the C++ corpus needs a Hugging Face account that has accepted the [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol) terms. Raw corpora, model weights and per-byte arrays are regenerated, not stored in git.

## Repository layout

| Path | Contents |
|---|---|
| `reports/mid_submission/` | **Submitted report**: PDFs, LaTeX, generated tables and figure, pubcheck record |
| `results/final/c-v1-2721954/` | **Real results**: calibration record, held-out scores, baselines, IoU, O4, O5, checksums |
| `stream_a/` | BLT patcher loading, sliding-window fix, fidelity gates, extraction (incl. O4) |
| `stream_b/` | Corpus restore, tree-sitter/benepar parsing, baselines, O4 noise builder, O5 regions |
| `escape_scoring/`, `schemas/v1/` | Frozen C-v1 scorer: matching, null, calibration, baselines, IoU, O4, O5 |
| `escape_common/`, `escape_eval/` | Shared I/O and offsets; earlier exploratory scorer (superseded) |
| `scripts/ada_final/` | SLURM pipeline for the full run on Ada |
| `tests/`, `golden_fixtures/`, `examples/` | Regression tests and small fixtures |
| `corpus/manifest.parquet`, `runs.json` | Frozen corpus manifest (hashes, splits) and run registry |
| `docs/` | Protocols, byte contract, fidelity notes, run guides, proposal |
| `reports/final/` | Template for the end submission (not submitted) |

`results/ada/` holds earlier **synthetic** software-validation runs, and the `results/prose_kgrid_*` directory holds an earlier **exploratory** prose analysis. Neither contributes to the report's held-out results.

## Team contributions

| Member | Main work |
|---|---|
| Priyanka Agarwal | **Stream B:** data contract, corpus construction (CodeSearchNet, The Stack, WikiText-103), tree-sitter and benepar parsing with byte-offset validation, whitespace baselines, P4 taxonomy, golden fixtures |
| Shashwat Mukadam | **Stream A:** BLT patcher pipeline, checkpoint-fidelity gates and the sliding-window fix, resumable extraction; first exploratory scorer |
| Divyansh Atri | **Stream C:** frozen scoring protocol and scorer, calibration and null, baselines, IoU, O4/O5 analyses, the Ada pipeline, and the report build |

## Limitations

The checkpoint is a third-party conversion, and we restored its sliding window ourselves. English targets come from predicted parses, and about 28% of C++ files trigger parser error recovery. Python samples are functions that may share repositories, so file bootstraps need not generalize across projects. O4 uses one noise rate per type. The report's Limitations section has the full list.
