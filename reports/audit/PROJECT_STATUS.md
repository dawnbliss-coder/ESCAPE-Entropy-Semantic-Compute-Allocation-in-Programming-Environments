# Project status: where ESCAPE stands (31 Oct 2026, submission day)

Companion files:
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md): what the results say;
- [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md): every finding, ranked.

## Verdict

**The project is complete and in good shape. The audit's fixes have been applied (see "Resolution" in [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md)); one wording choice is left for the team to confirm.**

- All five proposal deliverables exist, every planned analysis has run, and the extension (a distilled patcher) is finished.
- The numbers are reproducible: about 600 values were recomputed from the raw results and none was stale.
- The weaknesses the audit found were all in the writing, and none needed new compute. Status after the fixes:
  1. **Page limit (C1): fixed.** The body, Limitations included, now ends on page 8, and `check_final.py` enforces it.
  2. **AI-use disclosure (C2): added.** One sentence in Ethical Considerations; the team should confirm the wording against the course policy.
  3. **P3 ratio (C3): fixed.** The report now explains the ceiling and adds chance-corrected recall, which ranks English first.
- The eight major wording issues (M1–M8) and most minor ones are fixed in the report, the slides and the README.

## Snapshot

| Item | State |
|---|---|
| Deadline | 31 Oct 2026 (today) |
| Git | Audited at `d9ffc6b`; the audit and its fixes are the two commits after it, pushed to GitHub |
| Tags | `v1.0-final`, `v1.1-final`, and `v1.2-final` on the audit-fix commit |
| Final report | `reports/final_submission/Trimax-Final.pdf`: 11 pages, body ends on p.8, References start at the top of p.9; builds with 0 missing values; checker PASS with the 8-page rule |
| Slides | `reports/final_submission/slides/Trimax-Final-Viva.pdf`: 18 slides, 0 overfull boxes |
| Submission zip | `Trimax-Final.zip`, rebuilt after the fixes; contents identical to `main`. Not in git (GitHub's 100 MB limit); rebuild with `bash scripts/package_final.sh` |
| Tests | 198 pass; every module compiles |
| Frozen results | Unchanged since the freeze; 17 of 17 checksums pass |
| Ada cluster | Nothing running; all results copied back. The `/share1` file quota is at its hard limit (housekeeping) |

## What the project is

ESCAPE asks whether the Byte Latent Transformer's entropy-based patch boundaries fall on syntactic constituent starts in Python, C++ and English. Three things were measured:

- whether they beat a density-matched random null;
- whether they beat structure-blind rules (newlines, indentation changes, word starts);
- whether code differs from prose.

The answer: the boundaries beat chance by 3–5×, but lose to the simple rules. See [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## What was built

| Part | What it does | Size |
|---|---|---|
| `stream_a/` | BLT entropy patcher pipeline: checkpoint fidelity gates, restored 512-token sliding window, resumable extraction, `patch_files.py` for arbitrary files | 11 files, 2.3k lines |
| `stream_b/` | Corpus (CodeSearchNet Python, The Stack C++, WikiText-103), tree-sitter and benepar parsing with byte-offset validation, baselines, O4 noise, O5 regions, HTML tree viewers | 21 files, 2.5k lines |
| `escape_eval/`, `escape_scoring/` | Frozen scoring protocol: calibration, density-matched null (10,000 permutations), paired bootstraps, baselines, IoU, O4/O5 | 25 files, 3.9k lines |
| `escape_distill/` | Teacher–student distillation, extraction, re-thresholding, A-format scorer | 9 files, 1.6k lines |
| `scripts/` | Ada job scripts and the final analyses (P4 positions, O3 depth, O5 confound, τ sweeps, efficiency) | 14 Python files plus sbatch, 2.3k lines |
| `reports/` | Report and slide builders that generate every number from `results/` (no hand-typed results tables) | 6 files, 2.1k lines |
| `tests/` | 17 files, 198 tests | 1.6k lines |

**Compute:** 66 Ada jobs are recorded in `results/final/logs/full_run_jobs.txt`, all with logs. The largest was student extraction over all 16,300 files, about 83 GPU-hours on RTX 2080 Ti.

## Proposal deliverables

| # | Promised | Delivered | Status |
|---|---|---|---|
| 1 | Boundary extraction for arbitrary files (entropies, tagged boundaries, BPP) | `stream_a/` + `python -m stream_a.patch_files <file> --show` | Done |
| 2 | tree-sitter AST mapping for Python/C++ with memory-unsafe queries | `stream_b/` | Done |
| 3 | Evaluation harness: P1–P4, both baselines, permutation test, depth/type strata | `escape_eval/`, `escape_scoring/` | Done |
| 4 | Entropy heatmaps and BPP profiles for code and prose | Report Fig. 1 (Python, C++, English) | Done |
| 5 | Final report and presentation covering all objectives, incl. negative results | Report + 18-slide deck | Done (C1–C3 fixed) |

## Objectives

| Objective | Status | Result in one line |
|---|---|---|
| O1 syntactic alignment (P1, P2, P4) | Done | Beats chance, loses to simple rules; P2 holds; P4 reversed at the start byte |
| O2 compute-allocation ratio, code vs prose (P3) | Done | Reversed: English aligns most, also on chance-corrected recall |
| O3 Python vs C++ by depth | Done | Python > C++ at every shared depth |
| O4 robustness to code noise | Done | Degrades gracefully; indentation hurts most |
| O5 memory-unsafe compute vs identifier confound | Done | No semantic effect; small positive only after an identifier-entropy control |
| Extension: distilled patcher | Done | 14.3M params keeps 87–92% of the margin at about 5× the speed |

## Deviations from the proposal

All are disclosed in the report except the last.

- C++ data from The Stack: CodeSearchNet has no C++.
- float32 instead of bfloat16: 0.26% of boundaries differ.
- Training was added (the distillation extension); the proposal said none was needed.
- Patcher size measured at 99.5M (the proposal estimated about 150M).
- The original patch-length distribution was not compared (disclosed in Limitations).
- The mid-report's promised formatting-conditioned P4 test was not run; the report now says so (audit M3).

## Reproducibility and quality

- **Strong:**
  - every report number is generated from result files, and a checker blocks the build on any missing value;
  - the scoring protocol was frozen before test scoring;
  - the English holdout and τ\* rules were committed before they were scored;
  - checksums, logs and job ids are kept for every run.
- **Weak spots:**
  - the zip can't live in git (size);
  - the third-party checkpoint must be downloaded (`python -m stream_a.fetch_patcher`).

## Recent history worth knowing

- **30 Oct:** follow-up arms, completeness audit, more-permutation sensitivity check; tagged `v1.1-final`.
- **30–31 Oct (overnight), Priyanka:**
  - deleted the planning files (`plan.md`, `file_toimplement.md`) and the AI-generated docs (`docs/ai_assisted/`);
  - renamed `docs/ADA_CLAUDE_HANDOFF.md` to `docs/ADA_FULL_RUN_GATES.md`;
  - moved the heatmaps into the report body;
  - added HTML tree-viewer slides;
  - removed the AI-assistance sentence from the mid-report source.
- **31 Oct, morning:** the exploratory τ grid on test files was merged on top (`d9ffc6b`), keeping the body length equal to Priyanka's version.
- **GitHub:** Claude co-author lines were stripped from history. The contributor sidebar still shows `@claude` from GitHub's cache.

## Before you submit (in order)

1. Confirm the AI-use sentence in Ethical Considerations with the team and the course policy (C2). It is the last sentence of that section in `paper-body.tex`.
2. If anything changes, rebuild (`bash reports/final_submission/build.sh`, `build_slides.sh`, `bash scripts/package_final.sh`), commit, and tag the submitted commit.
3. Submit `Trimax-Final.pdf`, the slides and `Trimax-Final.zip`.
