# Project ESCAPE: Entropy and Semantic Compute Allocation in Programming Environments

**Team Trimax · CS7.501 Advanced NLP · IIIT Hyderabad · Final submission**

| Author           | Roll number |
| ---------------- | ----------- |
| Priyanka Agarwal | 2024101016  |
| Shashwat Mukadam | 2024102008  |
| Divyansh Atri    | 2024113001  |

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

|                                                                | Python                     | C++                        | English                    |
| -------------------------------------------------------------- | -------------------------- | -------------------------- | -------------------------- |
| **P1** start F1, BLT vs. density-matched null            | 0.152 vs. 0.032            | 0.165 vs. 0.036            | 0.495 vs. 0.147            |
| F1 margin over null [95% CI]                                   | +0.120 [0.117, 0.122]      | +0.129 [0.125, 0.134]      | +0.348 [0.341, 0.354]      |
| Precision / recall                                             | 0.083 / 0.822              | 0.095 / 0.646              | 0.342 / 0.894              |
| Strongest structure-blind baseline (F1)                        | indentation (0.394)        | word start (0.339)         | word start (0.769)         |
| BLT − best baseline F1 [95% CI]                               | −0.242 [−0.247, −0.237] | −0.173 [−0.181, −0.167] | −0.274 [−0.280, −0.268] |
| **P2** end-boundary margin (starts: rows above)          | −0.017                    | −0.035                    | −0.079                    |
| **P3** mean bytes per patch                              | 5.68                       | 7.86                       | 3.77                       |
| **P3** F1 / null F1 (scale-free; margins grow with target density) | 4.7×                 | 4.6×                       | 3.4×                       |
| **P4** recall margin, forced-opener vs. open-ended types | +0.759 vs. +0.611          | +0.691 vs. +0.451          | —                         |
| Span IoU (secondary): BLT / null / line segmentation           | 0.351 / 0.342 / 0.546      | 0.372 / 0.356 / 0.606      | 0.377 / 0.334 / 0.049      |
| **O4** margin on clean code → worst noise (indentation) | +0.115 → +0.099           | +0.131 → +0.118           | —                         |

**In short:**

| Prediction                                                                   | Verdict                                                                                                                                                       |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P1:** boundaries align with starts beyond the null                   | Supported in all three languages (margins far outside the null)                                                                                               |
| **Structure-blind baselines**                                          | Not supported: BLT is *beaten* by indentation or word-start splitting in every language, so most of the alignment is line and word segmentation, not grammar |
| **P2:** starts align better than ends                                  | Supported; end margins are negative                                                                                                                           |
| **P3:** alignment stronger in code than prose                          | Reversed on the F1 margin, which grows with target density; on the scale-free F1/null ratio code (4.6–4.7×) beats prose (3.4×). Prose "alignment" is mostly word onsets |
| **P4:** weaker for forced-opener constructs (`if`, `for`, `def`) | Reversed at the start byte, confirmed at the forced bytes: the byte after the keyword is a boundary at rate 0.000 (Python) / 0.035 (C++) vs a null ≈ 0.15, and the boundary returns where choice reopens (`scripts/final_analysis/position_scores.py`) |
| **Precision > recall** (entropy bounds branching)                      | Not supported: precision is far below recall; BLT proposes many more boundaries than there are targets                                                        |
| **O4:** robustness to code noise                                       | Supported: degrades gracefully. In a 5–40% rate sweep only indentation noise grows with rate (−0.005 → −0.029 Python, −0.004 → −0.023 C++); keyword typos lower BPP (−0.43 / −0.60 at 40%) but barely move alignment; every noised margin ≥ +0.086 |
| **O5:** compute near memory-unsafe code beyond the identifier confound | No semantic residual. With file fixed effects only, unsafe statements get *fewer* boundaries (−0.0083 [−0.0127, −0.0039]); the frozen +0.0069 [0.0043, 0.0095] appears only after conditioning on identifier entropy and is 4% of the mean density. Labels audited: `docs/O5_LABEL_AUDIT.md` |

Entropy is a surface statistic, and none of these results is evidence of semantic understanding.

## Objectives (from the proposal)

| Objective                                         | Status in this submission                                                                |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| O1 Syntactic alignment (P1, P2, P4)               | Done, with null, baselines, per-type and per-depth analysis                              |
| O2 Compute-allocation ratio (BPP, code vs. prose) | Done (P3)                                                                                |
| O3 Python vs. C++                                 | Done: side by side everywhere, plus scaffold-free (constituent) depth normalisation       |
| O4 Robustness to code noise                       | Done: four noise types on 1,000 held-out files per language, and a rate sweep (5–40%)    |
| O5 Memory-unsafe compute vs. identifier confound  | Done: label audit + nested confound decomposition (`results/final/analysis/o5_confound.json`) |

## Beyond the proposal: a distilled patcher

A 14.3M-parameter byte transformer distilled online from the 99.5M patcher (`escape_distill/`, α = 0.1, T = 2, on a
disjoint 29k-file extension corpus) was extracted on all 16,300 files and scored by the same frozen pipeline with its
own calibration record (k = 0 everywhere). Sources: `results/final/student-v1-*`, `docs/FULL_RUN_RESULTS.md`.

|                                                    | Python                  | C++                     | English                 |
| -------------------------------------------------- | ----------------------- | ----------------------- | ----------------------- |
| Student start F1 / margin over null                | 0.140 / +0.109          | 0.155 / +0.119          | 0.459 / +0.304          |
| Share of the teacher's margin kept                 | 91%                     | 92%                     | 87%                     |
| Student − teacher margin [95% CI, paired]          | −0.011 [−0.011, −0.010] | −0.011 [−0.012, −0.009] | −0.044 [−0.048, −0.040] |
| Student − best structure-blind baseline            | −0.253                  | −0.183                  | −0.310                  |
| Teacher τ\* / student τ\* (pre-registered sweep, calibration only) | 2.0 / 2.0         | 2.0 / 2.0               | 2.0 / 2.5               |
| Teacher margin at τ\* (one confirmatory test run)  | +0.153 (gain +0.033)    | +0.158 (gain +0.029)    | +0.426 (gain +0.078)    |
| Teacher at τ\* − best baseline                      | −0.210                  | −0.148                  | −0.215                  |
| Student margin at τ\* (one confirmatory test run per language) | +0.147 (gain +0.038) | +0.156 (gain +0.037) | +0.460 (gain +0.157) |
| Student at τ\* − best baseline                      | −0.216                  | −0.149                  | −0.186                  |
| **English holdout** (4,107 unseen paragraphs, scored once): margin teacher / student | — | — | +0.350 / +0.305 (eval set: +0.348 / +0.304) |

- Restricting the teacher to the student's 512-byte windows costs 0.004 F1 on 300 test files per code language, about 40% of the student's deficit there.
- Against the teacher's own boundaries on held-out windows: distilled student F1 0.871, no-teacher (α = 1) student 0.789, word starts 0.475. The α = 1 student compresses better (1.17 vs 1.24 bits/byte; teacher 1.14).
- Cost on an RTX 2080 Ti: the student's forward pass is 5.0× faster than the teacher's, and its rolling extraction 4.8× faster.
- Follow-ups on the same 600-file code subset plus the 240 English test files (exploratory; full-corpus extraction did not fit): a byte-balanced student trades code for prose (margin −0.002 Python, −0.002 C++, +0.010 English vs the distilled student), and the pre-registered teacher-adaptation verdict is not met (+0.006 / +0.005 / +0.029; only English clears the student's CI width). Scoring the teacher with HF's native full attention instead of the restored 512-token window cuts the test margin to +0.023 (Python) and +0.008 (C++ under 8 KB), against +0.120 and +0.131.

All numbers, sources and job ids: [`docs/FULL_RUN_RESULTS.md`](docs/FULL_RUN_RESULTS.md).

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

**2. Run the scorer tests** (seconds). These include an exact reproduction of a reference scoring run.

Use `python3.12` specifically (`escape_scoring/requirements.txt` now pins `numpy==2.5.3`, which requires
Python >= 3.12; a 3.10/3.11 default `python3` will fail to install it):

```bash
python3.12 -m venv .venv_c && .venv_c/bin/pip install -r escape_scoring/requirements.txt
.venv_c/bin/python -m pytest tests/scoring_v1 -q
```

**3. Re-run everything from scratch** on IIIT-H Ada (SLURM; a few hours with 4 × RTX 2080 Ti):

```bash
bash scripts/ada_final/sync_to_ada.sh                       # laptop → new Ada directory
ssh ada 'cd ~/escape-final-20260928 && bash scripts/ada_final/submit_all.sh'
bash scripts/ada_final/fetch_from_ada.sh                    # results back to the laptop
```

[`docs/FINAL_RUN.md`](docs/FINAL_RUN.md) walks through every job, including span IoU (`40_iou`) and O4 (`50–52_o4_*`). Restoring the C++ corpus needs a Hugging Face account that has accepted the [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol) terms. Raw corpora, model weights and per-byte arrays are regenerated, not stored in git.

## Repository layout

| Path                                            | Contents                                                                                       |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `reports/final_submission/`                   | **Final report and viva slides**: LaTeX sources, `build_final.py` (every number from `results/final/`), `build.sh`, `check_final.py` |
| `reports/`                                    | Mid-submission report `Trimax-Mid-Report.pdf` and its build (`reports/mid_submission/`) |
| `results/final/c-v1-2721954/`                 | **Real results** (frozen teacher run): calibration record, held-out scores, baselines, IoU, O4, O5, checksums |
| `results/final/` (other entries)              | Final-submission runs: student scoring, teacher-512 control, τ sweeps and confirmatory runs, holdout, `analysis/` (P4, O3, O4 sweep, O5, subsets, k, efficiency), Ada job logs |
| `escape_distill/`                             | Extension: student model, distillation training/eval, stride-1 extractor, re-thresholder, k = 0 scorer |
| `scripts/final_analysis/`                     | Final-submission analyses (P4 post-keyword, O3 depth, O4 sweep, O5 confound, subsets, efficiency) |
| `stream_a/`                                   | BLT patcher loading, sliding-window fix, fidelity gates, extraction (incl. O4)                 |
| `stream_b/`                                   | Corpus restore, tree-sitter/benepar parsing, baselines, O4 noise builder, O5 regions           |
| `escape_scoring/`, `schemas/v1/`            | Frozen C-v1 scorer: matching, null, calibration, baselines, IoU, O4, O5                        |
| `escape_common/`, `escape_eval/`            | Shared I/O and offsets; earlier exploratory scorer (superseded)                                |
| `scripts/ada_final/`                          | SLURM pipeline for the full run on Ada                                                         |
| `tests/`, `golden_fixtures/`, `examples/` | Regression tests and small fixtures                                                            |
| `corpus/manifest.parquet`, `runs.json`      | Frozen corpus manifest (hashes, splits) and run registry                                       |
| `docs/`                                       | Protocols, byte contract, fidelity notes, run guides, proposal, `FULL_RUN_RESULTS.md`, `O5_LABEL_AUDIT.md` |
| `docs/ai_assisted/`                           | AI-generated audits, reviews and handoff notes (working documents, not results; see its README) |

`results/ada/` holds earlier **synthetic** software-validation runs, and the `results/prose_kgrid_*` directory holds an earlier **exploratory** prose analysis. Neither contributes to the report's held-out results.

## Team contributions

| Member           | Main work                                                                                                                                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Priyanka Agarwal | **Stream B:** data contract, corpus construction (CodeSearchNet, The Stack, WikiText-103), tree-sitter and benepar parsing with byte-offset validation, whitespace baselines, P4 taxonomy, golden fixtures |
| Shashwat Mukadam | **Stream A:** BLT patcher pipeline, checkpoint-fidelity gates and the sliding-window fix, resumable extraction; first exploratory scorer                                                                   |
| Divyansh Atri    | **Stream C:** frozen scoring protocol and scorer, calibration and null, baselines, IoU, O4/O5 analyses, the Ada pipeline, and the report build                                                             |

## Limitations

The checkpoint is a third-party conversion, and we restored its sliding window ourselves. English targets come from predicted parses; the 300 evaluation paragraphs were seen during exploration, so the pre-registered holdout is the confirmatory English result. About 28% of C++ files trigger parser error recovery; their margin does not differ from clean parses (+0.004 [−0.005, 0.015]). The 8,000 Python functions come from 315 repositories; a repository-clustered bootstrap widens the Python interval 3.7× to [0.112, 0.127]. O5 labels are syntactic. The report's Limitations section has the full list.
