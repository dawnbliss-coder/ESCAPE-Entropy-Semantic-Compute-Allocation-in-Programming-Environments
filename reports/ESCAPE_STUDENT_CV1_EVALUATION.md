# ESCAPE Student C-v1 Evaluation (Golden-Fixture Variant)

**Date:** 2026-10-23 · **Experiment:** `results/student_cv1_golden_v1/` (scores.json,
config.json, run_golden_eval.py) · **Report authors:** OpenCode `escape-research` skill.
No training was performed. No committed, frozen, or user-owned files were modified or
overwritten; the experiment directory and this report are new, untracked files.

---

## 1. Scope and what this is not

This evaluates Divyansh's **already-trained distilled student** (`c4384eb`,
`results/distill/full/student/student.pt`, 14,262,144 params, 6,860 steps) under the
**frozen C-v1 scoring rules** — one-to-one matching at the frozen k = 0, the frozen
τ = 1.335442066192627 nats, the fixed-EOF density-matched null, and the structure-blind
baselines — on the **18 hand-verified golden fixtures** (6 Python, 6 C++, 6 English).

It is a **golden-fixture variant experiment, not a held-out result.** The real C-v1 test
split could not be scored locally: the corpus bytes and the A/B boundary/entropy
artifacts are not retained on this machine (they are regenerated on Ada), Ada is
unreachable from this session (no VPN), and restoring the corpora requires large or
gated Hugging Face downloads that this task prohibits. This report therefore answers
"does the student behave like the teacher under the frozen protocol?" at golden scale,
and documents exactly what is needed for the full test-split run.

## 2. Provenance (verified)

| Item | Value | Verification |
|---|---|---|
| Teacher | `itazap/blt-1b-hf` @ `91aa6b8e…cc01`, patcher only, float32 | sha256 of `patcher.safetensors` recorded in `config.json`; loaded via `stream_a.reference_lm` (the G3-verified pure-torch re-implementation, max \|Δlogits\| ≤ 1.05e-4 vs the HF port) |
| Student | `student.pt` from `c4384eb`, 14,262,144 params, config dim 384/8 layers/6 heads | sha256 recorded; `load_state_dict` strict; steps = 6,860 (matches committed `train_log.json`) |
| Training data | extension corpus train split, 27,153 files (13,975 py / 7,470 prose / 5,708 cpp) | from committed `results/distill/full/manifest_ext.parquet` + `DATA_NOTE.json` |
| Eval data (this run) | `golden_fixtures/{py,cpp,prose}/…` (18 files, 211–1932 B) | read directly; structures from the frozen `.structure.parquet` files |
| Frozen k / τ | k = 0 (all languages), τ = 1.335442066192627 | from `results/final/c-v1-2721954/calibration.json` (record hash re-verified in the prior audit) |

The earlier audit's verification of the extension corpus (0 hash overlap, 0
repo/article overlap with the C-v1 evaluation set, group-pure train/val) still holds;
no new evidence contradicts it.

## 3. Inference and context policy

- **teacher_native:** `[BOS] + bytes` in one forward with the swa512 banded mask
  (True = attend for `0 ≤ i−j < 512`), float32, entropy in nats. Files are ≪ 8,192
  tokens, so no chunking is needed; for sequences ≤ 512 tokens this equals causal
  attention (fidelity gate G3). This is the checkpoint's own regime.
- **teacher_512 (control):** the *same weights* with the student's context budget —
  per-byte predictions from fresh windows of ≤ 512 preceding bytes (stride 1). For
  files ≤ 512 B this is **bit-identical** to teacher_native (same input per byte), so
  it is copied, not recomputed; it was computed only for the 3 longer Python files.
  It isolates the context-length effect. It is a control, not a replacement for
  teacher_native.
- **student:** per-byte predictions from fresh windows of ≤ 512 preceding bytes
  (stride 1), absolute RoPE positions 0..511 — its trained regime. **No non-overlapping
  512-byte chunking is used anywhere**: every byte is predicted with the model's own
  full context budget, so no artificial boundary resets are introduced.

Numerical caveat, quantified: the teacher runs through the reference re-implementation
on CPU rather than the HF port on GPU (max logit divergence ≈ 1e-4; entropy divergence
≈ 1e-6 nats). Boundary-flip risk near τ is negligible at this scale: 0–2 bytes per
domain lie within 1e-3 nats of τ (recorded per stream in scores.json diagnostics).

## 4. Scoring configuration

- Targets: golden structure rows filtered by the C-v1 target rules (depth > 0, not a
  wrapper type, non-empty span, interior starts), deduplicated and attributed to the
  outermost node (same tie-breaks as `escape_scoring.metrics.target_nodes`).
- Matching: greedy one-to-one interval matching at k = 0 (`escape_scoring.metrics.match`).
- Null: per-file fixed-EOF shuffle of `(length, termination_reason)` pairs, R = 10,000,
  sequential PCG64 draws seeded by SHA256(`20260927`, `sample_id`, purpose), pooled
  counts per permutation → exact pooled permutation p. teacher_512 reuses teacher_native
  draws where layouts are identical (documented above).
- Baselines: newline / indentation-change / word-start, interior offsets, from the raw
  bytes (`escape_scoring.baselines`).
- Uncertainty: paired file percentile bootstrap, B = 2,000. **n = 6 files per domain →
  CIs are wide and all stream-to-stream differences are within each other's intervals.**

## 5. Results (golden fixtures; NOT held-out)

| Domain (files, targets) | Stream | P | R | F1 [95% CI] | margin vs null | p | vs strongest baseline (its F1) |
|---|---|---|---|---|---|---|---|
| **py** (6, 71) | teacher_native | 0.080 | 0.803 | 0.145 [0.119, 0.173] | +0.112 | ≤1e-4 | −0.332 vs indent (0.477) |
| | teacher_512 | 0.079 | 0.817 | 0.145 [0.119, 0.175] | +0.111 | ≤1e-4 | −0.333 vs indent |
| | student | 0.072 | 0.789 | 0.131 [0.102, 0.163] | +0.101 | ≤1e-4 | −0.346 vs indent |
| **cpp** (6, 55) | teacher_native | 0.142 | 0.873 | 0.245 [0.106, 0.399] | +0.198 | ≤1e-4 | −0.142 vs word (0.387) |
| | teacher_512 | 0.142 | 0.873 | 0.245 [0.106, 0.404] | +0.198 | ≤1e-4 | −0.142 vs word |
| | student | 0.126 | 0.836 | 0.220 [0.105, 0.340] | +0.172 | ≤1e-4 | −0.167 vs word |
| **prose** (6, 217) | teacher_native | 0.318 | 0.922 | 0.473 [0.401, 0.523] | +0.326 | ≤1e-4 | −0.287 vs word (0.760) |
| | teacher_512 | 0.318 | 0.922 | 0.473 [0.400, 0.524] | +0.326 | ≤1e-4 | −0.287 vs word |
| | student | 0.276 | 0.968 | 0.429 [0.371, 0.477] | +0.274 | ≤1e-4 | −0.331 vs word |

**Diagnostics (pooled over the 6 golden files per domain):**

| Domain | Stream | entropy mean/median | frac > τ | boundaries (density) | bits/byte | runtime |
|---|---|---|---|---|---|---|
| py | teacher_native | 0.565 / 0.086 | 0.172 | 713 (0.171) | — | 3.1 s |
| | teacher_512 | 0.577 / 0.091 | 0.177 | 732 (0.175) | 0.93 | 880.6 s (CPU) |
| | student | 0.613 / 0.108 | 0.188 | 782 (0.187) | 1.02 | 155.1 s (CPU) |
| cpp | teacher_native | 0.562 / 0.090 | 0.166 | 337 (0.163) | — | 1.3 s |
| | teacher_512 | == native (all files ≤512 B) | | | — | 0 |
| | student | 0.596 / 0.099 | 0.179 | 364 (0.177) | 0.93 | 34.5 s |
| prose | teacher_native | 0.934 / 0.503 | 0.293 | 629 (0.290) | — | 1.5 s |
| | teacher_512 | == native (all files ≤512 B) | | | — | 0 |
| | student | 1.093 / 0.731 | 0.354 | 762 (0.351) | 1.78 | 40.5 s |

Commands used (reproduced, not estimated): `run_golden_eval.py` (copy saved in the
experiment dir) with the audit venv `torch 2.14.1+cpu`, 4 threads, batch 4 windows,
8 GB virtual-memory cap. Runtime figures above are measured wall times on a
CPU-only laptop.

## 6. Errors, missing artifacts, and limitations

1. **Full held-out evaluation blocked (documented, not silent).** The C-v1 test-split
   bytes, entropies and boundaries are not on this machine, Ada is unreachable (VPN
   off), and corpus restoration requires large/gated downloads. What exists locally is
   the golden-fixture set used here. The full-split run needs Ada (one GPU-hour-class
   job: student stride-1 extraction is ~49 ms/byte on CPU, far less on GPU) or a
   restored corpus.
2. **n = 6 per domain.** All stream-vs-stream differences (student vs teacher, teacher
   vs teacher_512) are smaller than the bootstrap width; only the null comparisons are
   decisive at this scale. Golden fixtures are 211–1,932 B — mostly ≤ 512 B, so the
   context-length control only exercises 3 Python files.
3. The teacher is the reference re-implementation on CPU float32 (see §3 caveat;
   flip-risk quantified ≤ 2 bytes/domain).
4. English: this golden evaluation does **not** change the documented exploratory
   status of the original English P1 (prior `prose_kgrid` exposure of the 300-paragraph
   corpus remains; see §8).

## 7. Distinctions this report keeps separate

- **Teacher imitation** (window-level, prior audit): boundary F1 0.871 vs teacher,
  entropy r = 0.946 — measured on fresh ≤512-B windows, says nothing about syntax.
- **Predictive quality:** student bits/byte is *worse* than the teacher's on golden
  data (py 1.02 vs 0.93; prose 1.78 vs teacher_512-level ≈ 1.2–1.4) — the student is a
  lossy approximation, and its entropy is systematically higher (mean entropy +0.03–0.16
  nats; boundary density +7–21%).
- **Syntactic alignment (this run):** the student **retains** a positive margin over
  the density-matched null in all three domains and **loses to the strongest
  structure-blind baseline in all three** — exactly the original project's qualitative
  pattern, at golden scale. None of this is evidence of semantic understanding.

## 8. Answers to the Phase-5 questions

1. *Does the student recover the teacher's segmentation on complete files?*
   Approximately, at golden scale: same qualitative structure, F1 a few points lower
   (py −0.014, cpp −0.025, prose −0.044; prose the weakest, matching the window eval).
   Differences are within bootstrap noise at n = 6.
2. *Positive margin over the null?* **Yes, all three domains** (+0.101 / +0.172 /
   +0.274 vs the teacher's +0.112 / +0.198 / +0.326).
3. *Outperforms the strongest structure-blind baseline?* **No**, in any domain —
   indent (py) and word (cpp, prose) beat it by 0.33–0.35 F1. Same verdict as the
   original report.
4. *How much is context length?* **Almost none at this scale**: teacher_512 ≈
   teacher_native everywhere (Δ ≤ 0.001 on the only files that differ). The student's
   shortfall is therefore attributable to its **distillation/predictive gap**, not its
   512-byte context — an informative negative result for the context-length
   explanation.
5. *Efficiency justified?* Not measured here (CPU timings only); a proper answer needs
   GPU FLOPs/latency. The student is 7× smaller (14.3M vs 99.5M) and its boundary
   quality is a few F1 points lower; whether that trade is worth it is the efficiency
   experiment, not this one.
6. *Gains on unseen projects / independent English?* Not testable at golden scale.
   The path exists: **552 extension English articles (11,942 files) were never used**
   for training or the window eval (verified from the manifests). Pre-registered
   candidate holdout rule (deterministic, group-level, before any scoring):
   `article_rank = min over article files of int(sha256(bytes)[:8],16)`; select the
   **120 lowest article_rank among the 552 unused articles** (≈ 4,107 files, 3.06 MB;
   example articles: "Noisy miner", "James Robert Baker", "Directed acyclic graph",
   …). Their bytes are not materialized locally; scoring them requires the WikiText
   restore. The original 300-paragraph English set remains **exploratory**.
7. *Drop-in entropy model?* Not yet justified as a drop-in for the teacher (worse
   predictive quality and alignment everywhere), but it is a **credible lightweight
   approximation** for analyses that need the teacher's segmentation *pattern*
   (null/baseline contrasts reproduce qualitatively at ~70–90% of the teacher's
   margins).
8. *Which original conclusions are unchanged?* All of them: the frozen C-v1 results
   stand; margins-over-null are genuine; structure-blind baselines dominate; the
   Grammatical Branching Hypothesis remains contradicted by its sharpest tests; no new
   semantic claim is made.

## 9. Ranked next experiments (no new training needed for #1–#3)

1. **Full test-split student scoring on Ada (go).** Same pipeline as this run, GPU,
   student stride-1 extraction. Answers Q1–Q4 with real statistical power. Go/no-go:
   proceed; abandon only if a correctness bug is found in extraction (Q: can the
   student's boundaries be produced deterministically at scale?).
2. **Teacher-512 control at full scale (with #1).** The golden result says context
   length is a small effect; confirm on files > 512 B where the control actually
   differs. Go/no-go: if teacher_512 stays within CI of teacher_native at scale, the
   context-length explanation is dead and attention turns to the distillation gap.
3. **Efficiency measurement (with #1).** Record student vs teacher GPU FLOPs/latency/
   memory per byte and per boundary-quality point. Answers whether the 7× size cut
   justifies the F1 loss. Go/no-go: justified if student boundary quality is within
   a pre-agreed bound (e.g., its null margin ≥ 90% of the teacher's).
4. **Pristine English holdout (after #1).** Materialize the 120 pre-registered unused
   articles (§8.6), score student and teacher on them; this is the only route to a
   *confirmatory* English result. Go/no-go: proceed only if the selection rule above
   is accepted as pre-registered before anyone scores those articles.
5. **Distillation vs domain adaptation (medium term, training required).** Only after
   #1–#2: if the student's gap is the distillation regime rather than the architecture,
   direct domain adaptation of the teacher on the same train split is the decisive
   comparison. New training on Ada; separate experiment; do not reuse the frozen k.
6. **Language-specific students (only if #1 shows prose is the outlier at scale).**
7. **Syntax-supervised training:** a distinct research direction with its own
   hypothesis; not on the critical path.

**Blocker for #1–#4:** Ada access (VPN) and/or local corpus restoration. Until then,
the golden-fixture evidence above is the strongest claim that can be made honestly.
