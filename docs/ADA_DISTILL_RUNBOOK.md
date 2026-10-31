> **Status: EXECUTED** (Ada, 28–31 Oct 2026). Results, numbers and job ids: [`docs/FULL_RUN_RESULTS.md`](FULL_RUN_RESULTS.md). The document is retained as the pre-registration/plan record.

# ADA Runbook — Distilled Student: Training, C-v1 Scoring, and Next Experiments

**For:** Divyansh · **Repo:** ESCAPE (main @ `c4384eb`) · **Last updated:** 2026-10-23
**Context:** this runbook turns the audit + golden-fixture evaluation
(`reports/ESCAPE_STUDENT_CV1_EVALUATION.md`, `results/student_cv1_golden_v1/`) into exact
Ada commands. Nothing here overwrites the frozen C-v1 results, the frozen calibration
record, or the committed student checkpoint.

---

## 0. Sync and environment (once per Ada session)

```bash
# VPN on, then on the login node:
cd ~/escape-final-20260928          # or wherever the project root lives on Ada
git fetch origin && git checkout origin/main --detach   # OR: git pull  (main is at c4384eb)
source scripts/ada_final/env.sh     # defines a_py (torch 2.6.0+cu124, transformers 5.17.0) and c_py
CK=$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01
```

Verify identity hashes before trusting anything:

```bash
sha256sum results/distill/full/student/student.pt
# expected: aeef99da84fd0e97d499717e891b189f7f3ada883810ec9dddf99afbb560cf95
sha256sum "$CK/patcher.safetensors"
# expected: 888df74288bb8cc332bb57895a100e2e26031c21ba24f91caf0ef3a33e2e1b4d
```

If `corpus/` and `structure/` were not restored yet, run the Stream B restore first
(see `docs/FINAL_RUN.md`, job `10_data`): the frozen evaluation corpus bytes are needed
for Part 2. Do **not** regenerate or rewrite `corpus/manifest.parquet`.

---

## Part 1 — Reproduce the existing distillation (sanity, ~3 GPU-hours)

This reproduces the committed run exactly (same seed, same limits, same flags):

```bash
bash scripts/ada_distill/submit_all.sh            # materialize 15k py / 8k prose / 6k cpp, train, eval
```

Or, piece by piece (what `run_all.sbatch` executes):

```bash
# 1a. materialize the extension corpus subset into node-local scratch
a_py -m stream_b.build_ext_corpus py    --limit 15000 --out "$SCR/corpus_ext"
a_py -m stream_b.build_ext_corpus prose --limit 8000  --out "$SCR/corpus_ext"
a_py -m stream_b.build_ext_corpus cpp   --limit 6000  --out "$SCR/corpus_ext"
MANIFEST="$SCR/corpus_ext/manifest_ext.parquet"

# 1b. train (6,860 steps, 2 epochs; this is the exact committed configuration)
a_py -m escape_distill.train \
  --ckpt-dir "$CK" --data "$MANIFEST" --out results/distill/full/student \
  --train-split train --domains cpp,prose,py \
  --epochs 2 --batch-size 48 --lr 5e-4 --weight-decay 0.1 --warmup 200 \
  --temperature 2.0 --alpha 0.1 --dim 384 --layers 8 --heads 6 --log-every 100

# 1c. evaluate on the held-out val split (all 1,847 val files)
a_py -m escape_distill.evaluate \
  --ckpt-dir "$CK" --student results/distill/full/student/student.pt \
  --data "$MANIFEST" --out results/distill/full/eval \
  --eval-split val --domains cpp,prose,py --limit-files 3000
```

**Pass criteria (must match the committed `results/distill/full/`):**
train_log final loss ≈ 0.297 (KL ≈ 0.237, CE ≈ 0.838 nats) at step 6859; eval boundary F1
≈ 0.871 (py 0.869 / cpp 0.880 / prose 0.834), entropy Pearson ≈ 0.946. Small
last-digit differences from GPU kernel selection are fine (same tolerance class as
fidelity gate G3). If the numbers disagree materially, **stop and investigate before
anything downstream.**

---

## Part 2 — The missing experiment: student under the frozen C-v1 protocol on the REAL test split

This is the highest-value run (Option A in the evaluation report). It needs no training.

### 2a. Write a student extraction module (Stream-A format)

The rolling-window inference is already implemented and verified at golden scale in
`results/student_cv1_golden_v1/run_golden_eval.py` (committed). On Ada, wrap it as a
batch extractor `escape_distill/extract.py` (new file, do NOT touch `stream_a/extract.py`):

- **Context policy (do not change):** per-byte predictions from fresh windows of the
  last ≤512 bytes (`[BOS] + ctx`, absolute RoPE positions 0..511). Stride 1, i.e. every
  byte is predicted with the student's full context budget. **Never** use non-overlapping
  512-byte chunks — that would insert artificial boundary resets. Batch windows
  (batch 256 on GPU), right-pad with `PAD_ID=0`, causal attention.
- **Output format (identical to Stream A):**
  - `entropies/{run_id}/{file_id}.npy` — float32, shape `(n_bytes,)`, nats
  - `boundaries/{run_id}/{file_id}.parquet` — columns exactly
    `file_id, byte_offset, trigger, entropy, patch_index, patch_length`
    (one row per patch; `byte_offset` = first byte of patch; triggers from
    `compute_patches(ent, tau)`; tau = 1.335442066192627, strict float64 comparison)
- **Register the run** in `runs.json` with a new, honest identity, e.g.:
  `"student-v1-14326k"`: `backend: "patcher"`,
  `entropy_definition: "entropies[i] = -sum_v p(v) ln p(v) over all 260 patcher logits ..."`,
  `boundary_convention: "one row per patch; byte_offset = first byte of the patch ..."`,
  `tau: 1.335442066192627`, `max_patch_length: null`, `dtype: "float32"`,
  `context_mode: "student-rolling-512"`, `checkpoint: "escape_distill/student-bl-t-384-8-6"`,
  `checkpoint_revision: "c4384eb81f7d801aab2746dc453b6b1679136f9a"`,
  `student_pt_sha256: "aeef99da84fd0e97d499717e891b189f7f3ada883810ec9dddf99afbb560cf95"`.
- **Sanity gate before the full run:** extract the 18 golden fixtures with the new
  module and confirm the numbers in `results/student_cv1_golden_v1/scores.json`
  reproduce (they were produced with the reference implementation of the same policy).

```bash
# resumable extraction over all 16,300 corpus files (4x 2080 Ti job like 20_extract.sbatch)
a_py -m escape_distill.extract --run-id student-v1-14326k --domains py cpp prose
```

### 2b. Convert, validate, and score (C-v1 pipeline, untouched scorer)

```bash
DATA="$SCR/c-v1-student"
IDS=$(a_py -c "import pandas as pd; m=pd.read_parquet('corpus/manifest.parquet'); print(' '.join(sorted(m.file_id)))")
c_py -m escape_scoring convert-legacy --root . --run-id student-v1-14326k --output "$DATA" --sample-ids $IDS
c_py -m escape_scoring validate --data "$DATA" --workers 32
```

**Threshold policy — do both, keep them separate:**
1. **Student-specific calibration (protocol-faithful):** freeze k for the student on
   the SAME 20% calibration files with the frozen criterion:
   ```bash
   c_py -m escape_scoring calibrate --data "$DATA" --output results/final/student-v1-calibration.json --workers 32
   c_py -m escape_scoring score --data "$DATA" --calibration results/final/student-v1-calibration.json \
       --output results/final/student-v1-eval --workers 32
   c_py -m escape_scoring baselines --data "$DATA" --calibration results/final/student-v1-calibration.json \
       --output results/final/student-v1-baselines.json --workers 32
   ```
   (Expected: the criterion selects k=0 again — the teacher's calibration margins are
   strictly monotone in k — but whatever it selects, that is the student's frozen k.)
2. **Fixed-threshold k=0 compatibility row** (comparability with the teacher's frozen
   k=0): if the student calibration selects k=0, the row above IS the fixed-k row.
   Otherwise compute it with the standalone scorer in
   `results/student_cv1_golden_v1/run_golden_eval.py` adapted to the JSONL (it already
   implements k=0 matching + fixed-EOF null + baselines + bootstrap).

**Never** score the student against the teacher's frozen
`results/final/c-v1-2721954/calibration.json` — the scorer's model-signature check
rejects it by design (correctly). The teacher record stays untouched.

### 2c. Teacher-512 control (same protocol, one more run_id)

Repeat 2a–2b with the *teacher's* weights under the student's 512-byte rolling budget
(run_id `teacher-512-ctl`, backend reference or transformers with `mask=None` on
≤513-token windows). This isolates context-length effects; the golden result says it
will track teacher_native closely, which is itself the finding to confirm at scale.

### 2d. Report rows

For each stream (teacher_native frozen values, teacher-512, student): P/R/F1, margin
over the null with CI and p, margin vs the strongest structure-blind baseline with CI,
entropy/boundary-density diagnostics, BPP, and wall-clock/VRAM per stream. Use the
existing `reports/mid_submission/build_results.py` conventions for phrasing; write the
student numbers into a NEW report section — do not edit the submitted PDFs.

---

## Part 3 — Efficiency measurement (falls out of 2a, no extra run)

While extracting, record per-GPU: tokens/s, wall-clock per file-size bucket, peak VRAM,
and parameter count (student: 14,262,144 vs teacher patcher: 99,512,064). Compare
against the teacher's recorded 3,400 tokens/s float32 at 1.36 GiB peak
(`docs/STREAM_A_FIDELITY.md` §5.4). Report **boundary-quality-per-unit-compute**, not
raw speed: the student is 7× smaller but loses a few F1 points (golden evidence).

---

## Part 4 — Domain adaptation comparison (ONLY after Part 2; requires new training)

Question: is the student's gap a distillation artifact or an architecture limit?
Plan (do not start before Part 2 results are in):

1. Add `--init-from <checkpoint_dir>` to `escape_distill/train.py` so the teacher
   architecture (dim 768, 14 layers, 12 heads) is initialized from the frozen patcher
   and trained on the SAME extension train split with the SAME budget (6,860 steps ≈
   2.8 GPU-hours) — this is the "domain adaptation" arm. Use the same seed/optimizer
   schedule. Never overwrite the frozen checkpoint file: save to
   `results/adapt/<run>/teacher.pt`.
2. Evaluate with the exact `escape_distill/evaluate.py` harness, then score under C-v1
   (Part 2 pipeline) with its own calibration record.
3. Verdict rule (pre-declared): adaptation is worth pursuing only if its C-v1 margin
   over the null exceeds the student's by more than the bootstrap CI width in at least
   two domains. A negative result here is still a result (distillation ≈ adaptation).

---

## Part 5 — Pre-registered English holdout (data task, no scoring yet)

The evaluation report pre-registered this rule (deterministic, before any outcomes):

- Pool: the **552** extension-prose articles (11,942 files) that were never used for
  training or the window eval (verify with
  `results/distill/full/manifest_ext.parquet` ∪ `corpus_ext/manifest_ext.parquet`).
- Selection: `article_rank = min over the article's files of int(sha256(bytes)[:8],16)`;
  take the **120 articles with the lowest article_rank** (≈4,107 files, ≈3.06 MB).
- Materialize those bytes + benepar structure now (`stream_b/build_ext_corpus` for
  prose already computes rank; run `build_prose_structure.py` on the selected files),
  store under `corpus_ext/holdout/`, and **do not score them** until the team signs off
  the selection. This is the only route to a confirmatory English result.

---

## Safety checklist (non-negotiable)

- Do not modify, move, or overwrite `results/final/c-v1-2721954/` (frozen held-out
  results) or `results/final/stream_a_fidelity_ada/`.
- Do not overwrite `results/distill/full/student/student.pt` or `runs.json` entries
  that already exist; new runs get new run_ids.
- Do not recalibrate the teacher; do not touch `escape_scoring/` or `schemas/v1/`.
- Do not tune anything against the held-out test split. All model/hyperparameter
  decisions happen on train/val or the calibration split.
- Keep large corpora and JSONL on node-local `$SCR`; home quota is tight.
- Review before `git push` from Ada; do not commit checkpoints or scratch artifacts
  (only manifests, JSONs, CSVs, and code).
