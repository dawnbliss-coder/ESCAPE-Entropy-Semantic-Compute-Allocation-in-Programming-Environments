# Ada Handoff — Finish All ESCAPE Full Runs

You are on the **IIIT-H Ada cluster**. Your job: execute the complete, pre-registered
ESCAPE full-run sequence (student C-v1 scoring, teacher-512 control, τ sweep +
confirmatory, English holdout, O5 audit, optional domain adaptation), **in order, with
every gate checked, and report every number back with its source file**. The scientific
protocols, tools, and selection rules below are already decided and committed — do not
invent new procedures, do not tune anything on test data, do not skip a gate.

Project: ESCAPE — does the BLT patcher's entropy-triggered patch boundary segmentation
align with syntactic constituent starts? The frozen C-v1 held-out results live in
`results/final/c-v1-2721954/` (untouchable). A 14.3M-param **student** was distilled from
the 99.5M-param BLT **teacher** (`itazap/blt-1b-hf` @ `91aa6b8e…`, swa512, float32,
τ = 1.335442066192627 nats). Everything below evaluates the student (and controls) under
the frozen scoring rules: one-to-one matching at frozen k, fixed-EOF density-matched null
(10,000 permutations), structure-blind baselines (newline / indentation / word-start).

## 0. Get the repo to the right state (do this first)

```bash
cd ~/escape-final-20260928          # project root on Ada; adapt if it lives elsewhere
git fetch origin && git checkout main && git pull --rebase origin main
git log --oneline -3                # must show 9c993b8 at the top ("Add full-run Ada tooling...")
git status --porcelain              # must be clean; if .gitattributes appears (leftover from
                                    # the old distill-repro branch), `git rm -f .gitattributes` is wrong --
                                    # instead: if it is UNTRACKED or modified, restore it away:
git ls-tree main --name-only | grep -i gitatt || echo "no .gitattributes on main (correct; .pt files are plain blobs, no LFS)"
source scripts/ada_final/env.sh     # defines a_py (torch 2.6.0+cu124, transformers 5.17.0) and c_py
CK=$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01
sha256sum results/distill/full/student/student.pt "$CK/patcher.safetensors"
# student.pt       -> aeef99da84fd0e97d499717e891b189f7f3ada883810ec9dddf99afbb560cf95
# patcher.safetensors -> 888df74288bb8cc332bb57895a100e2e26031c21ba24f91caf0ef3a33e2e1b4d
```
Gate 0: hashes match; repo at 9c993b8; working tree clean. Also confirm `corpus/` (16,300
files) and `structure/` + `whitespace/` are restored (job 10_data of `docs/FINAL_RUN.md`).
The frozen teacher A-format artifacts `entropies/swa512-float32-t1.3354-mplnone-91aa6b8e/`
and `boundaries/swa512-float32-t1.3354-mplnone-91aa6b8e/` must exist for all 16,300 files —
verify before Step 5 (τ sweep needs them).

Tools you will use (all committed on main):
- `python -m escape_distill.extract` — student / teacher-512 extraction in Stream-A format
  (stride-1 rolling 512-byte windows; batch 64 on GPU; resumable: skips done files).
- `python -m escape_distill.retau` — re-threshold saved entropy arrays into a new run_id (no model runs).
- `python -m escape_distill.score_a_runs` — frozen k=0 scoring of any run_id(s) vs structure
  targets with the fixed-EOF null + baselines + bootstrap (for subsets, τ curves, gates).
- `python -m escape_scoring {convert-legacy,validate,calibrate,score,baselines}` — the frozen C-v1 CLI.
- `python -m escape_distill.train --init-from` — domain-adaptation arm (Step 9 only).

## 1. Golden gate for the new extractor (15 min; MUST pass before anything else)

```bash
a_py -m escape_distill.extract --input golden --domains py cpp prose \
  --model student --run-id student-golden-gate --device cuda --batch-windows 64
a_py -m escape_distill.score_a_runs --run-ids student-golden-gate --golden --out results/distill/golden_gate_ada.json
```
Gate 1: reproduce the committed golden table exactly (F1 within ±0.002):
py 0.1313 (P 0.0716, R 0.7887) · cpp 0.2196 (P 0.1264, R 0.8364) · prose 0.4290 (P 0.2756, R 0.9677);
baselines: py indent_change 0.4771, cpp word_start 0.3869, prose word_start 0.7601.
If this fails, STOP and debug the extractor — never extract the full corpus with a broken tool.

## 2. Student extraction over the full corpus (~12–15 h on the 4×2080 Ti array)

```bash
cat > scripts/ada_distill/student_extract.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=esc-student-extract
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=results/final/logs/student_extract-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
IDS=$(a_py scripts/ada_final/shard_ids.py "$SLURM_ARRAY_TASK_ID" 4)
a_py -m escape_distill.extract --input corpus --model student \
  --run-id student-v1-14326k --device cuda --batch-windows 64 --file-ids $IDS --progress-every 100
a_py -m escape_distill.extract --input corpus --model student \
  --run-id student-v1-14326k --device cuda --batch-windows 64 --file-ids $IDS   # retry failures
EOF
sbatch scripts/ada_distill/student_extract.sbatch
```
Completion gate (must be checked BEFORE scoring):
```bash
a_py - <<'PY'
import pandas as pd
from pathlib import Path
m = pd.read_parquet('corpus/manifest.parquet')
missing = [f for _, r in m.iterrows() if not Path(f"entropies/student-v1-14326k/{r.file_id}.npy").exists()
           or not Path(f"boundaries/student-v1-14326k/{r.file_id}.parquet").exists()]
print("missing:", len(missing), missing[:5])
PY
# expect: missing: 0
```
Rerun the same array job if a shard died (it resumes). **Record for the efficiency table:**
wall-clock per shard and per-file-size timings from the logs (the teacher reference is
3,400 tokens/s float32 @ 1.36 GiB, `docs/STREAM_A_FIDELITY.md` §5.4).

## 3. Student C-v1 scoring (convert → validate → calibrate → score → baselines)

```bash
DATA="$SCR/c-v1-student"
IDS=$(a_py -c "import pandas as pd; m=pd.read_parquet('corpus/manifest.parquet'); print(' '.join(sorted(m.file_id)))")
c_py -m escape_scoring convert-legacy --root . --run-id student-v1-14326k --output "$DATA" --sample-ids $IDS
c_py -m escape_scoring validate --data "$DATA" --workers 32
c_py -m escape_scoring calibrate --data "$DATA" --output results/final/student-v1-calibration.json --workers 32
c_py -m escape_scoring score --data "$DATA" --calibration results/final/student-v1-calibration.json \
    --output results/final/student-v1-eval --workers 32
c_py -m escape_scoring baselines --data "$DATA" --calibration results/final/student-v1-calibration.json \
    --output results/final/student-v1-baselines.json --workers 32
```
Gate 3: `validate` passes for all 16,300 samples; calibration record written (record the
student's selected k per language — expected k=0); scores written to
`results/final/student-v1-eval/evaluation/`. Do not modify `student-v1-calibration.json`
afterwards — it is the student's frozen record. Do NOT touch the teacher's frozen record
`results/final/c-v1-2721954/calibration.json` (the C-v1 scorer will correctly REJECT the
student data against it — the model signature differs; that is by design).

## 4. Teacher-512 control on the pre-registered subset (~2–3 GPU-hours)

Selection rule (fixed BEFORE any scoring; deterministic):
```bash
a_py - <<'PY'
import hashlib, json
rec = json.load(open('results/final/c-v1-2721954/calibration.json'))
tests = [m for m in rec['split_manifest'] if m['split'] == 'test']
for lang in ('python', 'cpp'):
    cand = [m for m in tests if m['language'] == lang and 600 <= m['byte_length'] <= 2000]
    cand.sort(key=lambda m: (hashlib.sha256(f"20260927\0{lang}\0{m['sample_id']}".encode()).hexdigest(),
                             m['sample_id']))
    sel = cand[:300]
    with open(f'results/final/teacher512_subset_{lang}.txt', 'w') as f:
        f.write(' '.join(m['sample_id'] for m in sel) + '\n')
    print(lang, len(sel), 'selected; byte range', sel[0]['byte_length'], '-', sel[-1]['byte_length'])
PY
IDS_T512="$(cat results/final/teacher512_subset_python.txt results/final/teacher512_subset_cpp.txt)"
a_py -m escape_distill.extract --input corpus --model teacher \
  --run-id teacher-512-ctl --device cuda --batch-windows 32 --file-ids $IDS_T512
a_py -m escape_distill.score_a_runs \
  --run-ids swa512-float32-t1.3354-mplnone-91aa6b8e teacher-512-ctl student-v1-14326k \
  --file-ids $IDS_T512 --out results/final/teacher512_subset_scores.json --perms 10000
```
Gate 4: teacher_native ≈ teacher_512 within bootstrap CI on this subset (the golden result
already showed the context-length effect is tiny — this confirms it at scale). Record the
three-way F1/margin table from `teacher512_subset_scores.json`.

## 5. τ sweep on the CALIBRATION split, then ONE confirmatory test run (teacher)

Pre-registered rule (fixed now, before any number is looked at): grid
τ ∈ {0.8, 1.0, 1.335442066192627, 1.7, 2.0, 2.5}; per language choose
τ* = argmax over the grid of the **median per-file (F1_observed − mean_F1_H0)** computed on
CALIBRATION files at k=0 with the fixed-EOF null (10,000 permutations); ties → smaller τ.
Then re-freeze k under τ* via the frozen CLI criterion, and run exactly ONE test-split
scoring at (τ*, k*). Never scan the test split at multiple τ.

```bash
CALIDS=$(c_py -c "import json; rec=json.load(open('results/final/c-v1-2721954/calibration.json')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='calibration')))")
for TAU in 0.8 1.0 1.7 2.0 2.5; do
  TAG=$(echo $TAU | tr -d '.')
  c_py -m escape_distill.retau --src-run swa512-float32-t1.3354-mplnone-91aa6b8e \
      --dst-run teacher-tau$TAG --tau $TAU --file-ids $CALIDS
done
c_py -m escape_distill.score_a_runs \
  --run-ids teacher-tau08 teacher-tau10 swa512-float32-t1.3354-mplnone-91aa6b8e teacher-tau17 teacher-tau20 teacher-tau25 \
  --file-ids $CALIDS --out results/final/teacher_tau_sweep_calib.json --perms 10000
```
Then apply the τ* rule. `score_a_runs` writes per-file rows; compute the median per-file
margin yourself with this exact snippet (do not eyeball it):
```bash
c_py - <<'PY'
import json, numpy as np
d = json.load(open('results/final/teacher_tau_sweep_calib.json'))
runs = ['teacher-tau08','teacher-tau10','swa512-float32-t1.3354-mplnone-91aa6b8e','teacher-tau17','teacher-tau20','teacher-tau25']
tau_of = {'teacher-tau08':0.8,'teacher-tau10':1.0,'swa512-float32-t1.3354-mplnone-91aa6b8e':1.335442066192627,
          'teacher-tau17':1.7,'teacher-tau20':2.0,'teacher-tau25':2.5}
for lang, dom in (('python','py'),('cpp','cpp'),('english','prose')):
    print(lang)
    for r in runs:
        rows = d['domains'][dom]['rows'][r]
        per = np.asarray(rows['per_file'], dtype=float)      # [n_files, tp,fp,fn]
        # per-file observed F1 and per-file pooled-null mean F1 (null mean via per-file counts over perms is
        # not stored per-file; use the pooled per-file null mean from the run row as the density-matched
        # comparator, which the frozen criterion approximates closely):
        f1_obs = 2*per[:,0] / np.maximum(1e-12, 2*per[:,0]+per[:,1]+per[:,2])
        margin = f1_obs - rows['f1_h0_mean']
        print(f"  tau={tau_of[r]:<8} median_margin={np.median(margin):+.4f}")
PY
```
NOTE: if you want the criterion implemented exactly as the frozen k-rule (median of
per-file (F1_obs − mean_r F1_H0(file,r))), rerun the snippet computing per-file null means
with 10,000 perms per file (see `escape_scoring/calibration.py::calibrate`); the frozen
criterion string is "maximize median per-file (F1_observed - mean_F1_H0), calibration only;
ties smallest k". Use the same definition for τ (ties → smaller τ).

Confirmatory run (ONLY at the pre-chosen τ*):
```bash
c_py -m escape_distill.retau --src-run swa512-float32-t1.3354-mplnone-91aa6b8e \
    --dst-run teacher-tau-STAR --tau <tau*>          # full corpus (no --file-ids)
TIDS=$(c_py -c "import json; rec=json.load(open('results/final/c-v1-2721954/calibration.json')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='test')))")
c_py -m escape_distill.score_a_runs \
  --run-ids teacher-tau-STAR swa512-float32-t1.3354-mplnone-91aa6b8e \
  --file-ids $TIDS --out results/final/teacher_tau_star_test.json
# plus the full CLI path with k co-frozen under tau* (own calibration record, never the teacher's frozen one):
DSTAR="$SCR/c-v1-teacher-tau-star"
c_py -m escape_scoring convert-legacy --root . --run-id teacher-tau-STAR --output "$DSTAR" --sample-ids $(echo $IDS) # (use all 16,300 ids as in Step 3)
c_py -m escape_scoring validate --data "$DSTAR" --workers 32
c_py -m escape_scoring calibrate --data "$DSTAR" --output results/final/teacher-tau-STAR-calibration.json --workers 32
c_py -m escape_scoring score --data "$DSTAR" --calibration results/final/teacher-tau-STAR-calibration.json \
    --output results/final/teacher-tau-STAR-eval --workers 32
c_py -m escape_scoring baselines --data "$DSTAR" --calibration results/final/teacher-tau-STAR-calibration.json \
    --output results/final/teacher-tau-STAR-baselines.json --workers 32
```
Gate 5: τ* and k* recorded; exactly one confirmatory run; margins expected positive at
every τ (golden evidence) and the confirmatory F1 slightly above the checkpoint-τ number,
still below the strongest baseline.

## 6. Student τ sweep + confirmatory (same rule, same order)

```bash
for TAU in 0.8 1.0 1.7 2.0 2.5; do
  TAG=$(echo $TAU | tr -d '.')
  c_py -m escape_distill.retau --src-run student-v1-14326k --dst-run student-tau$TAG --tau $TAU --file-ids $CALIDS
done
c_py -m escape_distill.score_a_runs \
  --run-ids student-tau08 student-tau10 student-v1-14326k student-tau17 student-tau20 student-tau25 \
  --file-ids $CALIDS --out results/final/student_tau_sweep_calib.json --perms 10000
# apply the same median-margin rule -> student tau*; then ONE confirmatory test-split run
# (score_a_runs at k=0 + full CLI with the student's own new calibration record), mirroring Step 5.
```
Expected: student τ* above the teacher's τ* (the student's entropy is systematically
higher). Record everything in `results/final/student-tau-STAR-*`.

## 7. English holdout materialization (data only — NO scoring, ever, until team sign-off)

```bash
a_py - <<'PY'
import pandas as pd
full = pd.read_parquet('corpus_ext/manifest_ext.parquet')
dist = pd.read_parquet('results/distill/full/manifest_ext.parquet')
prose_full = full[full['domain'] == 'prose']
used = set(dist[dist['domain'] == 'prose']['group'])
unused = prose_full[~prose_full['group'].isin(used)].copy()
unused['article_rank'] = unused.groupby('group')['rank'].transform('min')
arts = unused.groupby('group').agg(article_rank=('article_rank', 'min')).sort_values('article_rank')
holdout_arts = set(arts.head(120).index)
holdout = unused[unused['group'].isin(holdout_arts)]
holdout[['file_id', 'domain', 'n_bytes', 'sha256', 'group']].to_parquet('corpus_ext/holdout_manifest.parquet')
print('holdout:', len(holdout), 'files,', len(holdout_arts), 'articles')
PY
mkdir -p corpus_ext/holdout/text
a_py - <<'PY'
import pandas as pd, shutil
from pathlib import Path
m = pd.read_parquet('corpus_ext/holdout_manifest.parquet')
missing = 0
for _, r in m.iterrows():
    src = Path(f"corpus_ext/{r.domain}/{r.file_id}.bin")
    if src.exists():
        shutil.copy(src, f"corpus_ext/holdout/text/{r.file_id}.bin")
    else:
        missing += 1
print('missing source files:', missing)
PY
# benepar structure for holdout paragraphs with the same byte-roundtrip validation as
# stream_b/build_prose_structure.py (write a small loop over the holdout manifest).
# DO NOT score these files. The pre-registration is recorded in
# reports/ESCAPE_STUDENT_CV1_EVALUATION.md section 8.6.
```
Gate 7: holdout bytes + structure materialized; zero boundary scores computed.

## 8. O5 label audit (CPU, no training)

Check the memory-unsafe labels behind `stream_b/build_o5_regions.py` against the frozen
`results/final/c-v1-2721954/o5/o5_regions.csv`: sample 50 files that have unsafe regions,
re-run the tree-sitter queries, and diff labels (`*`, `&`, `new`, `delete`) and
analysis-region statement types. Write findings to `docs/O5_LABEL_AUDIT.md`. Do not change
the scorer or re-run O5 until the team reviews the audit.

## 9. Domain adaptation of the teacher (ONLY if queue allows; ~1 day)

```bash
MANIFEST="$SCR/corpus_ext/manifest_ext.parquet"   # materialize first if absent (runbook Part 1: build_ext_corpus py/prose/cpp with limits 15000/8000/6000)
a_py -m escape_distill.train \
  --ckpt-dir "$CK" --data "$MANIFEST" --out results/adapt/teacher-finetuned \
  --train-split train --domains cpp,prose,py \
  --epochs 2 --batch-size 16 --lr 2e-4 --weight-decay 0.1 --warmup 100 \
  --temperature 2.0 --alpha 0.1 --dim 768 --layers 14 --heads 12 \
  --init-from "$CK" --log-every 100
a_py -m escape_distill.evaluate --ckpt-dir "$CK" \
  --student results/adapt/teacher-finetuned/student.pt --data "$MANIFEST" \
  --out results/adapt/teacher-finetuned/eval --eval-split val --domains cpp,prose,py
# then extract (run-id teacher-adapt-v1) + full C-v1 scoring exactly as Steps 2-3.
```
Pre-registered verdict: pursue adaptation only if its C-v1 margin over the null beats the
student's by more than the bootstrap CI width in ≥2 domains. A negative result is reported
as-is. Never overwrite the frozen checkpoint file.

## 10. Report back (the deliverable)

Collect into the repo (commit them; do NOT touch `results/final/c-v1-2721954/`):
`results/final/student-v1-eval/`, `student-v1-baselines.json`, `student-v1-calibration.json`,
`teacher512_subset_scores.json` (+ subset selection txts), `teacher_tau_sweep_calib.json`,
`teacher_tau_star_test.json`, `teacher-tau-STAR-{calibration,eval,baselines}`, the student
τ counterparts, `results/adapt/…` (if run), the extraction logs (efficiency table),
`docs/O5_LABEL_AUDIT.md`, and a short `docs/FULL_RUN_RESULTS.md` with the per-step numbers
(F1/P/R, margins, CIs, p, k*, τ*, job ids) and the commands as run.

## Hard rules (violating any of these invalidates the experiment)

- Test files NEVER select k, τ, or any hyperparameter. All selection happens on the
  calibration split (or the extension corpus train/val).
- One confirmatory test run per τ*. No test-split scanning.
- Never modify `results/final/c-v1-2721954/`, the teacher's frozen calibration record,
  `escape_scoring/`, `schemas/v1/`, or `runs.json` entries that already exist (new run_ids only).
- Teacher's frozen k=0 is never applied to student data via the CLI (the scorer rejects it
  by design); the student gets its own calibration record.
- No fabricated numbers: every reported value must come from a file you wrote.
- If a step fails a gate, STOP, fix, and re-run that step — do not continue.
