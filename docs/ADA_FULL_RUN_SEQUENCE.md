# ADA FULL-RUN SEQUENCE — Student C-v1, τ sweep, Teacher-512 control, English holdout, Domain adaptation

> **Status: EXECUTED** (Ada, 28–31 Oct 2026). Every step below was run and gated;
> results, numbers and job ids are in [`docs/FULL_RUN_RESULTS.md`](FULL_RUN_RESULTS.md),
> logs in `results/final/logs/`. The document is retained as the pre-registration record.

**For:** Divyansh · **Run everything in order. Do not skip a step or a gate.** · **Base:** repo `main`
with `escape_distill/extract.py`, `retau.py`, `score_a_runs.py` and `train.py --init-from` (all
committed; `git pull` first). Companion docs: `docs/ADA_DISTILL_RUNBOOK.md` (details),
`reports/ESCAPE_STUDENT_CV1_EVALUATION.md` (golden baselines), `docs/SCORING_PROTOCOL.md` (frozen).

**Hard rules**
- Never modify `results/final/c-v1-2721954/` or the frozen calibration record.
- New runs get NEW run_ids. Never re-run an existing run_id with different settings.
- k and τ are selected on **calibration files only**. Test files never select anything.
- Record every job id, the commands exactly as run, and the resulting file hashes.

---

## STEP 0 — Sync and identity hashes (10 min)

```bash
cd ~/escape-final-20260928          # the project root on Ada (adapt if different)
git pull --rebase origin main
# No `git lfs pull` needed: main has no .gitattributes and student.pt is a plain git blob
# (57,076,600 bytes). Only the old `distill-repro-39379` branch tracks *.pt with LFS.
git ls-tree main --name-only | grep -i gitatt || echo "no .gitattributes on main (correct; .pt files are plain blobs, no LFS)"
source scripts/ada_final/env.sh
CK=$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01
sha256sum results/distill/full/student/student.pt "$CK/patcher.safetensors"
# student.pt -> aeef99da84fd0e97d499717e891b189f7f3ada883810ec9dddf99afbb560cf95
# patcher.safetensors -> 888df74288bb8cc332bb57895a100e2e26031c21ba24f91caf0ef3a33e2e1b4d
# ALSO verify corpus/ and structure/ are restored (job 10_data of docs/FINAL_RUN.md); if not, restore them FIRST.
```
Gate: hashes match exactly. If `student.pt` is a 133-byte LFS pointer (a leftover from the
`distill-repro-39379` branch) or its hash differs, stop and run
`git checkout main -- results/distill/full/student/student.pt` before anything else.

## STEP 1 — Golden gate for the new extractor on Ada (15 min, MUST PASS)

```bash
a_py -m escape_distill.extract --input golden --domains py cpp prose \
  --model student --run-id student-golden-gate --device cuda --batch-windows 64
a_py -m escape_distill.score_a_runs --run-ids student-golden-gate --out results/distill/golden_gate_ada.json
```
Gate: per-domain F1/P/R must reproduce the committed golden table
(`results/student_cv1_golden_v1/scores.json`): py F1 0.1313 (P 0.072/R 0.789), cpp 0.2196
(0.126/0.836), prose 0.4290 (0.276/0.968). Tolerance: ±0.002. If it fails, **stop** — do not
extract 16,300 files with a broken extractor.

## STEP 2 — Student extraction, full corpus (PRIMARY; ~12–15 h on 4×2080 Ti)

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
# retry failures once
a_py -m escape_distill.extract --input corpus --model student \
  --run-id student-v1-14326k --device cuda --batch-windows 64 --file-ids $IDS
EOF
sbatch scripts/ada_distill/student_extract.sbatch
```
Completion gate: every one of the 16,300 corpus files has
`entropies/student-v1-14326k/{fid}.npy` (float32, shape n_bytes) and
`boundaries/student-v1-14326k/{fid}.parquet` with `patch_length.sum() == n_bytes`. Check with:

```bash
a_py - <<'PY'
import pandas as pd
from pathlib import Path
m = pd.read_parquet('corpus/manifest.parquet')
missing = [f for _, r in m.iterrows() if not Path(f"entropies/student-v1-14326k/{r.file_id}.npy").exists()
           or not Path(f"boundaries/student-v1-14326k/{r.file_id}.parquet").exists()]
print("missing:", len(missing), missing[:5])
PY
```
**Record** in the job log: wall-clock and `tokens/s` per file-size bucket (needed for the
efficiency table). If a shard dies, rerun the same array job — it resumes (skips done files).

## STEP 3 — C-v1 conversion, validation, calibration, scoring (student)

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
cp "$DATA"/*.jsonl results/final/student-v1-inputs.jsonl.bak 2>/dev/null || true
```
Gate: `validate` passes for all 16,300 samples; `calibrate` writes a record (record the
selected k per language — expect k=0); `score` writes evaluation/{scores,per_file,diagnostics}.csv+json.
Keep `student-v1-calibration.json` untouched once written (it is the student's frozen record).

## STEP 4 — Teacher-512 control (subset; ~2–3 GPU-hours)

The stride-1 teacher control is infeasible on the full corpus, so it runs on a
**pre-registered subset** (this selection rule is fixed before any scoring):

```bash
a_py - <<'PY'
import hashlib, json, pandas as pd
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
```
```bash
IDS_T512="$(cat results/final/teacher512_subset_python.txt results/final/teacher512_subset_cpp.txt)"
a_py -m escape_distill.extract --input corpus --model teacher \
  --run-id teacher-512-ctl --device cuda --batch-windows 32 --file-ids $IDS_T512
a_py -m escape_distill.score_a_runs \
  --run-ids swa512-float32-t1.3354-mplnone-91aa6b8e teacher-512-ctl student-v1-14326k \
  --file-ids $IDS_T512 --out results/final/teacher512_subset_scores.json
```
This scores the frozen teacher run, the teacher-512 control, and the student on the SAME
subset at k=0 with the fixed-EOF null and baselines. Gate: teacher_native ≈ teacher_512
within bootstrap CI (expected from the golden result); record the deltas.

## STEP 5 — τ sweep on the CALIBRATION split (teacher; minutes, no model runs)

Pre-registered rule (fixed before looking at any number): grid
τ ∈ {0.8, 1.0, 1.335442066192627, 1.7, 2.0, 2.5}; per language choose
τ* = argmax of median per-file (F1_observed − mean_F1_H0) over calibration files, ties to the
smaller τ; then re-freeze k on calibration under τ* with the frozen k-criterion; then run
**one** test-split scoring at (τ*, k*).

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
The calibration-side score uses k=0 per the frozen record; the τ* rule is applied to the
**median per-file margin** reported there. Then:

```bash
# ONE confirmatory run at the pre-chosen (tau*, k*). Fill in tau* from the sweep output.
c_py -m escape_distill.retau --src-run swa512-float32-t1.3354-mplnone-91aa6b8e \
    --dst-run teacher-tau-STAR --tau <tau*>          # full corpus (no --file-ids)
TIDS=$(c_py -c "import json; rec=json.load(open('results/final/c-v1-2721954/calibration.json')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='test')))")
c_py -m escape_distill.score_a_runs \
  --run-ids teacher-tau-STAR swa512-float32-t1.3354-mplnone-91aa6b8e \
  --file-ids $TIDS --out results/final/teacher_tau_star_test.json
```
Also run the full C-v1 CLI for the confirmatory point (its own calibration record, k
co-frozen under τ*): `convert-legacy --run-id teacher-tau-STAR` → `validate` →
`calibrate` → `score` → `baselines` (exactly as Step 3). Gate: record k* and τ*; never
rescan the test split at another τ afterwards.

## STEP 6 — τ sweep + confirmatory for the STUDENT (same rule, same order)

```bash
for TAU in 0.8 1.0 1.7 2.0 2.5; do
  TAG=$(echo $TAU | tr -d '.')
  c_py -m escape_distill.retau --src-run student-v1-14326k --dst-run student-tau$TAG --tau $TAU --file-ids $CALIDS
done
c_py -m escape_distill.score_a_runs \
  --run-ids student-tau08 student-tau10 student-v1-14326k student-tau17 student-tau20 student-tau25 \
  --file-ids $CALIDS --out results/final/student_tau_sweep_calib.json --perms 10000
# pick student tau* with the same rule, then ONE test-split run (score_a_runs + full CLI) as in Step 5.
```
Expected (from the golden sweep): student τ* above the teacher's τ*; margins stay positive at
every τ; neither model beats the strongest baseline at any τ.

## STEP 7 — English holdout materialization (data only, NO scoring)

```bash
a_py - <<'PY'
import pandas as pd, hashlib
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
# write the selected bytes (requires corpus_ext materialized first; see runbook Part 1)
import pandas as pd
from pathlib import Path
m = pd.read_parquet('corpus_ext/holdout_manifest.parquet')
missing = 0
for _, r in m.iterrows():
    src = Path(f"corpus_ext/{r.domain}/{r.file_id}.bin")
    if src.exists():
        import shutil; shutil.copy(src, f"corpus_ext/holdout/text/{r.file_id}.bin")
    else:
        missing += 1
print('missing source files:', missing)
PY
# benepar structure for the holdout paragraphs (same walker as the evaluation prose):
mkdir -p corpus_ext/holdout/structure
# (run build_prose_structure.py pointed at the holdout manifest -- see stream_b/build_prose_structure.py;
#  do it with the same byte-roundtrip validation. DO NOT score these files.)
```
**Do not compute any boundary scores for holdout files** until the team signs off the
selection (recorded in `reports/ESCAPE_STUDENT_CV1_EVALUATION.md` §8.6).

## STEP 8 — O5 label audit (CPU, no training)

Re-run the tree-sitter queries behind `stream_b/build_o5_regions.py` against the frozen
`results/final/c-v1-2721954/o5/o5_regions.csv` labels: check every unsafe mask
(`*`, `&`, `new`, `delete`) and analysis-region statement type on a stratified sample of
50 files with unsafe regions. Record discrepancies in a short note
(`docs/O5_LABEL_AUDIT.md`). Do not change the scorer or re-run O5 until the team reviews
the audit.

## STEP 9 — Domain adaptation of the teacher (ONLY if queue allows; ~1 day)

```bash
MANIFEST="$SCR/corpus_ext/manifest_ext.parquet"      # materialized in runbook Part 1
a_py -m escape_distill.train \
  --ckpt-dir "$CK" --data "$MANIFEST" --out results/adapt/teacher-finetuned \
  --train-split train --domains cpp,prose,py \
  --epochs 2 --batch-size 16 --lr 2e-4 --weight-decay 0.1 --warmup 100 \
  --temperature 2.0 --alpha 0.1 --dim 768 --layers 14 --heads 12 \
  --init-from "$CK" --log-every 100
a_py -m escape_distill.evaluate --ckpt-dir "$CK" \
  --student results/adapt/teacher-finetuned/student.pt --data "$MANIFEST" \
  --out results/adapt/teacher-finetuned/eval --eval-split val --domains cpp,prose,py
```
Then extract + C-v1-score the adapted model exactly as Steps 2–3 (new run_id
`teacher-adapt-v1`). Pre-registered verdict: continue with adaptation only if its C-v1
margin over the null beats the student's by more than the bootstrap CI width in ≥2 domains.
A negative result is reported as-is.

## STEP 10 — Collect everything for the report

Copy back to the laptop (or commit): `results/final/student-v1-eval/`,
`student-v1-baselines.json`, `student-v1-calibration.json`, `teacher512_subset_scores.json`,
`teacher_tau_sweep_calib.json`, `teacher_tau_star_test.json`, student sweep counterparts,
`results/adapt/...` (if run), extraction logs (for the efficiency table), and
`docs/O5_LABEL_AUDIT.md`. List every number against its source file; do not edit the
frozen `c-v1-2721954` directory.

---

## Do-not-skip checklist
- [ ] STEP 0 hashes match; `student.pt` is the real 57 MB checkpoint (plain blob, not an LFS pointer)
- [ ] STEP 1 golden gate reproduces the committed golden table
- [ ] STEP 2 all 16,300 files have complete artifacts (checked before scoring)
- [ ] STEP 3 validate passes; student calibration record frozen
- [ ] STEP 4 subset selection rule recorded BEFORE scoring
- [ ] STEP 5/6 τ* chosen by the pre-registered rule on calibration only; exactly one confirmatory run
- [ ] STEP 7 holdout materialized, zero scoring
- [ ] STEP 8 O5 label audit written up
- [ ] STEP 9 (if run) verdict applied as pre-registered
