#!/bin/bash
# plan.md section 3 (P2): the alpha = 1 control. Same data, architecture, seed and schedule
# as the distilled student (run_all.sbatch), but the loss is hard-label CE only (no teacher
# KL), so the comparison isolates what distillation adds. Both students are evaluated on
# the SAME held-out val windows against the teacher in this job.
#SBATCH --job-name=esc-alpha1
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/alpha1-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
CKPT="$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"
CORPUS="$SCR/corpus_ext"
OUT=results/distill/alpha1
mkdir -p "$OUT"
stamp "materialize ext corpus (same limits as run_all.sbatch)"
a_py -m stream_b.build_ext_corpus py    --limit 15000 --out "$CORPUS"
# The prose step can abort at interpreter exit after writing everything ("Fatal Python error:
# PyGILState_Release", rc 134; see run_all.sbatch). Accept that only if the output is complete.
a_py -m stream_b.build_ext_corpus prose --limit 8000  --out "$CORPUS" || {
  rc=$?
  a_py - "$CORPUS" <<'PY' || { echo "ERROR: prose materialize failed (rc=$rc) and its output is incomplete"; exit "$rc"; }
import json, sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
m = pd.read_parquet(out / "manifest_ext.parquet")
ids = m.loc[m["domain"] == "prose", "file_id"]
missing = [f for f in ids if not (out / "prose" / f"{f}.bin").exists()]
note = json.loads((out / "DATA_NOTE.json").read_text())
assert len(ids) > 0 and not missing and note["prose"]["files"] == len(ids), (len(ids), len(missing))
print(f"prose output complete: {len(ids)} manifest rows, all .bin present")
PY
  echo "WARN: prose materialize exited rc=$rc at interpreter exit; output verified complete, continuing"
}
a_py -m stream_b.build_ext_corpus cpp   --limit 6000  --out "$CORPUS"
cp "$CORPUS/manifest_ext.parquet" "$CORPUS/DATA_NOTE.json" "$OUT/"
stamp "data identity vs the distilled student's manifest"
a_py - "$CORPUS/manifest_ext.parquet" <<'PY'
import sys, pandas as pd
a = pd.read_parquet(sys.argv[1]); b = pd.read_parquet("results/distill/full/manifest_ext.parquet")
key = ["file_id", "sha256", "split"] if "split" in a.columns else ["file_id", "sha256"]
same = set(map(tuple, a[key].values)) == set(map(tuple, b[key].values))
print("manifest identical to results/distill/full:", same, len(a), len(b))
PY
MANIFEST="$CORPUS/manifest_ext.parquet"
stamp "train alpha=1 (CE only), otherwise identical to the distilled student"
a_py -m escape_distill.train \
  --ckpt-dir "$CKPT" --data "$MANIFEST" --out "$OUT/student" \
  --train-split train --domains cpp,prose,py \
  --epochs 2 --batch-size 48 --lr 5e-4 --weight-decay 0.1 --warmup 200 \
  --temperature 2.0 --alpha 1.0 --dim 384 --layers 8 --heads 6 --log-every 100
for WHO in alpha1 distilled; do
  PT=$([ $WHO = alpha1 ] && echo "$OUT/student/student.pt" || echo results/distill/full/student/student.pt)
  stamp "evaluate $WHO on held-out val (same windows)"
  a_py -m escape_distill.evaluate --ckpt-dir "$CKPT" --student "$PT" --data "$MANIFEST" \
    --out "$OUT/eval_$WHO" --eval-split val --domains cpp,prose,py --limit-files 3000
done
rm -rf "$SCR"
stamp "ALPHA1 CONTROL DONE"
