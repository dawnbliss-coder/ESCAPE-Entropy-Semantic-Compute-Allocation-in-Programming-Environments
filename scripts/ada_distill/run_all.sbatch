#!/bin/bash
#SBATCH --job-name=esc-distill-full
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=runs/distill/logs/full_%j.out
# One-shot improved distillation: materialize a large slice of the extension corpus into
# node-local scratch (no 70k inodes on the tight home quota), distill a bigger student from
# the BLT teacher with tuning, evaluate on the held-out split. Only the small results
# (student.pt + JSON/CSV) are written to home under results/distill/.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ada_final/env.sh
mkdir -p runs/distill/logs results/distill/full

CKPT="${CKPT:-$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01}"
CORPUS="$SCR/corpus_ext"                        # node-local scratch
PY_LIMIT="${PY_LIMIT:-15000}"
PROSE_LIMIT="${PROSE_LIMIT:-8000}"
CPP_LIMIT="${CPP_LIMIT:-6000}"

stamp "materialize into scratch: $CORPUS  (py=$PY_LIMIT prose=$PROSE_LIMIT cpp=$CPP_LIMIT)"
a_py -m stream_b.build_ext_corpus py    --limit "$PY_LIMIT"    --out "$CORPUS"
a_py -m stream_b.build_ext_corpus prose --limit "$PROSE_LIMIT" --out "$CORPUS"
a_py -m stream_b.build_ext_corpus cpp   --limit "$CPP_LIMIT"   --out "$CORPUS" || \
  echo "WARN: cpp materialize failed (The Stack terms?); continuing with py+prose"
cp "$CORPUS/manifest_ext.parquet" "$CORPUS/DATA_NOTE.json" results/distill/full/ 2>/dev/null || true

MANIFEST="$CORPUS/manifest_ext.parquet"
DOMAINS="$(a_py - "$MANIFEST" <<'PY'
import sys, pandas as pd
print(",".join(sorted(pd.read_parquet(sys.argv[1])["domain"].unique())))
PY
)"
stamp "distill: domains=$DOMAINS  (bigger student 384/8/6, 2 epochs, T=2, alpha=0.1)"
a_py -m escape_distill.train \
  --ckpt-dir "$CKPT" --data "$MANIFEST" --out results/distill/full/student \
  --train-split train --domains "$DOMAINS" \
  --epochs 2 --batch-size 48 --lr 5e-4 --weight-decay 0.1 --warmup 200 \
  --temperature 2.0 --alpha 0.1 --dim 384 --layers 8 --heads 6 --log-every 100

stamp "evaluate on held-out val"
a_py -m escape_distill.evaluate \
  --ckpt-dir "$CKPT" --student results/distill/full/student/student.pt \
  --data "$MANIFEST" --out results/distill/full/eval \
  --eval-split val --domains "$DOMAINS" --limit-files 3000

stamp "DONE -> results/distill/full/eval/distill_eval.json"
a_py - <<'PY'
import json
d = json.load(open("results/distill/full/eval/distill_eval.json"))
t, b = d["trained"]["all"], d["untrained_baseline"]["all"]
print("=== FULL RUN RESULT (overall) ===")
print(f"  boundary F1       trained {t['boundary_f1']:.3f}  untrained {b['boundary_f1']:.3f}")
print(f"  entropy Pearson r trained {t['entropy_pearson_r']:.3f}  untrained {b['entropy_pearson_r']:.3f}")
print(f"  KL(teacher||stud) trained {t['kl_teacher_student_nats_per_byte']:.3f}  untrained {b['kl_teacher_student_nats_per_byte']:.3f}")
print(f"  bits/byte student {t['bits_per_byte_student']:.3f} vs teacher {t['bits_per_byte_teacher']:.3f}")
PY
