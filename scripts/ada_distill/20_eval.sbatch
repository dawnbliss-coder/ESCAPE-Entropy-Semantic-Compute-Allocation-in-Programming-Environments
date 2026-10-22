#!/bin/bash
#SBATCH --job-name=esc-distill-eval
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/distill/logs/eval_%j.out
# Evaluate the distilled student against the BLT teacher on the held-out (val) split of the
# extension corpus: entropy agreement, KL, bits/byte, and boundary F1 (patches derived with
# the frozen tau), all against an untrained-student baseline.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ada_final/env.sh
mkdir -p runs/distill/logs results/distill

CKPT="${CKPT:-$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01}"
DATA="${DATA:-$PWD/corpus_ext/manifest_ext.parquet}"
STUDENT="${STUDENT:-$PWD/results/distill/student/student.pt}"
OUT="${OUT:-$PWD/results/distill/eval}"
DOMAINS="${DOMAINS:-py,prose,cpp}"
LIMIT="${LIMIT:-2000}"           # cap val files for a fast, representative eval; empty = all

stamp "eval: student=$STUDENT data=$DATA domains=$DOMAINS limit=${LIMIT:-all}"
a_py -m escape_distill.evaluate \
  --ckpt-dir "$CKPT" --student "$STUDENT" --data "$DATA" --out "$OUT" \
  --eval-split val --domains "$DOMAINS" ${LIMIT:+--limit-files "$LIMIT"}
stamp "eval done -> $OUT/distill_eval.json"
