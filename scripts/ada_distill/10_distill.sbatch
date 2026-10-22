#!/bin/bash
#SBATCH --job-name=esc-distill
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/distill/logs/distill_%j.out
# Online knowledge distillation of the BLT teacher into the from-scratch StudentBLT,
# on the extension corpus (disjoint from the evaluation set). Teacher = the exact checkpoint
# used for the report; backend auto-selects transformers on Ada.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ada_final/env.sh
mkdir -p runs/distill/logs results/distill

CKPT="${CKPT:-$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01}"
DATA="${DATA:-$PWD/corpus_ext/manifest_ext.parquet}"
OUT="${OUT:-$PWD/results/distill/student}"
DOMAINS="${DOMAINS:-py,prose,cpp}"
LIMIT="${LIMIT:-}"               # e.g. LIMIT=4000 for a quick real run; empty = all train files
EPOCHS="${EPOCHS:-1}"

stamp "distill: teacher=$CKPT data=$DATA domains=$DOMAINS limit=${LIMIT:-all}"
a_py -m escape_distill.train \
  --ckpt-dir "$CKPT" --data "$DATA" --out "$OUT" \
  --train-split train --domains "$DOMAINS" ${LIMIT:+--limit-files "$LIMIT"} \
  --epochs "$EPOCHS" --batch-size 32 --lr 5e-4 --temperature 2.0 --alpha 0.1 \
  --dim 384 --layers 8 --heads 6 --log-every 100
stamp "distill done -> $OUT/student.pt"
