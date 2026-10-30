#!/bin/bash
# file_toimplement.md (optional, P2): byte-balanced distillation. Same data, architecture, seed,
# loss and schedule as the distilled student (results/distill/full, 6,860 steps), but windows are
# drawn so every domain gets an equal expected byte share (C++ is 73% of the unbalanced bytes).
# Then (1) both students on the same held-out val windows against the teacher, and (2) rolling
# 512-byte extraction (run student-bal-v1) of the pre-registered teacher-512 subset plus the
# English test split (scripts/ada_distill/newarm_ids.py subset), scored by newarms_score.sbatch.
#SBATCH --job-name=esc-bal
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=0-14:00:00
#SBATCH --output=results/final/logs/balanced-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
source scripts/ada_distill/materialize_ext.sh
nvidia-smi -L
a_py -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in this allocation"; print("cuda:", torch.cuda.get_device_name(0))'
CKPT="$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"
CORPUS="$SCR/corpus_ext"
OUT=results/distill/balanced
mkdir -p "$OUT"
if [[ ! -f "$OUT/student/student.pt" ]]; then
  materialize_ext "$CORPUS"
  cp "$CORPUS/manifest_ext.parquet" "$CORPUS/DATA_NOTE.json" "$OUT/"
  stamp "train byte-balanced student (otherwise identical to the distilled student)"
  a_py -m escape_distill.train \
    --ckpt-dir "$CKPT" --data "$CORPUS/manifest_ext.parquet" --out "$OUT/student" \
    --train-split train --domains cpp,prose,py --balance bytes \
    --epochs 2 --batch-size 48 --lr 5e-4 --weight-decay 0.1 --warmup 200 \
    --temperature 2.0 --alpha 0.1 --dim 384 --layers 8 --heads 6 --log-every 100 --device cuda
  for WHO in balanced distilled; do
    PT=$([ $WHO = balanced ] && echo "$OUT/student/student.pt" || echo results/distill/full/student/student.pt)
    stamp "evaluate $WHO on held-out val (same windows)"
    a_py -m escape_distill.evaluate --ckpt-dir "$CKPT" --student "$PT" --data "$CORPUS/manifest_ext.parquet" \
      --out "$OUT/eval_$WHO" --eval-split val --domains cpp,prose,py --limit-files 3000 --device cuda
  done
fi
IDS=$(a_py scripts/ada_distill/newarm_ids.py subset)
stamp "rolling extraction student-bal-v1 on $(wc -w <<< "$IDS") subset + English test files"
a_py -m escape_distill.extract --input corpus --model student --student-pt "$OUT/student/student.pt" \
  --run-id student-bal-v1 --device cuda --file-ids $IDS --progress-every 100 \
  || echo "first pass reported failures; retrying them"
a_py -m escape_distill.extract --input corpus --model student --student-pt "$OUT/student/student.pt" \
  --run-id student-bal-v1 --device cuda --file-ids $IDS
rm -rf "$SCR"
stamp "BALANCED CONTROL DONE"
