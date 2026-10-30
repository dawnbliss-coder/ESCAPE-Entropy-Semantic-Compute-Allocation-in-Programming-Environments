#!/bin/bash
# docs/ADA_FULL_RUN_SEQUENCE.md STEP 9 (optional): domain adaptation of the teacher. The teacher
# architecture (dim 768, 14 layers, 12 heads) is initialised from the frozen patcher and trained
# with the pre-registered command on the distilled student's training split. The frozen
# checkpoint is never written; the model goes to results/adapt/teacher-finetuned/student.pt.
# Then (1) adapted model and distilled student on the same held-out val windows, and (2) rolling
# 512-byte extraction (run teacher-adapt-v1) of the pre-registered teacher-512 subset plus the
# English test split, scored by newarms_score.sbatch. The pre-registered verdict asked for full
# C-v1 scoring of all files; rolling extraction of a 99.5M model over the 121 MB corpus (about
# 400 GPU-hours) did not fit before the deadline, so the verdict is applied on this subset and
# reported as a deviation.
#SBATCH --job-name=esc-adapt
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --time=1-08:00:00
#SBATCH --output=results/final/logs/adapt-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
source scripts/ada_distill/materialize_ext.sh
nvidia-smi -L
a_py -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in this allocation"; print("cuda:", torch.cuda.get_device_name(0))'
CKPT="$PWD/checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"
CORPUS="$SCR/corpus_ext"
OUT=results/adapt/teacher-finetuned
ARCH=(--dim 768 --layers 14 --heads 12 --init-from "$CKPT")
mkdir -p "$OUT"
if [[ ! -f "$OUT/student.pt" ]]; then
  stamp "GPU memory probe: two full-length steps at the pre-registered batch size"
  a_py -m escape_distill.train --ckpt-dir "$CKPT" --data golden --out "$SCR/probe" \
    --max-steps 2 --batch-size 16 "${ARCH[@]}" --log-every 1 --device cuda
  materialize_ext "$CORPUS"
  cp "$CORPUS/manifest_ext.parquet" "$CORPUS/DATA_NOTE.json" "$OUT/"
  stamp "train adapted teacher (STEP 9 command)"
  a_py -m escape_distill.train \
    --ckpt-dir "$CKPT" --data "$CORPUS/manifest_ext.parquet" --out "$OUT" \
    --train-split train --domains cpp,prose,py \
    --epochs 2 --batch-size 16 --lr 2e-4 --weight-decay 0.1 --warmup 100 \
    --temperature 2.0 --alpha 0.1 "${ARCH[@]}" --log-every 100 --device cuda
  for WHO in adapted distilled; do
    PT=$([ $WHO = adapted ] && echo "$OUT/student.pt" || echo results/distill/full/student/student.pt)
    stamp "evaluate $WHO on held-out val (same windows)"
    a_py -m escape_distill.evaluate --ckpt-dir "$CKPT" --student "$PT" --data "$CORPUS/manifest_ext.parquet" \
      --out "$OUT/eval_$WHO" --eval-split val --domains cpp,prose,py --limit-files 3000 --device cuda
  done
fi
IDS=$(a_py scripts/ada_distill/newarm_ids.py subset)
stamp "rolling extraction teacher-adapt-v1 on $(wc -w <<< "$IDS") subset + English test files"
a_py -m escape_distill.extract --input corpus --model student --student-pt "$OUT/student.pt" \
  --run-id teacher-adapt-v1 --device cuda --batch-windows 32 --file-ids $IDS --progress-every 100 \
  || echo "first pass reported failures; retrying them"
a_py -m escape_distill.extract --input corpus --model student --student-pt "$OUT/student.pt" \
  --run-id teacher-adapt-v1 --device cuda --batch-windows 32 --file-ids $IDS
rm -rf "$SCR"
stamp "ADAPT TEACHER DONE"
