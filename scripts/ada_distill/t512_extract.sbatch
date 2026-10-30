#!/bin/bash
# Full-run Step 4 (docs/ADA_CLAUDE_HANDOFF.md), GPU part: teacher with fresh 512-byte
# stride-1 windows (run teacher-512-ctl) on the pre-registered 600-file subset
# (results/final/teacher512_subset_{python,cpp}.txt, fixed before any scoring).
#SBATCH --job-name=esc-t512-extract
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=1-00:00:00
#SBATCH --output=results/final/logs/t512_extract-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
IDS_T512="$(cat results/final/teacher512_subset_python.txt results/final/teacher512_subset_cpp.txt)"
echo "subset files: $(wc -w <<< "$IDS_T512")"
a_py -m escape_distill.extract --input corpus --model teacher \
  --run-id teacher-512-ctl --device cuda --batch-windows 32 --file-ids $IDS_T512 --progress-every 50 \
  || echo "first pass reported failures; retrying them"  # extract exits non-zero on any failed file
a_py -m escape_distill.extract --input corpus --model teacher \
  --run-id teacher-512-ctl --device cuda --batch-windows 32 --file-ids $IDS_T512
rm -rf "$SCR"
