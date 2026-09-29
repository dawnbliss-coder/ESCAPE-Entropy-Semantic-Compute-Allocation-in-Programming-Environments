#!/bin/bash
# Secondary span-IoU metric on the frozen C-v1 test split (escape_scoring/iou.py).
# Needs a completed 30_score run: pass its directory as RUN_DIR=results/final/c-v1-<job>.
#SBATCH --job-name=escape-iou
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/final/logs/40_iou-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
RUN_DIR=${RUN_DIR:?set RUN_DIR=results/final/c-v1-<job>}
stamp "span IoU on $RUN_DIR test split"
c_py -m escape_scoring.iou --root . --run-id "$RUN_ID" --calibration "$RUN_DIR/calibration.json" \
  --output "$RUN_DIR/iou.json" --resamples 100 --bootstrap 2000 --workers "${SLURM_CPUS_PER_TASK:-32}"
( cd "$RUN_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
rm -rf "$SCR"
stamp "IOU DONE"
