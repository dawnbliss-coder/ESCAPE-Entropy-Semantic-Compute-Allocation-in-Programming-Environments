#!/bin/bash
# P4 post-keyword follow-up + O3 depth normalisation (scripts/final_analysis/position_scores.py)
# on the frozen test split, 10,000 frozen-scorer null draws per file. CPU only.
#   sbatch --export=ALL,RUN=swa512-float32-t1.3354-mplnone-91aa6b8e,TAG=teacher scripts/final_analysis/positions.sbatch
#SBATCH --job-name=esc-positions
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=12
#SBATCH --mem=36000M
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/positions-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
stamp "positions RUN=${RUN:?} TAG=${TAG:?}"
c_py scripts/final_analysis/position_scores.py --run-id "$RUN" \
  --out "results/final/analysis/positions_${TAG}.json" --workers "${SLURM_CPUS_PER_TASK:-12}"
rm -rf "$SCR"
stamp "POSITIONS $TAG DONE"
