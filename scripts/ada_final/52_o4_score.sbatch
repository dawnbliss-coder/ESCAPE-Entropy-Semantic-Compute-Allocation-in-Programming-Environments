#!/bin/bash
# Objective 4: robustness scoring (escape_scoring/o4.py) under the frozen k.
#SBATCH --job-name=escape-o4-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/final/logs/52_o4_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
RUN_DIR=${RUN_DIR:?set RUN_DIR=results/final/c-v1-<job>}
stamp "O4 scoring"
c_py -m escape_scoring.o4 --root . --calibration "$RUN_DIR/calibration.json" --output "$RUN_DIR/o4.json" \
  --permutations 1000 --bootstrap 2000 --workers "${SLURM_CPUS_PER_TASK:-32}"
cp o4/build_summary.json "$RUN_DIR/o4_build_summary.json"
( cd "$RUN_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
rm -rf "$SCR"
stamp "O4 SCORING DONE"
