#!/bin/bash
# k sweep, test-split SENSITIVITY rows only (k is frozen at 0 by the calibration record;
# nothing here selects k): score a run on the test split at k = 1 and 2, 1,000 null draws.
#   sbatch --export=ALL,RUN=swa512-float32-t1.3354-mplnone-91aa6b8e,TAG=teacher scripts/final_analysis/k_sensitivity.sbatch
#SBATCH --job-name=esc-k-sens
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/k_sensitivity-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
for K in 1 2; do
  stamp "score_a_runs ${RUN:?} k=$K (test split, sensitivity only)"
  c_py -m escape_distill.score_a_runs --run-ids "$RUN" --k $K --perms 1000 --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out "results/final/analysis/${TAG:?}_test_k${K}.json"
done
rm -rf "$SCR"
stamp "K SENSITIVITY $TAG DONE"
