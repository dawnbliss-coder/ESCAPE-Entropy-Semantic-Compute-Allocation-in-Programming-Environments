#!/bin/bash
# Full-run Step 4 (docs/ADA_CLAUDE_HANDOFF.md), scoring part: teacher native (swa512,
# regenerated + verified) vs teacher-512 vs student on the pre-registered subset, k=0,
# fixed-EOF null with 10,000 permutations.
#SBATCH --job-name=esc-t512-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=24
#SBATCH --mem=72000M
#SBATCH --time=12:00:00
#SBATCH --output=results/final/logs/t512_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
IDS_T512="$(cat results/final/teacher512_subset_python.txt results/final/teacher512_subset_cpp.txt)"
c_py -m escape_distill.score_a_runs \
  --run-ids swa512-float32-t1.3354-mplnone-91aa6b8e teacher-512-ctl student-v1-14326k \
  --file-ids $IDS_T512 --out results/final/teacher512_subset_scores.json --perms 10000 \
  --workers "${SLURM_CPUS_PER_TASK:-24}"
rm -rf "$SCR"
