#!/bin/bash
# The one confirmatory English result (scripts/ada_distill/holdout_score.py): extract the
# frozen teacher (swa512) and the student on the pre-registered holdout, then score both
# once with escape_distill.score_a_runs at the frozen k = 0 (10,000-permutation fixed-EOF
# null, structure-blind baselines, paired file bootstrap).
#SBATCH --job-name=esc-holdout-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18000M
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/holdout_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
stamp "extract teacher + student on the holdout"
a_py scripts/ada_distill/holdout_score.py extract --device cuda
VIEW="$SCR/holdout_view"
a_py scripts/ada_distill/holdout_score.py view "$VIEW"
stamp "score once (k=0, 10,000 permutations)"
mkdir -p results/final/analysis
c_py -m escape_distill.score_a_runs --root "$VIEW" --run-ids holdout-teacher-swa512 holdout-student-v1 \
  --file-ids $(cat "$VIEW/ids.txt") --perms 10000 --workers "${SLURM_CPUS_PER_TASK:-6}" \
  --out "$PWD/results/final/analysis/english_holdout_scores.json"
rm -rf "$SCR"
stamp "HOLDOUT SCORE DONE"
