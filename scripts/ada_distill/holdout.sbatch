#!/bin/bash
# Full-run Step 7 (docs/ADA_FULL_RUN_GATES.md): English holdout bytes + benepar structure.
# DATA ONLY: no boundary scores are computed. Needs .venv_b from 00_setup.
#SBATCH --job-name=esc-holdout
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=results/final/logs/holdout-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
stamp "rebuild the prose extension corpus on scratch (tracked manifest untouched)"
a_py -m stream_b.build_ext_corpus prose --out "$SCR/ext_prose"
stamp "select"
a_py scripts/ada_distill/holdout_prose.py select
stamp "materialize"
a_py scripts/ada_distill/holdout_prose.py materialize --src "$SCR/ext_prose"
stamp "structure (benepar)"
b_py scripts/ada_distill/holdout_prose.py structure
rm -rf "$SCR"
stamp "HOLDOUT DONE"
