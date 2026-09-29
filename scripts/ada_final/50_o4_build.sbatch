#!/bin/bash
# Objective 4: noised variants of 1,000 held-out test files per code language, with
# clean targets mapped to noised offsets (stream_b/build_o4_perturbations.py).
#SBATCH --job-name=escape-o4-build
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/final/logs/50_o4_build-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
RUN_DIR=${RUN_DIR:?set RUN_DIR=results/final/c-v1-<job>}
stamp "build O4 perturbations"
PYTHONPATH="$PWD/stream_b:$PWD" .venv_b/bin/python stream_b/build_o4_perturbations.py --calibration "$RUN_DIR/calibration.json"
rm -rf "$SCR"
stamp "O4 BUILD DONE"
