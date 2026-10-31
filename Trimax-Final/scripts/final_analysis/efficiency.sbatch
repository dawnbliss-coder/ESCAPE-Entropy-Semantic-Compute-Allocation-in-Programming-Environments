#!/bin/bash
# Student vs teacher efficiency benchmark (scripts/final_analysis/efficiency_bench.py), one 2080 Ti.
#SBATCH --job-name=esc-efficiency
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=0-02:00:00
#SBATCH --output=results/final/logs/efficiency-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
a_py scripts/final_analysis/efficiency_bench.py --out results/final/analysis/efficiency.json
rm -rf "$SCR"
stamp "EFFICIENCY DONE"
