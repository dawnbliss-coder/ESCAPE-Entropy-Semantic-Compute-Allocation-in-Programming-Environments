#!/bin/bash
# Full-run Step 8 (docs/ADA_CLAUDE_HANDOFF.md): O5 label audit (CPU, no training, no
# scoring). Needs the restored C++ corpus (10_data) and .venv_b (tree_sitter).
#SBATCH --job-name=esc-o5-audit
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=6000M
#SBATCH --time=04:00:00
#SBATCH --output=results/final/logs/o5_audit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
b_py scripts/ada_distill/o5_label_audit.py
rm -rf "$SCR"
