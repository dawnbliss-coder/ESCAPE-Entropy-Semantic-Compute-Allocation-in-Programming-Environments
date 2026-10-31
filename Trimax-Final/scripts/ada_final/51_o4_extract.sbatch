#!/bin/bash
# Objective 4: BLT patcher on the noised files, same settings as the primary run.
#SBATCH --job-name=escape-o4-extract
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=results/final/logs/51_o4_extract-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
nvidia-smi -L
stamp "O4 extract shard $SLURM_ARRAY_TASK_ID/4"
a_py -m stream_a.extract_o4 --shard "$SLURM_ARRAY_TASK_ID" --n-shards 4 --device cuda
rm -rf "$SCR"
stamp "O4 EXTRACT SHARD $SLURM_ARRAY_TASK_ID DONE"
