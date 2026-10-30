#!/bin/bash
# Full-run Step 2 (docs/ADA_FULL_RUN_GATES.md): student entropies + boundaries for every
# restored corpus file in Stream-A format, 4 byte-balanced shards. Resumable: rerun the
# array if a shard dies. Resources differ from the handoff, never outputs: u22 charges
# max(cpus, mem/3000M) CPUs and medium QoS caps a user at 40 CPUs, so 4 CPUs / 12000M
# per task (not 6 / 32G) lets all 4 tasks run beside one 24-CPU / 72000M scoring job;
# --time is 2 days because stride-1 windows are compute-bound (~20-25 h per shard
# expected on a 2080 Ti).
#SBATCH --job-name=esc-student-extract
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=results/final/logs/student_extract-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
IDS=$(a_py scripts/ada_final/shard_ids.py "$SLURM_ARRAY_TASK_ID" 4)
echo "shard $SLURM_ARRAY_TASK_ID/4: $(wc -w <<< "$IDS") files"
stamp "extract"
a_py -m escape_distill.extract --input corpus --model student \
  --run-id student-v1-14326k --device cuda --batch-windows 64 --file-ids $IDS --progress-every 100 \
  || echo "first pass reported failures; retrying them"  # extract exits non-zero on any failed file
stamp "retry failures"
a_py -m escape_distill.extract --input corpus --model student \
  --run-id student-v1-14326k --device cuda --batch-windows 64 --file-ids $IDS
rm -rf "$SCR"
stamp "STUDENT EXTRACT SHARD $SLURM_ARRAY_TASK_ID DONE"
