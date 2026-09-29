#!/bin/bash
# Stream A on GPU: BLT patcher entropies + boundaries for every restored corpus file,
# primary regime swa512 / float32 / checkpoint tau / no length cap. Resumable.
#SBATCH --job-name=escape-extract
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=results/final/logs/20_extract-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
N_SHARDS=4
TASK=${SLURM_ARRAY_TASK_ID:?}
nvidia-smi -L
a_py -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
if [[ "$TASK" == 0 ]]; then
  stamp "checkpoint-fidelity gate under this environment"
  a_py -m stream_a.fidelity --device cuda --out results/final/stream_a_fidelity_ada || echo "FIDELITY GATE NONZERO EXIT: inspect before using results"
fi
IDS=$(a_py scripts/ada_final/shard_ids.py "$TASK" "$N_SHARDS")
echo "shard $TASK/$N_SHARDS: $(wc -w <<< "$IDS") files"
stamp "extract"
a_py -m stream_a.extract --input corpus --split all --file-ids $IDS --device cuda --progress-every 200
stamp "retry failures once"
a_py -m stream_a.extract --input corpus --split all --file-ids $IDS --device cuda --only-failed
rm -rf "$SCR"
stamp "EXTRACT SHARD $TASK DONE"
