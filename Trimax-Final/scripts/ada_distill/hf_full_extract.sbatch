#!/bin/bash
# Optional: hf_full alignment sensitivity. Native HF full causal
# attention (docs/STREAM_A_FIDELITY.md: not faithful beyond 512 tokens, sensitivity only) on the
# test-split files of scripts/ada_distill/newarm_ids.py hf_full (all Python and English, C++
# under 8,192 bytes). Scored k=0 against the primary swa512 run by newarms_score.sbatch PART=hf_full.
#SBATCH --job-name=esc-hffull
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000M
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/hf_full_extract-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
a_py -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in this allocation"; print("cuda:", torch.cuda.get_device_name(0))'
IDS=$(a_py scripts/ada_distill/newarm_ids.py hf_full)
stamp "extract hf_full on $(wc -w <<< "$IDS") test files"
a_py -m stream_a.extract --input corpus --split all --context-mode hf_full --file-ids $IDS \
  --device cuda --progress-every 500
stamp "retry failures once"
a_py -m stream_a.extract --input corpus --split all --context-mode hf_full --file-ids $IDS \
  --device cuda --only-failed
rm -rf "$SCR"
stamp "HF_FULL EXTRACT DONE"
