#!/bin/bash
# Submit the distillation pipeline on Ada with SLURM dependencies:
#   00 materialize ext corpus (skip if corpus_ext/<domain>/*.bin already present)
#   10 distill student  (after 00)
#   20 evaluate student (after 10)
# Usage, from the project root on the Ada login node:
#   bash scripts/ada_distill/submit_all.sh            # full: materialize + distill + eval
#   SKIP_MATERIALIZE=1 bash scripts/ada_distill/submit_all.sh   # reuse existing .bin
#   LIMIT=4000 bash scripts/ada_distill/submit_all.sh           # quick real run on a subset
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs/distill/logs

export LIMIT="${LIMIT:-}" DOMAINS="${DOMAINS:-py,prose,cpp}" EPOCHS="${EPOCHS:-1}"
dep=""
if [ "${SKIP_MATERIALIZE:-0}" != "1" ]; then
  j00=$(sbatch --parsable --export=ALL,DOMAINS="${DOMAINS//,/ }" scripts/ada_distill/00_materialize.sbatch)
  echo "materialize: $j00"; dep="--dependency=afterok:$j00"
fi
j10=$(sbatch --parsable $dep --export=ALL scripts/ada_distill/10_distill.sbatch)
echo "distill: $j10"
j20=$(sbatch --parsable --dependency=afterok:$j10 --export=ALL scripts/ada_distill/20_eval.sbatch)
echo "eval: $j20"
echo "Watch: squeue -u \$USER ; results land in results/distill/"
