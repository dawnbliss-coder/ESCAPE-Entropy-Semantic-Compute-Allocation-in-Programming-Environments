#!/bin/bash
# On the Ada login node, from the project root: bash scripts/ada_final/submit_all.sh
# Submits setup -> data -> 4 GPU extraction shards -> scoring, chained with afterok.
# SKIP_SETUP=1 / SKIP_DATA=1 skip stages already completed. Extra sbatch args for
# scoring (e.g. ESCAPE_DOMAINS) are taken from the environment via --export=ALL.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/final/logs
dep=()
if [[ "${SKIP_SETUP:-0}" != 1 ]]; then
  j=$(sbatch --parsable scripts/ada_final/00_setup.sbatch); echo "setup   $j"; dep=(--dependency=afterok:$j)
fi
if [[ "${SKIP_DATA:-0}" != 1 ]]; then
  j=$(sbatch --parsable "${dep[@]}" scripts/ada_final/10_data.sbatch); echo "data    $j"; dep=(--dependency=afterok:$j)
fi
j=$(sbatch --parsable "${dep[@]}" scripts/ada_final/20_extract.sbatch); echo "extract $j (array 0-3)"
j=$(sbatch --parsable --export=ALL --dependency=afterok:$j scripts/ada_final/30_score.sbatch); echo "score   $j"
echo "Watch: squeue -u \$USER ; tail -f results/final/logs/*.log"
