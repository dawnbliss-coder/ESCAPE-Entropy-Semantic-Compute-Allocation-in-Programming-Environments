#!/bin/bash
# Optional item "more permutations": re-score with 10x the null draws, as a
# sensitivity check only. Frozen outputs are not touched; results go to results/final/analysis/.
#   PART=o4   escape_scoring.o4 on the O4 rate-sweep set (includes the frozen 20%/50% rates),
#             10,000 null permutations instead of 1,000 -> o4_rate_sweep_perm10k.json
#   PART=iou  escape_scoring.iou on the frozen test split, 1,000 null resamples instead of 100
#             -> iou_resamples1000.json
#   sbatch --export=ALL,PART=o4 scripts/final_analysis/more_perms.sbatch
#SBATCH --job-name=esc-more-perms
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=20
#SBATCH --mem=58000M
#SBATCH --time=0-14:00:00
#SBATCH --output=results/final/logs/more_perms-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
CAL=results/final/c-v1-2721954/calibration.json
W=${SLURM_CPUS_PER_TASK:-20}
mkdir -p results/final/analysis
case "${PART:?set PART=o4 or iou}" in
  o4)
    VIEW="$SCR/o4_sweep_root"
    mkdir -p "$VIEW"
    ln -s "$PWD/o4_sweep" "$VIEW/o4"
    ln -s "$PWD/boundaries" "$VIEW/boundaries"
    stamp "O4 rate sweep, 10,000 permutations"
    c_py -m escape_scoring.o4 --root "$VIEW" --calibration "$CAL" \
      --output results/final/analysis/o4_rate_sweep_perm10k.json --permutations 10000 --bootstrap 2000 --workers "$W"
    ;;
  iou)
    stamp "span IoU, 1,000 resamples"
    c_py -m escape_scoring.iou --root . --run-id "$RUN_ID" --calibration "$CAL" \
      --output results/final/analysis/iou_resamples1000.json --resamples 1000 --bootstrap 2000 --workers "$W"
    ;;
  *) echo "unknown PART=$PART"; exit 2 ;;
esac
rm -rf "$SCR"
stamp "MORE PERMS $PART DONE"
