#!/bin/bash
# Objective 4 rate sweep: the same 1,000 held-out files per code
# language as the frozen O4 build, every noise type at several rates, scored with the
# unchanged escape_scoring.o4 (1,000 null permutations, paired bootstrap vs clean).
#   STAGE=build    CPU: stream_b/build_o4_perturbations.py --out o4_sweep
#   STAGE=extract  GPU array 0-3: stream_a.extract_o4 --o4-dir o4_sweep
#   STAGE=score    CPU: escape_scoring.o4 on a view root whose o4/ is o4_sweep/
# Rates: missing_semicolon / indentation / keyword_typo 0.05 0.1 0.2 0.4; case_inversion 0.25 0.5 1.0
#   (0.2 and 0.5 are the frozen O4 rates, rebuilt identically so every curve shares its files).
#SBATCH --job-name=esc-o4-sweep
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --time=0-12:00:00
#SBATCH --output=results/final/logs/o4_sweep-%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
CAL=results/final/c-v1-2721954/calibration.json
case "${STAGE:?set STAGE=build, extract or score}" in
  build)
    stamp "O4 sweep build"
    PYTHONPATH="$PWD/stream_b:$PWD" .venv_b/bin/python stream_b/build_o4_perturbations.py --calibration "$CAL" \
      --out o4_sweep --rates 0.05 0.1 0.2 0.4 --case-rates 0.25 0.5 1.0
    ;;
  extract)
    nvidia-smi -L
    stamp "O4 sweep extract shard $SLURM_ARRAY_TASK_ID/4"
    a_py -m stream_a.extract_o4 --o4-dir o4_sweep --shard "$SLURM_ARRAY_TASK_ID" --n-shards 4 --device cuda
    ;;
  score)
    VIEW="$SCR/o4_sweep_root"
    mkdir -p "$VIEW"
    ln -s "$PWD/o4_sweep" "$VIEW/o4"
    ln -s "$PWD/boundaries" "$VIEW/boundaries"
    stamp "O4 sweep scoring"
    mkdir -p results/final/analysis
    c_py -m escape_scoring.o4 --root "$VIEW" --calibration "$CAL" --output results/final/analysis/o4_rate_sweep.json \
      --permutations 1000 --bootstrap 2000 --workers "${SLURM_CPUS_PER_TASK:-24}"
    cp o4_sweep/build_summary.json results/final/analysis/o4_rate_sweep_build_summary.json
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
rm -rf "$SCR"
stamp "O4 SWEEP $STAGE DONE"
