#!/bin/bash
# Full-run Steps 5/6 (docs/ADA_FULL_RUN_GATES.md), CALIBRATION split only: re-threshold
# the saved entropies at each grid tau, score the six runs at k=0 with the fixed-EOF null
# (10,000 permutations), and apply the pre-registered tau* rule. No test file is touched;
# the single confirmatory test run is submitted separately once the tau* record is reviewed.
#   sbatch --export=ALL,MODEL=teacher scripts/ada_distill/tau_sweep.sbatch   (or MODEL=student)
#SBATCH --job-name=esc-tau-sweep
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=24
#SBATCH --mem=72000M
#SBATCH --time=2-00:00:00
#SBATCH --output=results/final/logs/tau_sweep-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-24}
case "${MODEL:?set MODEL=teacher or MODEL=student}" in
  teacher) SRC=swa512-float32-t1.3354-mplnone-91aa6b8e ;;
  student) SRC=student-v1-14326k ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
echo "MODEL=$MODEL SRC=$SRC"
CALIDS=$(c_py -c "import json; rec=json.load(open('results/final/c-v1-2721954/calibration.json')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='calibration')))")
echo "calibration files: $(wc -w <<< "$CALIDS")"
# SKIP_RETAU=1 rescores grid runs whose re-thresholded files already exist (retau is deterministic).
for TAU in 0.8 1.0 1.7 2.0 2.5; do
  TAG=$(echo $TAU | tr -d '.')
  [[ "${SKIP_RETAU:-0}" == 1 ]] && continue
  stamp "retau $MODEL-tau$TAG"
  c_py -m escape_distill.retau --src-run "$SRC" --dst-run "$MODEL-tau$TAG" --tau $TAU --file-ids $CALIDS
done
RUNS="$MODEL-tau08 $MODEL-tau10 $SRC $MODEL-tau17 $MODEL-tau20 $MODEL-tau25"
stamp "score the grid on calibration files"
c_py -m escape_distill.score_a_runs --run-ids $RUNS --file-ids $CALIDS \
  --out "results/final/${MODEL}_tau_sweep_calib.json" --perms 10000 --workers "$W"
stamp "tau* (pre-registered median-margin rule)"
c_py scripts/ada_distill/tau_star.py "results/final/${MODEL}_tau_sweep_calib.json" --runs $RUNS \
  --taus 0.8 1.0 1.335442066192627 1.7 2.0 2.5 --out "results/final/${MODEL}_tau_star.json"
rm -rf "$SCR"
stamp "TAU SWEEP $MODEL DONE (confirmatory run NOT submitted; review the tau* record first)"
