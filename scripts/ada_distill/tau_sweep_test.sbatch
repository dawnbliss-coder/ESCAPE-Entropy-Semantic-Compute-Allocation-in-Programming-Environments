#!/bin/bash
# Exploratory (requested 2026-10-30): the tau grid scored on the TEST split, after tau* was fixed
# on calibration files (tau_sweep.sbatch) and each model's single confirmatory test run was
# scored (tau_confirm.sbatch). Descriptive only: tau* and k are not re-selected and no frozen
# output is touched. The grid runs are re-thresholded into a view on node-local scratch (home
# quota) under their own run ids, registered only in a copy of runs.json there, and scored at
# k=0 with 1,000 null permutations (10x the draws moved O4 margins by < 1e-4,
# results/final/analysis/).
#   sbatch --export=ALL,MODEL=teacher scripts/ada_distill/tau_sweep_test.sbatch   (or MODEL=student)
#SBATCH --job-name=esc-tau-test
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=20
#SBATCH --mem=58000M
#SBATCH --time=0-14:00:00
#SBATCH --output=results/final/logs/tau_sweep_test-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-20}
case "${MODEL:?set MODEL=teacher or MODEL=student}" in
  teacher) SRC=swa512-float32-t1.3354-mplnone-91aa6b8e ;;
  student) SRC=student-v1-14326k ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
CAL=results/final/c-v1-2721954/calibration.json
TESTIDS=$(c_py -c "import json; rec=json.load(open('$CAL')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='test')))")
echo "MODEL=$MODEL SRC=$SRC test files: $(wc -w <<< "$TESTIDS")"
VIEW="$SCR/tau_test_root"
mkdir -p "$VIEW/entropies" "$VIEW/boundaries"
ln -s "$PWD/corpus" "$VIEW/corpus"
ln -s "$PWD/structure" "$VIEW/structure"
ln -s "$PWD/entropies/$SRC" "$VIEW/entropies/$SRC"
ln -s "$PWD/boundaries/$SRC" "$VIEW/boundaries/$SRC"
cp runs.json "$VIEW/runs.json"
for TAU in 0.8 1.0 1.7 2.0 2.5; do
  TAG=$(echo $TAU | tr -d '.')
  stamp "retau $MODEL-test-tau$TAG (test files, scratch view)"
  c_py -m escape_distill.retau --root "$VIEW" --src-run "$SRC" --dst-run "$MODEL-test-tau$TAG" --tau $TAU --file-ids $TESTIDS
done
# Same order of taus as the calibration record (tau_sweep_test.py pairs them by position).
RUNS="$MODEL-test-tau08 $MODEL-test-tau10 $SRC $MODEL-test-tau17 $MODEL-test-tau20 $MODEL-test-tau25"
stamp "score the grid on test files (1,000 permutations)"
c_py -m escape_distill.score_a_runs --root "$VIEW" --run-ids $RUNS --file-ids $TESTIDS \
  --out "$PWD/results/final/analysis/${MODEL}_tau_sweep_test.json" --perms 1000 --workers "$W"
c_py scripts/ada_distill/tau_sweep_test.py "results/final/analysis/${MODEL}_tau_sweep_test.json" \
  --tau-star "results/final/${MODEL}_tau_star.json" --out "results/final/analysis/${MODEL}_tau_sweep_test_summary.json"
rm -rf "$SCR"
stamp "TAU SWEEP TEST $MODEL DONE"
