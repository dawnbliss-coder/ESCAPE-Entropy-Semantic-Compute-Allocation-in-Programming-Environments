#!/bin/bash
# file_toimplement.md optional arms, scoring (CPU): escape_distill.score_a_runs at the frozen
# k = 0, fixed-EOF null with 10,000 permutations, 2,000 paired file bootstraps.
#   PART=subset   teacher native / teacher-512 / student-v1 / byte-balanced student / adapted teacher
#                 on the pre-registered 600-file code subset, and the same runs minus teacher-512
#                 (not extracted for English) on the English test split
#   PART=hf_full  hf_full vs the primary swa512 run on newarm_ids.py hf_full
#   sbatch --export=ALL,PART=subset scripts/ada_distill/newarms_score.sbatch
#SBATCH --job-name=esc-newarms-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M
#SBATCH --time=0-08:00:00
#SBATCH --output=results/final/logs/newarms_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-8}
NATIVE=swa512-float32-t1.3354-mplnone-91aa6b8e
OUT=results/final/analysis
mkdir -p "$OUT"
case "${PART:?set PART=subset or hf_full}" in
  subset)
    for D in code prose; do
      if [[ $D == code ]]; then
        IDS="$(a_py scripts/ada_distill/newarm_ids.py subset --domain py) $(a_py scripts/ada_distill/newarm_ids.py subset --domain cpp)"
        RUNS="$NATIVE teacher-512-ctl student-v1-14326k student-bal-v1 teacher-adapt-v1"
      else
        IDS=$(a_py scripts/ada_distill/newarm_ids.py subset --domain prose)
        RUNS="$NATIVE student-v1-14326k student-bal-v1 teacher-adapt-v1"
      fi
      stamp "score $D: $(wc -w <<< "$IDS") files, runs $RUNS"
      c_py -m escape_distill.score_a_runs --run-ids $RUNS --file-ids $IDS \
        --out "$OUT/newarms_subset_$D.json" --perms 10000 --workers "$W"
    done
    ;;
  hf_full)
    IDS=$(a_py scripts/ada_distill/newarm_ids.py hf_full)
    stamp "score hf_full vs swa512 on $(wc -w <<< "$IDS") test files"
    c_py -m escape_distill.score_a_runs --run-ids hf_full-float32-t1.3354-mplnone-91aa6b8e "$NATIVE" \
      --file-ids $IDS --out "$OUT/hf_full_test_k0.json" --perms 10000 --workers "$W"
    ;;
  *) echo "unknown PART=$PART"; exit 2 ;;
esac
rm -rf "$SCR"
stamp "NEWARMS SCORE $PART DONE"
