#!/bin/bash
# Full-run Steps 5/6, second half (docs/ADA_FULL_RUN_GATES.md): the ONE confirmatory
# test-split run at the pre-registered tau*. tau* is read from the reviewed record
# results/final/${MODEL}_tau_star.json (never typed by hand); the job refuses to run if
# the two comparators disagree. Different tau* per language -> one run per language.
#   PART=retau  re-threshold the source entropies at tau* for the full corpus -> ${MODEL}-tau-STAR
#   PART=cli    frozen C-v1 CLI with k co-frozen under tau* (its own calibration record)
#   PART=k0     escape_distill.score_a_runs at k=0 on the test split, paired with the source run
# Submit retau first, then cli and k0 with --dependency=afterok:<retau job>:
#   sbatch --export=ALL,MODEL=teacher,PART=retau --cpus-per-task=4 --mem=12000M scripts/ada_distill/tau_confirm.sbatch
#SBATCH --job-name=esc-tau-confirm
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=24
#SBATCH --mem=72000M
#SBATCH --time=2-00:00:00
#SBATCH --output=results/final/logs/tau_confirm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-24}
case "${MODEL:?set MODEL=teacher or MODEL=student}" in
  teacher) SRC=swa512-float32-t1.3354-mplnone-91aa6b8e ;;
  student) SRC=student-v1-14326k ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
# One "lang tau dst" line per language. A single tau* for all languages -> one run
# ${MODEL}-tau-STAR scored by one CLI pass (as pre-registered). Different tau* per language ->
# one run ${MODEL}-tau-STAR-<lang> per language, each scored by its own CLI pass on that
# language's files only (the calibration split is per-language, so it is unchanged).
PLAN=$(c_py - "results/final/${MODEL}_tau_star.json" "$MODEL" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
langs = rec["languages"]
if not rec.get("all_agree") or not all(v["agree"] for v in langs.values()):
    sys.exit("tau* comparators disagree; the confirmatory run waits for a decision")
single = len({v["tau_star_frozen"] for v in langs.values()}) == 1
for lang, v in sorted(langs.items()):
    print(lang, v["tau_star_frozen"], f"{sys.argv[2]}-tau-STAR" + ("" if single else f"-{lang}"))
PY
)
echo "MODEL=$MODEL SRC=$SRC PART=${PART:?set PART=retau, cli or k0}"
echo "$PLAN"
DSTS=$(awk '{print $3}' <<< "$PLAN" | sort -u)
lang_ids() {  # file ids of one language (all languages for a single-tau* run)
  a_py - "$1" "$2" <<'PY'
import sys, pandas as pd
m = pd.read_parquet("corpus/manifest.parquet")
dom = {"python": "py", "cpp": "cpp", "english": "prose"}
if sys.argv[2].endswith("-tau-STAR"):
    print(" ".join(sorted(m.file_id)))
else:
    print(" ".join(sorted(m[m.domain == dom[sys.argv[1]]].file_id)))
PY
}

case "$PART" in
  retau)
    while read -r LANG TAU DST <&3; do
      [[ "$DST" == *-tau-STAR && "$LANG" != english ]] && continue   # single run: retau once (all files)
      stamp "retau $DST at tau*=$TAU"
      c_py -m escape_distill.retau --src-run "$SRC" --dst-run "$DST" --tau "$TAU" --file-ids $(lang_ids "$LANG" "$DST")
    done 3<<< "$PLAN"
    ;;
  cli)
    while read -r LANG TAU DST <&3; do
      [[ "$DST" == *-tau-STAR && "$LANG" != english ]] && continue   # single run: one CLI pass
      # ONLY_LANGS="cpp python" runs a subset, so per-language runs can go in parallel jobs.
      [[ -n "${ONLY_LANGS:-}" && " $ONLY_LANGS " != *" $LANG "* ]] && continue
      if [[ -f "results/final/$DST-eval/scores.json" && -f "results/final/$DST-baselines.json" ]]; then
        stamp "$DST already scored; skipping"; continue
      fi
      DATA="$SCR/c-v1-$DST"
      # REUSE_DATA=<dir> reuses a converted C-v1 dir left on this node by an interrupted job
      # (submit with --nodelist=<that node>); it is re-validated before use.
      if [[ -n "${REUSE_DATA:-}" && -f "$REUSE_DATA/patcher_output.jsonl" && "$(basename "$REUSE_DATA")" == "c-v1-$DST" ]]; then
        DATA="$REUSE_DATA"
        stamp "reusing converted $DATA; validate $DST"
      else
        stamp "convert + validate $DST"
        c_py -m escape_scoring convert-legacy --root . --run-id "$DST" --output "$DATA" --sample-ids $(lang_ids "$LANG" "$DST")
      fi
      c_py -m escape_scoring validate --data "$DATA" --workers "$W"
      if [[ -f "results/final/$DST-calibration.json" ]]; then
        # Calibration is deterministic and `score` re-verifies the record against these inputs
        # (verify_frozen), so a record left by an interrupted job is reused, never rewritten.
        stamp "calibrate $DST: reusing existing record (verified at scoring)"
      else
        stamp "calibrate $DST (k co-frozen under tau*; own record)"
        c_py -m escape_scoring calibrate --data "$DATA" --output "results/final/$DST-calibration.json" --workers "$W"
      fi
      stamp "score $DST (the single confirmatory test run)"
      c_py -m escape_scoring score --data "$DATA" --calibration "results/final/$DST-calibration.json" \
        --output "results/final/$DST-eval" --workers "$W"
      stamp "baselines $DST"
      c_py -m escape_scoring baselines --data "$DATA" --calibration "results/final/$DST-calibration.json" \
        --output "results/final/$DST-baselines.json" --workers "$W"
      rm -rf "$DATA"
    done 3<<< "$PLAN"
    ;;
  k0)
    TIDS=$(c_py -c "import json; rec=json.load(open('results/final/c-v1-2721954/calibration.json')); print(' '.join(sorted(m['sample_id'] for m in rec['split_manifest'] if m['split']=='test')))")
    stamp "score_a_runs k=0 on $(wc -w <<< "$TIDS") test files"
    for DST in $DSTS; do
      c_py -m escape_distill.score_a_runs --run-ids "$DST" "$SRC" --file-ids $TIDS \
        --out "results/final/${DST}_test_k0.json" --perms 10000 --workers "$W"
    done
    ;;
  *) echo "unknown PART=$PART"; exit 2 ;;
esac
rm -rf "$SCR"
stamp "TAU CONFIRM $MODEL $PART DONE"
