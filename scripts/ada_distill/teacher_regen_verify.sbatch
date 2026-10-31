#!/bin/bash
# Reproduction check for the teacher A-format artifacts (entropies/ + boundaries/ for run
# swa512-float32-t1.3354-mplnone-91aa6b8e), re-extracted with scripts/ada_final/20_extract.sbatch.
# Before Steps 4-6 use them, check them against the frozen C-v1 record: `score` first runs verify_frozen (every calibration
# file's patcher output + structure must hash to the stored fingerprint), then the
# test-split per-file and summary scores are compared byte-for-byte with the frozen
# ones. results/final/c-v1-2721954/ is only read.
#SBATCH --job-name=esc-teacher-verify
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=24
#SBATCH --mem=72000M
#SBATCH --time=1-00:00:00
#SBATCH --output=results/final/logs/teacher_regen_verify-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-24}
FROZEN=results/final/c-v1-2721954
OUT=results/final/teacher-regen-verify

stamp "extraction completeness"
a_py - <<'PY'
import sys
from pathlib import Path
from stream_a.extract import artifacts_done, select_files
root, run = Path("."), "swa512-float32-t1.3354-mplnone-91aa6b8e"
sel = select_files(root, "corpus", ["py", "cpp", "prose"], "all", None, None)
missing = [f for f, n in zip(sel["file_id"], sel["n_bytes"]) if not artifacts_done(root, run, f, int(n))]
print(len(sel), "files;", len(sel) - len(missing), "with complete artifacts")
sys.exit(1 if missing else 0)
PY

stamp "convert + validate"
DATA="$SCR/c-v1-teacher-regen"
IDS=$(a_py -c "import pandas as pd; m=pd.read_parquet('corpus/manifest.parquet'); print(' '.join(sorted(m.file_id)))")
c_py -m escape_scoring convert-legacy --root . --run-id "$RUN_ID" --output "$DATA" --sample-ids $IDS
c_py -m escape_scoring validate --data "$DATA" --workers "$W"

stamp "score against the FROZEN calibration record (verify_frozen runs first)"
c_py -m escape_scoring score --data "$DATA" --calibration "$FROZEN/calibration.json" \
  --output "$OUT/evaluation" --workers "$W"

stamp "compare with the frozen evaluation"
ok=1
for f in per_file.csv scores.csv diagnostics.csv; do
  a=$(sha256sum "$FROZEN/evaluation/$f" | cut -c1-64); b=$(sha256sum "$OUT/evaluation/$f" | cut -c1-64)
  if [[ "$a" == "$b" ]]; then echo "IDENTICAL $f"; else echo "DIFFERS   $f"; ok=0; fi
done
rm -rf "$SCR"
[[ $ok == 1 ]] && stamp "TEACHER REGEN VERIFIED" || { stamp "TEACHER REGEN DIFFERS FROM FROZEN"; exit 1; }
