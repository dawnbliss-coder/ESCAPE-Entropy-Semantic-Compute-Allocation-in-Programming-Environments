#!/bin/bash
# Stream C v1 on the real corpus: convert A/B outputs, validate, freeze k on the
# calibration split, score the held-out test split (10,000 permutations, 2,000
# file bootstraps), structure-blind baselines, O5. JSONL lives on node-local scratch.
#SBATCH --job-name=escape-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/final/logs/30_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-32}
# Domains to score. Default all three; if C++ could not be restored, set
# ESCAPE_DOMAINS="py prose" explicitly: the selection is recorded, never silent.
DOMAINS=${ESCAPE_DOMAINS:-"py cpp prose"}
OUT=results/final/c-v1-${SLURM_JOB_ID}
mkdir -p "$OUT"
df -h "$SCR" | cat

stamp "extraction completeness over the whole selection"
# Judged from the artifacts themselves (atomic per file), not from status.jsonl:
# concurrent appends from several nodes over NFS tear some of its lines.
a_py - "$DOMAINS" <<'PY'
import sys
from pathlib import Path
from stream_a.extract import artifacts_done, select_files, write_status_snapshot
root, run = Path("."), "swa512-float32-t1.3354-mplnone-91aa6b8e"
sel = select_files(root, "corpus", sys.argv[1].split(), "all", None, None)
write_status_snapshot(root, run, sel)  # event-log view, kept for the record
missing = [f for f, n in zip(sel["file_id"], sel["n_bytes"]) if not artifacts_done(root, run, f, int(n))]
print(len(sel), "files selected;", len(sel) - len(missing), "with complete, consistent artifacts")
if missing:
    Path("results/final/extraction_missing.txt").write_text("\n".join(missing) + "\n")
    sys.exit(f"{len(missing)} files lack complete artifacts; see results/final/extraction_missing.txt")
PY
a_py -m stream_a.build_bpp_summary --runs "$RUN_ID"
for split in calib main; do
  a_py -m escape_common.validate --run-id "$RUN_ID" --split "$split" --domain $DOMAINS
done
a_py scripts/validate_ab_offsets.py --run-id "$RUN_ID"

stamp "convert A/B -> C v1 JSONL (scratch)"
IDS=$(a_py -c "import pandas as pd, sys; m = pd.read_parquet('corpus/manifest.parquet'); print(' '.join(sorted(m[m.domain.isin(sys.argv[1].split())].file_id)))" "$DOMAINS")
echo "domains=$DOMAINS samples=$(wc -w <<< "$IDS")" | tee "$OUT/selection.txt"
DATA="$SCR/c-v1-inputs"
c_py -m escape_scoring convert-legacy --root . --run-id "$RUN_ID" --output "$DATA" --sample-ids $IDS
du -sh "$DATA" | cat
c_py -m escape_scoring validate --data "$DATA" --workers "$W"

stamp "freeze k on calibration files only"
c_py -m escape_scoring calibrate --data "$DATA" --output "$OUT/calibration.json" --workers "$W"
stamp "held-out test scoring"
c_py -m escape_scoring score --data "$DATA" --calibration "$OUT/calibration.json" --output "$OUT/evaluation" --workers "$W"
stamp "newline / indent / word baselines under the frozen k"
c_py -m escape_scoring baselines --data "$DATA" --calibration "$OUT/calibration.json" --output "$OUT/baselines.json" --workers "$W"
if [[ " $DOMAINS " == *" cpp "* ]]; then
  stamp "O5 (C++ regions): boundary density and inverse-BPP outcomes"
  cp analysis_regions/cpp/_selection.json "$OUT/o5_selection.json"
  c_py -m escape_scoring o5 --data "$DATA" --output "$OUT/o5" --workers "$W" \
    --controls sample_id control_statement_type
  c_py -m escape_scoring o5 --data "$DATA" --output "$OUT/o5_inverse_bpp" --workers "$W" \
    --outcome inverse_bpp_compute_density --controls sample_id control_statement_type
fi
cp bpp_summary.parquet "$OUT/"
cp "runs/$RUN_ID/status.parquet" "$OUT/extraction_status.parquet"
cp runs.json "$OUT/runs.json"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
rm -rf "$SCR"
stamp "SCORING DONE: $OUT"
