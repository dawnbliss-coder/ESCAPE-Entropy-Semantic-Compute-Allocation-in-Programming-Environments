#!/bin/bash
# Full-run Step 3 (docs/ADA_FULL_RUN_GATES.md): student C-v1 scoring with the student's
# own calibration record (never the teacher's frozen one). Starts with the Step 2
# completion gate (zero missing files).
#SBATCH --job-name=esc-student-score
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=24
#SBATCH --mem=72000M
#SBATCH --time=2-00:00:00
#SBATCH --output=results/final/logs/student_score-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
W=${SLURM_CPUS_PER_TASK:-24}
RUN=student-v1-14326k

stamp "Step 2 completion gate"
a_py - <<'PY'
import sys
import pandas as pd
from pathlib import Path
m = pd.read_parquet('corpus/manifest.parquet')
missing = [r.file_id for _, r in m.iterrows() if not Path(f"entropies/student-v1-14326k/{r.file_id}.npy").exists()
           or not Path(f"boundaries/student-v1-14326k/{r.file_id}.parquet").exists()]
print("missing:", len(missing), missing[:5])
sys.exit(1 if missing else 0)
PY

stamp "convert + validate"
DATA="$SCR/c-v1-student"
IDS=$(a_py -c "import pandas as pd; m=pd.read_parquet('corpus/manifest.parquet'); print(' '.join(sorted(m.file_id)))")
c_py -m escape_scoring convert-legacy --root . --run-id "$RUN" --output "$DATA" --sample-ids $IDS
c_py -m escape_scoring validate --data "$DATA" --workers "$W"
stamp "calibrate (student's own record)"
c_py -m escape_scoring calibrate --data "$DATA" --output results/final/student-v1-calibration.json --workers "$W"
stamp "score"
c_py -m escape_scoring score --data "$DATA" --calibration results/final/student-v1-calibration.json \
  --output results/final/student-v1-eval --workers "$W"
stamp "baselines"
c_py -m escape_scoring baselines --data "$DATA" --calibration results/final/student-v1-calibration.json \
  --output results/final/student-v1-baselines.json --workers "$W"
rm -rf "$SCR"
stamp "STUDENT SCORE DONE"
