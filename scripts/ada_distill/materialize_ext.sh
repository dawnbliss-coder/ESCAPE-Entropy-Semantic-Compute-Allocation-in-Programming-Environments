# Sourced by the post-submission training jobs (balanced_control.sbatch, adapt_teacher.sbatch):
# materialise the extension corpus to node-local scratch with the same limits as
# run_all.sbatch / alpha1_control.sbatch, then check that the manifest is identical to the
# distilled student's (results/distill/full/manifest_ext.parquet). Needs env.sh (a_py, stamp, SCR).
#   materialize_ext "$CORPUS"
materialize_ext() {
  local CORPUS=$1
  stamp "materialize ext corpus (same limits as run_all.sbatch)"
  a_py -m stream_b.build_ext_corpus py    --limit 15000 --out "$CORPUS"
  # The prose step can abort at interpreter exit after writing everything ("Fatal Python error:
  # PyGILState_Release", rc 134; see run_all.sbatch). Accept that only if the output is complete.
  a_py -m stream_b.build_ext_corpus prose --limit 8000  --out "$CORPUS" || {
    local rc=$?
    a_py - "$CORPUS" <<'PY' || { echo "ERROR: prose materialize failed (rc=$rc) and its output is incomplete"; return "$rc"; }
import json, sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
m = pd.read_parquet(out / "manifest_ext.parquet")
ids = m.loc[m["domain"] == "prose", "file_id"]
missing = [f for f in ids if not (out / "prose" / f"{f}.bin").exists()]
note = json.loads((out / "DATA_NOTE.json").read_text())
assert len(ids) > 0 and not missing and note["prose"]["files"] == len(ids), (len(ids), len(missing))
print(f"prose output complete: {len(ids)} manifest rows, all .bin present")
PY
    echo "WARN: prose materialize exited rc=$rc at interpreter exit; output verified complete, continuing"
  }
  a_py -m stream_b.build_ext_corpus cpp   --limit 6000  --out "$CORPUS"
  stamp "data identity vs the distilled student's manifest"
  a_py - "$CORPUS/manifest_ext.parquet" <<'PY'
import sys, pandas as pd
a = pd.read_parquet(sys.argv[1]); b = pd.read_parquet("results/distill/full/manifest_ext.parquet")
key = ["file_id", "sha256", "split"] if "split" in a.columns else ["file_id", "sha256"]
same = set(map(tuple, a[key].values)) == set(map(tuple, b[key].values))
print("manifest identical to results/distill/full:", same, len(a), len(b))
assert same, "training data differs from the distilled student's; refusing to train"
PY
}
