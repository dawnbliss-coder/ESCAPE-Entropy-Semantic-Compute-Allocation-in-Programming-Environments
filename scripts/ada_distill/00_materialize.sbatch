#!/bin/bash
#SBATCH --job-name=esc-ext-materialize
#SBATCH --partition=u22
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/distill/logs/materialize_%j.out
# Materialize the extension corpus (.bin files) that manifest_ext.parquet indexes, by
# re-running stream_b/build_ext_corpus.py for each domain. Re-downloads from Hugging Face:
#   - py / prose are ungated; cpp (The Stack) needs a HF token that has accepted the
#     bigcode/the-stack-smol terms (keep it at ~/.cache/huggingface/token).
# The .bin files are gitignored and regenerated, never committed. Run this once before the
# first distillation on Ada if corpus_ext/<domain>/*.bin are not present.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/ada_final/env.sh
mkdir -p runs/distill/logs

DOMAINS="${DOMAINS:-py prose cpp}"
for dom in $DOMAINS; do
  stamp "materialize $dom"
  a_py -m stream_b.build_ext_corpus "$dom" --out "$PWD/corpus_ext" || {
    echo "WARN: $dom failed (cpp needs The Stack terms accepted); continuing"; }
done
stamp "materialize done"
a_py - <<'PY'
import pandas as pd, pathlib
m = pd.read_parquet("corpus_ext/manifest_ext.parquet")
have = sum(1 for _, r in m.iterrows()
           if (pathlib.Path("corpus_ext")/str(r.get("domain"))/f"{r['file_id']}.bin").exists())
print(f"manifest rows={len(m)} materialized_bin={have}")
PY
