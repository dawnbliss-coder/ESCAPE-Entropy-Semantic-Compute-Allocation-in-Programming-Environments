#!/bin/bash
# Stream B: restore the frozen corpus bytes (checked against corpus/manifest.parquet,
# which is never rewritten), parse structure, whitespace, prose constituents, O5 regions.
#SBATCH --job-name=escape-data
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/final/logs/10_data-%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
STATUS=results/final/data_status.txt
: > "$STATUS"
step() {  # step NAME CMD...: run, log outcome, keep going so every domain is attempted
  local name=$1; shift
  stamp "$name"
  if "$@"; then echo "$name OK" >> "$STATUS"; else echo "$name FAILED ($?)" >> "$STATUS"; fi
}
R=scripts/restore_stream_b_artifacts.py
step code-py    b_py $R --code --code-domains py
# Gated: the HF account behind ~/.cache/huggingface/token must have accepted the
# bigcode/the-stack-smol terms, and a fine-grained token needs gated-repo read access.
step code-cpp   b_py $R --code --code-domains cpp
step prose      b_py $R --prose
step structure  b_py $R --structure
step whitespace b_py $R --whitespace py cpp prose
step prose-structure env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" .venv_b/bin/python stream_b/build_prose_structure.py
if grep -q '^code-cpp OK' "$STATUS"; then
  step o5-regions b_py stream_b/build_o5_regions.py
fi
step verify     b_py $R --verify
rm -rf "$SCR"
stamp "data status"; cat "$STATUS"
! grep -q FAILED "$STATUS"
