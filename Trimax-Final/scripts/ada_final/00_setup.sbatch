#!/bin/bash
# Environments and model files. Needs internet (compute nodes have it); CPU only.
#SBATCH --job-name=escape-setup
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=results/final/logs/00_setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/ada_final/env.sh
hostname; df -h "$HOME" "$SCR_BASE" | cat

stamp "C deps (.deps-c), pinned to the base venv's numpy/pandas/scipy-free stack"
if ! c_py -c 'import numba, scipy, statsmodels' 2>/dev/null; then
  "$ESCAPE_BASE_PYTHON" -m pip freeze | grep -iE '^(numpy|pandas|packaging|pyarrow)==' > "$SCR/constraints.txt"
  cat "$SCR/constraints.txt"
  "$ESCAPE_BASE_PYTHON" -m pip install --disable-pip-version-check --no-cache-dir --upgrade \
    --target "$PWD/.deps-c" -c "$SCR/constraints.txt" numba scipy statsmodels
fi
c_py -c 'import numba, scipy, statsmodels, numpy, pandas, pyarrow, jsonschema
from escape_scoring.fast import HAVE_NUMBA
assert HAVE_NUMBA, "numba did not import"
print("C OK", numba.__version__, scipy.__version__, statsmodels.__version__, numpy.__version__)'

stamp "Stream B venv (.venv_b): stream_b/requirements.txt with CPU torch"
if [[ ! -x .venv_b/bin/python ]]; then python3.12 -m venv .venv_b; fi
.venv_b/bin/pip install --disable-pip-version-check --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# stream_b/requirements.txt pins huggingface_hub==1.27.0, which benepar's transformers
# 4.30.2 rejects (<1.0); datasets 5.0.1 accepts >=0.25,<2.0, so a 0.x hub satisfies both.
grep -v '^torch==' stream_b/requirements.txt | sed 's/^huggingface_hub==.*/huggingface_hub<1.0/' > "$SCR/req_b.txt"
# tokenizers 0.13.3 (benepar's transformers 4.30.2) has no cp312 wheel; build it. Newer
# rustc rejects one of its casts by default (lint only; code unchanged since 2023).
module load u22/rust/1.94.0
export CARGO_HOME="$SCR/cargo" RUSTFLAGS="-A invalid_reference_casting"
.venv_b/bin/pip install --disable-pip-version-check --no-cache-dir -r "$SCR/req_b.txt"
.venv_b/bin/pip install --disable-pip-version-check --no-cache-dir \
  "en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
mkdir -p "$NLTK_DATA"
b_py scripts/fetch_benepar_model.py
b_py -c 'import benepar, spacy, tree_sitter, tree_sitter_cpp, tree_sitter_python, datasets
nlp = spacy.load("en_core_web_md"); nlp.add_pipe("benepar", config={"model": "benepar_en3"})
print("B OK", list(nlp("The model parses this sentence.").sents)[0]._.parse_string)'

stamp "BLT patcher tensors at the pinned revision"
a_py -m stream_a.fetch_patcher
a_py -c 'import torch, transformers; print("A", torch.__version__, torch.version.cuda, transformers.__version__)'

stamp "Stream B golden fixtures and CPU test suite"
b_py stream_b/validate_golden_fixtures.py
c_py -m pytest tests -q -m 'not gpu'
stamp "SETUP OK"
