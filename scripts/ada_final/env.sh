# Sourced by every scripts/ada_final job, from the project root on an Ada compute node.
# Interpreters:
#   A_PY  existing ~/anlp_a2 venv, used read-only: torch 2.6.0+cu124 (Ada's driver caps
#         CUDA at 12.8), transformers 5.17.0, numpy/pandas/pyarrow/jsonschema.
#   C_PY  same interpreter with .deps-c (numba, scipy, statsmodels) first on the path.
#   B_PY  .venv_b built by 00_setup from stream_b/requirements.txt (CPU torch; benepar
#         needs transformers 4.30.2, which cannot share an environment with A).
module load u22/python/3.12.4
export ESCAPE_BASE_PYTHON="${ESCAPE_BASE_PYTHON:-$HOME/anlp_a2/venv/bin/python}"
export RUN_ID="swa512-float32-t1.3354-mplnone-91aa6b8e"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NLTK_DATA="$PWD/.nltk_data"
# Keep the HF token location (~/.cache/huggingface/token); move only the big caches.
SCR_BASE="/scratch/$USER"
if ! mkdir -p "$SCR_BASE" 2>/dev/null || [[ ! -w "$SCR_BASE" ]]; then SCR_BASE="/tmp/$USER"; mkdir -p "$SCR_BASE"; fi
export SCR="$SCR_BASE/escape-${SLURM_JOB_ID:-manual}"
mkdir -p "$SCR"
export HF_HUB_CACHE="$SCR/hf/hub" HF_DATASETS_CACHE="$SCR/hf/datasets" PIP_CACHE_DIR="$SCR/pip"
a_py() { PYTHONPATH="$PWD" "$ESCAPE_BASE_PYTHON" "$@"; }
c_py() { PYTHONPATH="$PWD/.deps-c:$PWD" "$ESCAPE_BASE_PYTHON" "$@"; }
b_py() { PYTHONPATH="$PWD" "$PWD/.venv_b/bin/python" "$@"; }
stamp() { printf '\n=== %s  %s ===\n' "$(date '+%F %T')" "$*"; }
