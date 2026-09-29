#!/bin/bash
#SBATCH --job-name=escape-c
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=results/ada/c-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load u22/python/3.12.4
PYTHON_C="${ESCAPE_PYTHON:-python3}"
if [[ -d .c-deps ]]; then
    export PYTHONPATH="$PWD/.c-deps${PYTHONPATH:+:$PYTHONPATH}"
fi
RUN_DIR="results/ada/synthetic-${SLURM_JOB_ID:?}"
mkdir -p "$RUN_DIR"
hostname
"$PYTHON_C" --version
"$PYTHON_C" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["numpy", "pandas", "pyarrow", "scipy", "jsonschema", "pytest"]})'
"$PYTHON_C" -m pytest tests -q -m 'not model and not gpu'
"$PYTHON_C" -m escape_scoring synthetic --output "$RUN_DIR/data"
"$PYTHON_C" -m escape_scoring validate --data "$RUN_DIR/data"
"$PYTHON_C" -m escape_scoring calibrate --data "$RUN_DIR/data" --output "$RUN_DIR/calibration.json"
"$PYTHON_C" -m escape_scoring score --data "$RUN_DIR/data" --calibration "$RUN_DIR/calibration.json" --output "$RUN_DIR/evaluation"
"$PYTHON_C" -m escape_scoring o5 --data "$RUN_DIR/data" --output "$RUN_DIR/o5"
echo "SYNTHETIC demonstration finished: $RUN_DIR"
