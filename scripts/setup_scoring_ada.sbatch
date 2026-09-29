#!/bin/bash
# Run only with approval to install C's test dependencies on Ada.
#SBATCH --job-name=escape-c-deps
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --output=results/ada/setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load u22/python/3.12.4
PYTHON_C="${ESCAPE_PYTHON:?set to existing Ada Python environment}"
# --target leaves the existing assignment venv unchanged.
"$PYTHON_C" -m pip install --disable-pip-version-check --no-cache-dir --target "$PWD/.c-deps" pytest scipy
export PYTHONPATH="$PWD/.c-deps${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_C" -c 'import pytest, scipy, numpy, jsonschema, pandas, pyarrow; print("C dependencies import successfully")'
