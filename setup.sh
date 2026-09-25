#!/usr/bin/env bash
# Stream B setup: venv, deps, models, env var, full corpus + extraction pipeline.
#
# Precondition (manual, can't be scripted — do these first, see README.md):
#   1. Create a token at huggingface.co/settings/tokens
#   2. `hf auth login --force`, paste the token
#   3. Visit huggingface.co/datasets/bigcode/the-stack-smol, click "Agree and
#      access repository" (same account as the token) — needed for C++ only
set -e

if ! hf auth whoami >/dev/null 2>&1; then
    echo "ERROR: not logged into Hugging Face. Run 'hf auth login --force' first, then re-run this script." >&2
    exit 1
fi

echo "--- venv ---"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r stream_b/requirements.txt

echo "--- models ---"
.venv/bin/python -m spacy download en_core_web_md
.venv/bin/python -c "import benepar; benepar.download('benepar_en3')"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "--- corpus + pipeline ---"
.venv/bin/python stream_b/build_corpus.py          # C++, the-stack-smol
.venv/bin/python stream_b/build_py_corpus.py        # Python, CodeSearchNet
.venv/bin/python stream_b/build_structure.py
.venv/bin/python stream_b/whitespace_extractor.py
.venv/bin/python stream_b/build_taxonomy.py
.venv/bin/python stream_b/build_prose_corpus.py
.venv/bin/python stream_b/build_prose_structure.py
.venv/bin/python stream_b/build_prose_taxonomy.py
.venv/bin/python stream_b/build_golden_fixtures.py

echo "--- sanity check ---"
.venv/bin/python stream_b/validate_golden_fixtures.py

cat <<'EOF'

=== Done. ===
Run `source .venv/bin/activate` to use the venv interactively.
Remember to `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in any new
shell before importing benepar (this script only sets it for itself).
EOF
