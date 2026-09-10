## Setup

- Clone the repo, `cd` into it
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r stream_b/requirements.txt`
- `hf auth login`
- `python -m spacy download en_core_web_md`
- `python -c "import benepar; benepar.download('benepar_en3')"`
- `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- `python stream_b/build_corpus.py`

## Progress so far

- (Priyanka) Data contract drafted — `docs/SCHEMA.md`
- (Priyanka) C++ source resolved — both languages now use `bigcode/the-stack-smol`
- (Priyanka) tree-sitter smoke test — Python + C++ grammars
- (Priyanka) Corpus v0 built — 300 Python + 300 C++ files (50 calib / 250 main each)
- (Priyanka) Selection made growth-safe — fixed-position prefixes, not a reshuffle
