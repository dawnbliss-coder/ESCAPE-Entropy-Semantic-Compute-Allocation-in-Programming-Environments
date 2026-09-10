## Setup

- Clone the repo, `cd` into it
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r stream_b/requirements.txt`
- Create a token at huggingface.co/settings/tokens
- `hf auth login --force`, paste the token
- Visit huggingface.co/datasets/bigcode/the-stack-smol, click "Agree and access repository" (same account as the token)
- `python -m spacy download en_core_web_md`
- `python -c "import benepar; benepar.download('benepar_en3')"`
- `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- `python stream_b/build_corpus.py`

## Progress so far

- (Priyanka) Data contract drafted — `docs/SCHEMA.md`
- (Priyanka) C++ source resolved — both languages now use `bigcode/the-stack-smol`
- (Priyanka) tree-sitter smoke test — Python + C++ grammars
- (Priyanka) Corpus built — 8,000 Python + 8,000 C++ files (100 calib / 7,900 main each)
- (Priyanka) Selection made growth-safe — fixed-position prefixes, not a reshuffle
- (Priyanka) Real tree-sitter extraction pipeline — `structure/{domain}/{file_id}.parquet` for all 16,000 files (node_type, parent_type, depth, start_byte, end_byte)
- (Priyanka) `parse_ok` tracked per file in the manifest — 28.1% of C++ files hit tree-sitter error-recovery (isolated files missing macro/header context), kept and flagged rather than dropped
- (Priyanka) AST tree viewer (`stream_b/tree_viewer.py`) — generates a local interactive HTML view of any file's full parse tree next to its source, for inspection/debugging. 4 example views committed in `stream_b/tree_views/` (2 clean Python, 1 clean C++, 1 C++ with a parse error) — just open the `.html` files directly, no setup needed. Run `python stream_b/tree_viewer.py <file_id>` to generate one for any other file in `corpus/manifest.parquet`
- (Priyanka) Whitespace baseline extracted — `whitespace/{domain}/{file_id}.parquet` for all 16,000 files: `newline` positions (the required P1 baseline) and `indent_change` positions (feeds the R2 confound question) as separate tagged rows
