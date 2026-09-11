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
- (Priyanka) `taxonomy.json` built for P4 — empirically checked opener-token diversity per node type first (`stream_b/taxonomy_analysis.py`) rather than only trusting the proposal's stated examples; found `return_statement`'s opener is 100% fixed ("return") yet the proposal classifies it open-ended, which corrected the classification principle used for the node types the proposal never explicitly covers (`call`/`binary_operator`/C++ `declaration`)
- (Priyanka) Prose corpus built for R3/P3 — 300 WikiText-103 paragraphs frozen to `corpus/prose/`, same discipline as code. WikiText-103 isn't discrete per-document rows like the code source, so this required reconstructing article boundaries from heading markup first; found and fixed a real bug in nested-heading detection (level-2 headings were misread as level-1 article titles) before running at scale
- Memory-unsafe region tagging / identifier spans (O5) deliberately deferred — the proposal's own timeline places O5 at weeks 9-11, after mid-submission (weeks 7-8), so this isn't in scope for the current checkpoint
