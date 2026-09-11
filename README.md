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
- `python stream_b/build_structure.py`
- `python stream_b/whitespace_extractor.py`
- `python stream_b/build_taxonomy.py`
- `python stream_b/build_prose_corpus.py`
- `python stream_b/build_prose_structure.py`
- `python stream_b/build_prose_taxonomy.py`
- `python stream_b/build_golden_fixtures.py`

## Progress so far

- (Priyanka) Data contract drafted — `docs/SCHEMA.md`
- (Priyanka) C++ source resolved — both languages use `bigcode/the-stack-smol`, not CodeSearchNet
- (Priyanka) Code corpus built — 8,000 Python + 8,000 C++ files (100 calib / 7,900 main each), growth-safe and reproducible
- (Priyanka) Tree-sitter extraction pipeline — `structure/{domain}/{file_id}.parquet` for all 16,000 files
- (Priyanka) `parse_ok` tracked per file — 28.1% of C++ hits tree-sitter error-recovery (isolated files missing macro/header context), kept and flagged rather than dropped
- (Priyanka) AST tree viewer (`stream_b/tree_viewer.py`) — interactive local HTML view of any code file's parse tree + source; 4 examples committed in `stream_b/tree_views/`
- (Priyanka) Whitespace baseline extracted — `whitespace/{domain}/{file_id}.parquet`, `newline` (P1 baseline) and `indent_change` (R2) as tagged rows
- (Priyanka) `taxonomy.json` built for P4 — empirically checked opener-token diversity per node type rather than trusting examples alone; corrected the classification test itself (`stream_b/taxonomy_analysis.py`)
- (Priyanka) Prose corpus built for R3/P3 — 300 WikiText-103 paragraphs in `corpus/prose/`, same discipline as code
- (Priyanka) Prose parsed with benepar — `structure/prose/{file_id}.parquet`, 1,778 sentences, byte-offset conversion validated per-span
- (Priyanka) Prose tree viewer (`stream_b/prose_tree_viewer.py`) — same interactive style as the code viewer; 2 examples committed (one hand-verified, one with real multi-byte UTF-8 divergence)
- (Priyanka) `prose_taxonomy.json` built for P3 — mirrors code's statement/expression split as clause_level vs phrase_level (not the deterministic/open-ended axis, which is code-keyword-specific), verified against all 32 real observed labels
- (Priyanka) Golden fixtures — 18 files (6 py/6 cpp/6 prose), hand-picked and spot-checked by eye, `stream_b/validate_golden_fixtures.py` is the shared regression test (all pass)
- Memory-unsafe region tagging / identifier spans (O5) deliberately deferred — proposal's own timeline places O5 after mid-submission
