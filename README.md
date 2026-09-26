## Setup

- Clone the repo, `cd` into it
- Create a token at huggingface.co/settings/tokens
- `hf auth login --force`, paste the token
- Visit huggingface.co/datasets/bigcode/the-stack-smol, click "Agree and access repository" (same account as the token) — needed for C++ only; Python (CodeSearchNet) and prose (WikiText-103) are not gated
- `bash setup.sh` (Stream B only; `STREAM_A=1 bash setup.sh` also builds the Stream A/C venv and fetches the patcher)
- `source .venv/bin/activate`

## Progress so far

- (Priyanka) Data contract drafted — `docs/SCHEMA.md`
- (Priyanka) Code sources: C++ from `bigcode/the-stack-smol` (a sampled subset of the proposal's "The Stack" — CodeSearchNet has no C++ split); Python from **CodeSearchNet**, matching proposal §6 exactly. **Note for Stream C:** Python rows are isolated functions (not whole files like C++) and carry no license metadata at all (CodeSearchNet has no per-example license field, unlike C++'s the-stack-smol rows) — both are known limitations, flagged for the write-up and relevant to O3's cross-language granularity comparison.
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
- (Priyanka) R2 quantified (`stream_b/quantify_r2.py`) — py 31.8%, cpp 22.5% of AST node starts sit immediately after newline+indent; Python shows the stronger confound, as expected
- (Priyanka) Implementation-complete pass — audited all scripts, fixed real bugs (not just style): an empty-output schema bug affecting ~1.3% of files, an idempotency bug in `build_corpus.py` that would have silently dropped prose rows on a re-run, and a real miscounting bug in the prose tree viewer (caught by cross-checking its output against validated `structure/prose/` data).
- (Priyanka) Python corpus reverted from `the-stack-smol` to CodeSearchNet (26 Sep) — closes out the earlier deviation from proposal §6. C++ untouched. Full rebuild for Python only: `corpus/py`, `structure/py`, `whitespace/py`, `golden_fixtures/py` regenerated under new file_ids; `taxonomy.json` re-verified empirically against the new corpus (no change); all 18 golden fixtures still pass; R2 re-quantified. Only the write-up and TA repo access remain.
- Memory-unsafe region tagging / identifier spans (O5) deliberately deferred — proposal's own timeline places O5 after mid-submission

---

## Restoring Stream B data without rewriting the manifest

`corpus/*.bin`, `structure/` and `whitespace/` are gitignored. `build_corpus.py`,
`build_prose_corpus.py` and `build_structure.py` also rewrite the committed manifest.
`scripts/restore_stream_b_artifacts.py` reuses their functions unchanged, checks every
pulled record against `corpus/manifest.parquet` (n_bytes, sha256, split, source path), and
never writes the manifest. Run it inside `.venv`:

- `HF_TOKEN=... .venv/bin/python scripts/restore_stream_b_artifacts.py --code` restores py from CodeSearchNet (no token needed) and cpp from gated `bigcode/the-stack-smol` (token needed).
- `.venv/bin/python scripts/restore_stream_b_artifacts.py --prose` works on WikiText-103 without a token.
- `.venv/bin/python scripts/restore_stream_b_artifacts.py --structure --whitespace py cpp prose`
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES="" .venv/bin/python stream_b/build_prose_structure.py`
- `.venv/bin/python scripts/restore_stream_b_artifacts.py --verify`

## Stream A: BLT patcher → per-byte entropy → patch boundaries → BPP

The model is [`itazap/blt-1b-hf`](https://huggingface.co/itazap/blt-1b-hf) at revision
`91aa6b8e168046ad91e517d2191e1e974f50cc01`, a conversion of `facebook/blt-1b`
([paper](https://arxiv.org/abs/2412.09871)). Only the patcher (entropy model,
99.5M parameters) is needed, and the pipeline never reads parse data.

Setup uses a separate venv, because benepar pins `transformers==4.30.2`:

- `python3.12 -m venv .venv_a`
- `.venv_a/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130`
- `.venv_a/bin/pip install -r stream_a/requirements.txt`
- `.venv_a/bin/python -m stream_a.fetch_patcher` fetches only the patcher tensors (~199 MB) at the pinned revision, with per-tensor sha256 in `checkpoints/.../provenance.json`.

Pipeline commands, all run from the repo root:

- `.venv_a/bin/python -m stream_a.fidelity --device cuda` runs the checkpoint-fidelity gate. Output is `results/stream_a_fidelity/fidelity.json`; the evidence is written up in `docs/STREAM_A_FIDELITY.md`.
- `.venv_a/bin/python -m stream_a.extract --input golden` extracts the 18 golden fixtures.
- `.venv_a/bin/python -m stream_a.extract --input corpus --domains py cpp --split calib` extracts the calib split (then `--split main`, then `--domains prose --split main`). Extraction is resumable: re-run the same command to continue, and add `--only-failed` to retry failures.
- `.venv_a/bin/python -m stream_a.build_bpp_summary` builds `bpp_summary.parquet`.
- `.venv_a/bin/python -m escape_common.validate --run-id RUN --golden` (or `--split calib`) validates the output contract.
- `.venv_a/bin/python scripts/validate_ab_offsets.py --run-id RUN` cross-checks Stream A and Stream B byte offsets.
- `.venv_a/bin/python -m stream_a.figures --run-id RUN` draws entropy heatmaps, patch-length plots and BPP plots from saved artifacts.

Outputs follow the `docs/SCHEMA.md` contract (conventions in `docs/STREAM_A_FIDELITY.md` §3):

| Path | Contents |
|---|---|
| `entropies/{run_id}/{file_id}.npy` | `float32[n_bytes]`, nats |
| `boundaries/{run_id}/{file_id}.parquet` | `file_id, byte_offset, trigger, entropy, patch_index, patch_length`; `byte_offset` is the first byte of a patch; `trigger` is `init`, `entropy` or `length_cap` |
| `bpp_summary.parquet` | per-file BPP and entropy statistics |
| `runs.json` | every setting that affects outputs |
| `runs/{run_id}/status.parquet` | per-file status, failure tracebacks in `failures/` |

The primary configuration is `context_mode=swa512`: the reference 512-token sliding window
plus 8192-token chunks, which the HF port does not implement. It runs in float32 with the
checkpoint τ = 1.335442066192627 nats and no length cap. The run id is
`swa512-float32-t1.3354-mplnone-91aa6b8e`.

## Stream C: minimum evaluation (PRELIMINARY)

Definitions and open decisions are in `docs/STREAM_C_METHODS.md`. The code is in
`escape_common/` (schema, atomic I/O, byte↔char offsets, validator) and `escape_eval/`.
Every output table is marked `preliminary`.

- `.venv_a/bin/python -m escape_eval.calibrate_k --run-id RUN --source corpus` fixes k per language on the calib split only. It writes `results/k_calibration/selected_k.json`.
- `.venv_a/bin/python -m escape_eval.run_eval --run-id RUN --source corpus --split main --iou` scores the main split. It computes:
  - P1 overall and per depth, P2 and P4, all against H0 (10,000-resample permutation test), against the whitespace baseline and against the **word-boundary baseline** (the P3 prose confound control; paired randomisation tests);
  - 95% file-bootstrap confidence intervals;
  - P3 code vs prose;
  - span IoU as a secondary metric.
  - `--parse-ok {all,clean,recovered}` restricts code files by the manifest `parse_ok` flag (tree-sitter error-recovery; ≈28% of C++) for sensitivity runs; prose is unaffected.
- `.venv_a/bin/python -m escape_eval.figures --analysis-dir results/<analysis_id> --k-dir results/k_calibration` draws the figures.
- Smoke runs on golden fixtures need `--smoke` or `--allow-non-calib` and are written only under `results/smoke/`.
- `bpp_summary.parquet` carries `chunk_len`, `n_chunk_edges` and `n_bytes_truncated_context` per file: the swa512 8192-token chunk reset affects only files longer than 8191 bytes, and these columns let Stream C stratify or flag the affected bytes.

## Tests

- `.venv_a/bin/python -m pytest tests` covers the Stream A logic, the extraction driver, the Stream C matching, H0 and statistics code, the validator and offsets, and a mock end-to-end run. Model tests skip if the patcher weights have not been fetched.
- `.venv/bin/python stream_b/validate_golden_fixtures.py` is the shared Stream B regression test.

## Status as of Mid Submission

The detailed ledger is `docs/IMPL_STATUS.md` and the handoff is `docs/MID_HANDOFF.md`.
Everything in this section was verified on 2026-09-15.

- **Stream A:** implemented, tested and validated on real model outputs.
  - Runs completed: the 18 golden fixtures, and all 300 prose paragraphs under both `swa512` and native-HF `hf_full` context. The validator reports 0 problems.
  - Checkpoint fidelity: see `docs/STREAM_A_FIDELITY.md`.
  - Stream A ↔ Stream B byte offsets were cross-checked on 312 files with 0 problems.
- **Code corpus (py/cpp):** not restored in this clone. The HF token available to this session is rejected (HTTP 401), so the calib split, k calibration and every code-alignment number (P1/P2/P4/H0/whitespace) have **not been produced yet**. All of that code is implemented and tested; the commands are in `docs/IMPL_STATUS.md` §10.
- **Stream C:** implemented and tested, with an end-to-end SMOKE run on golden fixtures using real Stream A outputs. The SMOKE numbers are pipeline checks, not results.
- **P3, prose side (preliminary, real data):** results are in `results/prose_kgrid_swa512-float32-t1.3354-mplnone-91aa6b8e/`.
  - Data: 300 WikiText-103 paragraphs scored against 25,185 benepar constituent starts. `structure/prose` was restored with unchanged Stream B code; all 6 golden prose structures regenerate identically.
  - Method: every tolerance k = 0–8 is reported and none is selected, because prose has no calib split. Each k gets a 10,000-resample permutation test and file-bootstrap CIs.
  - Not yet possible: a code-side comparison (blocked on the token).
  - Confound: prose constituent starts largely coincide with word starts, and no current baseline controls for that (`docs/MID_HANDOFF.md` §6).
