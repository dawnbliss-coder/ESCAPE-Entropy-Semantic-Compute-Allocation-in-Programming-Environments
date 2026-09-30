# Reproduce the Ada run used in the mid-submission

The retained completed run is `results/final/c-v1-2721954/`. The steps below
reproduce extraction and scoring; cluster paths are historical defaults to adapt
to your account. Scripts are in `scripts/ada_final/`; the report is in
`reports/mid_submission/`.

## Before you start (once)

1. **C++ data access.** Log in to huggingface.co as the account whose token is on Ada
   (`Devatri`, in `~/.cache/huggingface/token`). Open
   <https://huggingface.co/datasets/bigcode/the-stack-smol> and click
   **Agree and access repository**. Access is granted immediately (gate: `auto`).
   On 2026-09-28 the token got `403 GatedRepo: not in the authorized list`.
   The token is fine-grained. If it still gets 403 after you agree, edit it at
   huggingface.co/settings/tokens and enable *Read access to contents of all public
   gated repos you can access*.
   Check from the Ada login node:
   ```bash
   T=$(cat ~/.cache/huggingface/token); curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $T" \
     https://huggingface.co/datasets/bigcode/the-stack-smol/resolve/4a6938ce94446f324c6629e7de00ac591710044b/data/cpp/data.json
   ```
   The result should be `200` or `302`.
2. VPN on, so `ssh ada` works.

## Run

```bash
# laptop, repo root: copy code into a NEW Ada directory (existing ones are untouched)
bash scripts/ada_final/sync_to_ada.sh            # -> ada:~/escape-final-20260928
# Ada login node
cd ~/escape-final-20260928 && bash scripts/ada_final/submit_all.sh
squeue -u $USER; tail -f results/final/logs/*.log
```

| Job | Resources | What it does |
|---|---|---|
| `00_setup` | CPU, ~1 h | `.deps-c` (numba, scipy, statsmodels) layered on the existing `~/anlp_a2/venv` (torch 2.6.0+cu124, transformers 5.17.0, used read-only); `.venv_b` from `stream_b/requirements.txt` with CPU torch (tokenizers 0.13.3 is built with the Rust module); spaCy/benepar models; patcher tensors; CPU tests |
| `10_data` | CPU, ~1–3 h | restores py (CodeSearchNet), cpp (the-stack-smol), prose (WikiText-103) and checks every file against `corpus/manifest.parquet` (never rewritten); tree-sitter structure, whitespace, benepar prose parses, O5 regions; `--verify`. Status: `results/final/data_status.txt` |
| `20_extract` | 4 × 2080 Ti | fidelity-gate rerun (shard 0) → `results/final/stream_a_fidelity_ada/`; swa512 entropies and boundaries for all files, split into 4 byte-balanced shards; resumable, retries failures |
| `30_score` | 32 CPU, scratch | checks extraction is complete; builds the BPP summary; runs the A/B validators; converts to C-v1 JSONL on node-local scratch; validates; freezes k on the 20% calibration split; scores the 80% test split (10,000 permutations, 2,000 bootstraps); baselines; O5 (two outcomes). Output: `results/final/c-v1-<jobid>/` with `SHA256SUMS` |

Home quota: about 4.5 GB is added (the B venv, the C deps, models, data). HF and pip caches
and the multi-GB JSONL stay on node-local scratch, and are deleted when each job ends.

**Re-running parts:** `SKIP_SETUP=1 bash scripts/ada_final/submit_all.sh` or
`SKIP_SETUP=1 SKIP_DATA=1 …`. Extraction resumes where it stopped. A scoring
output directory is never overwritten; each job writes a new one.

**If C++ cannot be restored** the data job exits non-zero and nothing downstream
runs. To score only Python and prose, say so explicitly:
`SKIP_SETUP=1 SKIP_DATA=1 ESCAPE_DOMAINS="py prose" bash scripts/ada_final/submit_all.sh`.
The selection is recorded in `selection.txt`. The report then shows MISSING-RESULT
for C++/O5 and fails its check, by design.

## Report

```bash
# laptop, repo root
bash scripts/ada_final/fetch_from_ada.sh
python3 reports/mid_submission/build_results.py        # newest results/final/c-v1-*/
TECTONIC_BIN=/path/to/tectonic bash reports/mid_submission/build.sh   # or a TeX Live with bibtex
python reports/mid_submission/check_report.py                # official aclpubcheck, --paper_type short
```

`build_results.py` writes every number and every data-dependent phrase
(`generated/results.tex`, `evidence.json`, `depth_margin.pdf`). Its verdict rules are
fixed in code and do not depend on the results: a difference counts as positive or
negative only when its 95% bootstrap interval excludes 0. It refuses synthetic inputs
and anything other than the frozen 10,000/2,000 protocol. `check_report.py` fails on
MISSING-RESULT, synthetic watermarks, unresolved references, non-A4 pages, unembedded
fonts, a body longer than 4 pages, or any pubcheck finding.

Read the generated sentences against the tables before submitting. The Discussion is
written to hold whatever the direction of the results, but the authors are responsible
for the interpretation.

## Changes to C made for this run

- `escape_scoring/fast.py`: numba matching and a vectorised fixed-EOF null. Its output
  is identical to the reference: `tests/scoring_v1/test_fast.py` reproduces Ada job
  2719654's `calibration.json` and `scores.json` exactly, with 1 and with 3 workers.
- `--workers` for `validate`, `calibrate`, `score`, `baselines`, `o5`. The results do
  not depend on the number of workers.
- `escape_scoring baselines`: newline, indentation-change and word-start baselines
  under the frozen k, with paired bootstrap of BLT − baseline F1.
- `stream_b/build_o5_regions.py`: O5 analysis units, memory-unsafe masks and identifier
  spans (a documented default operationalisation; revise it as a team if needed).
  The adapter now reads `analysis_regions/`.
- `scripts/restore_stream_b_artifacts.py --code-domains`: restore py without cpp.
