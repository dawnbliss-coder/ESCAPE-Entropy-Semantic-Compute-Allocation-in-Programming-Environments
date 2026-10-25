# ESCAPE Independent Audit Report

**Scope:** deep-read of the implementation, verification of frozen results, code-level
correctness tracing, and independent recomputation of every reported headline number.
**Date:** 2026-10-19. **Auditor:** OpenCode `escape-research` skill (read-only audit;
no scientific code, protocol, results, datasets or claims were modified).

Verdict up front: the measurement pipeline is **unusually rigorous for a course
project**. Every reported headline number was independently reproduced from retained
artifacts, the frozen calibration/protocol machinery works as documented, and the
project's claims are accurately bounded in the report. The scientifically weak parts are
**not bugs**; they are (a) an English holdout that is not pristine, (b) an O5 residual
that is statistically significant but unpublished pending a label audit, (c) the
structure-blind-baseline result, which reframes the headline question rather than
supporting H1, and (d) several small documentation drifts and unquantified replication
artefacts.

---

## Phase 0 — Ground truth and architecture

- **Git:** repo `ESCAPE-Entropy-Semantic-Compute-Allocation-in-Programming-Environments/`,
  branch `main` @ `38c6a80`, working tree clean except the untracked
  `.opencode/skills/escape-research/SKILL.md` (the skill itself). No scientific files
  modified anywhere in this audit.
- **Retained artifacts:** `results/final/c-v1-2721954/` + `SHA256SUMS`. All 18 checksums
  present locally **verify OK**. Three checksummed files are not retained locally
  (`bpp_summary.parquet`, `o5/o5_design.csv`, `o5_inverse_bpp/o5_design.csv`) and are
  skipped by `--ignore-missing`; the README mentions only the two O5 design matrices,
  not `bpp_summary.parquet` — a small documentation drift.
- **Data flow:** corpus bytes → Stream A (`stream_a/`, BLT patcher
  `itazap/blt-1b-hf@91aa6b8e`, swa512 window, float32, τ = 1.335442066192627 nats, no cap,
  8192-token chunks) → per-byte entropies + patch boundaries → Stream B (`stream_b/`,
  tree-sitter ASTs for py/cpp, benepar parses for prose, whitespace/word baselines,
  O4 noise, O5 regions) → `escape_scoring` C-v1 (`convert-legacy` → `validate` →
  `calibrate` → `score` → `baselines` / `iou` / `o4` / `o5`) → `reports/mid_submission/`
  build pipeline (`build_results.py` → macros → LaTeX → PDF, `check_report.py`).
- **Reproduction:** `scripts/ada_final/` SLURM jobs; `docs/FINAL_RUN.md` is the runbook.
  Raw corpora, weights, and per-byte arrays are regenerated, not stored in git.

## Phase 1 — Scientific question vs. what is actually measured

**Question asked:** do BLT entropy-triggered patch boundaries land on syntactic
constituent starts — beyond a density-matched null, beyond trivial line/word splitting,
and more in code than prose? **Measured:** exactly that, via start-offset F1 margin over
a fixed-EOF patch-shuffle null, structure-blind baselines, and a code/prose contrast.
**Motivated but not measured:** "semantic understanding" (explicitly disclaimed), and
true adaptive-compute allocation (only BPP / inverse-BPP proxies; no FLOP accounting).

| Item | Written claim | Operationalization | Supporting evidence | Falsifying evidence | Status on retained data |
|---|---|---|---|---|---|
| **P1** | boundaries align with constituent starts above density-matched null | start F1 vs fixed-EOF null, k frozen on calibration | F1 margin over null | margin CI containing 0 | **Supported** — py +0.1196 [0.1174, 0.1216]; cpp +0.1294 [0.1246, 0.1335]; en +0.3478 [0.3406, 0.3544] |
| **R2** (baseline burden, proposal §5.3) | alignment survives structure-blind baselines | BLT − best of newline/indent/word F1 | BLT > baselines | BLT < baselines | **Not supported** — py −0.242 [−0.247, −0.237] (indent 0.394); cpp −0.173 [−0.181, −0.167] (word 0.339); en −0.274 [−0.280, −0.268] (word 0.769) |
| **P2** | starts align better than ends | start vs end F1 margin | start > end | end ≥ start | **Supported** — end margins −0.017 / −0.035 / −0.079 |
| **P3** | BPP variance high in code, low in prose; prose near chance | within-file BPP variance; domain margin contrast | code variance > prose; code margin > prose | reverse | **Not supported** — prose margin (+0.348) > code margin (+0.129); C++ variance is high but the code/prose *alignment* contrast runs the wrong way |
| **P4** | weaker alignment for deterministic openers (`if`, `for`, `def`) | mean per-type recall margin, det vs open | det < open | det ≥ open | **Reversed** — py det +0.759 vs open +0.611; cpp +0.691 vs +0.451 |
| Precision > recall | entropy bounds branching | P vs R at starts | P > R | P ≤ R | **Not supported** — P ≈ 0.08–0.34, R ≈ 0.65–0.89 in all languages |
| **O1/O3** | alignment, Python vs C++ | P1 tables side-by-side | — | — | Done |
| **O2** | compute-allocation ratio | mean BPP 5.68 / 7.86 / 3.77 (unweighted per-file means) | — | — | Done |
| **O4** | robustness to code noise | margin/BPP shift on 4 noise types, 1000 held-out files/lang | small degradation | large/systemic failure | **Supported** — worst drop −0.016 (py indent), −0.013 (cpp indent); keyword typos move BPP most (−0.23 / −0.34) |
| **O5** | memory-unsafe effect shrinks to zero under identifier controls | region OLS, file fixed effects + statement controls | is_memory_unsafe ≈ 0 | significant nonzero residual | **Computed but deferred** — see Phase 2 finding F5 |

Drift check: the project has drifted from "entropy bounds grammatical branching" toward
"entropy tracks line/word segmentation"; the report **says so explicitly** and states
the baseline comparison is the burden of proof. The drift is honest, not silent.

## Phase 2 — Code-level correctness audit

### Verified correct (traced to code, not assumed)

1. **Entropy computation** (`stream_a/patcher_model.py`, `patching.py`): nats via
   `log_softmax` in float32 over all 260 logits; causal (no future information);
   `entropies[i] = H(p(b_i | BOS, b_0..b_{i−1}))`; threshold compared strictly in float64
   on the stored float32 — every trigger decision is reproducible from saved arrays.
2. **Byte-offset contract** (`escape_common/offsets.py`, `escape_scoring/contract.py`):
   all offsets are bytes into frozen UTF-8; manifest/text/patcher byte-length cross-check;
   node/region spans must land on codepoint edges; prose char→byte conversion is
   validated **per span** against benepar's own span text (raises on mismatch).
3. **Matching/scoring** (`escape_scoring/metrics.py`, `fast.py`): greedy ordered
   interval matching is maximum-cardinality (correct for 1-D intervals); interior-only
   boundaries; outermost-node attribution; zero-denominator → 0; `fast.py` verified
   byte-identical to the reference by `tests/scoring_v1/test_fast.py`.
4. **Fixed-EOF null** (`metrics.shuffled_boundaries`, `fast.FixedEofNull`): preserves
   length multiset, labels, count, file span; final EOF patch fixed; deterministic
   per-(seed, sample_id, purpose) PCG64 streams; separate purposes for calibration,
   test, bootstrap, IoU, O4 — no cross-purpose draw contamination.
5. **Procedural integrity:** k frozen on calibration-only (record checksum verified on
   recompute; 1600/6400 + 60/240 splits; monotone margins select k=0); `score`,
   `baselines`, `iou`, `o4` all verify `calibration_record_sha256` linkage; report build
   refuses synthetic inputs, enforces the 10,000/2,000 protocol, derives every
   data-dependent sentence from CIs, and renders `MISSING-RESULT` rather than silently
   dropping macros; verdict rule fixed: CI must exclude 0.
6. **τ provenance:** 1.335442066192627 nats is the checkpoint's patching threshold,
   recorded in `runs.json`, not tuned on any split.
7. **Two-scorer separation:** `escape_eval/` remains preliminary-only and feeds no
   report numbers.

### Findings (weaknesses, drifts, and items to fix — none is a silent scientific bug)

- **F1 — English holdout is not pristine (procedural).** The legacy scorer ran
  `results/prose_kgrid_*` over the English paragraphs **before** the C-v1 20/80 split
  was frozen; SCORING_PROTOCOL and the paper's Limitations disclose this. Consequence:
  English P1/P3 numbers are exploratory, not confirmatory. Any final-submission claim
  resting on the English +0.348 margin must say so, or acquire unseen English data.
- **F2 — O5 has a statistically significant positive residual that is unpublished.**
  `o5/o5.json` (statsmodels 0.15.0, file-cluster robust, n=12,609 regions/1,500 files):
  `is_memory_unsafe` = +0.00693 [0.00431, 0.00955], p = 2.4e-7 (boundary density) and
  +0.01123 (inverse-BPP), *after* identifier-entropy controls and file fixed effects.
  The proposal predicted shrinkage toward zero. The paper body defers the reading
  "pending a label audit" — the correct decision — but the final submission cannot both
  keep O5 as an objective and leave this number out: the audit priority is validating
  the memory-unsafe labels (unsafe 5,271 vs comparison 7,338 units) and ruling out
  region-selection effects (e.g., unsafe regions clustering in longer/short
  statements despite `statement_type` controls).
- **F3 — Chunk-reset artefact is real and unquantified in the paper.** 2,046/13,040
  test files exceed 8,191 bytes (2,016 C++, 30 Python) — 31.5% of C++ test files —
  so a third of C++ files contain bytes whose context is truncated by the 8,192-token
  chunk boundary. Recorded per file in run metadata, disclosed in prose, but no
  sensitivity table (e.g., margins excluding reset-affected bytes) exists.
- **F4 — Independence assumption is optimistic for Python.** CodeSearchNet functions
  can share repositories; the file bootstrap treats files as independent and may
  understate variance (disclosed in Limitations). A repository-grouped bootstrap would
  be the cheap, principled check.
- **F5 — Small definitional edges (document, don't fix):**
  - `indent_change` baseline initializes `prev=0`, so the first non-blank line with
    nonzero indent counts as a "change" against a virtual predecessor (edge case).
  - `word_start` baseline excludes offset 0 (interior convention), consistent with BLT
    boundary exclusions.
  - P3 "mean bytes per patch" is the **unweighted mean of per-file mean BPP** (verified
    in `build_results.py`), which overweights small files vs the pooled byte-weighted
    mean (py 5.99, cpp 7.22 pooled vs 5.68/7.86 reported). Legitimate choice, but the
    paper should say "mean per-file BPP" to be exact.
  - IoU excludes files with zero spans (6,188/6,400 C++ files), documented.
  - O4 `missing_semicolon` drops 69 C++ target nodes that fall inside edited ranges
    (counted in `o4_build_summary.json`).
- **F6 — Retention/doco drift (small):** `bpp_summary.parquet` is checksummed but not
  retained locally (README only explains the two O5 CSVs); `docs/FINAL_RUN.md` still
  says the report build pipeline "was removed after submission" although
  `build_results.py`/`build.sh`/`check_report.py` are present and documented in
  `reports/mid_submission/README.md`.
- **F7 — tests assume pinned env:** `escape_scoring/requirements.txt` documents that
  byte-exact reproduction needs numpy 2.5.3 (float64 reduction differences vs 1.26.x).
  Reproduced here in an isolated 3.14 venv: **32/32 scoring_v1 pass** (including the
  byte-exact Ada reference reproduction), **90/90** legacy eval/common pass, **51/52**
  stream_a pass (the single failure is matplotlib missing from the audit venv — a UI
  figure test, not scoring logic).

## Phase 3 — Independent recomputation (all verified)

| Quantity | Reported | Recomputed | Source |
|---|---|---|---|
| P1 margin py / cpp / en | +0.120 / +0.129 / +0.348 | +0.1196 / +0.1294 / +0.3478 | per_file.csv pooled tp/fp/fn |
| P1 CI py / cpp / en | [0.117,0.122] / [0.125,0.134] / [0.341,0.354] | identical | scores.json |
| Strongest baseline py / cpp / en | indent 0.394 / word 0.339 / word 0.769 | 0.3936 / 0.3386 / 0.7689 | baselines.json |
| BLT − best CI | −0.242 / −0.173 / −0.274 | [−0.247,−0.237] / [−0.181,−0.167] / [−0.280,−0.268] | baselines.json |
| P2 end margins | −0.017 / −0.035 / −0.079 | −0.0167 / −0.0354 / −0.0787 | scores.json |
| P3 BPP py / cpp / en | 5.68 / 7.86 / 3.77 | 5.679 / 7.860 / 3.770 | diagnostics.csv (unweighted per-file) |
| P4 det vs open py / cpp | +0.759 vs +0.611 / +0.691 vs +0.451 | identical | scores.json × taxonomy.json |
| IoU py (BLT/null/line) | 0.351/0.342/0.546 | identical | iou.json |
| O4 worst drop py / cpp | −0.016 / −0.013 (indent) | −0.0164 / −0.0133 | o4.json |
| O4 keyword-typo ΔBPP | −0.23 / −0.34 | −0.228 / −0.338 | o4.json |
| k calibration margins | monotone, k=0 best | py 0.1214→0.0097; cpp 0.126→0.014; en 0.3565→0.0019 | calibration.json |
| calibration record hash | d5f08e4… | recomputed, matches | calibration.json |

Every aggregate matched the retained JSON to the last digit; no rounding drift between
retained artifacts, generated report macros, and the README was found.

## What is correct / broken / uncertain / weak

- **Correct:** the byte contract, entropy/boundary pipeline, matching rule, null model,
  calibration machinery, baseline scoring, statistics code, report-generation
  discipline, and every published number.
- **Broken:** nothing in the scientific path. (One local doc drift in F6; one test
  dependency outside the C env — matplotlib — neither affects results.)
- **Uncertain:** the O5 residual (label audit pending); English confirmatory status;
  the generalisation of file-level bootstrap CIs across shared CodeSearchNet
  repositories; the fraction of C++ alignment that survives excluding chunk-reset bytes.
- **Weak:** the positive P1 result does not survive the project's own burden-of-proof
  test — entropy boundaries track line/word segmentation more than grammar. This is the
  project's central scientific fact, and it is honestly reported. The mechanism claims
  that failed (precision > recall, P4 direction) are the sharpest tests, and both
  failed, which means the proposal's "Grammatical Branching Hypothesis" is refuted by
  its own evidence while the weaker "entropy is a genuine but shallow structural
  signal" reading stands.

## What should happen next (in priority order)

1. **O5 label audit** (already on the timeline): validate memory-unsafe region labels
   against the tree-sitter queries, then decide whether the +0.0069 significant
   residual survives corrections. This is the only place a *new quantitative claim*
   may still emerge.
2. **Structure-blind reframing:** report the baseline result as the primary finding,
   and run the planned "byte after the opening keyword" analysis to explain the P4
   reversal (starts vs forcing positions).
3. **Cheap robustness tables:** re-score excluding (a) chunk-reset-affected bytes and
   (b) `parse_ok=False` C++ files, as sensitivity rows; add a repository-grouped
   bootstrap for Python.
4. **English status:** either acquire unseen English paragraphs for the final
   submission or relabel the English P1 numbers as exploratory in the paper.
5. **Housekeeping:** restore or regenerate `bpp_summary.parquet` alongside the two O5
   CSVs (or drop them from SHA256SUMS); fix the two documentation drifts in F6; state
   "mean per-file BPP" explicitly in the P3 caption.
