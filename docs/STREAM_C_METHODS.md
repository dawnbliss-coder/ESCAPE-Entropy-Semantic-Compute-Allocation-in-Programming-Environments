# Stream C: minimum evaluation methods (PRELIMINARY, mid submission)

This is a technical specification of what `escape_common/` and `escape_eval/` compute. It is
not report prose. Every number these modules produce is **preliminary**, and every table
carries `preliminary=True`. Stream C methodology belongs to the Stream C owner. The last
section lists the decisions that owner still needs to confirm. Each one is a single
configurable parameter, so changing it does not require restructuring the code.

## 1. Inputs (all read from disk, never recomputed)

- **Stream A.** Read from `boundaries/{run_id}/{file_id}.parquet`, `runs.json` and `bpp_summary.parquet`. The contract is in `docs/SCHEMA.md` and `docs/STREAM_A_FIDELITY.md` §3:
  - one row per patch;
  - `byte_offset` is the first byte of the patch;
  - `trigger` is one of init, entropy or length_cap.
- **Stream B.** Read from:
  - `structure/{domain}/{file_id}.parquet` (end offsets are exclusive);
  - `whitespace/{domain}/{file_id}.parquet`;
  - `taxonomy.json` and `prose_taxonomy.json`;
  - the golden fixtures, for smoke runs only.
- **Validation.** Every boundaries file is validated before scoring (`escape_common/validate.py`). Scoring aborts on any contract violation.

## 2. Sets per file *f*

| Symbol | Definition |
|---|---|
| X_f | `init` offsets, read from the file and never hard-coded. For this checkpoint X_f = {0}. |
| B_f | `entropy` offsets: the scored BLT boundaries. |
| C_f | `length_cap` offsets. Scored separately (`length_cap` columns) and never mixed into B_f. There are none at the checkpoint's native `max_patch_length = null`. |
| T_f | unique `start_byte` values minus X_f. Nested nodes sharing a start count once. |
| E_f | unique `end_byte` values minus X_f (P2). |
| W_f | unique `byte_offset` of `kind == "newline"` rows minus X_f. This is the only whitespace baseline; there is no second definition. Prose has no whitespace baseline (paragraphs contain no newlines), so it is reported as N/A. |
| S_f | unique word-start offsets minus X_f: byte i where byte i is a word byte (`[A-Za-z0-9_]`) and byte i-1 is not (or i == 0). This is the word-boundary baseline, added as the control for the P3 prose confound: constituent starts in prose largely coincide with word starts, and W_f is empty for prose. Not density matched. |

**Outermost attribution.**
- Each start is attributed to its outermost node, taken as the smallest `depth`, then the larger `end_byte`, then the lexicographically smallest `node_type` (`stream_b/ast_walker.py` docstring).
- Each end is attributed by smallest depth, then smaller `start_byte`, then `node_type`.
- Attribution supplies the stratum values `depth` (raw depth column), `node_type` and `category` (taxonomy).

## 3. Metrics

**Tolerance match.** Boundary *b* matches target *t* iff |b − t| ≤ k. Counts are **any-match**:
- TPp = #{b ∈ B : ∃ t, |b − t| ≤ k}
- TPr = #{t ∈ T : ∃ b, |b − t| ≤ k}

**Pooled (micro) metrics over the files of a domain.**
- P = ΣTPp / Σ|B|
- R = ΣTPr / Σ|T|
- F1 = 2PR/(P+R)

**Strata.**
- Depth, node_type and category restrict the target side only. Precision_s uses the matches to T^s over all of |B|, and recall_s is taken over T^s.
- Per-depth scoring uses the raw depth. Cross-language depth normalisation is post-mid.

**Secondary metric (span IoU).** For each unique non-empty AST span, take the best IoU with any single patch, then the span-weighted mean. It is reported for BLT, H0 layouts and whitespace line segments.

**Implementation.** Precision uses dilated target signatures and recall uses row-offset `searchsorted`, both in `escape_eval/matching.py`. Both are checked exactly against a naive pure-Python reference in `tests/test_eval_matching.py`.

## 4. Tolerance k (calibrated a priori)

- **Where.** The calib split only (100 py + 100 cpp), per language. Calibration refuses any non-calib file.
- **How.** Over the grid k ∈ {0,…,8}: margin(k) = F1_BLT(k) − mean pooled F1 over R_cal = 1000 H0 resamples.
- **Selection.** k* = argmax margin, with ties going to the smallest k.
- **Record.** `results/k_calibration/selected_k.json` stores:
  - the method string, grid, R_cal and seed;
  - the run_id and run config;
  - the sorted file_ids and their sha256;
  - the manifest sha256 and git state.
- **Use in scoring.** `run_eval` takes k only from a non-smoke `selected_k.json`. Overrides are rejected outside `--smoke`.
- **Prose.** Prose has no calib split, so it is scored at every calibrated code k.
- **Rationale.** Only the margin over H0 is informative, because boundary density is pinned by τ. Maximising that margin on held-out calib files picks the tolerance at which structure, rather than density, is most visible, without looking at main.

## 5. Null model H0 (density-matched)

**Per file** (`escape_eval/baselines.py`):
- Let a = max(X_f). BLT patches starting before a keep their layout.
- The lengths of the patches starting at ≥ a are **uniformly permuted** and laid out from a. The candidate boundaries are a + cumsum(perm)[:−1].
- If length caps exist, |B_f| candidates are chosen uniformly without replacement.

**What is preserved and what is randomised.**
- Preserved: the boundary count |B_f|, the multiset of patch lengths (BLT's own empirical distribution for that file), the exact file span and the forced prefix.
- Randomised: positions and order.

**Randomness.**
- Each file has its own streams, `PCG64(SeedSequence([master_seed, sha256(file_id)[:8], purpose]))`, with master seed 20260915.
- Results do not depend on file order or chunk size (tests: `tests/test_eval_targets_baselines.py`).

## 6. Statistical tests and intervals

- **Permutation test, BLT vs H0** (default R = 10,000).
  - Each resample draws one independent H0 set per file, and pooled P/R/F1 are recomputed.
  - Reported: p = (1 + #{null ≥ observed}) / (R + 1), one-sided (BLT > H0); null mean, sd and 2.5/97.5% quantiles; Δ = obs − null mean; z = Δ/sd; ratio = obs / null mean.
  - Pooling: H0 draws feed pooled per-resample sums (memory O(R × strata)). One draw per file per resample is shared by all kinds, k and strata, so comparisons are paired.
- **Confidence intervals: paired cluster bootstrap over files** (default B = 10,000, percentile 95%).
  - Files are the independent sampling unit, while boundaries inside a file are dependent, so whole files are resampled with replacement.
  - One weight matrix per (domain, k) drives every quantity: the BLT, E[H0] and whitespace metrics, the BLT−H0 and BLT−whitespace deltas, and the P2/P4 contrasts. All contrasts are therefore paired.
  - E[H0] uses each file's mean H0 counts over the R resamples.
  - Percentile intervals are simple and transformation-invariant. The bootstrap also handles the ratio-of-sums estimators without any normality assumption.
- **BLT vs whitespace: paired randomisation test** (R = 10,000). Per file, BLT and whitespace counts are swapped (numerators and denominators) with probability ½. The statistic is the pooled difference, tested one-sided.
- **BLT vs word baseline: paired randomisation test** (R = 10,000, same procedure, separate purpose stream). This is the P3 confound control; it is reported for every domain, not just prose.
- **P3, BPP.**
  - Difference of the medians of per-file `var_bpp` (and `mean_bpp`) between code and prose, with bootstrap CIs that resample within each group.
  - Two-sided Mann–Whitney U test.
  - Sensitivity subset: code files whose `n_bytes` falls inside the prose size range (prose paragraphs are 300–3,000 bytes, code files are much longer).

## 7. Predictions as tables (`escape_eval/run_eval.py`)

| Table | Content |
|---|---|
| `alignment` | P1 (kind=start, axis=all), per depth (axis=depth), per node_type and category; P2's end rows (kind=end). BLT / E[H0] / whitespace / word-boundary with CIs, permutation and randomisation p-values, deltas with CIs, length-capped rows separately |
| `p2_start_end` | start − end for BLT, and for the margin over H0, with CIs; whether precision > recall at starts, plus the bootstrap share |
| `p4_openers` | recall lift over H0 for deterministic openers vs open-ended constructs; open − deterministic contrast with CI; P4 predicts a positive contrast |
| `p3_bpp` | code vs prose BPP variance and mean |
| `p3_alignment` | F1 margin over H0 at the same k, code − prose, with CI |
| `iou` | secondary |

Every output directory has a `params.json` recording all parameters, seeds, file lists and
hashes, the run config snapshot, git state and package versions. It is registered in
`results/manifest.json`. Smoke runs (golden fixtures, overridden k) go only under
`results/smoke/`.

## 8. Decisions for the Stream C owner to confirm

Each item below is implemented as a single configurable parameter. **Provisional
defaults (set by Shashwat on 2026-09-15, pending Divyansh's confirmation)** are
marked "PROVISIONAL". Changing any of them does not require restructuring the
code; re-run `run_eval` with the new parameter.

1. **Any-match counts rather than one-to-one bipartite matching.** The two coincide at k = 0. One-to-one matching would lower recall where one boundary sits within k of two nested starts. **PROVISIONAL: any-match (as implemented).**
2. **Per-file permutation of BLT's own patch lengths as H0.** The alternative is drawing lengths from the pooled per-language distribution, which gives only an approximate count and span. **PROVISIONAL: per-file (as implemented).**
3. **k selection criterion.** The F1 margin over E[H0]; alternatives are pure F1 or a precision margin. Grid 0..8. **PROVISIONAL: F1 margin over E[H0] (as implemented).**
4. **Targets at init offsets are excluded** for all methods symmetrically. **PROVISIONAL: excluded (as implemented).**
5. **End targets at `n_bytes` are kept** (`exclude_eof_end` is available). **PROVISIONAL: kept (as implemented).**
6. **Per-depth and per-type precision use the full |B| denominator.** Recall is the primary per-stratum quantity. **PROVISIONAL: full |B| denominator (as implemented).**
7. **Percentile bootstrap CIs.** BCa is the alternative. **PROVISIONAL: percentile (as implemented).**
8. **Prose is scored at the code k values** (it has no calib split). **PROVISIONAL: as implemented.**
9. **BPP counts all patches**, including the init patch (the `n_init` column allows exclusion). **PROVISIONAL: all patches (as implemented).**
10. **Word-boundary baseline S_f (NEW).** Definition in §2. It exists because the prose constituent-start signal was observed to overlap with word segmentation (`docs/MID_HANDOFF.md` §6 caveat), and W_f is empty for prose. **PROVISIONAL: byte-level `[A-Za-z0-9_]` word starts, not density matched.**
11. **parse_ok stratification (NEW).** `run_eval --parse-ok {all,clean,recovered}` restricts code files by the manifest `parse_ok` flag (tree-sitter error-recovery, ≈28% of C++). Prose is unaffected. **PROVISIONAL: `all` is the default; clean/recovered are sensitivity runs.**
