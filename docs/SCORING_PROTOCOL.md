# Workstream C scoring protocol v1.0.0

What this is: the authoritative methodology behind every number in the
submitted mid-submission report, implemented by `escape_scoring/` and tested
against the frozen run `results/final/c-v1-2721954/`. This document freezes the
protocol before its new test run. It supersedes the
preliminary choices in `STREAM_C_METHODS.md` **only for `escape_scoring`**. Existing
`escape_eval`, A/B implementations, results and golden files remain unchanged.
Earlier prose results were already examined by the team; this document cannot
retroactively preregister them or turn a resplit of those data into an untouched
confirmatory holdout. All new real-data outputs remain preliminary.

## Byte and entropy contract

`schemas/v1/*.schema.json` define Draft 2020-12 schemas. Each JSONL line is one
record; `sample_manifest` and `patcher_output` have one record per sample, nodes
and regions have one per node/region. Every record has `schema_version=1.0.0`.
No producer needs to import C. Validation checks schemas plus cross-record joins,
source SHA256, UTF-8 decoding, complete entropy coverage and patch tiling.

Offset i means immediately before byte i. Spans are half-open `[start_byte,end_byte)`.
Nodes/regions must end on UTF-8 codepoint edges; model patches may split a multibyte
codepoint. Text length is measured on frozen raw bytes, never character counts.
Edge validation cannot detect every character/byte mistake when both happen to be
valid edges: B must still round-trip parser offsets against the frozen text.

`entropy_records[i]` predicts byte i from preceding bytes. Values are the model's
global next-byte Shannon entropy, never normalized relative to local entropy.
The model's finite context is permitted; local entropy normalization is not.
Units are declared nats or bits. C does not infer entropy from threshold or labels.
A must supply checkpoint/revision/run/config metadata. Its existing full 260-logit
distribution is retained as documented in `STREAM_A_FIDELITY.md`.

Patches tile the file. A patch's **termination reason** describes its right endpoint.
Boundary records are exactly these endpoints and labels, including diagnostic EOF.
The existing A format instead labels patch starts; `convert-legacy` moves the next
row's trigger to the preceding patch's end (`length_cap` → `max_length`). It never
changes A's files. Empty files have no patches or entropy records.

## Primary metrics and strata

Only entropy boundaries at `0 < i < byte_length` enter P1/P2. Cap boundaries are
diagnostics only. Target nodes must have `is_primary_alignment_target=true`, a
positive span and positive depth. Exclude root/program/translation_unit/module/
document/paragraph wrappers and endpoint offsets 0 and file end. The primary
target is the **start**, never span IoU. End scores are a secondary P2 diagnostic
using the same frozen tolerance and exclusions.

Deduplicate target starts before matching. At each offset attribution goes to the
smallest depth, then longest span, then lexical node type and node ID. Sorted
greedy feasible pairing computes maximum-cardinality one-to-one matching under
`abs(prediction-target) <= k`; each endpoint is used at most once.

TP is the number of pairs, FP = predictions − TP, FN = unique targets − TP.
P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). A zero denominator yields 0;
empty files remain in manifests and file-resampling units. Global, language and
domain results pool counts (micro). Calibration uses a different, explicitly
file-level criterion below. Depth/type scores restrict target sets and rematch;
each uses all predictions for all files of that language, including files with
no targets in that stratum. Strata are separate analyses, not additive partitions
of TP. Raw depth is reported separately by language, not assumed comparable.

## Frozen tolerance rule

For **each language**, including English prose:

1. Sort files by SHA256 of `seed + NUL + language + NUL + sample_id`, then ID.
2. Assign the first `ceil(n/5)` files to calibration, the remainder to test.
   At least five distinct files per language are required. This is exact 20/80
   when n is divisible by five; upward rounding is explicitly recorded otherwise.
   Duplicate text hashes within a language are rejected for deduplication, so
   identical texts cannot leak between splits. No random byte-level split.
3. On calibration files **only**, evaluate K = [0, 1, 2, 4, 8], with the same seeded
   H0 draws reused across k. Default seed = 20260927; default R = 10,000.
4. Choose the k maximizing the median over files of
   `(F1_observed(file,k) - mean_r F1_H0(file,k,r))`. Ties within floating-point
   tolerance 1e-12 choose the smallest k.
5. Write a new calibration JSON containing every split assignment, text/source
   hash, seed, K, criterion, per-file curves, chosen k, model configuration,
   input hashes and a record checksum. Refuse to overwrite it.
6. Final scoring accepts only that frozen record and its original dataset. It
   verifies manifest, split, calibration inputs and model identity; there is no
   k override or retuning flag. Test values are never used in selecting k.

The old 100/7900 code split and no-calibration prose split are incompatible with
this rule. Conversion preserves old assignments as metadata, assigns `unassigned`
for C, and leaves B's manifest untouched. Changing the input corpus requires a
new named experiment and transparent reporting; it is not permission to tune on
test outcomes. Checksums detect drift, not intentional record forgery.

## H0 and inference

**Explicit user decision, 2026-09-27:** keep the final EOF patch last; uniformly
shuffle all other `(patch_length, termination_reason)` tuples within each file.
This resolves a contradiction in the initial request: unrestricted EOF shuffling
can move entropy to excluded file-end and change the scored density. The approved
conditional shuffle preserves the entire length multiset, all labels, total byte
length, final patch and the exact number of scored entropy boundaries. It also
fixes the start of the final patch; disclose this conditioning when interpreting H0.

Reconstruct cumulative endpoints, score only interior entropy-labeled endpoints.
Never shuffle lengths separately from labels or sample labels afterwards. Malformed
lengths/labels/coverage fail; no silent truncation, resampling or repair. A regression
test also demonstrates why unrestricted EOF shuffling is invalid under the strict
original interpretation. The final CLI only runs the approved fixed-EOF rule.

Each of R = 10,000 permutations supplies an independent shuffle per evaluation
file, pools the same counts as observed, and shares the layout across strata.
Streams use NumPy PCG64 seeded by the first 128 bits of
SHA256(`seed + NUL + sample_id + NUL + purpose`). File iteration order does not
change draws. Purposes separate calibration, test permutations and bootstrap.

For precision, recall and F1 report null mean, sample SD (ddof=1), linear 2.5/97.5
percentiles, observed−null mean, observed/null mean and
`p = (1 + count(null >= observed)) / (R + 1)` (one-sided).
Undefined ratios are JSON null, never infinity. Null percentiles describe the
permutation distribution; **they are not confidence intervals**.

Observed metric and F1 effect/ratio CIs use paired percentile bootstrap over
whole files (default B=2,000), recomputing pooled counts. The paired null comparator
uses per-file mean null counts. Its F1 equals mean pooled null F1 because the
approved null fixes both denominators. Intervals are unavailable with fewer than
two independent files; ratio intervals are unavailable if any replicate has zero
null denominator. P-values across depth/type strata are exploratory and unadjusted;
no automatic H1 acceptance decision is made.

Diagnostics: entropy count, capped count, both internal boundary densities per
byte, mean/median BPP and population BPP variance. BPP means **bytes per patch**,
not bits per byte; these diagnostics use all complete patches including final EOF.
JSON records exact configuration, software versions and test-input fingerprints;
tidy CSVs contain scores, per-file results and diagnostics.

## O5 specification

Only memory-unsafe **C++** is selected. B supplies `analysis_region` sampling units,
`memory_unsafe` masks and `identifier` spans; C does not label code or choose safe
comparison regions. Analysis units should include both unsafe and comparison
regions under a team-approved sampling rule. A unit is unsafe when it overlaps a
supplied unsafe mask by at least one byte. Masks and identifiers use half-open
intersection; identifier bytes are unioned before averaging.

Default outcome `all_boundary_density` = count of entropy **and cap** boundaries
in `[region_start, region_end)`, excluding file edges, divided by region length.
Optional `inverse_bpp_compute_density` = average over region bytes of the inverse
length of their containing patch (fractional patch-mass allocation). It is a
patching compute proxy, not a measured FLOP count. Unit lengths are not clipped
to whole patches.

OLS predictors: intercept, `is_memory_unsafe`, `mean_identifier_entropy_nats`,
`identifier_byte_proportion`, raw `region_length`, and configurable categorical
language/file controls or numeric B-provided controls. Default categorical control
is sample/file fixed effects; a constant language contributes no dummy. Identifier
entropy is 0 when identifier proportion is 0; both enter jointly. Bits are converted
to nats with ln(2), with no local normalization.

Always export the joined region table, design matrix and least-squares estimates.
If installed, statsmodels provides file-cluster robust covariance and t-based
intervals with finite-sample correction. Otherwise inference is explicitly
unavailable; no package is installed automatically. Rank-deficient designs produce
minimum-norm estimates with `identified=false`, no inferential claims. Fewer than
two files or no residual degrees of freedom also precludes inference. Overlap
dependence is handled only at the file cluster level; very few clusters remain a
limitation. The purpose is **confound isolation**, never semantic risk understanding.

## Remaining decisions / scope limits

- B's analysis-region sampling rule and identifier/unsafe annotations are needed
  for real O5. C cannot invent these labels or comparable safe units.
- The proposal's full H1 claim also needs its whitespace/word controls, P3/P4 and
  fidelity sign-off. Preserved legacy implementations exist, but their results
  use different statistics and cannot be merged into v1 primary tables.
- Existing prose exploration limits confirmatory interpretation of a new split.
  The team must decide whether to acquire genuinely unexamined test data later.
- Per-file duplicate-byte checks do not establish independence of multiple
  CodeSearchNet functions from the same upstream source file or repository.
  B must supply grouping metadata if source-file/repository-level splitting is
  required; do not claim these extracted functions are independent original files.
