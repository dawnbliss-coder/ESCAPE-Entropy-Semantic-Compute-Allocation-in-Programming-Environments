# ESCAPE: Project Context, Status and Critical Review

**As of:** 2026-10-27 (repo `main` @ `1bf1e1b`) · **Final deadline in the mid-report:** 31 Oct 2026, four days from now.
**Sources:** the code, `results/final/c-v1-2721954/`, `results/distill/`, the proposal, the mid-report, `AUDIT_REPORT.md`, `AUDIT_REPORT_2.md` and `reports/ESCAPE_STUDENT_CV1_EVALUATION.md`. Section 5 adds two checks run for this review on the retained CSVs.

---

## 0. TL;DR

1. **The engineering is strong.** Byte-exact pipeline, frozen calibration, a density-matched null, bootstrap CIs, reproducible results and two independent audits. Every headline number reproduces from the retained artifacts.
2. **The hypothesis as written (H1, the Grammatical Branching Hypothesis) does not hold.** It fails its own sharpest tests:
   - BLT loses to trivial indentation or word-start splitting in every language, by 0.17 to 0.27 F1.
   - Precision is far below recall, the opposite of the prediction.
   - P4 comes out reversed.
   - P3 as measured comes out reversed.

   Only P1 (beats a random null) and P2 (starts beat ends) survive.
3. **Why P1 "passes" is the core weakness.** Almost every constituent start is also a word or token start, and a byte-level LM's entropy spikes at word starts. So beating a random shuffle is close to guaranteed and does not separate *syntax* from *lexical segmentation*. The project never runs the test that would separate them.
4. **Two results in the current story are fragile, and the checks in this review show it:**
   - **P3 flips with the metric.** On the scale-free ratio over the null, code beats prose: 4.6–4.7× against 3.4×.
   - **The O5 "memory-unsafe residual" only appears after controlling for a variable computed from the same entropy.** Without that control, unsafe regions get *fewer* boundaries.
5. **Most of the post-mid plan was not implemented.** The planned items were the P4 post-keyword test, depth normalization, the O4 noise-rate sweep and the O5 label audit. The work went into a teacher→student distillation instead. That experiment works technically but does not test the research question.
6. **What makes this defensible:** reframe the question from "does BLT find syntax?" to "how much of BLT's segmentation is syntax *beyond* lexical segmentation?". Then run the one decisive experiment, a lexically controlled analysis (§6, item 1). A clean answer either way is a solid result.

---

## 1. Context: what ESCAPE is

### 1.1 The idea in plain words

The **Byte Latent Transformer (BLT)** reads raw bytes, not tokens. A small byte-level language model, the **patcher** (99.5M parameters here), predicts each next byte. At every position it computes the **entropy** of its prediction:

```
H(b_i) = − Σ_v p(v | b_<i) · log p(v | b_<i)        (nats)
```

When `H > τ` (τ = 1.3354 nats, the checkpoint's own threshold), BLT starts a new **patch**. Each patch gets one step of the expensive large transformer. So boundaries decide *where compute is spent*.

**ESCAPE asks:** do these entropy-triggered boundaries land at the **starts of syntactic constituents**? That means AST nodes in code (tree-sitter) and phrases in prose (benepar parses). And do they do so more than chance, more than trivial line or word splitting, and more in code than in prose?

### 1.2 Hypothesis and predictions (from the proposal)

**H1, the Grammatical Branching Hypothesis.** Boundaries concentrate at constituent **starts** rather than interiors. The effect should be stronger in code than in prose, and stronger at starts than at ends.

| ID | Prediction | Why the proposal expected it |
|---|---|---|
| P1 | Boundaries match constituent starts above a density-matched null | Entropy is high where the grammar opens a choice |
| P2 | Starts align better than ends | Closing `;` `)` `}` are forced, so low entropy |
| P3 | Code ≫ prose (prose near chance) | Code grammar has sharper choice points |
| P4 | Weaker for forced openers (`if`, `for`, `def`) | A forced next token means low entropy |
| "P/R" | Precision > recall | `H > τ` certifies ≥ e^τ ≈ 3.8 live continuations, so boundaries are a *subset* of branching points |
| R2 | Must beat structure-blind baselines (newline, indentation, word start) | Otherwise it is line or word segmentation, not grammar |

**Objectives:** O1 alignment, O2 bytes-per-patch (BPP) code vs prose, O3 Python vs C++, O4 robustness to code noise, O5 memory-unsafe C++ compute vs the identifier-entropy confound.

### 1.3 Pipeline

```
corpus bytes (16,300 files: 8,000 Python / 8,000 C++ / 300 English paragraphs)
   │
   ├─► Stream A (Shashwat): BLT patcher itazap/blt-1b-hf @ 91aa6b8e, swa512 window restored,
   │                        float32, τ = 1.3354 ─► per-byte entropy + patch boundaries
   │
   ├─► Stream B (Priyanka): tree-sitter ASTs (code), benepar parses (prose),
   │                        newline / indent / word baselines, O4 noised files, O5 regions
   │
   └─► Stream C (Divyansh): escape_scoring C-v1 — byte contract, one-to-one matching,
                            20/80 calibration/test split, k frozen on calibration (k = 0),
                            fixed-EOF patch-shuffle null (10,000 perms), file bootstrap (2,000)
   +   Extension (Divyansh, post-mid): 69,975-file disjoint corpus + 14.3M distilled student
```

### 1.4 Key concepts you must be able to explain

| Concept | One-line meaning | Why it matters here |
|---|---|---|
| Density-matched null | Shuffle the file's own patch lengths. Same number of boundaries, random positions | Removes "more boundaries means more hits". Says nothing about *where* lexically sensible positions are |
| Margin vs ratio | Margin = F1 − F1_null (a difference). Ratio = F1 / F1_null (a lift) | A margin grows with target density, so it is not comparable across languages. A ratio is scale-free |
| Structure-blind baseline | Splits at newlines, indentation changes or word starts, with no grammar | The real burden of proof: grammar must beat *these* |
| Confound | A third variable that explains both | Word starts explain both "is a constituent start" and "has high entropy" |
| Bad control | Controlling for a variable caused by, or built from, the outcome | Breaks regression estimates (see O5, §5.7) |
| k (tolerance) | How many bytes off a boundary may be and still count | Frozen at k = 0 on calibration. Exact hits only |

---

## 2. What is implemented (done)

| Component | Status | Evidence |
|---|---|---|
| Corpus: CodeSearchNet Python, The Stack C++, WikiText-103 English; hashes and splits frozen | Done | `corpus/manifest.parquet`, `stream_b/build_*corpus.py` |
| Stream A: patcher loading, 512-token sliding window restored (missing from the HF port), fidelity gates, resumable extraction | Done | `stream_a/`, `docs/STREAM_A_FIDELITY.md`, `results/final/stream_a_fidelity_ada/` |
| Stream B: tree-sitter + benepar targets with validated byte offsets, baselines, P4 taxonomy, O4 noise, O5 regions | Done | `stream_b/`, `taxonomy.json`, `golden_fixtures/` |
| Stream C: frozen C-v1 scorer (calibrate, score, baselines, IoU, O4, O5), tests | Done | `escape_scoring/`, `tests/scoring_v1` (32/32 pass per audit) |
| Held-out results, P1–P4, baselines, IoU, O4 | Done | `results/final/c-v1-2721954/` + `SHA256SUMS` |
| O5 regression (computed, not interpreted) | Computed | `results/final/c-v1-2721954/o5*/` |
| Mid-submission report (ACL format) | Done | `reports/Trimax-Mid-Report.pdf` |
| Audit 1 (pipeline) and audit 2 (distillation + corpus) | Done | `AUDIT_REPORT.md`, `AUDIT_REPORT_2.md` |
| Extension corpus: 69,975 files, group-level train/val, 0 overlap with the eval set | Done | `corpus_ext/manifest_ext.parquet`, `stream_b/build_ext_corpus.py` |
| Distillation: 14.3M student mimics the 99.5M patcher (online KD, ≤ 512-byte windows) | Done; reproduced (job 39379) | `escape_distill/`, `results/distill/full/`, `results/distill/repro_39379/` |
| Student under C-v1 on 18 golden fixtures | Done (n = 6 per domain) | `results/student_cv1_golden_v1/`, `reports/ESCAPE_STUDENT_CV1_EVALUATION.md` |

### 2.1 Headline results (held-out test: 6,400 Python / 6,400 C++ / 240 English)

| | Python | C++ | English |
|---|---|---|---|
| P1 F1, BLT vs null | 0.152 vs 0.032 | 0.165 vs 0.036 | 0.495 vs 0.147 |
| F1 margin [95% CI] | +0.120 [0.117, 0.122] | +0.129 [0.125, 0.134] | +0.348 [0.341, 0.354] |
| Precision / recall | 0.083 / 0.822 | 0.095 / 0.646 | 0.342 / 0.894 |
| Best structure-blind baseline | indent 0.394 | word 0.339 | word 0.769 |
| BLT − best baseline | **−0.242** | **−0.173** | **−0.274** |
| P2 end margin | −0.017 | −0.035 | −0.079 |
| P3 mean BPP (per-file mean) | 5.68 | 7.86 | 3.77 |
| P4 recall margin, forced vs open | +0.759 vs +0.611 | +0.691 vs +0.451 | — |
| O4 margin, clean → worst noise | +0.115 → +0.099 | +0.131 → +0.118 | — |

### 2.2 Scorecard against the proposal

| Prediction | Verdict |
|---|---|
| P1 beats null | Supported (but see §5.2: weak test) |
| R2 beats structure-blind baselines | **Not supported**, loses in all three languages |
| P2 starts > ends | Supported |
| P3 code > prose | Not supported on the margin; **flips on the ratio** (§5.4) |
| P4 forced openers weaker | **Reversed** (§5.5) |
| Precision > recall | **Not supported**: P ≈ 0.08–0.34, R ≈ 0.65–0.89 |
| O4 robust to noise | Supported (worst drop −0.016) |
| O5 residual shrinks to zero | Computed; the positive residual is an artifact of the control (§5.7) |

### 2.3 Distillation results (what they do and do not show)

| Metric (student vs teacher, held-out windows) | Python | C++ | Prose | All |
|---|---|---|---|---|
| Boundary F1 vs *teacher* boundaries | 0.869 | 0.880 | 0.834 | **0.871** (untrained 0.386) |
| Entropy Pearson r | 0.949 | 0.954 | 0.899 | 0.946 |
| Bits/byte, student vs teacher | 1.14 vs 1.05 | 1.19 vs 1.14 | 1.83 vs 1.37 | 1.24 vs 1.14 |

On the 18 golden files under C-v1, the student keeps a positive margin over the null: +0.101 / +0.172 / +0.274, against the teacher's +0.112 / +0.198 / +0.326. It **still loses to the baselines** by 0.33–0.35 F1. The pattern is the same as the teacher's, at about 70–90% of the margin.

---

## 3. What is still in progress or not done

**Planned in the mid-report (timeline table) against the repo state.** None of the planned analyses appears in a commit after the mid-submission. The commits since are the extension corpus, distillation, audits and a reproduction run.

| Planned (dates from the mid-report) | State on 27 Oct |
|---|---|
| 3–10 Oct: O5 label audit + interpretation | **Not done.** Numbers computed only. §5.7 shows the current model is mis-specified |
| 10–17 Oct: P4 test at the byte *after* the opening keyword | **Not done** |
| 10–17 Oct: cross-language depth normalization (O3) | **Not done** (`docs/STREAM_C_METHODS.md`: "post-mid") |
| 17–24 Oct: O4 noise-rate sweep (one rate per type today) | **Not done** |
| 17–24 Oct: freeze protocol amendments from reviewer feedback | **Not done** |
| 24–31 Oct: final ablations, 8-page write-up, `Trimax-Final.zip`, viva | Pending |

**Started post-mid but unfinished:**
- Student C-v1 scoring on the **full test split**: golden files only so far (n = 6 per domain, CIs overlap everything). Needs Ada.
- **Efficiency** of the student (latency, FLOPs, memory against boundary quality): not measured. Without it, "7× smaller" has no payoff.
- **Pristine English holdout**: a selection rule was proposed (120 unused articles) but never materialised or scored.
- Robustness rows recommended by audit 1: results excluding chunk-reset bytes, excluding `parse_ok = False` C++ files, and a repository-grouped bootstrap for Python.
- Housekeeping:
  - `corpus_ext/DATA_NOTE.json` describes the 29k subset, not the 69,975-file manifest.
  - `the-stack-smol-xl` is not revision-pinned.
  - `bpp_summary.parquet` is checksummed but not retained.
  - `docs/FINAL_RUN.md` has a stale line.

**Practical blocker:** per-byte entropy arrays and corpus bytes are not stored locally. They are regenerated on Ada. Most new analyses below need either Ada (VPN) or a local restore of a subset. The teacher checkpoint *is* local, and a subset fits on the laptop GPU.

---

## 4. Why the original hypothesis was likely to fail (the intuition)

This is the most useful thing to understand before facing the professor.

A byte-level LM's next-byte uncertainty is highest at **the first byte of a new word or token**: which word comes next? It is also high **inside identifiers seen for the first time**: how is this name spelled? Once the first byte or two of a word are seen, the rest is cheap. So entropy boundaries are, to a first approximation, **"word starts + novel-identifier bytes"**. That is *lexical* structure.

Constituent starts are almost always also token starts. So:
- **Recall is high for free.** Every constituent start is a word start, and BLT fires at word starts. Python recall is 0.82, English 0.89.
- **Precision is low by construction.** Most word starts are not constituent starts, and identifiers add many more boundaries. BLT proposes **9.8 boundaries per target in Python, 6.8 in C++ and 2.6 in English** (computed from `per_file.csv`).
- **Indentation and word baselines win.** They capture the same lexical signal without the extra identifier noise. 31.8% of Python and 22.5% of C++ constituent starts sit exactly at an indentation-change byte.

The proposal's own scope section partly anticipated this ("identifier entropy could swamp the syntactic signal"). The data say that is exactly what happened.

---

## 5. Flaws: why the hypothesis is not yet defensible

Ordered by how much damage each does in a viva.

### 5.1 H1 is refuted by its own sharpest tests
The proposal says H1 is supported only if P1 clears **both** baselines with non-overlapping CIs, and P2, P3 and P4 all hold. Today:
- R2 fails in all three languages.
- P4 is reversed.
- Precision < recall.
- P3 fails on the reported metric.

**What to say:** "H1 as stated is not supported. What survives is a weaker claim: entropy is a genuine but shallow structural signal." Do not present H1 as "partially supported".

### 5.2 P1's "support" is nearly guaranteed: the null is too weak
The null shuffles patch lengths. Random positions land mid-word, which no reasonable segmenter does. Any word-aware splitter beats it, and the word-start baseline beats BLT itself. So "+0.12 F1 over the null" shows boundaries are *not random*, not that they are *syntactic*.

**A professor will ask:** "Would a dumb model that splits at word starts also beat your null?" The answer is yes, by more than BLT.

### 5.3 The central confound is never tested directly
No analysis in the repo asks the conditional question: **among positions that are all token starts, are constituent starts more likely to get a boundary than non-constituent token starts?** A search of the code finds no conditional or lexically matched analysis. This is the one experiment that can show "syntax beyond lexis". Without it, the structural claim cannot be made either way.

### 5.4 Cross-domain comparisons use a metric that depends on target density (P3)
English benepar targets are dense: **0.102 targets per byte**, nearly every word starts a phrase. Code has 0.017 (Python) and 0.020 (C++). A random boundary hits a target far more often in English, so the absolute margin is inflated in English.

The retained `scores.csv` already holds the scale-free ratio over the null:

| | F1 ratio over null [95% CI] | F1 margin |
|---|---|---|
| Python | **4.73** [4.69, 4.78] | +0.120 |
| C++ | **4.61** [4.39, 4.76] | +0.129 |
| English | 3.36 [3.31, 3.42] | +0.348 |

On the ratio, **code beats prose with non-overlapping CIs**, the direction P3 predicted. The verdict flips with the metric. So P3 is **undetermined** today, not "refuted".

**Caution:** switching metrics after seeing results is HARKing. Present both, explain why margins are not comparable across target densities, and say the protocol should have pre-specified a scale-free metric. Also, benepar phrase trees and tree-sitter ASTs differ in granularity, so "code vs prose" compares different *kinds* of targets.

### 5.5 P4 is confounded with line position
The forced-opener types are **statement-level, line-initial** (`if`, `for`, `while`). Their first byte follows indentation, where "which statement comes next?" is genuinely uncertain. Several "open-ended" types are **mid-line expressions** (`call`, `binary_operator`, `binary_expression`).

Recall from `scores.csv`:
- Python: `while` 0.954, `for` 0.934, `if` 0.903, against `binary_operator` 0.683 and `call` 0.766.
- C++: `while` 0.942, `for` 0.823, against `binary_expression` 0.506.

The forcing the proposal predicted applies to the byte *after* the keyword. Start offsets never measure that byte. The planned post-keyword test was not run.

### 5.6 One threshold, binary boundaries, and baselines not density-matched
- Everything uses the single checkpoint τ. There is no PR curve, no τ sweep and no threshold-free measure (for example AUROC of entropy at constituent starts).
- The baselines propose different numbers of boundaries than BLT. "BLT loses to word start" mixes *how many* boundaries with *where* they are. A fair comparison matches density: top-N entropy bytes against N word starts, or word starts subsampled to BLT's density.

### 5.7 O5: the "significant memory-unsafe residual" is an artifact of the control
**Raw means** (12,609 regions, 1,500 files):
- Unsafe regions get *fewer* boundaries: density 0.155 vs 0.166 for comparison regions.
- Unsafe regions also have lower identifier entropy: 0.493 vs 0.554 nats.

**Re-fit of the regression for this review** (file fixed effects, file-clustered SEs, `o5_regions.csv`). Model M3 reproduces the reported coefficient exactly:

| Model | Unsafe coefficient on boundary density [95% CI] | On inverse BPP |
|---|---|---|
| M0: unsafe + file FE | **−0.0083** [−0.0127, −0.0039] | −0.0024 [−0.0066, +0.0017] |
| M1: + region length + statement type | −0.0068 [−0.0112, −0.0023] | −0.0014 [−0.0056, +0.0027] |
| M2: + identifier byte share | −0.0068 [−0.0112, −0.0024] | −0.0015 [−0.0056, +0.0027] |
| M3: + mean identifier **entropy** (reported) | **+0.0069** [+0.0043, +0.0096] | +0.0112 [+0.0088, +0.0137] |

The positive "residual" appears **only** once you condition on mean identifier entropy. That variable is computed from the *same per-byte entropy* that defines the boundaries, so the outcome is partly regressed on itself. This is a bad control. The sign flip shows it, not a semantic effect.

The labels are also purely syntactic (`*`, `&`, `new`/`delete`, `malloc`/`memcpy`, casts). No types, aliasing or bounds are known.

**Honest reading:** without the entropy-derived control, memory-unsafe statements get the same or fewer boundaries. That is consistent with the proposal's prediction that BLT does not allocate compute by risk.

### 5.8 One model, so "BLT" cannot be separated from "any byte-level surprisal"
Only one checkpoint is tested, a third-party HF port whose window the team had to restore. There is no comparison entropy model, such as a byte n-gram model. If a byte 5-gram model's surprisal gives the same alignment, the finding is about byte-level statistics in general, not BLT's learned patcher. The distilled student is effectively a second model, but it was trained to copy the teacher, so it is not independent.

### 5.9 Data and statistics validity
- **English** is not a clean holdout: the paragraphs were explored before the protocol froze. It is also small (240 test paragraphs from 18 articles) and uses predicted parses. English numbers are exploratory.
- **Python** functions share repositories, so the file bootstrap understates variance. Use a repository-grouped bootstrap.
- **C++:**
  - 28.1% of files trigger parser error recovery.
  - 31.5% of C++ test files exceed the 8,192-token chunk, so their context resets. There is no sensitivity table for either.
- O4 uses one noise rate per type, 1,000 permutations, and 100 for IoU.

### 5.10 The distillation experiment does not answer the research question
- It is scored against **teacher boundaries**, not syntax. On syntax it has only 18 golden files.
- There are no efficiency numbers, so the "smaller and cheaper" payoff is unproven.
- The student is a worse LM (1.24 vs 1.14 bits/byte), and its context regime (≤ 512-byte windows) differs from the teacher's full-file regime.

**A professor will ask:** "Why did you do this, and what does it tell us about the hypothesis?" You need a sentence that ties it in (see §7.2).

### 5.11 The title over-claims
"Entropy and **Semantic Compute Allocation**" promises semantics and compute allocation. The project measures *segmentation*, never measures compute (FLOPs or latency), and explicitly disclaims semantics. The professor will notice that the title promises what the data refute.

---

## 6. What to implement next, by priority

Effort estimates are rough. "Ada" means it needs per-byte arrays regenerated on the cluster, or a local subset restore.

### Must do before the final (highest value per hour)

**1. The decisive experiment: a lexically controlled alignment test.** Ada, about 1 day.
- **Candidates:** every token-start byte (word starts plus punctuation and operator token starts). Restricting to candidates removes the lexical advantage by design.
- **Conditional rates:** P(boundary | constituent start) against P(boundary | other token start), per language, with a file bootstrap.
- **Byte-level logistic regression:** `boundary ~ is_constituent_start + line_initial + after_indent + first_occurrence_identifier + token_class (keyword/identifier/punct/literal) + token_length + file FE`. Report the odds ratio for `is_constituent_start` with clustered CIs.
- **Lexically matched null:** shuffle BLT's boundaries **only among token-start positions**, keeping the count, and recompute the P1 margin.

| Result | Meaning |
|---|---|
| Effect > 0 after controls | A defensible "syntax beyond lexical segmentation" claim |
| Effect ≈ 0 | A clean, publishable-style negative: BLT's segmentation is lexical |

**2. Threshold-free and density-matched comparisons.** Same data as item 1, a few hours.
- AUROC of per-byte entropy: constituent-start bytes against other token-start bytes.
- PR curves over a τ sweep for BLT, plotted against the baselines' single points.
- A matched-density comparison: top-N entropy bytes against N word or indent starts.

**3. Fix the P3 analysis.** Local, about 1 hour: the numbers already exist in `scores.csv`.
- Report the ratio over the null next to the margin, with an explanation of target density (§5.4).
- Frame it as "P3 depends on the metric". Do not claim "P3 supported".

**4. Redo O5 properly.** Local for the re-fit (done in §5.7); Ada for new covariates.
- Report M0–M3 side by side.
- Replace the entropy-derived control with non-entropy covariates: identifier count, identifier length, first-occurrence flag, corpus token frequency.
- Hand-audit about 100 regions for label correctness.
- Conclude "no evidence of extra compute for memory-unsafe code".

**5. Explain the P4 reversal.** Ada, half a day.
- (a) Recall at the byte immediately after `keyword + space`, where forcing was predicted.
- (b) Stratify node types by line-initial against mid-line starts.

If forced openers stop looking special once line position is controlled, the reversal is explained.

### Should do if time allows (strengthens the story)

6. **Byte n-gram entropy baseline.** For example a Kneser–Ney 5-gram trained on `corpus_ext`, scored with the same protocol. It tests whether BLT is special or whether any surprisal works. CPU only, about half a day.
7. **Student on the full test split + efficiency.** One Ada job with stride-1 extraction. Record latency, memory and FLOPs per byte against margin, which turns distillation into a result (§7.2).
8. **Robustness rows:**
   - excluding chunk-reset bytes
   - excluding `parse_ok = False` C++ files
   - a repository-grouped bootstrap for Python
   - the pre-registered English holdout (120 articles)

   Mostly scoring reruns, about half a day.
9. **Housekeeping:** rename or reframe the title; fix `DATA_NOTE.json`; pin `the-stack-smol-xl`; restore `bpp_summary.parquet` or drop it from `SHA256SUMS`; write "mean per-file BPP" in the P3 caption.

### Future work (say it, don't build it now)
- Train a code-specialised entropy model and test whether alignment beyond lexis increases.
- Measure real compute allocation end-to-end with BLT's latent transformer: FLOPs per region.
- Syntax-aware patching as a separate hypothesis.

### A realistic four-day plan (28–31 Oct)

| Day | Work |
|---|---|
| 28 Oct | Get per-byte arrays: rerun extraction on Ada, or extract a 1–2k-file subset locally on the RTX 3050. Write the item 1 and 2 scripts. Do item 3 now |
| 29 Oct | Run items 1, 2 and 5. Finish item 4 (O5 write-up). Start item 6 if the arrays are ready |
| 30 Oct | Item 7 Ada job and item 8 robustness reruns. Freeze all numbers |
| 31 Oct | Write-up (8 pages), slides, `Trimax-Final.zip` |

If Ada is unreachable, do items 1, 2 and 5 on a locally restored subset and say so. A smaller sample that tests the right question beats a large sample that tests the wrong one.

---

## 7. How to present this to the professor

### 7.1 Claims you can and cannot make today

| You **can** say | You **cannot** say |
|---|---|
| BLT boundaries are far from random: 4.6–4.7× the null's F1 in code, CIs tight | BLT's patcher "discovers syntax" |
| Starts align better than ends (P2) | Boundaries track grammar beyond word or line segmentation |
| Trivial indentation and word splitting beat BLT, so most alignment is lexical | Prose is more structural than code (metric-dependent) |
| Patching degrades gracefully under code noise (O4) | BLT allocates compute by semantic risk (O5); the residual is a control artifact |
| A 7× smaller student reproduces the teacher's boundaries (F1 0.87, r 0.95) | The student is a drop-in or more efficient replacement (not measured) |

### 7.2 Reframed research question and one-paragraph pitch

> **What does an entropy-based byte patcher actually segment in code, and how much of it is syntax beyond lexical segmentation?**
>
> "We tested whether BLT's entropy boundaries track syntactic constituents in Python, C++ and English under a frozen, calibration-first protocol. Boundaries beat a density-matched null by 4.6–4.7× in code, but structure-blind indentation and word splitting beat BLT in every language. Precision is far below recall, and the predicted forced-opener deficit is reversed. A lexically controlled analysis [item 1] isolates the share of alignment that is syntactic beyond token starts: [result]. A distilled 14M-parameter student reproduces the same pattern, which suggests the segmentation is a low-complexity surface function. Entropy is a genuine but shallow structural signal."

The distillation fits this story as **evidence that segmentation is cheap to learn and does not need syntax**. Frame it that way, not as a separate project.

### 7.3 Questions to expect, with answers

| Question | Answer |
|---|---|
| "Your null is random; isn't beating it trivial?" | Yes, which is why we added baselines (BLT loses) and the lexically controlled test (§6, item 1) |
| "Why does prose look better than code?" | Target density: 0.10 vs 0.02 targets per byte. On the scale-free ratio code wins (4.7 vs 3.4). We report both |
| "Why is P4 reversed?" | Forced openers are line-initial statements. Forcing applies after the keyword, which start offsets don't measure. Item 5 tests this |
| "Is the O5 effect real?" | No. It appears only when controlling for an entropy-derived variable; raw and non-entropy-controlled effects are ≤ 0 |
| "Is your checkpoint faithful?" | Third-party port; we restored the 512 window and matched an independent re-implementation to 1e-4 logits. Not bitwise against the gated original |
| "What's the contribution if H1 is false?" | A rigorous negative result with a protocol (null + baselines + lexical control) that separates syntax from lexis, plus evidence that BLT's segmentation is lexical |

---

## Appendix: where the numbers come from

- Main scores: `results/final/c-v1-2721954/evaluation/scores.csv` (`axis = language/domain/node_type`) and `scores.json`.
- Baselines: `results/final/c-v1-2721954/baselines.json`.
- Density per byte (§4, §5.4): `evaluation/per_file.csv` (tp + fn = targets, tp + fp = boundaries) joined with `evaluation/diagnostics.csv` (`byte_length`).
- O5 re-fit (§5.7): `o5/o5_regions.csv`. OLS after within-file demeaning, file-clustered sandwich SEs, 95% CI = ±1.96 SE. M3 reproduces `o5/o5.json` (`is_memory_unsafe` = +0.00693; inverse BPP +0.01123).
- Distillation: `results/distill/full/eval/distill_eval.csv`; reproduction `results/distill/repro_39379/`.
- Student golden C-v1: `results/student_cv1_golden_v1/scores.json`.
