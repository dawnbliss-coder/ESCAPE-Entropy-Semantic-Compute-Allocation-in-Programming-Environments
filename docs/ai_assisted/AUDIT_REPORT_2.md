# ESCAPE Audit Report 2 — Teacher–Student Distillation and Dataset Expansion

**Date:** 2026-10-23 · **Auditor:** OpenCode `escape-research` skill · **Scope:** read-only
investigation of the two new contributions on `origin/main` (not present in the local
clone), plus compatibility, research-value and readiness assessment. No scientific code,
protocol, results, datasets or checkpoints were modified. A read-only `git fetch` was
performed; the working tree was never changed.

---

## Phase 1 — Current repository state

- **Local clone:** `main` @ `38c6a80`, working tree clean apart from the untracked
  `.opencode/skills/` and `AUDIT_REPORT.md` (mine). The local clone does **not** contain
  any of the new work.
- **Remote `origin/main`** has advanced by 3 commits beyond the local HEAD:
  - `bf7bbbd` (Oct 2) — retire submitted mid-submission PDFs, track `Trimax-Mid-Report.pdf`.
  - `ea66092` (Oct 19, author **dawnbliss-coder**) — "Add extension corpus builder and
    manifest (disjoint from evaluation set)": `stream_b/build_ext_corpus.py`,
    `stream_b/build_ext_structure.py`, `corpus_ext/manifest_ext.parquet` (8.7 MB),
    `corpus_ext/DATA_NOTE.json`, `.gitignore` changes.
  - `c4384eb` (Oct 22, author **Divyansh-Atri**) — "Added teacher student distillation":
    `escape_distill/` package (6 modules + README + requirements), `results/distill/`
    (full + smoke runs, incl. `student.pt` 57 MB and `train_log.json`), a comparison
    figure, and `scripts/ada_distill/` SLURM pipeline.
- Git authorship matches the roles the user described (Divyansh: distillation commit;
  the corpus commit comes from the repo-owner account that also authored all prior
  Stream-B corpus work). No feature branches or PRs exist; both landed directly on
  `main`. Nothing on the `Shash` branch is relevant (pre-mid-submission history).
- **Prior audit artifacts** (`results/final/c-v1-2721954/`, calibration record, C-v1
  protocol, report pipeline) are all intact in both local and remote trees; the two new
  commits touch none of them.

## Phase 2 — Teacher–student model audit

### A. Architecture and objective
- **Teacher:** the exact frozen BLT patcher used for every report number
  (`itazap/blt-1b-hf` @ `91aa6b8e`, swa512, float32, 99.5M params), via either the
  transformers `BltPatcher` (used on Ada) or the repo's own pure-torch reference
  re-implementation (equal to the HF port to 1e-4, fidelity gate G3). Teacher is frozen
  (no fine-tuning) and run inference-only.
- **Student:** from-scratch byte-level transformer written in `escape_distill/student_blt.py`
  — vocab 260 (BOS=1, byte→id+4), pre-norm blocks, RMSNorm(1e-5), interleaved RoPE
  (θ=10000), SwiGLU FFN (8/3), causal attention with head_dim⁻¹ᐟ² scaling, tied embeddings.
  Full-run config: dim 384, 8 layers, 6 heads → **14,262,144 params** (~14.3M, 7× smaller
  than the teacher).
- **What is distilled:** the teacher's **next-byte soft distribution** (260-way logits) on
  ≤512-byte windows, online (teacher run per minibatch, no dumped tensors). Loss =
  `α·CE(student, true byte) + (1−α)·T²·KL(softmax(teacher/T) ‖ softmax(student/T))` with
  T=2, α=0.1 — a hybrid: genuine online knowledge distillation with a small hard-label
  anchor (not pure KD, not ordinary SFT). No syntax labels anywhere in training.
- **Context honesty:** teacher and student are compared on fresh ≤512-byte windows, where
  swa512 ≡ causal (gate G3). This is valid *within a window* — but see the critical
  limitation in §C.

### B. Completeness and reproducibility
- Entry point: `python -m escape_distill.train`; Ada run recorded in `train_log.json`
  (args snapshot included: 2 epochs, 6,860 steps, batch 48, lr 5e-4 cosine w/ 200-step
  warmup, AdamW(0.9, 0.95), wd 0.1, grad-clip 1.0, seed 20260927, T=2, α=0.1).
- **Completed successfully:** loss converged 12.99 → 0.297 (KL 13.8 → 0.24, hard CE
  5.58 → 0.84 nats), ~10,140 s on one Ada GPU. `student.pt` (57 MB = 14.26M float32
  params + config + args) and all eval artifacts are committed.
- **Reproducibility gaps:**
  1. `stream_b/build_ext_corpus.py` pins the-stack-smol, CodeSearchNet and WikiText
     revisions, **but not `the-stack-smol-xl`** (loaded unpinned) — the C++ extension
     component is not strictly reproducible if that dataset is revised on the Hub.
  2. Raw `corpus_ext/*.bin` are gitignored (regenerable; 156 MB total) — reproduction
     requires re-materializing from the datasets (~GBs, needs HF access).
  3. No validation-based checkpoint selection: only the final step is saved. Fine for
     the current claim, but any future "best" checkpoint claim would need a selection
     rule fixed in advance.
- Train/val separation is by file (group-level split; verified 0 groups span both).

### C. Does the student perform the intended task?
**On held-out 512-byte windows (val split, all 1,847 files):**

| metric (student vs teacher) | py | cpp | prose | all |
|---|---|---|---|---|
| boundary F1 (τ, k=0) | 0.869 | 0.880 | 0.834 | **0.871** |
| untrained-student baseline F1 | 0.364 | 0.382 | 0.463 | 0.386 |
| entropy Pearson r / MAE (nats) | 0.949 / 0.16 | 0.954 / 0.16 | 0.899 / 0.31 | 0.946 / 0.17 |
| KL(teacher‖student), nats/byte | 0.116 | 0.104 | 0.393 | 0.135 |
| bits/byte student vs teacher | 1.14 vs 1.05 | 1.19 vs 1.14 | 1.83 vs 1.37 | 1.24 vs 1.14 |

Interpretation: the distillation **clearly works** — the student reproduces the
teacher's window-level boundaries (F1 0.87 vs 0.39 untrained) and entropy values
(r ≈ 0.95), while its own next-byte predictions remain slightly worse than the teacher's
(1.24 vs 1.14 bits/byte). Prose is the weakest domain (largest KL, worst bits/byte).

**Critical limitations (all confirmed from code and data):**
1. **This is teacher-mimicry, not the ESCAPE question.** The eval scores student
   boundaries against *teacher boundaries*, not against syntactic targets. There is **no
   C-v1 scoring of the student yet**: no P1 alignment, no null, no structure-blind
   baselines, no BPP/IoU/O4. The README says so; nothing here is report-ready.
2. **Windowed context is not the pipeline context.** Training/eval windows are ≤512
   bytes with fresh context at every window start, but the real teacher on a full file
   has a 7154-token receptive field with 8192-token chunks. The student (max_seq_len
   513) can never see what the teacher sees on longer files. The extension corpus is
   mostly multi-window files (63% py, 75% prose, 93% C++ have >512 bytes; C++ median
   4 KB), so most training windows start mid-file with truncated teacher context. Any
   C-v1 test-set comparison therefore needs a **new, explicitly defined student context
   regime** (e.g., 512-byte causal chunks with per-chunk resets), and the window F1 of
   0.87 must not be extrapolated to full-file behavior.
3. Window-start blind spot: boundaries at byte 0 of each window are excluded for both
   models, so boundary agreement is measured only on interior window bytes.
4. The entropy/boundary conventions otherwise match the frozen definitions (nats, same
   formula, same τ via `stream_a.patching.compute_patches`, interior-only starts).

### D. Existing evaluation — what exists and what does not
- Measured: LM predictive quality (bits/byte), entropy agreement (r/MAE), KL, window
  boundary F1 vs teacher, untrained baseline. All recomputed/read from
  `results/distill/full/eval/distill_eval.json` (consistent with the CSV).
- **Not measured:** syntactic alignment (P1), null margins, baseline comparison,
  inference cost/latency/memory, and full-file behavior. Lower training loss must not be
  read as segmentation improvement — it isn't, and the repo doesn't claim it is.

## Phase 3 — Dataset expansion audit

### A. What was added
- `corpus_ext/manifest_ext.parquet`: **69,975 files** (40,000 Python from CodeSearchNet,
  20,000 English paragraphs from WikiText-103, 9,975 C++ from the-stack-smol + smol-xl),
  ≈156 MB of raw bytes (bin files gitignored, regenerable by `build_ext_corpus.py`).
  A 29,000-file subset (15k/8k/6k) was actually used for the distillation run.
- Lengths: py median 666 B (max 59 KB), prose 300–3,000 B (median 699 B, same filter as
  the evaluation prose), C++ median 4.1 KB (max 100 KB; 32% exceed the 8,191-byte chunk
  threshold).
- Splits: train/val assigned at **group level** (code: repository; prose: article) via a
  deterministic hash (≈5% of groups to val). Independently verified: **0 groups span
  both splits**.
- Provenance/licenses recorded per file (revisions pinned for 3 of 4 sources, see §2B).

### B. Disjointness and leakage (independently recomputed from committed manifests)
- **Exact-hash overlap with the C-v1 evaluation corpus: 0 files** (all 16,300 eval
  files checked).
- **Repository/article overlap: 0.** Python: 317 evaluation repos (recovered by parsing
  the eval manifest's GitHub URLs — exactly the builder's recorded count) vs 7,241
  extension repos → 0 shared. Prose: 18 evaluation articles vs 926 extension articles →
  0 shared. C++: enforced at build time by matching eval hashes back to the-stack-smol
  rows (1,185 candidates dropped; recorded in DATA_NOTE); not locally re-verifiable
  without the dataset, but the builder hard-asserts it.
- **No cross-split contamination:** extension splits are group-pure, and the extension
  corpus excludes every repo/article used by the evaluation set (including the
  calibration files — conservative).
- English exposure: the extension deliberately excludes the 18 articles behind the 300
  evaluation paragraphs. The remaining **926 fresh articles are the single most valuable
  new asset** (see Phase 5): a subset could become a genuinely pristine English holdout,
  fixing the prior audit's biggest procedural weakness.
- Caveat: selection was hash-rank based and reproducible; nothing indicates selection on
  test outcomes.

### C. Compatibility with the three-stream evaluation
- Byte encoding, offsets, and the tree-sitter walker are **identical** to Stream B
  (`build_ext_structure.py` reuses `ast_walker.py` and `whitespace_extractor.py`); the
  extension manifest even carries `parse_ok`.
- Appropriate uses: **training and validation only.** The builder itself has no test
  split ("the evaluation files are the test"), which is the right stance. The new data
  must not be turned into a *new* test set for P1 just because it is recent; a new
  holdout would need its own pre-registered selection from untouched sources.
- Drift found: `corpus_ext/DATA_NOTE.json` describes the 29,000-file distill subset
  (limits 15k/8k/6k), **not** the committed 69,975-file manifest it sits next to.
  The self-consistent record of the run lives in `results/distill/full/`. Fix the
  provenance note.

## Phase 4 — Do the two contributions work together?

They are **already integrated**: the student was trained on the extension corpus train
split (27,153 files) and evaluated on its val split (1,847 files). Dependency map:

1. Data provenance (pinned revisions; smol-xl unpinned — gap) ✓
2. Partitioning (group-level train/val; disjoint from evaluation) ✓ verified
3. Teacher inference (frozen checkpoint, on-the-fly) ✓
4. Student training (completed, logged, checkpoint committed) ✓
5. Model selection — *missing* (final step only; no fixed selection rule)
6. Entropy estimation — implemented and convention-correct ✓
7. Boundary generation on the **C-v1 test set** — *missing*
8. Threshold/k handling under the frozen protocol — *untouched, correct stance*
9. C-v1 scoring of the student — *missing*
10. Statistics/report integration — *missing* (correctly not yet claimed)

No calibration or evaluation files overlap the training data (verified). The two
contributions are compatible and can be combined for the next step without further
provenance work.

## Phase 5 — What the new work changes

- **Nothing in the prior audit changes.** The C-v1 results, the soundness verdicts on the
  protocol, the baseline refutation, the exploratory status of English P1, the O5
  residual, and the chunk-reset caveats all stand. The new commits do not alter any
  frozen artifact.
- **What is now possible that wasn't before:** (a) a 14.3M-parameter student that mimics
  the teacher's window-level next-byte distribution (boundary F1 0.87 vs 0.39
  untrained); (b) a 69,975-file disjoint corpus with group-pure splits; (c) 926 fresh
  English articles — a route to a pristine English holdout; (d) an Ada pipeline for
  materialize→distill→evaluate.
- **What remains unproven:** that the student's boundaries align with syntax, beat the
  null or the baselines, that distillation beats direct domain adaptation, that the
  entropy mechanism (rather than surface statistics) explains anything, and that any of
  this improves adaptive compute allocation. These are separate claims needing separate
  evidence. The student's raw prediction is *worse* than the teacher's (1.24 vs 1.14
  bits/byte), so no improvement can be assumed from prediction quality.

## Phase 6/7 — Options, ranked

| # | Option | Scientific value | Cost | Distinguishing power | Info if negative |
|---|---|---|---|---|---|
| **1** | **A: score the existing student under C-v1 on the frozen test split** (define the student's context regime, extract boundaries, run `escape_scoring score` with the frozen k; no new training) | **high** | **low** (one GPU-hour class; no training) | **high** — substitutes for the teacher or not, against the null and the same baselines | shows the context mismatch is fatal; cheap |
| **2** | **G-reserve: pre-register a pristine English holdout** from the 926 untouched articles | high | trivial | fixes English confirmatory status | n/a |
| **3** | C: domain-adapt the teacher vs distill (Option B) — only after A shows the student is in the game | medium | medium (teacher fine-tune ≈ 100M params, hours on Ada) | distinguishes distillation from direct adaptation | tells us KD itself is unnecessary |
| **4** | E: language-specific students (prose is the weak domain: KL 0.39 vs ~0.11) | medium | medium | tests shared-vs-split modeling | per-language budgets are enough |
| **5** | H: efficiency — student vs teacher FLOPs/latency/memory at equal boundary quality | medium | low (falls out of A) | needed only if A succeeds | n/a |
| **6** | D: entropy calibration of the student (reliability curves) | low now | low | calibration ≠ alignment | expected: student is underdispersed |
| **7** | F: structure-aware training | new direction | high | conflicts with the original entropy-only claim; keep separate | n/a |

**Recommendation.**
1. *Best immediate action (no training):* **Option A** — a student C-v1 scoring run on
   the frozen test split (new named experiment; frozen k; same null and baselines), plus
   **Option G-reserve** (write the pristine English holdout selection rule now, before
   anyone looks at those articles).
2. *Best next experiment with the existing student:* A, followed by a small distillation
   variant sweep **only if** A's margin survives the baselines (e.g., pure-KD α=0,
   larger student, or 1k-byte windows with the teacher given the previous window's tail
   — the cheapest fix for the truncated-context mismatch).
3. *Best medium-term direction if A succeeds:* language-specific students (prose) and the
   domain-adaptation comparison (C), then efficiency (H). Structure-aware training (F)
   only as a separately-motivated research question.

Go/no-go for A: proceed if the student can be made deterministic (fixed chunking, no
dropout) and the run stays out of the frozen directories; abandon A in favor of direct
teacher adaptation if the student margin over the null is not within a pre-agreed bound
of the teacher's (e.g., its 95% CI fails to overlap the teacher's margin).

## Phase 8 — Environment readiness

- **Local:** RTX 3050 (6 GB) laptop, 39 GB free disk, no torch installed locally, Ada
  unreachable from this shell (hostname does not resolve — VPN off). The teacher
  checkpoint (199 MB patcher.safetensors) **is present locally** and `student.pt` (57 MB)
  is committed, so a **local read-only evaluation of the saved student against the
  teacher is feasible** with only `torch` (reference backend; teacher fits in ~4 GB per
  the README). No training should be attempted on this GPU.
- **Ada:** the distillation pipeline (`scripts/ada_distill/`) exists and the full run
  already completed there; any re-run needs VPN + the gated the-stack datasets.
- Do not download large models or datasets during this audit; nothing further is
  required to make the audit complete.

## Bottom line

- **What is real and usable:** a genuinely trained 14.3M student (committed checkpoint,
  complete logs) whose window-level mimicry of the frozen BLT patcher is strong; a
  69,975-file disjoint, group-pure, leak-checked extension corpus; an Ada pipeline for
  both. Disjointness from the C-v1 evaluation set was independently verified (0 hash
  overlap, 0 repo/article overlap, pure splits).
- **What is unfinished:** any C-v1-valid evaluation of the student (no P1/null/baselines),
  a defined student context regime for real files, checkpoint selection, smol-xl revision
  pinning, and two provenance/documentation drifts (DATA_NOTE vs manifest; prose
  confirmatory status).
- **Highest research value:** score the existing student under the frozen protocol
  (Option A) and pre-register a pristine English holdout from the 926 untouched articles
  (Option G-reserve). Neither requires new training.
