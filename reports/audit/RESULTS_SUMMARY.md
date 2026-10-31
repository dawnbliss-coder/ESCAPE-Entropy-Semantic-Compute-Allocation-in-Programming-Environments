# Results summary: what the project found (31 Oct 2026)

A plain-language reading of the final results. Every number here was checked against the files under `results/final/` during the audit (see `AUDIT_FINDINGS.md`). Full tables and job ids are in `docs/FULL_RUN_RESULTS.md`.

## The question

The Byte Latent Transformer (BLT) reads raw bytes. It starts a new *patch*, a unit of expensive computation, wherever a small model is uncertain about the next byte, that is, where entropy exceeds a threshold τ. **Do these boundaries land where syntactic constituents begin** (statements, expressions, phrases)? Three languages were tested: Python, C++ and English.

## The answer in one paragraph

**Partly, and mostly for a shallow reason.**
- BLT's boundaries hit constituent starts **3–5× more often than chance** in every language, and the result holds on unseen data.
- But **simple rules do better.** Splitting at word starts or indentation changes matches the targets better than BLT, and word-start splitting beats BLT on both precision and recall while placing fewer boundaries.
- So the entropy model has learned mostly *where words and lines begin*, not grammar.
- Raising the threshold helps (best at τ ≈ 2.0) but never closes the gap.
- A 7× smaller distilled model keeps about 90% of the effect.
- None of the original hypothesis's four predictions holds cleanly except one (P2: boundaries align better with starts than with ends).

## Headline numbers (held-out test split: 6,400 Python, 6,400 C++, 240 English files)

| | Python | C++ | English |
|---|---|---|---|
| Start F1, BLT vs. random boundaries at the same density | 0.152 vs. 0.032 | 0.165 vs. 0.036 | 0.495 vs. 0.147 |
| Margin over chance [95% CI] | +0.120 [0.117, 0.122] | +0.129 [0.125, 0.134] | +0.348 [0.341, 0.354] |
| Precision / recall | 0.083 / 0.822 | 0.095 / 0.646 | 0.342 / 0.894 |
| Best simple rule (F1) | indentation (0.394) | word start (0.339) | word start (0.769) |
| BLT minus best simple rule | −0.242 | −0.173 | −0.274 |
| Mean bytes per patch | 5.68 | 7.86 | 3.77 |

How to read it: BLT clearly beats random placement, since the intervals are far from zero. It clearly loses to rules that know nothing about grammar.

## Prediction by prediction

| Prediction (from the proposal) | Result | Verdict |
|---|---|---|
| **P1**: boundaries align with constituent starts | Beats chance everywhere (+0.120 / +0.129 / +0.348); loses to the simple rules everywhere | Half met: yes vs. chance, no vs. baselines |
| **P2**: starts align better than ends | End margins are −0.017 / −0.035 / −0.079 | **Met** |
| **P3**: alignment stronger in code than prose | Reversed: English has the largest margin. The report's "scale-free ratio" rescue doesn't hold (see audit C3) | Not met |
| **P4**: weaker for forced openers (`if`, `for`, `def`) | Reversed at the start byte (forced +0.759 vs open +0.611 in Python). At the bytes *after* the keyword, the boundary rate drops to 0.000 (Python) / 0.035 (C++) against about 0.15 by chance, so the forcing mechanism is real, just not where P4 measured it | Not met as stated; mechanism confirmed |
| **Precision > recall** | Opposite: precision 0.08–0.34, recall 0.65–0.89. BLT proposes 2.6–9.8 boundaries per target | Not met |
| **O3**: Python vs C++ by depth | Python aligns better than C++ at every shared tree depth | Descriptive, done |
| **O4**: robust to code noise | Margins stay ≥ +0.086 under every noise type. Indentation noise hurts most (−0.029 at 40% in Python); typos and missing semicolons hurt a little and also grow with rate | **Met** (degrades gracefully) |
| **O5**: more compute near memory-unsafe C++ beyond identifier effects | With file effects only, unsafe statements get *fewer* boundaries (−0.0083). A small positive effect (+0.0069) appears only after controlling for identifier entropy, about 4% of the mean | No semantic effect found |

By the proposal's own rule (H1 needs P1–P4 all to hold), **H1 is not supported**. The report says "partly supported", which the audit flags as too soft (M2).

## The threshold τ

The checkpoint's default τ = 1.335 was tuned for compression, not alignment. A sweep over {0.8, 1.0, 1.335, 1.7, 2.0, 2.5} was run on calibration files, with the best τ fixed in advance by a written rule.

| | Python | C++ | English |
|---|---|---|---|
| Best τ, teacher | 2.0 | 2.0 | 2.0 |
| Best τ, student | 2.0 | 2.0 | 2.5 |
| Teacher margin at τ = 1.335 → at best τ (test) | +0.120 → +0.153 | +0.129 → +0.158 | +0.348 → +0.426 |
| Teacher at best τ minus best simple rule | −0.210 | −0.148 | −0.215 |

- **Exploratory check (30 Oct).** The whole grid was also scored on the test files, after the confirmatory runs, to see whether the calibration choice holds. It does: the best τ on test equals the calibration choice in **6 of 6** model/language pairs. At **no** τ does either model get within **0.148 F1** of the best simple rule.
- **Caveat.** At τ = 1.335, BLT places so many boundaries that its F1 is mathematically capped below the baselines (0.18 / 0.26 / 0.55 at best). The precision-and-recall comparison is the fairer argument that the simple rules win (audit M1).

## Beyond the proposal: a distilled patcher

A 14.3M-parameter byte model was trained to imitate the 99.5M-parameter BLT entropy model, then scored with the same frozen pipeline.

| | Python | C++ | English |
|---|---|---|---|
| Student F1 / margin over chance | 0.140 / +0.109 | 0.155 / +0.119 | 0.459 / +0.304 |
| Share of the teacher's margin kept | 91% | 92% | 87% |
| Student minus teacher [95% CI] | −0.011 [−0.011, −0.010] | −0.011 [−0.012, −0.009] | −0.044 [−0.048, −0.040] |

- **Speed:** the student's forward pass is 5.0× faster and its rolling extraction 4.8× faster on an RTX 2080 Ti.
- **Agreement:** the student agrees with the teacher's boundaries at F1 0.871. A student trained without the teacher reaches 0.789, and word starts 0.475.
- **Context window:** restricting the teacher to the student's 512-byte window costs 0.004 F1. That is about 40% of the student's deficit, so the shorter context explains part of the gap and capacity the rest.
- **Follow-ups (exploratory, on a 600-file subset plus English test):**
  - A byte-balanced student trades a little code accuracy for prose (−0.002 / −0.002 / +0.010).
  - Fine-tuning the teacher on the same data recovers its own performance but doesn't beat it. The pre-registered criterion was not met (1 of 3 languages).

## English holdout (the confirmatory English result)

The 300 English evaluation paragraphs had been looked at during development, so a fresh holdout of **4,107 paragraphs** from 120 unused articles was scored once.

| | Teacher | Student |
|---|---|---|
| Holdout margin | +0.350 | +0.305 |
| Eval-set margin | +0.348 | +0.304 |

The English result replicates almost exactly.

## A check on robustness: did fixing the sliding window matter?

The public HuggingFace port of the checkpoint drops BLT's 512-token attention window, and the team restored it. Scored with the port's native full attention instead, the test margin collapses from +0.120 to +0.023 (Python) and from +0.131 to +0.008 (C++, files under 8 KB). Without that fix, the project would have reported almost no alignment.

## How solid is this?

- **Solid**
  - The positive result against chance: huge margins, tight intervals, held-out data, replicated on the English holdout and by the student.
  - The negative result against the baselines: it holds at every τ and on precision and recall separately.
  - P2.
  - The O4 robustness.
- **Solid but reworded in the audit**
  - O4 (other noise types also grow with rate, just much less);
  - O5 (the "operator bytes" explanation is a guess);
  - P3 (the ratio argument is wrong);
  - the H1 verdict (not supported, rather than partly).
- **Thin**
  - English test set: only 240 files, and its intervals resample paragraphs, not articles. The holdout mitigates this.
  - Follow-up arms: scored only on subsets.
  - Multiple comparisons across the many strata: never corrected.

**Bottom line:** the measurements are careful and reproducible, and the conclusions follow from them once the four wording fixes in the audit are made. As a research finding, the answer is clear and honest: BLT's entropy boundaries track the surface of text and code (word and line starts) far more than its grammar.
