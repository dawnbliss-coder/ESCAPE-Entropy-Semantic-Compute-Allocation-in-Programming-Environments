# Audit findings (31 Oct 2026)

State audited: `main` at `d9ffc6b` (in sync with GitHub). This includes Priyanka's three commits from the night of 30–31 Oct and the exploratory τ test grid.

## How the audit was done

Three passes, all read-only.

1. **Numbers.** An independent reviewer traced every number in the README, the report PDF, the slides and `docs/FULL_RUN_RESULTS.md` back to the files under `results/`. About 600 values were recomputed from the raw JSON/CSV files, not from the build scripts.
2. **Claims.** An independent reviewer read the proposal, the submitted mid-report and the final report as a skeptical grader would.
3. **Repository health.** I ran the tests, the builds, the checksums, the zip and the git checks myself.

Before going in here, every Critical and Major finding was re-checked against its source file. The **Checked** column says how each finding was verified:

- **re-checked**: confirmed against the source by me;
- **reported**: found by a reviewer, with the source value cited, but not re-checked by me.

Line numbers refer to the current files: `paper` is `reports/final_submission/paper-body.tex`, and `slides` is `reports/final_submission/slides/Trimax-Final-Viva.tex`.

**Bottom line.**
- No stale or wrong macro was found: the generated numbers match their sources.
- The problems are in the words around the numbers:
  - one possible page-limit breach;
  - one missing disclosure;
  - one mathematically wrong framing (P3);
  - a handful of claims stronger than the data.
- All of these are text edits; nothing needs to be re-run.

| Severity | Count |
|---|---|
| Critical (fix before submitting) | 3 |
| Major (a grader would likely mark it) | 8 |
| Minor (polish) | 17 |
| Housekeeping (not part of the submission) | 5 |

---

## Critical

### C1. The report body runs past 8 pages
**Checked:** re-checked.

- **Where:** `Trimax-Final.pdf`, page 9. The top of the left column, before "References", holds the end of Limitations and all of Future Work. That is about 30 lines, roughly a quarter of the page.
- **Evidence:**
  - The course rule written in the repo is "7–8 pages excluding references" (`reports/mid_submission/check_report.py:28`, `docs/FINAL_RUN.md:74`).
  - `check_final.py` only asserts that References start on page ≤ 9, so it reports PASS even though `validation/report-checks.json` records `body_pages_upper_bound: 9`.
  - This came in with the 31 Oct commit that moved the heatmaps into the body. The body length before the τ addition was the same.
- **Risk:** under the ACL convention, Limitations does not count toward the page limit, and in that case this is fine. If the course counts Limitations, the report is over the limit.
- **Fix:** either confirm that Limitations is exempt, or cut about 30 lines. Candidates are the Cost paragraph in §5, or moving one heatmap back to the appendix. Then make `check_final.py` assert `body_pages <= 8`.

### C2. The AI-assistance disclosure was removed and nothing replaced it
**Checked:** re-checked.

- **Where:** commit `d4d8bc2` ("report", 31 Oct) deleted this sentence from `reports/mid_submission/paper-body.tex`: "This report draft and supporting implementation used AI assistance … under the course's disclosure … rules". The same day, `docs/ai_assisted/` was deleted.
- **Current state:** the final report, the slides and the README contain no disclosure of AI assistance, although AI tools were used substantially in the implementation, the analysis and the writing.
- **Risk:** if the course requires disclosure, as the mid-report's own wording says it does, this is an academic-integrity problem. That outweighs anything else in this list.
- **Fix:** add a one- or two-sentence disclosure to the final report's Ethical Considerations section, in the form the course requires. This is separate from GitHub contributor credit, which can stay removed.

### C3. The P3 "scale-free" ratio is not scale-free
**Checked:** re-checked (algebra plus the report's own numbers).

- **Where:**
  - abstract (`paper:5`, "relative to the null, code aligns more strongly");
  - `paper:31`;
  - the Table 10 caption ("does not grow with target density");
  - `slides:123`;
  - `README:41`, `README:53`.
- **Why it is wrong:**
  - Because the null is matched to BLT's boundary density, F1 / null-F1 equals recall / null recall.
  - Null recall is roughly 1 / bytes-per-patch, so the ratio is capped near the mean patch length: about 5.8 for Python, 7.1 for C++ and 3.8 for English.
  - Measured against those caps, English reaches 89%, Python 82% and C++ 65%. Prose still ranks first, so the "code beats prose on the ratio" reading doesn't hold.
- **Fix:** drop "relative to the null, code aligns more strongly" from the abstract and conclusion, and say instead that the ratio is bounded by patch length. Alternatively, report both normalisations: chance-corrected recall (R − R0)/(1 − R0) is 0.784 for Python, 0.588 for C++ and 0.856 for English.

---

## Major

### M1. At its boundary density, BLT cannot beat the baselines on F1, and the report doesn't say so
**Checked:** re-checked.

- **Where:** `paper:26`, `paper:63`, `paper:85` ("Fewer, higher-entropy boundaries are better aligned").
- **The constraint:** with r predictions per target, F1 ≤ 2/(r + 1). At τ = 1.335, r is 9.8 / 6.8 / 2.6, which caps F1 at 0.184 / 0.256 / 0.554 (Python / C++ / English). The best baselines are 0.394 / 0.339 / 0.769, so the comparison is decided by density before alignment comes into play.
- **The argument the report is missing:** a stronger, density-free argument is available. Word-start splitting beats BLT on precision and on recall in every language, with fewer boundaries (`c-v1-2721954/baselines.json`):

  | | Word start P / R | BLT P / R |
  |---|---|---|
  | Python | 0.142 / 0.892 | 0.083 / 0.822 |
  | C++ | 0.205 / 0.967 | 0.095 / 0.646 |
  | English | 0.627 / 0.994 | 0.342 / 0.894 |

- **Fix:** state the cap in one sentence, add the precision-and-recall comparison, and limit "better aligned at higher τ" to precision and margin.

### M2. The H1 verdict is softer than the proposal's own criterion
**Checked:** re-checked.

- **Where:** `paper:96` ("H1 is only partly supported"); `slides:229`.
- **What the proposal says** (`docs/proposal/Trimax-Proposal-Final.tex:284–287`): H1 is supported only if all four hold:
  - P1 clears both baselines;
  - P2 holds;
  - P3 shows dissociation;
  - P4's stratification appears.
- **Result:** only P2 holds as stated.
- **Fix:** "Not supported under the pre-stated criterion: P2 holds; P1 holds only against the null; P3 and P4 are reversed."

### M3. The report misattributes the mid-report's P4 explanation, and a promised test was never run
**Checked:** re-checked.

- **Where:** `paper:54` ("The mid-report's conjecture was that a start boundary sits *before* the keyword…").
- **Evidence:** the **submitted** `reports/Trimax-Mid-Report.pdf` raises that reading on p.6 only to reject it ("would predict no difference …"). It favours formatting consistency instead, and on p.7 it promises a test conditioned on formatting. That test was never run. The quoted conjecture exists only in the repo's `reports/mid_submission/paper-body.tex:103`, which differs from the submitted PDF.
- **Fix:** attribute it correctly ("the mid-report considered and set aside …"). Either say the formatting test was dropped, or run it.

### M4. A failed prediction is missing from the abstract and conclusion
**Checked:** re-checked.

- **Where:** the proposal states in advance that "precision against constituent starts should exceed recall" (`proposal:191`). It fails badly (Python 0.083 vs 0.822), and `paper:28` says so, but the abstract counts only three failures (P3, P4, O5).
- **Fix:** name the precision > recall failure in the abstract and conclusion.

### M5. The O5 explanation is a guess worded as fact
**Checked:** re-checked.

- **Where:** `paper:73` ("most plausibly carried by the unsafe operator bytes themselves"); `slides:162`; README ("no semantic residual").
- **Evidence:**
  - Nothing in `scripts/final_analysis/o5_confound.py` tests the operator-byte explanation. The proposal argues those tokens are low-entropy (`proposal:311–313`).
  - On the proposal's own BPP outcome, the raw unsafe effect with file fixed effects is −0.0024 [−0.0066, 0.0017], which is null. Only boundary density is "unconditionally negative" (`results/final/analysis/o5_confound.json`).
- **Fix:** label it a conjecture. Better, mask the operator bytes and refit; the refit is cheap.

### M6. A slide misstates how many times the test split was scored
**Checked:** re-checked.

- **Where:** `slides:94`: "Test split scored once; every later analysis reuses the frozen per-file counts …".
- **Evidence:** the test files were also scored for:
  - the student;
  - the τ\* confirmatory runs;
  - k = 1, 2;
  - the O4 sweep;
  - HF full attention;
  - the exploratory τ test grid (30 Oct).

  The report's own list at `paper:22` also leaves out the τ test grid.
- **Fix:** "The frozen teacher run was scored once; later runs (listed) scored the test files without selecting anything on them."

### M7. "Only indentation noise grows with rate" is contradicted by the sweep
**Checked:** re-checked.

- **Where:** `paper:65`, `README:56`, `slides:158`.
- **Evidence** (`results/final/analysis/o4_rate_sweep.json`), change in margin at 5 → 40% noise:

  | Noise | Change (5% → 40%) | 95% CI at 40% |
  |---|---|---|
  | Python keyword typo | −0.0011 → −0.0070 | [−0.0080, −0.0061] |
  | C++ missing semicolon | −0.0006 → −0.0031 | [−0.0038, −0.0024] |
  | Python identifier-case inversion | −0.0006 → −0.0013 (25 → 100%) | — |

  These are 4–9× smaller than indentation noise but not flat.
- **Fix:** "Indentation noise does by far the most damage, growing roughly in proportion to the lines touched; the other noise types also grow with rate but stay below 0.007."

### M8. "Genuine but shallow structural signal" is never tested
**Checked:** reported.

- **Where:** abstract (`paper:5`), `slides:257`.
- **Problem:** no null restricted to word or line starts separates syntax from token and line onsets, so "structural" goes beyond what was measured.
- **Fix:** call it a lexical/layout signal, or add the restricted null.

---

## Minor

| # | Finding | Where | Checked |
|---|---|---|---|
| m1 | "The 8,000 Python functions come from only 315 repositories": the 8,000 span **317** repositories; 315 is the test split alone | `paper:76` (`\xRepoN`), `README:189` | re-checked |
| m2 | The segmentation example is wrong: the real output is `\|if n \|< \|2:` (`2:` and the newline in one patch), not `\|2\|:` | `paper:36`; source `results/final/stream_a_fidelity_ada/fidelity.json` | re-checked |
| m3 | "3–5× above chance … in a distilled student": the student's English ratio is 2.96 (2.93 on the holdout) | `slides:249`, Conclusion | re-checked (0.459 / 0.155) |
| m4 | The slide says the 512-byte window "costs little"; the report says it explains 38% (Python) and 43% (C++) of the student's deficit | `slides:187` vs `paper:82` | re-checked |
| m5 | Three tables are never referenced in the text: the P3 ratio table (`tab:ratio`), starts vs ends (`tab:startend`) and span IoU (`tab:iou`, the proposal's secondary metric) | generated `final.tex` | re-checked |
| m6 | The O4 table caption says "the same held-out files"; the noise runs use 1,000 files per language (clean margin +0.115 there vs +0.120 in Table 1) | O4 table caption | re-checked |
| m7 | Negative zeros in tables (`[-0.00, …]`, `-0.000`), about 30 cells | generated tables, `docs/FULL_RUN_RESULTS.md` | re-checked |
| m8 | The slide prints the O5 coefficient as `+6.93e-03`; the report and README print `+0.0069` | `slides:165` (`\OFiveUnsafe`) | re-checked |
| m9 | "512-token" and "512-byte" are both used for the same window (2 vs 6 uses); the Fig. 1 caption starts lowercase; "C-v1" is used before it is defined | `paper` | re-checked |
| m10 | README and slides give P4 start margins as unweighted per-type means (+0.759 / +0.611), the report's Table 4 gives pooled values (+0.751 / +0.633), and only the report says which | `README:42`, `slides:132`, Table 4 | reported |
| m11 | Two different file-bootstrap intervals for the teacher's Python margin ([0.117, 0.122] in Table 1, [0.118, 0.122] in Table 11), so "widens 3.7×" is 3.6× against Table 1 | `paper:76`, `README:189`, `slides:95` | reported |
| m12 | The same quantity differs in the 4th decimal between documents because the null draws were re-run (e.g. English calibration median 0.3565 vs 0.3559; Table 12 shows +0.357, Fig. 5 plots 0.356) | `docs/FULL_RUN_RESULTS.md`, Table 12 | reported |
| m13 | O3 quotes depths 1→7 and leaves out C++ at depth 8+ (+0.151, n = 71,965), the steepest drop | `paper` O3 paragraph | reported |
| m14 | Multiple comparisons are never mentioned (many strata, three languages, many arms); small effects such as the balanced student's C++ interval [−0.0035, −0.0001] may not survive correction | Limitations | reported |
| m15 | Clustering: English intervals resample paragraphs, not articles (the holdout is 4,107 paragraphs from 120 articles); C++ is not clustered by repository | Limitations | reported |
| m16 | The Python newline baseline is 0.000 by construction at k = 0 (indentation always follows a newline); worth a footnote | Table 2 | re-checked (0.000) |
| m17 | Related work calls BLT the first model whose segmentation is decided at inference time; dynamic token pooling (Nawrot et al., 2023) predates it, so hedge or cite | `paper:16` | reported |

Smaller style points:
- l.93 and slide 14 mix 3- and 4-decimal intervals (`\xBalDiffCICpp`).
- Several sentences run to 50–69 words (`paper:13, 54, 76, 96`).
- Figures and tables are cited out of order (Fig. 4 before Figs. 2–3, Table 13 before Table 5).
- README rounds O5 to "4%" where the report says 4.2%.
- "About 84 GPU-hours" is 83.0.

---

## Housekeeping (outside the submission)

| # | Item | Status |
|---|---|---|
| h1 | GitHub's contributor sidebar still lists `@claude`, although no commit on any branch or tag mentions Claude (cached; GitHub recomputes it, or Support can purge it) | re-checked 31 Oct |
| h2 | No tag on the final commit `d9ffc6b`: `v1.1-final` is three commits behind | re-checked |
| h3 | `check_final.py` enforces "References start by page 9", not "body ≤ 8 pages" (see C1) | re-checked |
| h4 | Ada: the `/share1` file count is at the hard limit (3,200 files against a 3,000 quota); home is 88% full. Nothing is running | re-checked |
| h5 | Local-only branches `backup/pre-pull-1bf1e1b` and `backup/pre-strip-claude` (the latter still has Claude co-author lines; never pushed) can be deleted | re-checked |

---

## Checks that passed

- **Numbers.** About 600 values were recomputed from the raw results and match the generated macros:
  - frozen teacher run (~75);
  - k sweep (33);
  - student (~30);
  - τ\* calibration grids, confirmatory runs and the new test grid (~130);
  - English holdout (16);
  - teacher-512 control (14);
  - follow-up arms (30);
  - distillation (20);
  - efficiency (12);
  - HF full attention (12);
  - P4 positions (40);
  - O3 (6), O4 (52), O5 (24);
  - subsets (18), IoU (18), and others.

  All 62 macro intervals and 88 report-table intervals have the right order and sign, with the estimate inside the interval.
- **Exploratory τ test grid.** All 36 values match. The test maximum equals the calibration τ\* in 6 of 6 model/language pairs. The minimum gap to the best baseline is 0.1483 (teacher, C++, τ = 2), so "at least 0.148" is right. It is labelled exploratory in the report, the figure caption, the slides and the results doc.
- **Frozen protocol.** `results/final/c-v1-2721954/`, `escape_scoring/` and `schemas/v1/` are unchanged since `v1.0-final`, and `runs.json` lost no entries. The holdout rule (25 Oct) and the τ\* rule (28 Oct) were committed before scoring.
- **Checksums.** `sha256sum -c SHA256SUMS --ignore-missing` passes 17 of 17. The 3 skipped files are regenerable and documented.
- **Tests and code.**
  - 198 tests pass.
  - Every module compiles.
  - No tracked file refers to the files deleted or renamed on 31 Oct (`plan.md`, `file_toimplement.md`, `docs/ai_assisted/`, `ADA_CLAUDE_HANDOFF.md`).
- **Builds.**
  - The report builds with 0 missing inputs and 0 MISSING macros.
  - The slides build with 0 overfull boxes (18 slides).
  - The slide links to the HTML tree viewers resolve.
- **Zip.** `Trimax-Final.zip` has 577 files and passes `unzip -t`. The report PDF, slides PDF, README, results doc and report source inside it are byte-identical to `main`, and it contains no deleted or AI-labelled files.
- **Disclosed deviations.**
  - float32 instead of bf16 (0.26% of boundaries differ);
  - C++ from The Stack instead of CodeSearchNet;
  - the third-party checkpoint and the restored window;
  - the HF full-attention sensitivity check;
  - benepar targets for English;
  - earlier inspection of the 300 English paragraphs (hence the holdout);
  - follow-up arms scored only on subsets;
  - no subword baseline.

All of these are stated in the report.
