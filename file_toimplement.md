# ESCAPE: Remaining Work and Implementation Handoff

> **Status 30 Oct 2026 (evening).** Everything below that does not depend on missing material is done:
> - P0 text corrections are in the report; the page count was measured with no filler added (11 pages, References on page 8, so the body is 7 pages). No course rule other than the mid-submission's "8 pages excluding references" was found.
> - The optional arms are done: hf_full, the byte-balanced student, STEP 9 adaptation (verdict not met) and the decision not to build a parser teacher (`docs/PARSER_TEACHER_DECISION.md`). More permutations were run as a sensitivity check (jobs 48197 / 48198): no O4 margin moves by more than 4e-05, and the IoU intervals move only in the fourth decimal.
> - The team decisions were made by the user on 30 Oct:
>   - the AI files moved to `docs/ai_assisted/` and labelled;
>   - the stale branch deleted;
>   - the history rewrite declined;
>   - Ada logs committed;
>   - committed and tagged `v1.0-final` and pushed.
> - Still blocked: the reviewer amendment list and `reports/AUDIT_TEACHER_STUDENT_FILES.md` (M7). See `plan.md` for the per-item record.

**Purpose:** turn the open items in `plan.md` and the proposal/report comparison into concrete tasks. This file is a handoff for whoever implements or reviews the remaining work.

**Status basis:** current repository notes, especially `plan.md`, plus the final report source under `reports/final_submission/`. The plan says the report, slides, and `Trimax-Final.zip` are built. Confirm the live Git status before changing files; do not assume the working tree is clean.

## Already implemented (do not redo without a reason)

- The extraction pipeline, tree-sitter mapping, scoring harness, O1–O5 analyses, P1–P4 analyses, final report and slides.
- The student–teacher extension, including full student scoring, threshold sweeps, holdout analysis, alpha=1 control, efficiency results, and per-domain results.
- The report and slide PDFs and `Trimax-Final.zip` are already generated according to `plan.md`.
- The report checker was reported to pass; the Claude audit reported 187 tests passing. Re-run checks after changes that could affect them.

## Priority 0: proposal/report corrections before freezing

### 1. Disclose the missing subword-tokenizer comparison

**Gap:** The proposal's O4 discussion says robustness will be demonstrated relative to subword tokenizers (see `docs/proposal/Trimax-Proposal-Final.tex`, around the O4 description). The final report tests perturbations but does not compare against a subword tokenizer. Do not imply that the proposed comparison was performed.

**Recommended deadline action:** Add this sentence to the Limitations section in `reports/final_submission/paper-body.tex`:

> We did not evaluate a subword-tokenizer baseline, so O4 establishes robustness only for the tested perturbations and does not establish robustness relative to subword tokenizers.

Rebuild the report and ZIP and verify that the sentence appears in the PDF. Do not change experimental numbers.

**If the team chooses to run the comparison instead:** write down the tokenizer, version, vocabulary/training provenance, normalization, and segmentation-to-byte-offset mapping before scoring. Use the same frozen evaluation files and constituent targets; compare perturbation degradation with a clearly defined subword baseline. Keep this analysis separate from the existing confirmatory result unless a reviewer approves a revised protocol. Record seeds, commands, outputs, and limitations. This may require substantial compute and is not a safe last-minute substitute for disclosure.

### 2. Correct density claim in the report

In `reports/final_submission/paper-body.tex`, the P1 discussion currently says, “The margin is real and density-independent.” The report elsewhere explains that F1 margins scale with target density. Replace the sentence with a statement consistent with the results, for example:

> The margin is positive across languages, but its magnitude depends on target density.

Rebuild and inspect the report PDF after editing.

### 3. Confirm the page-count rule

The current report is noted as 11 pages total, with References starting on page 7 (page 8 after the 30 Oct additions). The repository checker allows up to 8 pages before References; that means six pages precede References in the current PDF. Find the course rubric or submission instructions and determine whether it requires a 7-page minimum for the body. If no such rule is available, report the measured page count to the team and do not add filler.

## Priority 0: package freeze

- [x] Confirm final report and slide sources/PDFs are present and included in the archive.
- [x] Rebuild report after approved edits: `bash reports/final_submission/build.sh`.
- [x] Inspect `scripts/package_final.sh`, then use its documented invocation to rebuild `Trimax-Final.zip`.
- [x] Run `python3 reports/final_submission/check_final.py` and an archive integrity check. Confirm the ZIP contains the updated report PDF and source.
- [x] Review `git status` and the exact staged file list with the team. Commit and tag only after the team approves the freeze. No history rewrite after tagging.

## Blocked on missing material

### Reviewer statistical amendments

`plan.md` says the reviewer amendment list is missing from this checkout. Obtain the original list from the reviewer/team, then make a table with each requested change, decision (adopt/defer/reject), evidence, and affected file. If a change affects a reported number, follow the frozen calibration/scoring protocol and record the new run; do not invent amendments or silently edit values.

### Teacher/student audit file and M7 corrections

`reports/AUDIT_TEACHER_STUDENT_FILES.md` is reported missing locally and on Ada. Obtain it from its author/team. Use its M7 evidence to correct the two cited statements in `docs/ai_assisted/AUDIT_REPORT_2.md` (moved 30 Oct) and `reports/ESCAPE_STUDENT_CV1_EVALUATION.md`, then link the correction to the audit item. Do not guess what M7 says. The requested commit/share of that audit file is blocked until the file is recovered.

## Team decisions / Git operations

These need an explicit team choice before action:

- Decide whether `AUDIT_REPORT*.md` and `.opencode/` belong in the submitted repository. If retained, label their purpose and status; if excluded, document the decision and remove them from the submission staging list.
- Decide whether to delete or merge the stale `distill-repro-39379` branch. The plan says it exists remotely and contains `run_all_repro.sbatch`; inspect its unique commits and needed files before any deletion. The sbatch-path fix applies only if that branch/script is retained.
- Commit and tag the frozen submission after final review. Agree the tag name and commit contents first.
- Do not rewrite history to remove duplicate `student.pt` after the freeze. The plan reports this as not done; assess repository size and collaborator impact before considering history surgery.
- Ada job logs are in the ZIP but ignored by Git. Decide whether the commit needs them. Prefer a concise provenance/job ledger over adding large logs unless the course explicitly requires raw logs.

## Optional extensions (P2; only do if time and compute are available)

These are not blockers for the current submission if the report accurately states they were not run:

- **`hf_full` alignment sensitivity:** score the existing `hf_full` run for alignment, using the frozen scoring protocol; compare it with the current primary result and label it a sensitivity analysis.
- **Parser-supervised teacher:** define the parser teacher, training targets, data split, and comparison to the BLT teacher before implementation. This is a new study, not a small patch.
- **Teacher domain adaptation:** only proceed if the required compute and extraction fit. Apply the pre-registered decision rule and report both positive and negative outcomes.
- **Byte-balanced distillation:** rerun training with sampling balanced by bytes across domains, then evaluate with the frozen test protocol. Current report gives per-domain results; C++ is 73% of training bytes.
- **More permutations:** O4 uses 1,000 permutations and IoU uses 100 according to the plan. Raising these counts is optional; it should not delay the freeze unless the reviewer requires it.

## Verification checklist after report edits

- [x] `bash reports/final_submission/build.sh` exits successfully.
- [x] `python3 reports/final_submission/check_final.py` passes.
- [x] Extracted PDF text contains the tokenizer limitation and corrected density wording; no unresolved `??` or `MISSING` markers.
- [x] Record total pages and References start page; compare them with the actual course rule.
- [x] Rebuild `Trimax-Final.zip`; verify archive integrity and confirm it contains the current PDFs and sources.
- [x] Review `git status` and diff; ensure no results, data, or unrelated files changed.
- [x] Run the project test suite if code changed. The latest audit reported 187 passing tests; report the actual command and result for any new run.
- [x] Commit/tag only after team review and confirmation of the blocked inputs and repository decisions.
