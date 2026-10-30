# Next task: finish the ESCAPE teacher-student evaluation

This is the user's follow-up, prepared by Codex for delivery after your current task finishes. Finish any existing work first, then implement, run and verify the remaining teacher-student work below. Do not restart work you have already completed or duplicate active Ada jobs. The final submission deadline is 31 October 2026; protect time for the report and slides.

## Start by reconciling the latest state

Read `plan.md`, `docs/ADA_FULL_RUN_SEQUENCE.md`, `docs/ADA_CLAUDE_HANDOFF.md`, `docs/ADA_DISTILL_RUNBOOK.md`, `docs/SCORING_PROTOCOL.md`, and `reports/ESCAPE_STUDENT_CV1_EVALUATION.md`. Inspect the current diff, existing results and Ada job state before changing anything. Some handoff instructions are stale. Use the frozen protocol and verified artifacts to resolve them; document actual methodological changes separately.

The distilled student, its training/reproduction and the 18-file golden evaluation already exist. The main objective is complete-file syntactic evaluation plus context and efficiency controls, not another training run. At Codex's inspection:

- There were uncommitted changes in `escape_distill/` and several new scripts under `scripts/ada_distill/`. Preserve and extend this work.
- `tau_confirm.sbatch` had been added during your current task. Review it rather than treating the confirmatory workflow as wholly absent. It currently refuses different per-language selected thresholds.
- `teacher512_subset_python.txt` and `teacher512_subset_cpp.txt` already matched the frozen selection rule (300 files each).
- `corpus_ext/holdout_manifest.parquet` already selected 4,107 paragraphs from 120 articles, matched the documented selection rule, and had no article overlap with the distillation manifest or exact-text overlap with the evaluation corpus. It is currently ignored by Git; preserve this selection and make its provenance retainable.
- Full student scores and Ada golden-gate results were not present locally. This is not evidence they have not run remotely. Inspect Ada and collect existing results first.

Create or update `docs/TEACHER_STUDENT_STATUS.md` with each component's state, artifact paths, job IDs and next action. Use this as the coordination record for the user and Codex. Distinguish implemented, submitted/running, failed and verified complete.

## 1. Validate the existing execution path

Keep `results/final/c-v1-2721954/`, its calibration and the original corpus/splits untouched. Give changed configurations new run IDs. Validate model hashes against the committed references; main currently stores the student checkpoint as a regular Git blob, so reconcile contradictory LFS instructions accordingly.

Recheck the working-tree student device fix, extraction resumption, and the parallel scoring changes. Reject unknown requested sample IDs and missing requested entropy arrays rather than silently dropping them or returning success. A complete extraction requires every expected ID, finite float32 entropy arrays of shape `(n_bytes,)`, valid boundary metadata and exact patch tiling, not merely existing filenames.

Retain actual code/checkpoint hashes, model identity, context policy, source run, sample IDs, split and threshold in outputs. Ensure comparison outputs describe each run's actual tau rather than a single hard-coded threshold. Add focused checks for serial/parallel numerical equivalence, the per-file null calculation, incomplete-input rejection and provenance conflicts. Run the relevant existing scorer/patching tests; do not silently change frozen scoring semantics.

## 2. Finish the full student C-v1 result

Follow the existing scripts and gates:

1. Ada golden gate: student F1 must match Python 0.1313, C++ 0.2196, prose 0.4290 within 0.002; also verify the recorded precision/recall and baseline checks.
2. Extract the original 16,300 corpus files with the registered primary rolling-window policy. Resume completed valid files and check all shards before scoring.
3. Convert, validate, calibrate, score and run baselines through `escape_scoring`, with a separate immutable student calibration record and the same deterministic corpus split.
4. Collect the calibration record, scores, per-file results, diagnostics, baselines, commands, logs and hashes under distinct `results/final/student-*` paths.

Use Ada for heavy inference/scoring. The runtime estimates in the handoffs conflict with the newer extraction script; use actual throughput and the queue to estimate completion. Preserve the primary context policy even if it is slow. If completion cannot fit the deadline, report verified completed scope and retain the golden-only claim instead of presenting a partial run as full-corpus evidence. Do not cancel or relaunch existing jobs without first checking their state.

## 3. Complete the teacher-512 context control and efficiency measurements

Use the existing 300 Python + 300 C++ selection lists, verifying their hashes and sample membership. Compare the original teacher, teacher-512 and student on exactly the same files at the prescribed settings. Verify restored teacher artifacts against the frozen reference before using them. Report paired differences with bootstrap intervals; overlapping individual intervals are not an equivalence test and the outcome must not be prescribed in advance.

Measure GPU wall time, useful source bytes per second, peak allocated/reserved GPU memory, hardware/software and batch/window settings on comparable workloads. Synchronize GPU timing and distinguish warmup, model loading and extraction costs. Compare student versus teacher-512 under the matched context workload, and report the native teacher's practical throughput separately. The student's roughly 14.3M parameters versus the teacher's 99.5M do not by themselves establish a speed advantage. Keep training cost, inference cost, bits per byte and bytes per patch distinct.

## 4. Finish calibration-only tau selection and the final test evaluation

Reuse saved entropies and the registered grid: 0.8, 1.0, 1.335442066192627, 1.7, 2.0, 2.5. For each model and language, select tau at k=0 by the median over calibration files of that file's observed F1 minus that file's mean null F1; ties within the documented tolerance choose the smaller tau. Use the documented per-file criterion. The pooled-null snippet in an older handoff is not an interchangeable selection rule.

At selected tau, re-freeze k on calibration using the existing C-v1 criterion, then run the planned final evaluation once. Record the selected settings and input hashes before opening the new test result. Keep any fixed-k comparison explicitly supplementary. If a test result has already been seen, disclose that fact rather than relabelling it as untouched confirmation.

Handle different per-language tau choices: C-v1 currently rejects mixed model/threshold configurations in one calibration dataset. Use separate model/language runs and records containing all original samples for that language so the original per-language split is preserved. Do not select a global tau just to make the CLI accept it, and do not weaken frozen-record validation.

Generate the existing calibration k-grid table directly from `calibration.json`. Additional test k=1/2 results are labelled sensitivity only. Confirmatory-run orchestration should record prerequisites, selections and job IDs and prevent accidental duplicate submission while allowing transparent recovery from failed infrastructure jobs.

## 5. Complete the English holdout if it fits the deadline

Use the existing holdout manifest rather than selecting again after seeing any results. Verify article and hash exclusions, materialize all selected bytes, and build benepar targets with byte-roundtrip validation. Implement explicit external-manifest/text/structure support: the current extractor and converter assume the main evaluation corpus, and frozen C-v1 dataset fingerprints cannot simply be reused for a different dataset.

Follow the recorded team sign-off requirement for the holdout selection before boundary scoring; check whether that sign-off already exists. Freeze model/settings and the external evaluation procedure using calibration/development data only. Score the untouched holdout once without calibrating on its paragraphs, and account for dependence within articles. The original English evaluation remains exploratory. If sign-off or implementation is still outstanding at the cutoff, retain the data-only artifacts and report this experiment as pending.

## 6. Close the evidence and reporting loop

Write `reports/TEACHER_STUDENT_FINAL.md` with source-linked tables for (a) imitation of teacher boundaries, (b) alignment against syntax/null/structure-blind baselines, (c) context-control effects, and (d) measured efficiency. Label golden, original test, calibration, sensitivity and fresh holdout results correctly. Include completed extension results in the final report/slides without disturbing already completed proposal analyses. Make the golden runner portable, pin the actual experiment environment and missing dataset revision where recoverable, and correct stale handoff statements based on evidence.

Do not invent timings, assume a queued job finished, claim a smaller model proves syntax is absent, or claim distillation improves upon independent training without the corresponding control. Leave alpha=1/no-teacher training, domain adaptation, parser-teacher work and byte-balanced retraining for after the required evaluation and submission deliverables unless already complete.

Finish with a compact checklist: completed artifacts and checks, jobs still running with IDs, actual blockers and exact next commands. Keep the coordination record up to date so Codex can review the result without reconstructing the experiment history. Work autonomously within the user's existing authorization and the session's normal permissions; raise only genuine unresolved methodological choices or missing required approvals.
