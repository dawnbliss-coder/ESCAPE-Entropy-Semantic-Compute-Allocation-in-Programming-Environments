# Teacher choice: BLT entropy patcher, not a parser (S2)

**Decision (2026-10-30):** the teacher for the distilled patcher is the BLT entropy model (`itazap/blt-1b-hf` @ 91aa6b8e). The parser-supervised teacher (S2) was not built or run for the final submission. The report's Limitations section says so.

## Why not a parser teacher here

1. **It cannot answer the research question.** ESCAPE asks whether boundaries learned from next-byte entropy line up with syntax (O1–O5). A parser-supervised teacher is trained on the constituent starts from tree-sitter (code) and benepar (English) that C-v1 scores against. Its alignment would measure how well it fits its own labels, not whether entropy encodes structure.
2. **It would not be comparable under the frozen protocol.** C-v1 compares each run with a density-matched null and with the newline, indentation and word-start baselines. A model trained on the targets would beat all of these by construction, so the comparison carries no information.
3. **Time.** The teacher-student extension, its controls (α = 1, teacher-512, byte balancing, domain adaptation) and the confirmatory runs used the remaining compute before 31 Oct.

## What S2 would need if the team picks it up

- **Purpose:** a deployment patcher for compute allocation, judged on a downstream task (compression or latent-model loss at equal patch budget), not on C-v1 alignment.
- **Labels:** constituent-start bytes from tree-sitter on the extension corpus (`results/distill/full/manifest_ext.parquet`), which is disjoint from the evaluation corpus. English labels would come from benepar on the prose part of the extension corpus.
- **Model:** the 14.3M student architecture (`escape_distill/student_blt.py`, dim 384, 8 layers, 6 heads) with a per-byte boundary head and binary cross-entropy. It should be trained with the same windows and schedule as the distilled student so that the two differ only in their supervision.
- **Comparison:** the BLT-distilled student and the BLT teacher at equal mean patch length, on held-out files, with a pre-registered metric written down before training.
- **Pre-registration:** the metric, the patch budget and the test split have to be fixed before any S2 model is scored.
