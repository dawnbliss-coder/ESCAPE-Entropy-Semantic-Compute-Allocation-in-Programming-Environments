# Project ESCAPE

**E**ntropy & **S**emantic **C**ompute **A**llocation in **P**rogramming **E**nvironments — does the Byte Latent Transformer's entropy-triggered patch boundaries line up with syntactic constituent starts in code?

Team Trimax · CS7.501 Advanced NLP · Final Project

(Proposal and other writeups are kept locally per-member, not in this repo — this repo holds shared project code and the data contract only.)

## Work is split into three independent streams

| Stream | Owner | Scope |
|---|---|---|
| **A** — Patcher, Entropy & Compute Allocation | | BLT patcher fidelity (sliding-window fix), boundary extraction, BPP profiling |
| **B** — Ground Truth, Corpora & Adversarial Data | Priyanka | Frozen corpora, tree-sitter AST offsets, memory-unsafe region tagging, prose ground truth, adversarial mutations |
| **C** — Scoring, Statistics & Confound Analysis | | Data contract, alignment scoring, null models/baselines, statistical inference |

Each stream is runnable independently against a shared data contract (file layout, `file_id` and byte-offset conventions), to be agreed with A and C and added here once finalized — **B has no upstream dependency and its Week-1 output is the project's critical path**, since A and C both index B's frozen bytes and offsets.

## Repo layout (as it fills in)

```
corpus/            B — frozen raw bytes + manifest (gitignored, regenerable)
adversarial/        B — mutated corpus for O4 (gitignored)
structure/           B — AST node offsets (gitignored)
whitespace/          B — newline/indent positions (gitignored)
memory_regions/      B — C++ memory-unsafe spans (gitignored)
identifiers/         B — identifier spans (gitignored)
golden_fixtures/     B — ~20 hand-verified files, tracked (shared regression test)
taxonomy.json        B — node_type -> deterministic_opener | open_ended

entropies/           A — per-byte entropy arrays (gitignored)
boundaries/          A — tagged boundary tables (gitignored)

escape_common/       C — schema definitions, offset utilities, validator
escape_eval/         C — scorer, baselines, permutation engine, figures
results/             C — output tables/figures
```

Large or regenerable data artifacts stay out of git (see `.gitignore`); only code and golden fixtures are tracked.
