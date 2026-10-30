# Local smoke run (not the main result)

This directory is a **local smoke test** of the `escape_distill` pipeline, run locally (not on
Ada) on 2026-10-22 to check that training and evaluation work end to end. It is **not** the
distillation result and its numbers must not be quoted as one.

| | Smoke (this directory) | Main run |
|---|---|---|
| Data | 76 train / 18 eval files of local repo bytes (py + prose only, no C++) | extension corpus `corpus_ext/`, 27,153 train / 1,847 held-out val files (py, cpp, prose) |
| Student | 4,886,784 params (4.9M; dim 256, 6 layers, 4 heads) | 14,262,144 params (14.26M; dim 384, 8 layers, 6 heads) |
| Steps | 426 | 6,860 |

The main result is [`results/distill/full/`](../full/) (Ada job 34241; `eval/distill_eval.json`).
