# escape_distill - teacher→student distillation of the BLT entropy model

An experiment extending Project ESCAPE: can a **small, from-scratch BLT entropy model
(the student)** be taught to reproduce the **frozen 100M-parameter BLT patcher (the
teacher)** - its next-byte distribution and, through that, where patch boundaries fall - on
a new dataset the teacher never produced labels for?

The teacher is the exact checkpoint used for every number in the report
(`itazap/blt-1b-hf` @ `91aa6b8e`, swa512, float32). The new dataset is the **extension
corpus** (`corpus_ext/`, 69,975 files: 40k Python, 20k prose, ~10k C++), built disjoint from
the evaluation set (repo-level dedup, 0 overlap with the test split).

## Design (faithful to the project)

- **Student** (`student_blt.py`): a from-scratch byte transformer with the BLT patcher's
  ingredients - vocab 260, pre-norm blocks, RMSNorm, interleaved RoPE, SwiGLU FFN,
  head-dim⁻¹ᐟ² attention scaling - but small (default dim 256 / 6 layers / 4 heads ≈ 4.9M
  params; the Ada run uses 384 / 8 / 6). It is *not* a copy of the checkpoint; the question
  is how much of the teacher a small student can absorb.
- **Windows ≤ 512 bytes.** On ≤512-token windows the teacher's 512-token sliding window is
  provably identical to full causal attention (fidelity gate G3), so teacher and
  limited-context student are compared on exactly the same context, with no 8192-token
  chunking.
- **Online KD** (`train.py`): `loss = α·CE(student, true byte) + (1−α)·T²·KL(softmax(teacher/T) ‖ softmax(student/T))`,
  over positions that predict a real byte. The teacher is run on the fly (no multi-TB dump
  of 260-way distributions). Defaults T=2, α=0.1.
- **Teacher backends** (`teacher.py`): `transformers` (the report's exact `BltPatcher`,
  used on Ada) or `reference` (the pure-torch `stream_a.reference_lm`, equal to the HF port
  to 1e-4; needs no transformers/xformers). Auto-selected.

## What the result reports (`evaluate.py`), per domain and overall

Student vs teacher on held-out windows, each against an **untrained-student baseline**:

| metric | meaning |
|---|---|
| boundary F1 | student's interior patch starts (τ, `compute_patches`) vs the teacher's |
| entropy Pearson r / MAE | per-byte entropy agreement (nats) |
| KL(teacher‖student) | the distillation objective, on held-out data (nats/byte) |
| bits/byte | each model's own next-byte cross-entropy vs the true byte |

## Run it

**Local smoke** (any machine with torch + the fetched checkpoint; teacher fits in ~4 GB):

```bash
CK=checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01
PYTHONPATH=. python -m escape_distill.train   --ckpt-dir $CK --data <dir-of-.bin> --out /tmp/stu --epochs 6
PYTHONPATH=. python -m escape_distill.evaluate --ckpt-dir $CK --data <dir-of-.bin> --student /tmp/stu/student.pt --out /tmp/stu
```

**Full run on Ada** (materialize the ext corpus, distill, evaluate - SLURM, dependency-chained):

```bash
bash scripts/ada_distill/submit_all.sh                 # full
LIMIT=4000 bash scripts/ada_distill/submit_all.sh      # quick real run on a subset
SKIP_MATERIALIZE=1 bash scripts/ada_distill/submit_all.sh   # reuse existing corpus_ext/*.bin
```

## Scope

This started as an experiment (“does distillation of the entropy model work on a new dataset?”).
For the final submission the student was scored with the frozen C-v1 pipeline on the test split,
swept for τ on calibration files with one confirmatory test run per τ*, and scored once on the
pre-registered English holdout (report §5, `docs/FULL_RUN_RESULTS.md`). The student is trained purely on the teacher's
outputs; no claim is made that either model understands syntax (the project's standing
caveat: entropy is a surface statistic).
