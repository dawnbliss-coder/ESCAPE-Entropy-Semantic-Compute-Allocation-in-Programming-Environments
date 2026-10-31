"""The ONE confirmatory English result: score the pre-registered English holdout.

Pre-registration: reports/ESCAPE_STUDENT_CV1_EVALUATION.md 8.6 (selection rule) and
scripts/ada_distill/holdout_prose.py (materialised by job 43883: 120 unused WikiText
articles, every paragraph sha256-verified, benepar structure with byte round-trip
validation). Team sign-off to score it once was given on 2026-10-29. Nothing here
selects anything: k = 0 is the frozen English tolerance of both the teacher and the
student calibration records, tau is each model's checkpoint tau, and the null, matching
and baselines are those of escape_distill.score_a_runs.

  extract  teacher: the frozen swa512 patcher (stream_a PatcherBackend, the C-v1 run's
           settings) -> run holdout-teacher-swa512; student: escape_distill.extract
           rolling windows -> run holdout-student-v1. Resumable.
  view     a view root for score_a_runs: corpus/manifest.parquet listing only the holdout
           paragraphs (domain prose), corpus/prose and structure/prose linking to the
           holdout bytes / structure, and boundaries/ linking to the real boundaries/.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = REPO_ROOT / "corpus_ext/holdout_manifest.parquet"
TEXT = REPO_ROOT / "corpus_ext/holdout/text"
STRUCTURE = REPO_ROOT / "corpus_ext/holdout/structure"
RUNS = {"teacher": "holdout-teacher-swa512", "student": "holdout-student-v1"}


def holdout():
    m = pd.read_parquet(MANIFEST).sort_values("file_id").reset_index(drop=True)
    for r in m.itertuples(index=False):
        content = (TEXT / f"{r.file_id}.bin").read_bytes()
        if len(content) != r.n_bytes or hashlib.sha256(content).hexdigest() != r.sha256:
            sys.exit(f"{r.file_id}: holdout bytes differ from the holdout manifest")
        if not (STRUCTURE / f"{r.file_id}.parquet").exists():
            sys.exit(f"{r.file_id}: no benepar structure")
    return m


def extract(device):
    import torch
    from escape_common.io import atomic_save_npy, atomic_to_parquet
    from escape_common.schema import boundary_path, entropy_path
    from escape_distill import extract as dx
    from stream_a.extract import PatcherBackend, artifacts_done, process_file

    m = holdout()
    backend = PatcherBackend(REPO_ROOT, "swa512", "float32", device)
    if abs(backend.tau - dx.TAU) > 0 or backend.config_max_patch_length is not None:
        sys.exit("checkpoint tau / length cap differ from the frozen C-v1 run")
    student, _ = dx.load_student(REPO_ROOT / "results/distill/full/student/student.pt", device)
    t = {"teacher": 0.0, "student": 0.0}
    for i, r in enumerate(m.itertuples(index=False), 1):
        content = (TEXT / f"{r.file_id}.bin").read_bytes()
        if not artifacts_done(REPO_ROOT, RUNS["teacher"], r.file_id, r.n_bytes):
            t0 = time.time()
            process_file(REPO_ROOT, RUNS["teacher"], backend, r.file_id, content, backend.tau, None)
            t["teacher"] += time.time() - t0
        if not artifacts_done(REPO_ROOT, RUNS["student"], r.file_id, r.n_bytes):
            t0 = time.time()
            with torch.inference_mode():
                ent = dx.rolling_entropies(lambda b: student(b), content, device, 64)
            if not np.isfinite(ent).all():
                sys.exit(f"{r.file_id}: non-finite student entropy")
            atomic_save_npy(entropy_path(REPO_ROOT, RUNS["student"], r.file_id), ent.astype(np.float32))
            atomic_to_parquet(dx.boundaries_frame(r.file_id, ent.astype(np.float32)),
                              boundary_path(REPO_ROOT, RUNS["student"], r.file_id))
            t["student"] += time.time() - t0
        if i % 500 == 0 or i == len(m):
            print(f"[holdout extract] {i}/{len(m)} teacher {t['teacher']:.0f}s student {t['student']:.0f}s", flush=True)
    missing = [(k, f) for f, n in zip(m.file_id, m.n_bytes) for k in RUNS
               if not artifacts_done(REPO_ROOT, RUNS[k], f, int(n))]
    print(f"[holdout extract] {len(m)} files, {int(m.n_bytes.sum())} bytes; missing artifacts: {len(missing)}")
    if missing:
        sys.exit(1)


def view(dst: Path):
    m = holdout()
    (dst / "corpus/prose").mkdir(parents=True)
    (dst / "structure/prose").mkdir(parents=True)
    os.symlink(REPO_ROOT / "boundaries", dst / "boundaries")
    for f in m.file_id:
        os.symlink(TEXT / f"{f}.bin", dst / f"corpus/prose/{f}.bin")
        os.symlink(STRUCTURE / f"{f}.parquet", dst / f"structure/prose/{f}.parquet")
    pd.DataFrame({"file_id": m.file_id, "domain": "prose", "n_bytes": m.n_bytes, "sha256": m.sha256}) \
        .to_parquet(dst / "corpus/manifest.parquet", index=False)
    (dst / "ids.txt").write_text(" ".join(m.file_id) + "\n")
    print(f"[holdout view] {len(m)} files at {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract").add_argument("--device", default="cuda")
    sub.add_parser("view").add_argument("dst", type=Path)
    args = ap.parse_args()
    extract(args.device) if args.cmd == "extract" else view(args.dst)


if __name__ == "__main__":
    main()
