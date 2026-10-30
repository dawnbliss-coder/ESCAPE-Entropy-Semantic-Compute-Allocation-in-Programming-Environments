"""Student vs teacher efficiency table (plan.md section 3): GPU time, memory, tokens/s.

Two measurements on one GPU, same process, same inputs:
  forward   one forward pass over B x 513 tokens ([BOS] + 512 bytes, the training window)
            for B in {1, 8, 32, 64}: tokens/s and peak allocated memory. This is the
            architecture's cost; it does not depend on how the corpus extraction is batched.
  rolling   escape_distill.extract.rolling_entropies (stride-1 fresh 512-byte windows, the
            extractor used for both student-v1-14326k and teacher-512-ctl) on the first
            N files of the pre-registered teacher-512 subset: bytes/s and peak memory.
The teacher's native single-pass swa512 throughput (13,245 tokens/s float32, 1,364 MiB on a
2080 Ti) is already recorded in results/final/stream_a_fidelity_ada/fidelity.json.

Usage (Ada GPU node, a_py):
  python scripts/final_analysis/efficiency_bench.py --out results/final/analysis/efficiency.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from escape_distill.extract import load_student, load_teacher, rolling_entropies  # noqa: E402
from escape_distill.student_blt import BOS_ID, BYTE_TOKEN_OFFSET  # noqa: E402

CK = REPO_ROOT / "checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"


def sync():
    torch.cuda.synchronize()


@torch.inference_mode()
def bench_forward(fn, batch, reps):
    for _ in range(2):
        fn(batch)
    sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(batch)
    sync()
    dt = (time.perf_counter() - t0) / reps
    return {"seconds_per_pass": dt, "tokens_per_s": batch.numel() / dt,
            "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", type=int, default=20)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args(argv)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    student, _ = load_student(REPO_ROOT / "results/distill/full/student/student.pt", dev)
    teacher, _ = load_teacher(CK, dev)
    models = {"student": (lambda b: student(b), sum(p.numel() for p in student.parameters()), 64),
              "teacher": (lambda b: teacher.logits(b, mask=None).float(), sum(v.numel() for v in teacher.w.values()), 32)}
    g = torch.Generator().manual_seed(20260927)
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "float32",
           "forward": {}, "rolling": {}}
    for name, (fn, n_params, _) in models.items():
        out["forward"][name] = {"n_params": n_params}
        for B in (1, 8, 32, 64):
            x = torch.randint(4, 260, (B, 513), generator=g)
            x[:, 0] = BOS_ID
            try:
                out["forward"][name][f"B{B}"] = bench_forward(fn, x.to(dev), args.reps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out["forward"][name][f"B{B}"] = "OOM"
            print(name, B, out["forward"][name][f"B{B}"], flush=True)
    m = pd.read_parquet(REPO_ROOT / "corpus/manifest.parquet").set_index("file_id")
    ids = []
    for lang in ("python", "cpp"):
        ids += (REPO_ROOT / f"results/final/teacher512_subset_{lang}.txt").read_text().split()[: args.files // 2]
    contents = [(REPO_ROOT / f"corpus/{m.loc[f, 'domain']}/{f}.bin").read_bytes() for f in ids]
    n_bytes = sum(len(c) for c in contents)
    for name, (fn, _, bw) in models.items():
        rolling_entropies(fn, contents[0][:600], dev, bw)
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for c in contents:
            rolling_entropies(fn, c, dev, bw)
        sync()
        dt = time.perf_counter() - t0
        out["rolling"][name] = {"files": len(contents), "bytes": n_bytes, "batch_windows": bw, "seconds": dt,
                                "bytes_per_s": n_bytes / dt, "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20}
        print(name, "rolling", out["rolling"][name], flush=True)
    fid = json.loads((REPO_ROOT / "results/final/stream_a_fidelity_ada/fidelity.json").read_text())
    out["teacher_native_swa512_single_pass"] = fid["checks"]["throughput"]["float32"]
    out["speedups"] = {
        "forward_B32_tokens_per_s_student_over_teacher":
            out["forward"]["student"]["B32"]["tokens_per_s"] / out["forward"]["teacher"]["B32"]["tokens_per_s"]
            if isinstance(out["forward"]["teacher"]["B32"], dict) else None,
        "rolling_bytes_per_s_student_over_teacher":
            out["rolling"]["student"]["bytes_per_s"] / out["rolling"]["teacher"]["bytes_per_s"]}
    path = REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["speedups"]))


if __name__ == "__main__":
    main()
