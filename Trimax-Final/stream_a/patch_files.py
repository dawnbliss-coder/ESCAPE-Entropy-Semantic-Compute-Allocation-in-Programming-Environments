"""Boundary extraction for arbitrary files (proposal deliverable 1).

stream_a.extract reads only manifest-verified corpus or golden files. This runs the same
patcher backend and the same entropy -> patch rule on any files given by path, outside the
corpus and runs.json bookkeeping, and writes per file:
  {out}/{name}.entropies.npy        float32[n_bytes]; entropies[i] belongs to byte i (nats)
  {out}/{name}.boundaries.parquet   same columns as stream_a.extract
  {out}/{name}.summary.json         BPP and entropy statistics, settings
--show prints each file with '|' before every patch start after byte 0.

Examples (repo root, .venv_a; checkpoint from python -m stream_a.fetch_patcher):
  python -m stream_a.patch_files src/foo.py --out /tmp/patches --show
  python -m stream_a.patch_files a.cpp b.txt --device cuda
  python -m stream_a.patch_files any.py --mock --show      # byte-only stand-in, no model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_common.io import atomic_save_npy, atomic_to_parquet, atomic_write_json  # noqa: E402
from stream_a.extract import REPO_ROOT, MockBackend, PatcherBackend, boundaries_frame  # noqa: E402
from stream_a.patching import (  # noqa: E402
    bpp_stats,
    byte_entropies_from_token_entropies,
    compute_patches,
    entropy_stats,
)


def patch_bytes(backend, content: bytes, tau: float, max_patch_length=None):
    """Per-byte entropies and patches of one byte string, exactly as stream_a.extract.process_file."""
    ent = byte_entropies_from_token_entropies(backend.token_entropies(content), len(content))
    if not np.all(np.isfinite(ent)):
        raise FloatingPointError("non-finite entropy")
    return ent, compute_patches(ent, tau, max_patch_length)


def show(content: bytes, patches) -> str:
    """The text with '|' before every patch start except byte 0 (undecodable bytes replaced)."""
    starts = patches.starts.tolist() + [len(content)]
    return "|".join(content[a:b].decode("utf-8", errors="replace") for a, b in zip(starts[:-1], starts[1:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("patch_files_out"))
    ap.add_argument("--context-mode", choices=("swa512", "hf_full"), default="swa512")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tau", type=float, default=None, help="default: the checkpoint's threshold (1.3354 nats)")
    ap.add_argument("--max-patch-length", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="deterministic byte-only stand-in; outputs are NOT model measurements")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    backend = MockBackend() if args.mock else PatcherBackend(REPO_ROOT, args.context_mode, args.dtype, args.device)
    tau = backend.tau if args.tau is None else args.tau
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        content = path.read_bytes()
        if not content:
            print(f"[patch_files] {path}: empty, skipped")
            continue
        ent, patches = patch_bytes(backend, content, tau, args.max_patch_length)
        name = path.name
        atomic_save_npy(args.out / f"{name}.entropies.npy", ent.astype(np.float32))
        atomic_to_parquet(boundaries_frame(name, ent, patches), args.out / f"{name}.boundaries.parquet")
        summary = {"file": str(path), "n_bytes": len(content), "n_patches": int(patches.n_patches),
                   "tau": tau, "max_patch_length": args.max_patch_length, "backend": backend.config["backend"],
                   "context_mode": backend.config["context_mode"], "dtype": backend.config["dtype"],
                   **bpp_stats(patches.lengths), **entropy_stats(ent, tau)}
        atomic_write_json(args.out / f"{name}.summary.json", summary)
        print(f"[patch_files] {path}: {len(content)} bytes, {patches.n_patches} patches, "
              f"mean BPP {summary['mean_bpp']:.2f}, mean entropy {summary['ent_mean']:.2f} nats -> {args.out}")
        if args.show:
            print(show(content, patches))


if __name__ == "__main__":
    main()
