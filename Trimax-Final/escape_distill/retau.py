"""Re-threshold saved per-byte entropy arrays into a NEW run's boundary parquets.

No model runs. Reads entropies/{src_run_id}/{file_id}.npy and writes
boundaries/{dst_run_id}/{file_id}.parquet (and copies the entropy array into
entropies/{dst_run_id}/) at the given tau, in the same Stream-A format as
stream_a/extract.py. Used for the tau sweep: each grid point and the pre-registered
tau* confirmatory run get their own run_id.

Usage:
  python -m escape_distill.retau --src-run swa512-float32-t1.3354-mplnone-91aa6b8e \
      --dst-run teacher-tau2000 --tau 2.0 --file-ids ... --shard i --n-shards k
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from escape_common.io import atomic_save_npy, atomic_to_parquet, update_json  # noqa: E402
from escape_common.schema import (RUNS_JSON_PATH, boundary_path,  # noqa: E402
                                  entropy_path, corpus_bin_path, golden_bin_path)  # noqa: E402
from stream_a.patching import compute_patches  # noqa: E402

BOUNDARY_COLUMNS = ["file_id", "byte_offset", "trigger", "entropy", "patch_index", "patch_length"]


def boundaries_frame(file_id: str, entropies: np.ndarray, tau: float) -> pd.DataFrame:
    patches = compute_patches(entropies, tau)
    return pd.DataFrame({
        "file_id": pd.Series([file_id] * patches.n_patches, dtype="string"),
        "byte_offset": patches.starts.astype(np.int64),
        "trigger": pd.Series(patches.triggers.tolist(), dtype="string"),
        "entropy": entropies[patches.starts].astype(np.float32),
        "patch_index": np.arange(patches.n_patches, dtype=np.int32),
        "patch_length": patches.lengths.astype(np.int32),
    }, columns=BOUNDARY_COLUMNS)


def register_run(root: Path, run_id: str, entry: dict) -> dict:
    def update(runs):
        if run_id in runs:
            old = runs[run_id]
            mismatched = {k: (old.get(k), entry.get(k)) for k in ("backend", "tau", "entropy_definition")
                          if old.get(k) != entry.get(k)}
            if mismatched:
                raise SystemExit(f"run_id {run_id} exists with different settings: {mismatched}")
            return runs
        runs[run_id] = entry
        return runs
    return update_json(root / RUNS_JSON_PATH, update)[run_id]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--src-run", required=True, help="run_id whose entropy arrays are re-thresholded")
    ap.add_argument("--dst-run", required=True, help="new run_id for the re-thresholded boundaries")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--file-ids", nargs="*")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    manifest = pd.read_parquet(root / "corpus/manifest.parquet")
    ids = list(manifest["file_id"]) if not args.file_ids else args.file_ids
    ids = sorted(ids)
    if args.shard is not None and args.n_shards:
        ids = ids[args.shard:: args.n_shards]

    # Inherit the source run's model identity (checkpoint, revision, context, entropy
    # definition) so escape_scoring convert-legacy accepts the re-thresholded run; only
    # tau and the provenance fields change.
    src_entry = json.loads((root / RUNS_JSON_PATH).read_text()).get(args.src_run, {})
    inherited = {k: v for k, v in src_entry.items()
                 if k not in ("invocations", "created_by", "date", "tau", "boundary_convention",
                              "commit", "commit_dirty", "code_sha256", "source_run")}
    register_run(root, args.dst_run, {
        **inherited,
        "backend": "patcher", "created_by": "escape_distill/retau.py",
        "entropy_definition": src_entry.get(
            "entropy_definition", "global_next_byte_shannon (arrays copied verbatim from the source run)"),
        "boundary_convention": "one row per patch; byte_offset = first byte of the patch; "
                               "trigger entropy = entropies[byte_offset] > tau (strict, float64 comparison)",
        "tau": args.tau, "max_patch_length": None, "source_run": args.src_run,
    })

    done = failed = 0
    for fid in ids:
        ep_src = entropy_path(root, args.src_run, fid)
        if not ep_src.exists():
            print(f"[retau] SKIP {fid}: no source entropy array", flush=True)
            continue
        try:
            ent = np.load(ep_src, allow_pickle=False)
            atomic_save_npy(entropy_path(root, args.dst_run, fid), ent)
            atomic_to_parquet(boundaries_frame(fid, ent.astype(np.float32), args.tau),
                              boundary_path(root, args.dst_run, fid))
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[retau] FAIL {fid}: {exc}", flush=True)
    print(f"[retau] DONE dst_run={args.dst_run} tau={args.tau} shard={args.shard}: "
          f"{done} done, {failed} failed", flush=True)
    if failed:
        raise SystemExit(f"{failed} files failed")


if __name__ == "__main__":
    main()
