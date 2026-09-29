"""Stream A for Objective 4: BLT patcher outputs for the noised files in o4/.

Same backend, context regime (swa512), dtype, threshold and patching code as the
primary run (stream_a/extract.py: PatcherBackend, process_file), so clean and noised
conditions differ only in their bytes. The clean condition reuses the primary run's
artifacts and is not recomputed. Outputs, per noised condition C:
  entropies/<primary run id>__o4-C/<sample_id>.npy, boundaries/<...>/<sample_id>.parquet
Resumable: files whose artifacts are complete are skipped. Sharded by --shard i/n.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stream_a.extract import PatcherBackend, artifacts_done, process_file  # noqa: E402

PRIMARY_RUN = "swa512-float32-t1.3354-mplnone-91aa6b8e"


def o4_run_id(condition: str) -> str:
    return f"{PRIMARY_RUN}__o4-{condition}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = Path(args.root)
    m = pd.read_parquet(root / "o4/manifest.parquet")
    m = m[m["condition"] != "clean"].sort_values(["n_bytes", "sample_id", "condition"], ascending=[False, True, True])
    # Longest-processing-time assignment by bytes, deterministic.
    load, mine = [0] * args.n_shards, []
    for row in m.itertuples(index=False):
        s = min(range(args.n_shards), key=lambda i: (load[i], i))
        load[s] += row.n_bytes
        if s == args.shard:
            mine.append(row)
    backend = PatcherBackend(root, "swa512", "float32", args.device)
    if backend.config_max_patch_length is not None:
        raise SystemExit("primary run used no length cap; checkpoint config now sets one")
    done = skipped = 0
    for i, row in enumerate(mine, 1):
        run_id = o4_run_id(row.condition)
        if artifacts_done(root, run_id, row.sample_id, int(row.n_bytes)):
            skipped += 1
            continue
        content = (root / f"o4/{row.condition}/text/{row.sample_id}.bin").read_bytes()
        if len(content) != row.n_bytes or hashlib.sha256(content).hexdigest() != row.sha256:
            raise SystemExit(f"{row.condition}/{row.sample_id}: bytes differ from o4/manifest.parquet")
        process_file(root, run_id, backend, row.sample_id, content, backend.tau, None)
        done += 1
        if i % 500 == 0 or i == len(mine):
            print(f"[{i}/{len(mine)}] done={done} skipped={skipped}", flush=True)
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} files, done={done}, skipped={skipped}")


if __name__ == "__main__":
    main()
