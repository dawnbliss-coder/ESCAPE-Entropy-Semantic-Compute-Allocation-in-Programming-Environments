"""Aggregate saved Stream A artifacts into bpp_summary.parquet, one row per (file_id, run_id).

Reads only boundaries/{run_id}/*.parquet, entropies/{run_id}/*.npy, runs.json and the manifest,
so it can be re-run at any time without the model.

Columns:
  - Contract (docs/SCHEMA.md) first: file_id, run_id, n_bytes, n_patches, mean_bpp, var_bpp.
  - Then additive: domain, split, median/min/max/p10/p90 bpp, trigger counts, entropy quantiles,
    frac_bytes_above_tau.
  - BPP counts ALL patches (init, entropy-triggered and length-capped); var_bpp is the population
    variance (ddof=0).

Usage: python -m stream_a.build_bpp_summary [--runs RUN_ID ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_common.io import atomic_to_parquet  # noqa: E402
from escape_common.schema import BPP_SUMMARY_CONTRACT_COLUMNS, BPP_SUMMARY_PATH, MANIFEST_PATH, RUNS_JSON_PATH  # noqa: E402
from stream_a.patching import bpp_stats, entropy_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def summarize_run(root: Path, run_id: str, tau: float, manifest: pd.DataFrame) -> list:
    meta = manifest.set_index("file_id")
    rows = []
    for bp in sorted((root / "boundaries" / run_id).glob("*.parquet")):
        fid = bp.stem
        ep = root / "entropies" / run_id / f"{fid}.npy"
        if not ep.exists():
            continue  # boundaries are written last, so this only happens after manual deletion
        b = pd.read_parquet(bp)
        ent = np.load(ep)
        n_bytes = int(ent.shape[0])
        row = {"file_id": fid, "run_id": run_id, "n_bytes": n_bytes, "n_patches": int(len(b))}
        row.update(bpp_stats(b["patch_length"].to_numpy()))
        trig = b["trigger"].value_counts()
        row.update({"domain": meta.at[fid, "domain"] if fid in meta.index else None,
                    "split": meta.at[fid, "split"] if fid in meta.index else None,
                    "n_init": int(trig.get("init", 0)), "n_entropy": int(trig.get("entropy", 0)),
                    "n_length_cap": int(trig.get("length_cap", 0))})
        row.update(entropy_stats(ent, tau))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--runs", nargs="*", help="run ids (default: every run in runs.json)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    runs = json.loads((root / RUNS_JSON_PATH).read_text())
    run_ids = args.runs or sorted(runs)
    manifest = pd.read_parquet(root / MANIFEST_PATH)
    rows = []
    for run_id in run_ids:
        run_rows = summarize_run(root, run_id, float(runs[run_id]["tau"]), manifest)
        print(f"{run_id}: {len(run_rows)} files")
        rows.extend(run_rows)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=BPP_SUMMARY_CONTRACT_COLUMNS)
    extra = [c for c in df.columns if c not in BPP_SUMMARY_CONTRACT_COLUMNS]
    df = df[BPP_SUMMARY_CONTRACT_COLUMNS + extra].sort_values(["run_id", "file_id"]).reset_index(drop=True)
    atomic_to_parquet(df, root / BPP_SUMMARY_PATH)
    print(f"wrote {BPP_SUMMARY_PATH}: {len(df)} rows")
    if not df.empty:
        print(df.groupby(["run_id", "domain"])[["n_bytes", "mean_bpp", "var_bpp", "ent_mean"]].median().to_string())


if __name__ == "__main__":
    main()
