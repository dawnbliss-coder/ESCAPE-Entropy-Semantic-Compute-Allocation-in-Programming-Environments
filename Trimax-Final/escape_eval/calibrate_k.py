"""Tolerance k calibration, per code language, on the calib split ONLY (PRELIMINARY).

The method is fixed before any main-split scoring (docs/STREAM_C_METHODS.md §4). For each
language and each k in the grid (default 0..8):
  F1_BLT(k)    pooled F1 of BLT entropy-triggered boundaries against AST start targets
  E[F1_H0(k)]  mean pooled F1 over R_cal density-matched H0 resamples (same sampler as scoring)
  margin(k)    F1_BLT(k) - E[F1_H0(k)]
k* = argmax margin(k); ties go to the smallest k.

Any file whose manifest split is not 'calib' is refused. With --allow-non-calib (smoke tests
only) the output goes under results/smoke/ and is marked SMOKE, and run_eval will not accept
it for non-smoke scoring.

Usage:
  python -m escape_eval.calibrate_k --run-id RUN --source corpus --domains py cpp
  python -m escape_eval.calibrate_k --run-id RUN --source golden --allow-non-calib   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_eval import data as D  # noqa: E402
from escape_eval import results_io as RIO  # noqa: E402
from escape_eval import stats as S  # noqa: E402
from escape_eval.engine import METHOD_BLT, EngineConfig, count_sets, run_engine  # noqa: E402

DEFAULT_GRID = tuple(range(0, 9))
DEFAULT_SEED = 20260915
METHOD_TEXT = ("argmax over k of pooled F1(BLT entropy-triggered vs AST start targets) minus the mean pooled F1 "
               "of density-matched H0 resamples; ties -> smallest k; calib split only")


def calibrate(root, stream_a_root, run_id: str, source: str, domains, grid=DEFAULT_GRID, n_resamples: int = 1000,
              seed: int = DEFAULT_SEED, allow_non_calib: bool = False, limit_per_domain=None,
              results_root="results", log=print) -> dict:
    root = Path(root)
    stream_a_root = Path(stream_a_root)
    entry = D.load_run_entry(stream_a_root, run_id)
    split_filter = None if (allow_non_calib or source == "golden") else "calib"
    table = D.build_file_table(root, source, domains, split=split_filter, limit_per_domain=limit_per_domain)
    splits = D.manifest_splits(root, table["file_id"])
    not_calib = sorted(f for f, s in splits.items() if s != "calib")
    if not_calib and not allow_non_calib:
        raise SystemExit(f"refusing to calibrate k on {len(not_calib)} non-calib files, e.g. {not_calib[:5]}")
    smoke = bool(allow_non_calib)

    curve_rows, selected, file_ids = [], {}, {}
    for domain in domains:
        files = table[table["domain"] == domain].reset_index(drop=True)
        if files.empty:
            log(f"{domain}: no files - skipped")
            continue
        missing = [f for f in files["file_id"] if not D.stream_a_outputs_exist(stream_a_root, run_id, f)]
        if missing:
            raise SystemExit(f"{domain}: {len(missing)} files lack Stream A outputs for {run_id}, e.g. {missing[:5]}")
        cfg = EngineConfig(ks=tuple(int(k) for k in grid), kinds=("start",), axes=("all",),
                           n_resamples=n_resamples, master_seed=seed, iou=False)
        log(f"{domain}: calibrating on {len(files)} files, grid={list(grid)}, R={n_resamples}")
        res = run_engine(files, stream_a_root, run_id, entry, {domain: D.load_taxonomy(root, domain)}, cfg, log=log)
        if res.problems:
            raise SystemExit(f"{domain}: engine problems: {res.problems[:5]}")
        ids = files["file_id"].tolist()
        file_ids[domain] = ids
        keys = [("all", "all")]
        for k in grid:
            cs = count_sets(res.counts, ids, "start", k, keys, (METHOD_BLT,))[METHOD_BLT]
            obs = cs.pooled()
            null = res.null.null_metrics("start", k, keys, cs.n_pred.sum(), cs.n_targets.sum(0))
            f1_null = null["f1"][:, 0]
            curve_rows.append({
                "domain": domain, "k": int(k), "n_files": len(ids), "n_pred": int(cs.n_pred.sum()),
                "n_targets": int(cs.n_targets.sum()),
                "precision_blt": float(obs["precision"][0]), "recall_blt": float(obs["recall"][0]),
                "f1_blt": float(obs["f1"][0]),
                "precision_h0_mean": float(S.nanmean_quiet(null["precision"][:, 0])),
                "recall_h0_mean": float(S.nanmean_quiet(null["recall"][:, 0])),
                "f1_h0_mean": float(S.nanmean_quiet(f1_null)),
                "f1_h0_sd": float(np.nanstd(f1_null, ddof=1)) if f1_null.size > 1 else float("nan"),
                "margin_f1": float(obs["f1"][0] - S.nanmean_quiet(f1_null)),
            })
        dom = pd.DataFrame([r for r in curve_rows if r["domain"] == domain])
        best = dom["margin_f1"].max()
        k_star = int(dom.loc[dom["margin_f1"] >= best - 1e-12, "k"].min())
        row = dom[dom["k"] == k_star].iloc[0]
        selected[domain] = {"k": k_star, "margin_f1": float(row["margin_f1"]), "f1_blt": float(row["f1_blt"]),
                            "f1_h0_mean": float(row["f1_h0_mean"])}
        log(f"{domain}: k* = {k_star} (margin {row['margin_f1']:.4f}: F1_BLT {row['f1_blt']:.4f} vs "
            f"E[F1_H0] {row['f1_h0_mean']:.4f}; {res.seconds:.1f}s)")

    out_root = RIO.effective_results_root(results_root, smoke)
    out_dir = out_root / "k_calibration"
    curve = pd.DataFrame(curve_rows)
    RIO.write_table(curve, out_dir, "curve")
    record = {
        "selected_k": selected,
        "method": METHOD_TEXT,
        "grid": [int(k) for k in grid],
        "n_resamples": int(n_resamples),
        "master_seed": int(seed),
        "run_id": run_id,
        "run_entry": {k: entry.get(k) for k in ("tau", "sliding_window", "chunk_len", "max_patch_length",
                                                 "context_mode", "dtype", "checkpoint", "checkpoint_revision")},
        "source": source,
        "file_ids": file_ids,
        "file_ids_sha256": {d: D.file_ids_sha256(v) for d, v in file_ids.items()},
        "manifest_sha256": D.manifest_sha256(root),
        "created": RIO.utc_now_iso(),
        "smoke": smoke,
        "preliminary": True,
        **RIO.git_info(root),
        "packages": RIO.package_versions(),
    }
    RIO.write_json(record, out_dir / "selected_k.json")
    RIO.register_analysis(out_root, "k_calibration", out_dir, "k_calibration", smoke=smoke,
                          extra={"run_id": run_id, "selected_k": {d: v["k"] for d, v in selected.items()}})
    log(f"wrote {out_dir / 'selected_k.json'}" + (" (SMOKE)" if smoke else ""))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate tolerance k per language on the calib split")
    ap.add_argument("--root", default=".")
    ap.add_argument("--stream-a-root", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", choices=D.SOURCES, default="corpus")
    ap.add_argument("--domains", nargs="+", default=["py", "cpp"])
    ap.add_argument("--grid", nargs="+", type=int, default=list(DEFAULT_GRID))
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--allow-non-calib", action="store_true", help="SMOKE ONLY: accept non-calib files")
    ap.add_argument("--limit-per-domain", type=int)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args(argv)
    if args.source == "golden" and not args.allow_non_calib:
        raise SystemExit("golden fixtures are not a calibration set: add --allow-non-calib for a SMOKE run")
    root = Path(args.root).resolve()
    calibrate(root, Path(args.stream_a_root).resolve() if args.stream_a_root else root, args.run_id, args.source,
              args.domains, grid=args.grid, n_resamples=args.n_resamples, seed=args.seed,
              allow_non_calib=args.allow_non_calib, limit_per_domain=args.limit_per_domain,
              results_root=root / args.results_root if not Path(args.results_root).is_absolute() else args.results_root)


if __name__ == "__main__":
    main()
