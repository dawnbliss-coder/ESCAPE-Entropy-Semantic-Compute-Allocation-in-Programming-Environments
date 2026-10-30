"""Subset and clustering re-analyses of a frozen C-v1 evaluation, from per_file.csv alone.

Exact identities used (no model or null is re-run):
  per file, the null keeps the prediction count P = tp+fp and target count G = tp+fn, so
  each permutation's F1 is 2T/(P+G) and the stored per-file mean null F1 gives the
  expected null match count E[T] = mean_h0_f1 * (P+G)/2. Pooled over any file subset,
  the null F1 mean is 2*sum(E[T]) / sum(P+G) -- the same quantity the frozen scorer
  reports -- and the margin's file bootstrap is the frozen scorer's paired bootstrap.

Analyses (plan.md section 2):
  cpp_parse      C++ files with tree-sitter error recovery (parse_ok False) vs clean parses
  long_files     files over 8,191 bytes (chunk-reset positions with truncated context) vs the rest
  py_repo_boot   Python margin CI from a repository-clustered bootstrap (resample GitHub
                 repositories, all their files) next to the frozen file bootstrap

Usage:
  python scripts/final_analysis/subset_scores.py --eval results/final/c-v1-2721954/evaluation \
      --out results/final/analysis/subsets_teacher.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20260927
CHUNK = 8191


def rng_for(*parts):
    key = hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "big")))


def load(eval_dir):
    pf = pd.read_csv(Path(eval_dir) / "per_file.csv")
    pf = pf[pf["kind"] == "start"].copy()
    man = pd.read_parquet(REPO_ROOT / "corpus/manifest.parquet").set_index("file_id")
    pf["n_bytes"] = man.loc[pf["sample_id"], "n_bytes"].to_numpy()
    pf["parse_ok"] = man.loc[pf["sample_id"], "parse_ok"].to_numpy()
    src = man.loc[pf["sample_id"], "source_path"].astype(str).to_numpy()
    pf["repo"] = [m.group(1) if (m := re.match(r"https://github\.com/([^/]+/[^/]+)/", s)) else s for s in src]
    P, G = pf["tp"] + pf["fp"], pf["tp"] + pf["fn"]
    pf["den"] = P + G
    pf["null_tp"] = pf["mean_h0_f1"] * pf["den"] / 2
    return pf


def pooled(df):
    den = df["den"].sum()
    f1, h0 = 2 * df["tp"].sum() / den, 2 * df["null_tp"].sum() / den
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    return {"n_files": int(len(df)), "precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)),
            "f1": float(f1), "f1_h0_mean": float(h0), "f1_margin": float(f1 - h0)}


def boot(df, key, boot_n, cluster=None):
    rng = rng_for(key, "subset_bootstrap" if cluster is None else "cluster_bootstrap")
    if cluster is None:
        tp, ntp, den = (df[c].to_numpy(float) for c in ("tp", "null_tp", "den"))
        idx = rng.integers(0, len(df), (boot_n, len(df)))
        f1, h0 = 2 * tp[idx].sum(1) / den[idx].sum(1), 2 * ntp[idx].sum(1) / den[idx].sum(1)
    else:
        g = df.groupby(cluster)[["tp", "null_tp", "den"]].sum()
        tp, ntp, den = (g[c].to_numpy(float) for c in ("tp", "null_tp", "den"))
        idx = rng.integers(0, len(g), (boot_n, len(g)))
        f1, h0 = 2 * tp[idx].sum(1) / den[idx].sum(1), 2 * ntp[idx].sum(1) / den[idx].sum(1)
    m = f1 - h0
    return {"f1_ci": [float(np.quantile(f1, .025)), float(np.quantile(f1, .975))],
            "f1_margin_ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))]}


def diff_boot(a, b, key, boot_n):
    """Unpaired file bootstrap of margin(a) - margin(b) for disjoint subsets."""
    rng = rng_for(key, "subset_difference")
    out = []
    arrs = [tuple(d[c].to_numpy(float) for c in ("tp", "null_tp", "den")) for d in (a, b)]
    ia = rng.integers(0, len(a), (boot_n, len(a)))
    ib = rng.integers(0, len(b), (boot_n, len(b)))
    for (tp, ntp, den), idx in zip(arrs, (ia, ib)):
        out.append(2 * (tp[idx].sum(1) - ntp[idx].sum(1)) / den[idx].sum(1))
    d = out[0] - out[1]
    return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", required=True, help="evaluation dir holding per_file.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    pf = load(args.eval)
    res = {"eval": args.eval, "boot": args.boot, "seed": SEED, "kind": "start", "analyses": {}}

    def row(df, key, cluster=None):
        return {**pooled(df), **boot(df, key, args.boot, cluster)}

    cpp = pf[pf["language"] == "cpp"]
    rec, ok = cpp[cpp["parse_ok"] == False], cpp[cpp["parse_ok"] == True]  # noqa: E712
    res["analyses"]["cpp_parse"] = {
        "all": row(cpp, "cpp_all"), "error_recovery": row(rec, "cpp_err"), "clean_parse": row(ok, "cpp_ok"),
        "share_error_recovery": float(len(rec) / len(cpp)),
        "margin_difference_err_minus_clean": float(pooled(rec)["f1_margin"] - pooled(ok)["f1_margin"]),
        "margin_difference_ci": diff_boot(rec, ok, "cpp_parse", args.boot)}
    lf = {}
    for lang in sorted(pf["language"].unique()):
        d = pf[pf["language"] == lang]
        long_, short = d[d["n_bytes"] > CHUNK], d[d["n_bytes"] <= CHUNK]
        lf[lang] = {"le_8191": row(short, f"{lang}_short") if len(short) else None,
                    "gt_8191": row(long_, f"{lang}_long") if len(long_) > 1 else
                    ({**pooled(long_)} if len(long_) else None),
                    "share_gt_8191": float(len(long_) / len(d)),
                    "share_bytes_gt_8191": float(d.loc[d["n_bytes"] > CHUNK, "n_bytes"].sum() / d["n_bytes"].sum())}
        if len(long_) > 1 and len(short) > 1:
            lf[lang]["margin_difference_long_minus_short"] = float(pooled(long_)["f1_margin"] - pooled(short)["f1_margin"])
            lf[lang]["margin_difference_ci"] = diff_boot(long_, short, f"{lang}_long_short", args.boot)
    res["analyses"]["long_files"] = lf
    py = pf[pf["language"] == "python"]
    res["analyses"]["py_repo_boot"] = {
        "n_files": int(len(py)), "n_repos": int(py["repo"].nunique()),
        "files_per_repo_median": float(py.groupby("repo").size().median()),
        "files_per_repo_max": int(py.groupby("repo").size().max()),
        **pooled(py),
        "file_bootstrap": boot(py, "py_file", args.boot),
        "repo_cluster_bootstrap": boot(py, "py_repo", args.boot, cluster="repo")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
