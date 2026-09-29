"""Objective 4 scoring: robustness of start alignment and BPP to code noise.

For every O4 file and condition (clean or one noise type), score interior entropy
boundaries against the condition's mapped clean-structure starts with the frozen
per-language k and one-to-one matching (fast.counts_fast), and compute the mean F1 of
the fixed-EOF null (fast.FixedEofNull) over R permutations. Targets follow the primary
rules (depth > 0, non-wrapper, non-empty, 0 < start < n, unique starts).

Pooled per language and condition: P, R, F1, null F1, margin = F1 - null F1, mean BPP.
Robustness is the paired file-bootstrap interval of (noised - clean) for the margin
and for mean BPP, on the same files. Clean-condition patches are the primary run's.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from escape_scoring.contract import WRAPPERS, write_json
from escape_scoring.fast import FixedEofNull, as_sorted, counts_fast, f1_of
from escape_scoring.metrics import metrics, rng_for

PRIMARY_RUN = "swa512-float32-t1.3354-mplnone-91aa6b8e"
DOMAIN = {"python": "py", "cpp": "cpp"}


def _run_id(condition):
    return PRIMARY_RUN if condition == "clean" else f"{PRIMARY_RUN}__o4-{condition}"


def _file(task):
    root, sid, language, condition, n_bytes, k, permutations, seed = task
    root = Path(root)
    nodes = pd.read_parquet(root / f"o4/{condition}/structure/{sid}.parquet")
    nodes = nodes[(nodes["depth"] > 0) & (~nodes["node_type"].str.lower().isin(WRAPPERS))
                  & (nodes["end_byte"] > nodes["start_byte"])
                  & (nodes["start_byte"] > 0) & (nodes["start_byte"] < n_bytes)]
    targets = as_sorted(nodes["start_byte"].tolist())
    b = pd.read_parquet(root / f"boundaries/{_run_id(condition)}/{sid}.parquet").sort_values("byte_offset")
    starts = b["byte_offset"].to_numpy(np.int64)
    triggers = b["trigger"].astype(str).tolist()
    if int(b["patch_length"].sum()) != n_bytes:
        raise ValueError(f"{condition}/{sid}: patches do not tile the noised file")
    ends = np.append(starts[1:], n_bytes)
    reasons = [{"entropy": "entropy", "length_cap": "max_length"}[t] for t in triggers[1:]] + ["eof"]
    patches = [{"start_byte": int(a), "end_byte": int(e), "termination_reason": r} for a, e, r in zip(starts, ends, reasons)]
    pred = as_sorted([int(s) for s, t in zip(starts, triggers) if t == "entropy" and 0 < s < n_bytes])
    observed = counts_fast(pred, targets, k)
    sampler = FixedEofNull(patches, n_bytes)
    rng = rng_for(seed, f"{sid}\0{condition}", "o4_null")
    null = np.array([counts_fast(sampler.draw(rng), targets, k) for _ in range(permutations)])
    return {"sample_id": sid, "language": language, "condition": condition,
            "tp": observed[0], "fp": observed[1], "fn": observed[2],
            "null_mean_counts": null.mean(axis=0).tolist(), "null_mean_f1": float(f1_of(null).mean()),
            "n_patches": len(patches), "n_bytes": n_bytes}


def run(root, calibration_path, output, *, permutations=1000, bootstrap=2000, workers=1):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite O4 results")
    record = json.loads(Path(calibration_path).read_text())
    seed, selected_k = record["seed"], record["selected_k"]
    manifest = pd.read_parquet(root / "o4/manifest.parquet")
    tasks = [(str(root), r.sample_id, r.language, r.condition, int(r.n_bytes), int(selected_k[r.language]),
              permutations, seed) for r in manifest.itertuples(index=False)]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            per_file = list(pool.imap(_file, tasks, chunksize=4))
    else:
        per_file = [_file(t) for t in tasks]
    df = pd.DataFrame(per_file)
    rows = []
    for language in sorted(df["language"].unique()):
        lang = df[df["language"] == language]
        clean = lang[lang["condition"] == "clean"].set_index("sample_id").sort_index()
        for condition in ["clean", *sorted(c for c in lang["condition"].unique() if c != "clean")]:
            cur = lang[lang["condition"] == condition].set_index("sample_id").loc[clean.index]
            obs = cur[["tp", "fp", "fn"]].to_numpy(float)
            nul = np.array(cur["null_mean_counts"].tolist())
            bpp = cur["n_bytes"].to_numpy(float) / cur["n_patches"].to_numpy(float)

            def margin(o, n):
                return float(metrics(o.sum(axis=0))["f1"] - metrics(n.sum(axis=0))["f1"])
            m_obs = metrics(obs.sum(axis=0))
            row = {"language": language, "condition": condition, "n_files": len(cur),
                   "precision": float(m_obs["precision"]), "recall": float(m_obs["recall"]), "f1": float(m_obs["f1"]),
                   "h0_f1": float(metrics(nul.sum(axis=0))["f1"]), "f1_margin": margin(obs, nul),
                   "mean_bpp": float(bpp.mean()), "n_targets": int(obs[:, 0].sum() + obs[:, 2].sum())}
            if condition != "clean":
                c_obs = clean[["tp", "fp", "fn"]].to_numpy(float)
                c_nul = np.array(clean["null_mean_counts"].tolist())
                c_bpp = clean["n_bytes"].to_numpy(float) / clean["n_patches"].to_numpy(float)
                rng = rng_for(seed, f"{language}\0{condition}", "o4_bootstrap")
                d_margin, d_bpp = [], []
                for _ in range(bootstrap):
                    idx = rng.integers(0, len(cur), len(cur))
                    d_margin.append(margin(obs[idx], nul[idx]) - margin(c_obs[idx], c_nul[idx]))
                    d_bpp.append(float(bpp[idx].mean() - c_bpp[idx].mean()))
                row.update({"delta_margin": row["f1_margin"] - margin(c_obs, c_nul),
                            "delta_margin_ci": [float(np.quantile(d_margin, .025)), float(np.quantile(d_margin, .975))],
                            "delta_bpp": float(bpp.mean() - c_bpp.mean()),
                            "delta_bpp_ci": [float(np.quantile(d_bpp, .025)), float(np.quantile(d_bpp, .975))]})
            rows.append(row)
    build = json.loads((root / "o4/build_summary.json").read_text())
    result = {"schema_version": "1.0.0", "synthetic": record["synthetic"], "permutations": permutations,
              "bootstrap": bootstrap, "selected_k": selected_k, "calibration_record_sha256": record["record_sha256"],
              "build": build, "definitions": __doc__, "rows": rows}
    write_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Objective 4 robustness scoring")
    ap.add_argument("--root", default=".")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    res = run(a.root, a.calibration, a.output, permutations=a.permutations, bootstrap=a.bootstrap, workers=a.workers)
    for r in res["rows"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
