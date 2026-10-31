"""Secondary metric: span IoU on the frozen C-v1 test split.

For each unique non-empty constituent span [s, e) of a test file, take the best IoU
with any single patch (escape_eval.matching.best_patch_iou, tested against a naive
reference); report the mean over all spans pooled per language. Compared layouts:
  blt      the observed patches
  h0       fixed-EOF permutations of the file's own (length, reason) pairs, i.e. the
           same null as the primary score, but keeping every patch start
  line     segments starting at 0 and after every newline
  word     segments starting at 0 and at every word start
Spans follow the primary-target rules: depth > 0, not a wrapper type, non-empty.
IoU rewards whole-constituent patches, so with patches far shorter than most
constituents it understates alignment by design; it is secondary to P1.

Reads A/B's on-disk artifacts directly (never regenerates them) and the split from
the frozen calibration record, so it needs no JSONL conversion.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from escape_eval.baselines import whitespace_segment_starts, word_segment_starts
from escape_eval.matching import best_patch_iou
from escape_scoring.baselines import newline
from escape_scoring.contract import WRAPPERS, write_json
from escape_scoring.metrics import rng_for

LANG_DOMAIN = {"python": "py", "cpp": "cpp", "english": "prose"}
LAYOUTS = ("blt", "h0", "line", "word")


def _file(task):
    root, run_id, sid, language, n_bytes, resamples, seed = task
    domain = LANG_DOMAIN[language]
    raw = (Path(root) / f"corpus/{domain}/{sid}.bin").read_bytes()
    if len(raw) != n_bytes:
        raise ValueError(f"{sid}: byte length changed")
    nodes = pd.read_parquet(Path(root) / f"structure/{domain}/{sid}.parquet")
    nodes = nodes[(nodes["depth"] > 0) & (~nodes["node_type"].str.lower().isin(WRAPPERS))
                  & (nodes["end_byte"] > nodes["start_byte"])]
    spans = nodes[["start_byte", "end_byte"]].drop_duplicates()
    s, e = spans["start_byte"].to_numpy(np.int64), spans["end_byte"].to_numpy(np.int64)
    sums = dict.fromkeys(LAYOUTS, 0.0)
    if s.size == 0 or n_bytes == 0:
        return sid, language, 0, sums
    b = pd.read_parquet(Path(root) / f"boundaries/{run_id}/{sid}.parquet").sort_values("byte_offset")
    starts = b["byte_offset"].to_numpy(np.int64)
    lengths = np.diff(np.append(starts, n_bytes))
    sums["blt"] = float(best_patch_iou(s, e, starts, n_bytes).sum())
    # Null layouts: shuffle all patches but the final (EOF) one, as the primary null.
    rng = rng_for(seed, sid, "iou_null")
    head = lengths[:-1]
    total = 0.0
    for _ in range(resamples):
        order = rng.permutation(head.size)
        layout = np.concatenate([[0], np.cumsum(head[order])]).astype(np.int64)
        total += float(best_patch_iou(s, e, layout, n_bytes).sum())
    sums["h0"] = total / resamples
    sums["line"] = float(best_patch_iou(s, e, whitespace_segment_starts(newline(raw), n_bytes), n_bytes).sum())
    sums["word"] = float(best_patch_iou(s, e, word_segment_starts(raw, n_bytes), n_bytes).sum())
    return sid, language, int(s.size), sums


def run(root, run_id, calibration_path, output, *, resamples=100, bootstrap=2000, workers=1):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite IoU results")
    record = json.loads(Path(calibration_path).read_text())
    seed = record["seed"]
    tests = [m for m in record["split_manifest"] if m["split"] == "test"]
    tasks = [(str(root), run_id, m["sample_id"], m["language"], m["byte_length"], resamples, seed) for m in tests]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            per_file = list(pool.imap(_file, tasks, chunksize=8))
    else:
        per_file = [_file(t) for t in tasks]
    rows = []
    for language in sorted({t[3] for t in tasks}):
        files = [(n, sums) for _, lang, n, sums in per_file if lang == language and n > 0]
        n = np.array([f[0] for f in files], dtype=float)
        S = {k: np.array([f[1][k] for f in files]) for k in LAYOUTS}
        mean = {k: float(S[k].sum() / n.sum()) for k in LAYOUTS}
        rng = rng_for(seed, f"{language}\0iou", "file_bootstrap")
        boot = {k: [] for k in ("blt", "blt_minus_h0", "blt_minus_line", "blt_minus_word")}
        for _ in range(bootstrap):
            idx = rng.integers(0, len(n), len(n))
            m = {k: S[k][idx].sum() / n[idx].sum() for k in LAYOUTS}
            boot["blt"].append(m["blt"])
            for other in ("h0", "line", "word"):
                boot[f"blt_minus_{other}"].append(m["blt"] - m[other])
        row = {"language": language, "n_files": len(files), "n_spans": int(n.sum()),
               **{f"mean_best_iou_{k}": v for k, v in mean.items()}}
        for k, values in boot.items():
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = float(np.quantile(values, .025)), float(np.quantile(values, .975))
        for other in ("h0", "line", "word"):
            row[f"blt_minus_{other}"] = mean["blt"] - mean[other]
        rows.append(row)
    result = {"schema_version": "1.0.0", "synthetic": record["synthetic"], "run_id": run_id,
              "calibration_record_sha256": record["record_sha256"], "resamples": resamples,
              "bootstrap": bootstrap, "definitions": __doc__, "rows": rows}
    write_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="span IoU on the frozen C-v1 test split")
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    res = run(a.root, a.run_id, a.calibration, a.output, resamples=a.resamples, bootstrap=a.bootstrap, workers=a.workers)
    for r in res["rows"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
