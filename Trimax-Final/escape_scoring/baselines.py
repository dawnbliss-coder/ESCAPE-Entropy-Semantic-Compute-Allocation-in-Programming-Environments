"""Deterministic structure-blind baselines scored exactly like BLT (proposal R2, P3).

Each baseline proposes interior boundaries from the raw bytes alone:
  newline        offset right after every b"\\n" (stream_b/whitespace_extractor.py)
  indent_change  first non-blank byte of a line whose indentation differs from the
                 previous non-blank line's (same module)
  word_start     byte i is [A-Za-z0-9_] and byte i-1 is not (escape_eval/baselines.py)
Targets, the frozen per-language k, one-to-one matching and pooling are those of
the primary score. H1 is only interesting where BLT beats these, not just H0.
The interval is a paired file bootstrap of BLT F1 minus baseline F1.
"""
from __future__ import annotations

import numpy as np

from .calibration import verify_frozen
from .contract import write_json
from .fast import as_sorted, counts_fast
from .metrics import metrics, predictions, rng_for, target_nodes
from .parallel import ordered_map

BASELINES = ("newline", "indent_change", "word_start")
_WORD = np.zeros(256, dtype=bool)
for _c in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_":
    _WORD[_c] = True


def newline(content: bytes):
    return [i + 1 for i, b in enumerate(content) if b == 0x0A and i + 1 < len(content)]


def indent_change(content: bytes):
    offsets, prev, pos = [], 0, 0
    for line in content.split(b"\n"):
        if line.rstrip(b" \t\r"):
            indent = len(line) - len(line.lstrip(b" \t"))
            if indent != prev:
                offsets.append(pos + indent)
            prev = indent
        pos += len(line) + 1
    return offsets


def word_start(content: bytes):
    if not content:
        return []
    w = _WORD[np.frombuffer(content, dtype=np.uint8)]
    starts = np.flatnonzero(w[1:] & ~w[:-1]) + 1
    return starts.tolist()


def _interior(offsets, n):
    return as_sorted([o for o in offsets if 0 < o < n])


def _file_counts(dataset, sample_id, k):
    sample = dataset.load(sample_id)
    n = sample.manifest["byte_length"]
    proposals = {"blt": as_sorted(predictions(sample))}
    for name in BASELINES:
        proposals[name] = _interior(globals()[name](sample.text), n)
    out = {}
    for kind in ("start", "end"):
        targets = as_sorted(target_nodes(sample, kind))
        out[kind] = {name: counts_fast(pred, targets, k) for name, pred in proposals.items()}
    return sample.manifest["language"], out


def score_baselines(dataset, record, output, *, bootstrap=2000, workers=1):
    verify_frozen(dataset, record, workers=workers)
    files = [m for m in record["split_manifest"] if m["split"] == "test"]
    jobs = [(m["sample_id"], record["selected_k"][m["language"]]) for m in files]
    per_lang = {}
    for language, out in ordered_map(_file_counts, jobs, dataset, workers):
        for lang in (language, "all"):
            for kind, by_name in out.items():
                bucket = per_lang.setdefault((lang, kind), {})
                for name, c in by_name.items():
                    bucket.setdefault(name, []).append(c)
    rows = []
    for (language, kind), bucket in sorted(per_lang.items()):
        blt = np.asarray(bucket["blt"])
        for name in ("blt", *BASELINES):
            c = np.asarray(bucket[name])
            m = metrics(c.sum(axis=0))
            row = {"language": language, "kind": kind, "proposer": name, "n_files": len(c),
                   "n_predictions": int(c[:, 0].sum() + c[:, 1].sum()),
                   "tp": int(c[:, 0].sum()), "fp": int(c[:, 1].sum()), "fn": int(c[:, 2].sum()),
                   **{key: float(v) for key, v in m.items()}}
            if name != "blt":
                rng = rng_for(record["seed"], f"{language}\0{kind}\0{name}", "baseline_bootstrap")
                diffs = []
                for _ in range(bootstrap):
                    idx = rng.integers(0, len(c), len(c))
                    diffs.append(float(metrics(blt[idx].sum(axis=0))["f1"] - metrics(c[idx].sum(axis=0))["f1"]))
                row["blt_minus_baseline_f1"] = float(metrics(blt.sum(axis=0))["f1"] - m["f1"])
                row["ci_low"], row["ci_high"] = (float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))) \
                    if len(c) >= 2 else (None, None)
            rows.append(row)
    result = {"schema_version": "1.0.0", "synthetic": record["synthetic"], "selected_k": record["selected_k"],
              "calibration_record_sha256": record["record_sha256"], "bootstrap": bootstrap,
              "definitions": __doc__, "rows": rows}
    write_json(output, result, exclusive=True)
    return result
