"""Micro-aggregated permutation scores and paired file-bootstrap intervals."""
from __future__ import annotations

import csv
import platform
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from .calibration import model_signature, verify_frozen
from .contract import digest, write_json
from .fast import FixedEofNull, as_sorted, counts_fast
from .metrics import diagnostics, metrics, predictions, rng_for, target_nodes
from .parallel import ordered_map


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary(observed, null, file_observed, file_mean_null, seed, key, bootstrap):
    obs_metrics, null_metrics = metrics(observed), metrics(null)
    result = {"tp": int(observed[0]), "fp": int(observed[1]), "fn": int(observed[2])}
    rng = rng_for(seed, str(key), "file_bootstrap")
    boot = {name: [] for name in ("precision", "recall", "f1", "f1_margin", "f1_ratio")}
    for _ in range(bootstrap):
        indices = rng.integers(0, len(file_observed), len(file_observed))
        om = metrics(file_observed[indices].sum(axis=0))
        nm = metrics(file_mean_null[indices].sum(axis=0))
        for name in ("precision", "recall", "f1"):
            boot[name].append(float(om[name]))
        boot["f1_margin"].append(float(om["f1"] - nm["f1"]))
        if nm["f1"] > 0:
            boot["f1_ratio"].append(float(om["f1"] / nm["f1"]))
    for name, val in obs_metrics.items():
        values = null_metrics[name]
        mean = float(np.mean(values))
        result.update({name: float(val), f"{name}_h0_mean": mean,
                       f"{name}_h0_sd": float(np.std(values, ddof=1)),
                       f"{name}_h0_p025": float(np.quantile(values, .025)),
                       f"{name}_h0_p975": float(np.quantile(values, .975)),
                       f"{name}_permutation_p": float((1 + np.sum(values >= val)) / (len(values) + 1)),
                       f"{name}_margin": float(val - mean),
                       f"{name}_ratio": float(val / mean) if mean > 0 else None})
    for name, values in boot.items():
        # Do not report a conditional ratio CI after discarding zero denominators.
        valid = len(file_observed) >= 2 and len(values) == bootstrap and bootstrap >= 2
        result[f"{name}_ci_low"] = float(np.quantile(values, .025)) if valid else None
        result[f"{name}_ci_high"] = float(np.quantile(values, .975)) if valid else None
    result["ci_method"] = "paired file percentile bootstrap" if len(file_observed) >= 2 else "unavailable: fewer than two files"
    return result


def _score_file(dataset, manifest, k, strata, permutations, seed, signature):
    sample = dataset.load(manifest["sample_id"])
    if model_signature(sample) != signature:
        raise ValueError(f"{sample.sample_id}: model configuration changed")
    input_hash = digest({"patcher": sample.patcher, "nodes": sample.nodes})
    pred = as_sorted(predictions(sample))
    file_sets = []
    for kind in ("start", "end"):
        nodes = target_nodes(sample, kind)
        common = [("global", "all", list(nodes)),
                  ("language", manifest["language"], list(nodes)),
                  ("domain", manifest["domain"], list(nodes))]
        for axis, field in (("ast_depth", "depth"), ("node_type", "node_type")):
            for value in sorted(strata[axis]):
                common.append((axis, value, [i for i, r in nodes.items() if str(r[field]) == value]))
        # Include language in depth/type label; raw depth is not comparable
        # across parsers. No silent cross-language depth normalization.
        for axis, value, targets in common:
            group_lang = manifest["language"] if axis in {"ast_depth", "node_type"} else "all"
            key = (kind, axis, group_lang, value)
            targets = as_sorted(targets)
            file_sets.append((key, targets, counts_fast(pred, targets, k), np.zeros((permutations, 3))))
    # Null counts depend only on the target set; compute each distinct set once.
    distinct = {}
    for _, targets, _, _ in file_sets:
        distinct.setdefault(targets.tobytes(), targets)
    null_sets = {sig: np.zeros((permutations, 3)) for sig in distinct}
    sampler = FixedEofNull(sample.patcher["patches"], manifest["byte_length"])
    rng = rng_for(seed, sample.sample_id, "test_permutation")
    for rep in range(permutations):
        drawn = sampler.draw(rng)
        for sig, targets in distinct.items():
            null_sets[sig][rep] = counts_fast(drawn, targets, k)
    file_sets = [(key, observed, null_sets[targets.tobytes()]) for key, targets, observed, _ in file_sets]
    return sample.sample_id, input_hash, file_sets, diagnostics(sample)


def _score_chunk(dataset, chunk, selected_k, strata, permutations, seed, signature):
    """Score a contiguous run of files; return pooled null sums and per-file summaries."""
    null_sums, files = {}, []
    for manifest in chunk:
        k = selected_k[manifest["language"]]
        sid, input_hash, file_sets, diag = _score_file(
            dataset, manifest, k, strata[manifest["language"]], permutations, seed, signature)
        summary = []
        for key, observed, null in file_sets:
            if key in null_sums:
                null_sums[key] += null
            else:
                null_sums[key] = null.copy()
            per_file = None
            if key[1] == "global":
                per_file = {"sample_id": sid, "language": manifest["language"],
                            "domain": manifest["domain"], "kind": key[0], "k": k,
                            "tp": int(observed[0]), "fp": int(observed[1]), "fn": int(observed[2]),
                            **{name: float(v) for name, v in metrics(observed).items()},
                            "mean_h0_f1": float(metrics(null)["f1"].mean())}
            summary.append((key, observed, null.mean(axis=0), per_file))
        files.append((sid, input_hash, diag, summary))
    return null_sums, files


def evaluate(dataset, record, output, *, permutations=10000, bootstrap=2000, seed=None, workers=1):
    """Final test evaluation: no k argument, no calibration or override path."""
    verify_frozen(dataset, record, workers=workers)
    if permutations < 2 or bootstrap < 2:
        raise ValueError("permutations and bootstrap must be >= 2")
    if record["eof_policy"] != "fixed_eof":
        raise ValueError("final scoring requires approved fixed_eof density-matched null")
    seed = record["seed"] if seed is None else seed
    files = [m for m in record["split_manifest"] if m["split"] == "test"]
    if not files:
        raise ValueError("empty test split")
    out = Path(output)
    if (out / "scores.json").exists():
        raise FileExistsError("refusing to overwrite existing scores.json")
    # Predeclare strata over the evaluation files, including zero-target files
    # in their denominators. This does not select or modify tolerance k.
    strata = {}
    for manifest in files:
        language = manifest["language"]
        by_axis = strata.setdefault(language, {"ast_depth": set(), "node_type": set()})
        sample = SimpleNamespace(manifest=manifest, nodes=dataset.rows("ground_truth_nodes", manifest["sample_id"]))
        for kind in ("start", "end"):
            for row in target_nodes(sample, kind).values():
                by_axis["ast_depth"].add(str(row["depth"]))
                by_axis["node_type"].add(row["node_type"])
    # Contiguous chunks, merged in order: per-file lists keep the serial file order,
    # and pooled null sums are sums of integer counts, hence exact in any grouping.
    n_chunks = 1 if workers <= 1 else min(len(files), 16 * workers)
    bounds = np.linspace(0, len(files), n_chunks + 1).astype(int)
    jobs = [(files[a:b], record["selected_k"], strata, permutations, seed, record["model_signature"])
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    groups, per_file, diag, input_hashes = {}, [], [], {}
    for null_sums, chunk_files in ordered_map(_score_chunk, jobs, dataset, workers):
        for key, null in null_sums.items():
            g = groups.setdefault(key, {"obs": np.zeros(3), "null": np.zeros((permutations, 3)),
                                        "file_obs": [], "file_null": []})
            g["null"] += null
        for sid, input_hash, file_diag, summary in chunk_files:
            input_hashes[sid] = input_hash
            diag.append(file_diag)
            for key, observed, null_mean, file_row in summary:
                g = groups[key]
                g["obs"] += observed
                g["file_obs"].append(observed)
                g["file_null"].append(null_mean)
                if file_row is not None:
                    per_file.append(file_row)
    rows = []
    for key, g in sorted(groups.items()):
        rows.append({"kind": key[0], "axis": key[1], "language": key[2], "stratum": key[3],
                     "n_files": len(g["file_obs"]),
                     **_summary(g["obs"], g["null"], np.asarray(g["file_obs"]),
                                np.asarray(g["file_null"]), seed, key, bootstrap)})
    result = {"schema_version": "1.0.0", "synthetic": record["synthetic"],
              "status": "synthetic demonstration" if record["synthetic"] else "preliminary; no automatic hypothesis claim",
              "configuration": {"seed": seed, "permutations": permutations, "bootstrap": bootstrap,
                                "rng": "NumPy PCG64 seeded by SHA256(seed, sample_id, purpose)",
                                "eof_policy": "fixed_eof", "selected_k": record["selected_k"],
                                "aggregation": "micro pooled one-to-one counts; file-bootstrap",
                                "calibration_record_sha256": record["record_sha256"],
                                "test_input_sha256": input_hashes,
                                "python": platform.python_version(), "numpy": np.__version__},
              "scores": rows, "per_file": per_file, "diagnostics": diag}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "scores.json").exists():
        raise FileExistsError("refusing to overwrite existing scores.json")
    write_json(out / "scores.json", result, exclusive=True)
    write_csv(out / "scores.csv", rows)
    write_csv(out / "per_file.csv", per_file)
    write_csv(out / "diagnostics.csv", diag)
    return result
