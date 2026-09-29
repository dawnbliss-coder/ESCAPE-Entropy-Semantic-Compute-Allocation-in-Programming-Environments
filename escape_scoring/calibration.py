"""File-level split and immutable calibration-only tolerance selection."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np

from .contract import digest, write_json
from .fast import FixedEofNull, as_sorted, counts_fast, f1_of
from .metrics import metrics, predictions, rng_for, target_nodes
from .parallel import ordered_map

SEED = 20260927
CANDIDATES = [0, 1, 2, 4, 8]
CRITERION = "maximize median per-file (F1_observed - mean_F1_H0), calibration only; ties smallest k"


def split_manifest(manifest, seed=SEED):
    ids = [m["sample_id"] for m in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id")
    result = []
    for language in sorted({m["language"] for m in manifest}):
        rows = [dict(m) for m in manifest if m["language"] == language]
        if len(rows) < 5:
            raise ValueError(f"{language}: at least five files required for a 20/80 split")
        # Same bytes never cross splits; reject duplicates rather than silently
        # making a byte-level or approximate grouped split.
        hashes = [m["text_sha256"] for m in rows]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"{language}: duplicate source bytes; deduplicate before splitting")
        rows.sort(key=lambda m: (hashlib.sha256(
            f"{seed}\0{language}\0{m['sample_id']}".encode()).hexdigest(), m["sample_id"]))
        n_cal = math.ceil(len(rows) / 5)
        for i, row in enumerate(rows):
            row["split"] = "calibration" if i < n_cal else "test"
            result.append(row)
    return sorted(result, key=lambda m: m["sample_id"])


def model_signature(sample):
    p = sample.patcher
    return {key: p[key] for key in ("model", "threshold", "entropy_definition", "entropy_unit")}


def _calibration_file(dataset, sample_id, byte_length, seed, permutations):
    sample = dataset.load(sample_id)
    targets = as_sorted(target_nodes(sample))
    pred = as_sorted(predictions(sample))
    obs = {k: float(metrics(counts_fast(pred, targets, k))["f1"]) for k in CANDIDATES}
    null = FixedEofNull(sample.patcher["patches"], byte_length)
    rng = rng_for(seed, sample_id, "calibration")
    per_rep = {k: np.empty(permutations) for k in CANDIDATES}
    for rep in range(permutations):
        drawn = null.draw(rng)
        for k in CANDIDATES:
            per_rep[k][rep] = f1_of(counts_fast(drawn, targets, k))
    # Plain running sum in permutation order, as the reference. Not sum(): since
    # Python 3.12 it compensates float rounding and so gives different last digits.
    null_sum = {}
    for k in CANDIDATES:
        total = 0.0
        for value in per_rep[k].tolist():
            total += value
        null_sum[k] = total
    return (sample_id, digest({"patcher": sample.patcher, "nodes": sample.nodes}),
            model_signature(sample), obs, null_sum)


def calibrate(dataset, output, *, seed=SEED, permutations=10000, eof_policy="fixed_eof", workers=1):
    if Path(output).exists():
        raise FileExistsError("refusing to overwrite frozen calibration")
    if eof_policy != "fixed_eof":
        raise ValueError("calibration requires approved fixed_eof null")
    if permutations < 2:
        raise ValueError("at least two permutations required")
    manifest = split_manifest(dataset.manifest, seed)
    # If the supplied manifest declares assignments they must agree, rather than
    # being treated as instructions to keep an old, incompatible 100/7900 split.
    original = {m["sample_id"]: m["split"] for m in dataset.manifest}
    if any(original[m["sample_id"]] not in {"unassigned", m["split"]} for m in manifest):
        raise ValueError("supplied splits disagree with deterministic 20/80 split")
    selected, curves, fingerprints, configs = {}, [], {}, []
    for language in sorted({m["language"] for m in manifest}):
        margins = {k: [] for k in CANDIDATES}
        rows = [row for row in manifest if row["language"] == language and row["split"] == "calibration"]
        jobs = [(row["sample_id"], row["byte_length"], seed, permutations) for row in rows]
        for sample_id, fingerprint, signature, obs, null_sum in ordered_map(_calibration_file, jobs, dataset, workers):
            fingerprints[sample_id] = fingerprint
            if configs and signature != configs[0]:
                raise ValueError("mixed model configurations in calibration")
            configs.append(signature)
            for k in CANDIDATES:
                mean = null_sum[k] / permutations
                margins[k].append(obs[k] - mean)
                curves.append({"sample_id": sample_id, "language": language, "k": k,
                               "observed_f1": obs[k], "mean_h0_f1": mean, "margin": obs[k] - mean})
        medians = {k: float(np.median(margins[k])) for k in CANDIDATES}
        best = max(medians.values())
        # Floating-point ties within 1e-12; persisted as part of protocol.
        selected[language] = min(k for k in CANDIDATES if medians[k] >= best - 1e-12)
    record = {"schema_version": "1.0.0", "seed": seed, "candidates": CANDIDATES,
              "criterion": CRITERION, "tie_tolerance": 1e-12, "selected_k": selected,
              "split_rule": "per-language SHA256 rank; ceil(n/5) calibration; minimum 5 files",
              "split_manifest": manifest, "input_manifest_sha256": digest(sorted(dataset.manifest, key=lambda m: m["sample_id"])),
              "calibration_input_sha256": fingerprints, "model_signature": configs[0],
              "permutations": permutations, "eof_policy": eof_policy, "curves": curves,
              "synthetic": all(m["synthetic"] for m in manifest)}
    record["record_sha256"] = digest(record)
    write_json(output, record, exclusive=True)
    return record


def _fingerprint(dataset, sample_id):
    sample = dataset.load(sample_id)
    return sample_id, digest({"patcher": sample.patcher, "nodes": sample.nodes})


def verify_frozen(dataset, record, workers=1):
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    if digest(payload) != record.get("record_sha256"):
        raise ValueError("calibration record changed; refusing test scoring")
    if record["candidates"] != CANDIDATES or record["criterion"] != CRITERION:
        raise ValueError("incompatible calibration protocol")
    if digest(sorted(dataset.manifest, key=lambda m: m["sample_id"])) != record["input_manifest_sha256"]:
        raise ValueError("manifest changed since calibration")
    if split_manifest(dataset.manifest, record["seed"]) != record["split_manifest"]:
        raise ValueError("split manifest changed")
    for language, k in record["selected_k"].items():
        if k not in CANDIDATES:
            raise ValueError(f"invalid frozen tolerance for {language}")
    expected = record["calibration_input_sha256"]
    for sid, actual in ordered_map(_fingerprint, [(sid,) for sid in expected], dataset, workers):
        if actual != expected[sid]:
            raise ValueError(f"{sid}: calibration inputs changed")
