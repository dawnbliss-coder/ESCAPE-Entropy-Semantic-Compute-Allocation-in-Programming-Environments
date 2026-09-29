"""Ordered maximum-cardinality matching and labeled patch permutation."""
from __future__ import annotations

import hashlib
import numpy as np

from .contract import WRAPPERS


def rng_for(seed, sample_id, purpose):
    key = hashlib.sha256(f"{seed}\0{sample_id}\0{purpose}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "big")))


def match(predictions, targets, k):
    """Greedy ordered interval matching is maximum cardinality.

    Discard an endpoint only when it is too far left to match any remaining
    endpoint. Otherwise pair the leftmost feasible endpoints; an exchange
    argument preserves an optimum. Both sides are sets, never node multisets.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")
    p, t = sorted(set(predictions)), sorted(set(targets))
    i = j = 0
    pairs = []
    while i < len(p) and j < len(t):
        if p[i] < t[j] - k:
            i += 1
        elif t[j] < p[i] - k:
            j += 1
        else:
            pairs.append((p[i], t[j]))
            i += 1
            j += 1
    return pairs


def counts(predictions, targets, k):
    p, t = set(predictions), set(targets)
    tp = len(match(p, t, k))
    return np.array([tp, len(p) - tp, len(t) - tp], dtype=float)


def metrics(c):
    c = np.asarray(c, dtype=float)
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn)}


def target_nodes(sample, kind="start"):
    n = sample.manifest["byte_length"]
    nodes = [r for r in sample.nodes if r["is_primary_alignment_target"]
             and r["node_type"].lower() not in WRAPPERS
             and r["depth"] > 0 and r["end_byte"] > r["start_byte"]
             and 0 < r[f"{kind}_byte"] < n]
    # Attribute duplicate offsets to outermost node, then widest, type, ID.
    nodes.sort(key=lambda r: (r["depth"], -(r["end_byte"] - r["start_byte"]),
                              r["node_type"], r["node_id"]))
    unique = {}
    for row in nodes:
        unique.setdefault(row[f"{kind}_byte"], row)
    return unique


def predictions(sample):
    n = sample.manifest["byte_length"]
    return [r["byte_offset"] for r in sample.patcher["boundaries"]
            if r["termination_reason"] == "entropy" and 0 < r["byte_offset"] < n]


class InvalidNull(ValueError):
    pass


def shuffled_boundaries(patches, n_bytes, rng, eof_policy="fixed_eof"):
    """Shuffle whole (length, reason) pairs, never reassign labels.

    strict: literal complete shuffle; fail if EOF moves inside the file.
    fixed_eof: explicit protocol amendment, final EOF pair remains fixed.
    unrestricted: literal shuffle with interior EOF labels retained as
    non-scoring markers; entropy at file end excluded (not density matched).
    No rejection sampling, retries, label replacement or silent repair.
    """
    if eof_policy not in {"strict", "fixed_eof", "unrestricted"}:
        raise ValueError("unknown EOF policy")
    pairs = [(p["end_byte"] - p["start_byte"], p["termination_reason"]) for p in patches]
    if any(length <= 0 or reason not in {"entropy", "max_length", "eof"}
           for length, reason in pairs) or sum(x[0] for x in pairs) != n_bytes:
        raise InvalidNull("invalid patch lengths/labels/span")
    if pairs and (pairs[-1][1] != "eof" or sum(x[1] == "eof" for x in pairs) != 1):
        raise InvalidNull("observed patches need exactly one final EOF pair")
    count = len(pairs) - (1 if pairs and eof_policy == "fixed_eof" else 0)
    shuffled = [pairs[int(i)] for i in rng.permutation(count)]
    if pairs and eof_policy == "fixed_eof":
        shuffled.append(pairs[-1])
    endpoint = 0
    out = []
    for length, label in shuffled:
        endpoint += length
        if eof_policy == "strict" and ((label == "eof") != (endpoint == n_bytes)):
            raise InvalidNull("complete tuple shuffle moves EOF: choose an explicit team-approved EOF policy")
        if label == "entropy" and 0 < endpoint < n_bytes:
            out.append(endpoint)
    return out


def diagnostics(sample):
    lengths = [p["end_byte"] - p["start_byte"] for p in sample.patcher["patches"]]
    n = sample.manifest["byte_length"]
    entropy = len(predictions(sample))
    capped = sum(r["termination_reason"] == "max_length" for r in sample.patcher["boundaries"])
    return {"sample_id": sample.sample_id, "language": sample.manifest["language"],
            "domain": sample.manifest["domain"], "byte_length": n,
            "entropy_triggered_count": entropy, "capped_boundary_count": capped,
            "entropy_boundary_density": entropy / n if n else None,
            "all_internal_boundary_density": (entropy + capped) / n if n else None,
            "mean_bpp": float(np.mean(lengths)) if lengths else None,
            "median_bpp": float(np.median(lengths)) if lengths else None,
            "bpp_variance": float(np.var(lengths)) if lengths else None,
            "n_patches": len(lengths)}
