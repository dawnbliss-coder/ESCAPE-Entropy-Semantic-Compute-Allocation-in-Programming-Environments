"""Vectorised tolerance matching and span IoU against naive pure-Python references."""

import numpy as np
import pandas as pd
import pytest

from escape_eval.matching import ToleranceScorer, best_patch_iou, build_layout, match_counts
from escape_eval.targets import AXES, build_targets

NODE_TYPES = ["if_statement", "call", "assignment", "return_statement", "function_definition"]
TAXONOMY = {"if_statement": "deterministic_opener", "function_definition": "deterministic_opener",
            "call": "open_ended", "assignment": "open_ended", "return_statement": "open_ended"}


def naive(pred, offsets, cols, n_strata, k):
    tp_p = np.zeros(n_strata, dtype=np.int64)
    tp_r = np.zeros(n_strata, dtype=np.int64)
    for b in pred:
        hit = set()
        for t, c in zip(offsets, cols):
            if abs(int(b) - int(t)) <= k:
                hit |= set(int(x) for x in c)
        for c in hit:
            tp_p[c] += 1
    for t, c in zip(offsets, cols):
        if any(abs(int(b) - int(t)) <= k for b in pred):
            for x in set(int(x) for x in c):
                tp_r[x] += 1
    return tp_p, tp_r


def random_structure(rng, n_bytes, n_rows):
    starts = rng.integers(0, n_bytes, size=n_rows)
    ends = np.minimum(n_bytes, starts + rng.integers(1, 40, size=n_rows))
    return pd.DataFrame({
        "node_type": rng.choice(NODE_TYPES, size=n_rows).astype(object),
        "depth": rng.integers(1, 7, size=n_rows).astype(np.int64),
        "start_byte": starts.astype(np.int64),
        "end_byte": ends.astype(np.int64),
    })


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kind", ["start", "end"])
def test_counts_match_naive_reference(seed, kind):
    rng = np.random.default_rng(seed)
    n_bytes = int(rng.integers(20, 300))
    structure = random_structure(rng, n_bytes, int(rng.integers(0, 40)))
    targets = build_targets(structure, kind, taxonomy=TAXONOMY, exclude_offsets=[0], n_bytes=n_bytes)
    layout = build_layout(targets, AXES)
    for k in (0, 1, 3, 11):
        scorer = ToleranceScorer(targets.offsets, layout.target_cols, layout.n_strata, n_bytes, k)
        rows = []
        for _ in range(5):
            m = int(rng.integers(0, max(1, n_bytes // 3)))
            rows.append(np.sort(rng.choice(np.arange(1, n_bytes), size=min(m, n_bytes - 1), replace=False)))
        width = max(len(r) for r in rows)
        # pad rows to equal width by repeating their last value would create duplicates; score row by row
        for r in rows:
            if r.size == 0:
                pred = np.empty((1, 0), dtype=np.int64)
            else:
                pred = r[None, :]
            tp_p, tp_r = scorer.counts(pred)
            exp_p, exp_r = naive(r, targets.offsets, layout.target_cols, layout.n_strata, k)
            np.testing.assert_array_equal(tp_p[0], exp_p)
            np.testing.assert_array_equal(tp_r[0], exp_r)
        assert width >= 0


@pytest.mark.parametrize("seed", range(6))
def test_multi_row_batches_equal_single_rows(seed):
    rng = np.random.default_rng(1000 + seed)
    n_bytes = 400
    structure = random_structure(rng, n_bytes, 60)
    targets = build_targets(structure, "start", taxonomy=TAXONOMY, exclude_offsets=[0], n_bytes=n_bytes)
    layout = build_layout(targets, AXES)
    scorer = ToleranceScorer(targets.offsets, layout.target_cols, layout.n_strata, n_bytes, 2)
    batch = np.stack([np.sort(rng.choice(np.arange(1, n_bytes), size=50, replace=False)) for _ in range(7)])
    tp_p, tp_r = scorer.counts(batch)
    for i in range(batch.shape[0]):
        p1, r1 = scorer.counts(batch[i:i + 1])
        np.testing.assert_array_equal(tp_p[i], p1[0])
        np.testing.assert_array_equal(tp_r[i], r1[0])


def test_simple_known_values():
    # targets at 10 and 20; boundaries at 9 (near 10 with k=1), 15 (far), 20 (exact)
    assert match_counts([9, 15, 20], [10, 20], n_bytes=30, k=0) == (1, 1)
    assert match_counts([9, 15, 20], [10, 20], n_bytes=30, k=1) == (2, 2)
    # one boundary between two close targets can match both (any-match recall)
    assert match_counts([11], [10, 12], n_bytes=30, k=1) == (1, 2)


def test_end_target_at_eof_is_reachable_within_k():
    assert match_counts([29], [30], n_bytes=30, k=1) == (1, 1)
    assert match_counts([29], [30], n_bytes=30, k=0) == (0, 0)


def brute_best_iou(s, e, starts, n):
    ends = list(starts[1:]) + [n]
    best = 0.0
    for a, b in zip(starts, ends):
        inter = max(0, min(e, b) - max(s, a))
        union = max(e, b) - min(s, a)
        best = max(best, inter / union)
    return best


@pytest.mark.parametrize("seed", range(10))
def test_best_patch_iou_matches_brute_force(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(5, 200))
    starts = np.concatenate(([0], np.sort(rng.choice(np.arange(1, n), size=int(rng.integers(0, n - 1)), replace=False))))
    s = rng.integers(0, n - 1, size=30)
    e = np.minimum(n, s + rng.integers(1, 60, size=30))
    got = best_patch_iou(s, e, starts, n)
    exp = [brute_best_iou(int(a), int(b), starts, n) for a, b in zip(s, e)]
    np.testing.assert_allclose(got, exp)
