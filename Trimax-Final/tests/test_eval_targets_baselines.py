"""Targets (outermost attribution, no double counting, init exclusion) and baselines
(H0 sampler properties, determinism, chunk invariance, whitespace)."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from escape_eval.baselines import (H0Sampler, h0_spec, whitespace_boundaries, whitespace_segment_starts,
                                   word_boundaries, word_segment_starts, word_start_offsets)
from escape_eval.targets import build_targets

TAXONOMY = {"expression_statement": "open_ended", "call": "open_ended", "assignment": "open_ended",
            "if_statement": "deterministic_opener"}


def frame(rows):
    return pd.DataFrame(rows, columns=["node_type", "depth", "start_byte", "end_byte"])


def test_shared_start_counts_once_and_goes_to_outermost():
    s = frame([("call", 3, 4, 9), ("expression_statement", 2, 4, 10), ("assignment", 3, 4, 10),
               ("if_statement", 1, 0, 30), ("call", 4, 15, 20)])
    t = build_targets(s, "start", taxonomy=TAXONOMY)
    assert t.offsets.tolist() == [0, 4, 15]
    assert t.node_type.tolist() == ["if_statement", "expression_statement", "call"]
    assert t.depth.tolist() == [1, 2, 4]
    assert t.category.tolist() == ["deterministic_opener", "open_ended", "open_ended"]


def test_tie_break_same_depth_prefers_longer_then_name():
    s = frame([("call", 2, 5, 8), ("assignment", 2, 5, 12), ("call", 2, 20, 25), ("assignment", 2, 20, 25)])
    t = build_targets(s, "start", taxonomy=TAXONOMY)
    assert t.node_type.tolist() == ["assignment", "assignment"]


def test_init_offsets_excluded_and_counted():
    s = frame([("if_statement", 1, 0, 10), ("call", 2, 3, 8)])
    t = build_targets(s, "start", taxonomy=TAXONOMY, exclude_offsets=[0])
    assert t.offsets.tolist() == [3] and t.n_forced_excluded == 1


def test_end_targets_outermost():
    s = frame([("call", 3, 4, 10), ("expression_statement", 2, 2, 10), ("if_statement", 1, 0, 30)])
    t = build_targets(s, "end", taxonomy=TAXONOMY, n_bytes=30)
    assert t.offsets.tolist() == [10, 30]
    assert t.node_type.tolist() == ["expression_statement", "if_statement"]
    assert t.n_eof == 1


def blt_like(n_bytes, rng, cap=None):
    """A realistic boundaries layout: init at 0, entropy starts, optional length caps."""
    from stream_a.patching import compute_patches

    ent = rng.exponential(0.8, size=n_bytes).astype(np.float32)
    p = compute_patches(ent, 1.335442066192627, cap)
    return p.starts, p.triggers, p.lengths


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cap", [None, 5])
def test_h0_sampler_properties(seed, cap):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 600))
    off, trig, plen = blt_like(n, rng, cap)
    spec = h0_spec(off, trig, plen, n)
    n_entropy = int((trig == "entropy").sum())
    assert spec.n_scored == n_entropy
    draws, candidates = H0Sampler(spec, 7, "py_000000000001").draw(25, return_candidates=True)
    assert draws.shape == (25, n_entropy)
    if n_entropy:
        assert np.all(np.diff(draws, axis=1) > 0)
        assert draws.min() > spec.a and draws.max() < n
        for row in candidates:
            lengths = np.diff(np.concatenate(([spec.a], row, [n])))
            assert Counter(lengths.tolist()) == Counter(spec.post_lengths.tolist())
        assert all(np.isin(r, c).all() for r, c in zip(draws, candidates))
        if cap is None:
            np.testing.assert_array_equal(draws, candidates)


def test_h0_determinism_seed_and_chunk_invariance():
    rng = np.random.default_rng(3)
    off, trig, plen = blt_like(400, rng)
    spec = h0_spec(off, trig, plen, 400)
    a = H0Sampler(spec, 11, "cpp_aaaaaaaaaaaa").draw(10)
    b = H0Sampler(spec, 11, "cpp_aaaaaaaaaaaa").draw(10)
    np.testing.assert_array_equal(a, b)
    s = H0Sampler(spec, 11, "cpp_aaaaaaaaaaaa")
    np.testing.assert_array_equal(a, np.concatenate([s.draw(4), s.draw(6)]))
    c = H0Sampler(spec, 12, "cpp_aaaaaaaaaaaa").draw(10)
    d = H0Sampler(spec, 11, "cpp_bbbbbbbbbbbb").draw(10)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)


def test_h0_preserves_forced_prefix_when_two_init_rows():
    off = np.array([0, 1, 4, 9])
    trig = np.array(["init", "init", "entropy", "entropy"], dtype=object)
    plen = np.array([1, 3, 5, 3])
    spec = h0_spec(off, trig, plen, 12)
    assert spec.a == 1 and spec.n_scored == 2
    layouts = H0Sampler(spec, 0, "py_000000000002").draw_layouts(20)
    assert np.all(layouts[:, 0] == 0) and np.all(layouts[:, 1] == 1)


def test_whitespace_baseline():
    assert whitespace_boundaries([5, 12, 5, 0], exclude_offsets=[0]).tolist() == [5, 12]
    assert whitespace_segment_starts([5, 12], 20).tolist() == [0, 5, 12]


def test_word_start_offsets_ascii():
    content = b"The cat, sat: on\tMAT_2 mats."
    assert word_start_offsets(content).tolist() == [0, 4, 9, 14, 17, 23]
    assert word_boundaries(content, exclude_offsets=[0]).tolist() == [4, 9, 14, 17, 23]


def test_word_start_offsets_multibyte_and_empty():
    # UTF-8 continuation bytes are never word bytes; word chars are ASCII only.
    assert word_start_offsets("café noir".encode("utf-8")).tolist() == [0, 6]
    assert word_start_offsets(b"123 !!!").tolist() == [0]
    assert word_start_offsets(b"").tolist() == []
    assert word_boundaries(b"", exclude_offsets=[0]).tolist() == []
    assert word_segment_starts(b"abc def", 7).tolist() == [0, 4]


def test_word_segment_starts_bounds():
    assert word_segment_starts(b"ab cd", 5).tolist() == [0, 3]
    assert word_segment_starts(b"", 0).tolist() == []
