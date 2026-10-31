import math

import numpy as np
import pytest

from stream_a.patching import (
    TRIGGER_ENTROPY,
    TRIGGER_INIT,
    TRIGGER_LENGTH_CAP,
    bpp_stats,
    byte_entropies_from_token_entropies,
    byte_starts_from_token_patch_lengths,
    bytes_to_token_ids,
    chunk_context_meta,
    compute_patches,
    entropy_stats,
    reference_start_ids,
)

TAU = 1.335442066192627


def split_large_numbers(lst, m):
    """facebookresearch/blt bytelatent/data/patcher.py:456-467, verbatim logic."""
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    return new_lst


def hand_entropy_nats(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def test_entropy_units_hand_computed():
    # the proposal's worked examples: 4 equiprobable continuations vs one dominant of ten
    assert hand_entropy_nats([0.25] * 4) == pytest.approx(math.log(4))  # 1.386 nats
    assert hand_entropy_nats([0.25] * 4) > TAU
    skewed = [0.99] + [0.01 / 9] * 9
    assert hand_entropy_nats(skewed) == pytest.approx(0.0780, abs=1e-3)
    assert TAU / math.log(2) == pytest.approx(1.9266, abs=1e-4)  # tau in bits


def test_tau_is_exact_float32():
    assert float(np.float32(TAU)) == TAU


def test_basic_triggers_and_lengths():
    ent = np.array([9.0, 0.1, 2.0, 0.5, 1.4, TAU, 3.0], dtype=np.float32)
    p = compute_patches(ent, TAU)
    assert p.starts.tolist() == [0, 2, 4, 6]
    assert p.triggers.tolist() == [TRIGGER_INIT, TRIGGER_ENTROPY, TRIGGER_ENTROPY, TRIGGER_ENTROPY]
    assert p.lengths.tolist() == [2, 2, 2, 1]
    assert p.lengths.sum() == len(ent)


def test_threshold_is_strict():
    above = np.nextafter(np.float32(TAU), np.float32(np.inf))
    ent = np.array([0.0, np.float32(TAU), above], dtype=np.float32)
    assert compute_patches(ent, TAU).starts.tolist() == [0, 2]


def test_byte0_always_init_even_if_low_entropy():
    p = compute_patches(np.zeros(5, np.float32), TAU)
    assert p.starts.tolist() == [0] and p.triggers.tolist() == [TRIGGER_INIT] and p.lengths.tolist() == [5]


def test_empty_and_single_byte():
    assert compute_patches(np.zeros(0, np.float32), TAU).n_patches == 0
    p = compute_patches(np.array([5.0], np.float32), TAU)
    assert p.starts.tolist() == [0] and p.lengths.tolist() == [1]


def test_length_cap_classification():
    ent = np.zeros(10, np.float32)
    p = compute_patches(ent, TAU, max_patch_length=4)
    assert p.starts.tolist() == [0, 4, 8]
    assert p.triggers.tolist() == [TRIGGER_INIT, TRIGGER_LENGTH_CAP, TRIGGER_LENGTH_CAP]
    assert p.lengths.tolist() == [4, 4, 2]

    ent[5] = 5.0
    p = compute_patches(ent, TAU, max_patch_length=4)
    assert p.starts.tolist() == [0, 4, 5, 9]
    assert p.triggers.tolist() == [TRIGGER_INIT, TRIGGER_LENGTH_CAP, TRIGGER_ENTROPY, TRIGGER_LENGTH_CAP]
    assert p.lengths.tolist() == [4, 1, 4, 1]


def test_exact_multiple_of_cap():
    p = compute_patches(np.zeros(8, np.float32), TAU, max_patch_length=4)
    assert p.lengths.tolist() == [4, 4]


@pytest.mark.parametrize("seed", range(20))
def test_length_cap_matches_reference_split(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    ent = rng.exponential(0.6, size=n).astype(np.float32)
    m = int(rng.integers(1, 12))
    uncapped = compute_patches(ent, TAU)
    capped = compute_patches(ent, TAU, max_patch_length=m)
    assert capped.lengths.tolist() == split_large_numbers(uncapped.lengths.tolist(), m)
    assert capped.lengths.sum() == n and capped.lengths.max() <= m
    assert np.all(np.diff(capped.starts) > 0)
    cap_rows = capped.triggers == TRIGGER_LENGTH_CAP
    # a cap boundary can never sit where entropy already exceeded tau
    assert np.all(ent[capped.starts[cap_rows]].astype(np.float64) <= TAU)
    # the entropy-triggered set is untouched by capping
    assert capped.starts[capped.triggers == TRIGGER_ENTROPY].tolist() == uncapped.starts[1:].tolist()


@pytest.mark.parametrize("seed", range(10))
def test_matches_reference_find_entropy_patch_start_ids(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(2, 300))
    token_ent = rng.exponential(0.8, size=n + 1).astype(np.float32)
    ref_ids = reference_start_ids(token_ent, TAU, include_next_token=True)
    token_lengths = np.diff(np.append(ref_ids, n + 2))  # seq_len_next_tok = (n+1) + 1
    byte_starts = byte_starts_from_token_patch_lengths(token_lengths, n)
    ours = compute_patches(byte_entropies_from_token_entropies(token_ent, n), TAU)
    assert byte_starts.tolist() == ours.starts.tolist()


def test_hf_trailing_zero_length_patch_is_ignored():
    # HF patch_lengths_from_entropies emits a zero-length final patch when the last
    # position's entropy exceeds the threshold; it maps past the last byte.
    token_lengths = np.array([1, 3, 2, 0])  # BOS | bytes 0-2 | bytes 3-4 | (empty)
    assert byte_starts_from_token_patch_lengths(token_lengths, 5).tolist() == [0, 3]


def test_token_ids_and_index_alignment_multibyte():
    content = "é€x".encode("utf-8")  # 2 + 3 + 1 bytes
    ids = bytes_to_token_ids(content)
    assert ids.tolist() == [1] + [b + 4 for b in content]
    token_ent = np.arange(len(content) + 1, dtype=np.float32)
    assert byte_entropies_from_token_entropies(token_ent, len(content)).tolist() == list(range(len(content)))
    with pytest.raises(ValueError):
        byte_entropies_from_token_entropies(token_ent, len(content) + 1)


def test_bpp_stats():
    s = bpp_stats([2, 2, 2, 1])
    assert s["mean_bpp"] == pytest.approx(7 / 4)
    assert s["var_bpp"] == pytest.approx(np.var([2, 2, 2, 1]))
    assert s["median_bpp"] == 2 and s["min_bpp"] == 1 and s["max_bpp"] == 2
    assert math.isnan(bpp_stats([])["mean_bpp"])


def test_entropy_stats():
    s = entropy_stats(np.array([0.0, 1.0, 2.0, 3.0], np.float32), TAU)
    assert s["ent_mean"] == pytest.approx(1.5) and s["frac_bytes_above_tau"] == pytest.approx(0.5)


def test_chunk_context_meta():
    assert chunk_context_meta(0, None) == {"chunk_len": None, "n_chunk_edges": 0, "n_bytes_truncated_context": 0}
    assert chunk_context_meta(8191, 8192) == {"chunk_len": 8192, "n_chunk_edges": 0, "n_bytes_truncated_context": 0}
    # 8192 bytes -> 8193 tokens: one chunk boundary at token 8192 (byte 8191 loses full context).
    assert chunk_context_meta(8192, 8192) == {"chunk_len": 8192, "n_chunk_edges": 1, "n_bytes_truncated_context": 1}
    assert chunk_context_meta(20000, 8192) == {"chunk_len": 8192, "n_chunk_edges": 2, "n_bytes_truncated_context": 11809}
    with pytest.raises(ValueError):
        chunk_context_meta(10, 0)
