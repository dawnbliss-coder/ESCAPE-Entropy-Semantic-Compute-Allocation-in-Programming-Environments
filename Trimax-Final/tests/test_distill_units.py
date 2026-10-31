"""Unit tests for the distillation extension: windowing, byte-balanced sampling, the A-format
readers of score_a_runs and the re-thresholder's boundary frame."""
import numpy as np
import pandas as pd
import pytest
import torch

from escape_distill.data import (
    BOS_ID,
    BYTE_TOKEN_OFFSET,
    Sample,
    WindowLoader,
    build_windows,
    byte_balanced_weights,
    collate,
    window_domains,
    windows_from_bytes,
)
from escape_distill.retau import BOUNDARY_COLUMNS, boundaries_frame
from escape_distill.score_a_runs import c_v1_patches_from_a, f1_of, interior_starts_from_a
from stream_a.patching import compute_patches

SAMPLES = [
    Sample("a", "cpp", b"x" * 1200),   # windows of 512, 512, 176 bytes
    Sample("b", "py", b"def f():\n    return 1\n"),
    Sample("c", "prose", b"The cat sat. " * 10),
    Sample("d", "cpp", b"int main() { return 0; }"),
]


def test_windows_cover_bytes_without_overlap():
    data = bytes(range(256)) * 5
    wins = list(windows_from_bytes(data, 512))
    assert [len(w) for w in wins] == [512, 512, 256]
    assert b"".join(wins) == data


def test_window_ids_and_domains_line_up():
    wins = build_windows(SAMPLES, max_bytes=512)
    doms = window_domains(SAMPLES, max_bytes=512)
    assert len(wins) == len(doms) == 6
    assert doms == ["cpp", "cpp", "cpp", "py", "prose", "cpp"]
    assert all(w[0] == BOS_ID for w in wins)
    assert wins[3][1] == ord("d") + BYTE_TOKEN_OFFSET
    # min_bytes drops short windows from both lists in the same way
    assert len(build_windows(SAMPLES, 512, min_bytes=100)) == len(window_domains(SAMPLES, 512, min_bytes=100)) == 4


def test_byte_balanced_weights_equalise_expected_bytes():
    wins = build_windows(SAMPLES, max_bytes=512)
    doms = window_domains(SAMPLES, max_bytes=512)
    w, rec = byte_balanced_weights(wins, doms)
    nbytes = np.array([len(x) - 1 for x in wins])
    w = np.array(w)
    expected = {d: float((nbytes * w)[np.array(doms) == d].sum()) for d in set(doms)}
    assert all(v == pytest.approx(1.0) for v in expected.values())
    assert rec["bytes_per_epoch_unbalanced"] == int(nbytes.sum())
    assert rec["domains"]["cpp"]["bytes"] == 1200 + 24
    assert sum(d["share_unbalanced"] for d in rec["domains"].values()) == pytest.approx(1.0)
    assert all(d["share_balanced"] == pytest.approx(1 / 3) for d in rec["domains"].values())


def test_window_loader_default_is_one_shuffled_pass():
    wins = build_windows(SAMPLES, max_bytes=512)
    seen = []
    for batch in WindowLoader(wins, 4, torch.device("cpu"), seed=1):
        seen.extend(batch["input_ids"][:, 1].tolist())
    assert sorted(seen) == sorted(int(w[1]) for w in wins)


def test_window_loader_weighted_draws_are_seeded():
    wins = build_windows(SAMPLES, max_bytes=512)
    doms = window_domains(SAMPLES, max_bytes=512)
    w, _ = byte_balanced_weights(wins, doms)

    def draw(seed):
        return [b["input_ids"].tolist() for b in WindowLoader(wins, 2, torch.device("cpu"), seed=seed, weights=w)]

    assert draw(7) == draw(7)
    assert sum(len(b) for b in draw(7)) == len(wins)


def test_collate_masks_positions_that_predict_real_bytes():
    wins = [np.array([BOS_ID, 10, 11, 12]), np.array([BOS_ID, 20])]
    batch = collate(wins, torch.device("cpu"))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["predict_mask"].tolist() == [[True, True, True, False], [True, False, False, False]]


def _a_frame(starts, triggers):
    return pd.DataFrame({"byte_offset": starts, "trigger": triggers})


def test_a_format_readers():
    b = _a_frame([0, 7, 3, 12], ["init", "length_cap", "entropy", "entropy"])
    assert interior_starts_from_a(b, 20) == [3, 12]
    assert c_v1_patches_from_a(b, 20) == [
        {"start_byte": 0, "end_byte": 3, "termination_reason": "entropy"},
        {"start_byte": 3, "end_byte": 7, "termination_reason": "max_length"},
        {"start_byte": 7, "end_byte": 12, "termination_reason": "entropy"},
        {"start_byte": 12, "end_byte": 20, "termination_reason": "eof"},
    ]
    with pytest.raises(ValueError):
        c_v1_patches_from_a(_a_frame([2], ["entropy"]), 5)


def test_f1_of_counts():
    c = np.array([[2, 1, 1], [0, 0, 0]])
    assert f1_of(c).tolist() == pytest.approx([2 * 2 / (4 + 1 + 1), 0.0])


def test_retau_frame_matches_compute_patches():
    ent = np.array([0.1, 2.0, 0.2, 0.3, 1.9, 0.1, 0.1, 2.5], dtype=np.float32)
    df = boundaries_frame("f", ent, 1.335)
    p = compute_patches(ent, 1.335)
    assert list(df.columns) == BOUNDARY_COLUMNS
    assert df["byte_offset"].tolist() == p.starts.tolist()
    assert df["patch_length"].tolist() == p.lengths.tolist()
    assert df["patch_length"].sum() == len(ent)
    assert interior_starts_from_a(df, len(ent)) == [s for s, t in zip(p.starts, p.triggers) if t == "entropy" and s > 0]
