"""escape_common.validate (Stream A output contract) and escape_common.offsets (byte/char)."""

import numpy as np
import pandas as pd
import pytest

from escape_common import offsets as O
from escape_common.validate import validate_boundaries, validate_entropy_array
from stream_a.extract import boundaries_frame
from stream_a.patching import compute_patches

TAU = 1.335442066192627


def make_valid(n=60, seed=0, cap=None):
    rng = np.random.default_rng(seed)
    ent = rng.exponential(0.9, size=n).astype(np.float32)
    patches = compute_patches(ent, TAU, cap)
    return ent, boundaries_frame("py_0123456789ab", ent, patches)


def problems(df, ent, cap=None, **kw):
    return validate_boundaries(df, n_bytes=len(ent), tau=TAU, max_patch_length=cap, file_id="py_0123456789ab",
                               entropies=ent, **kw)


def test_valid_frames_pass():
    for cap in (None, 4):
        ent, df = make_valid(cap=cap)
        assert problems(df, ent, cap=cap, check_completeness=True) == []


def test_parquet_roundtrip_keeps_contract(tmp_path):
    ent, df = make_valid()
    path = tmp_path / "b.parquet"
    df.to_parquet(path, index=False)
    assert problems(pd.read_parquet(path), ent, check_completeness=True) == []


@pytest.mark.parametrize("mutation", ["dtype", "unsorted", "low_entropy_trigger", "bad_sum", "first_row",
                                      "cap_without_max", "entropy_column", "unknown_trigger"])
def test_each_violation_is_caught(mutation):
    ent, df = make_valid(n=80, seed=3)
    df = df.copy()
    if mutation == "dtype":
        df["byte_offset"] = df["byte_offset"].astype("int32")
    elif mutation == "unsorted":
        df.loc[[1, 2], "byte_offset"] = df.loc[[2, 1], "byte_offset"].to_numpy()
    elif mutation == "low_entropy_trigger":
        low = int(np.flatnonzero(ent.astype(np.float64) <= TAU)[1])
        extra = pd.DataFrame({"file_id": ["py_0123456789ab"], "byte_offset": [low], "trigger": ["entropy"],
                              "entropy": np.array([ent[low]], np.float32), "patch_index": [0], "patch_length": [1]})
        df = pd.concat([df, extra]).sort_values("byte_offset").reset_index(drop=True)
    elif mutation == "bad_sum":
        df.loc[len(df) - 1, "patch_length"] = df.loc[len(df) - 1, "patch_length"] + 1
    elif mutation == "first_row":
        df.loc[0, "trigger"] = "entropy"
    elif mutation == "cap_without_max":
        df.loc[1, "trigger"] = "length_cap"
    elif mutation == "entropy_column":
        df.loc[1, "entropy"] = np.float32(df.loc[1, "entropy"] + 0.5)
    elif mutation == "unknown_trigger":
        df.loc[1, "trigger"] = "newline"
    assert problems(df, ent), mutation


def test_entropy_array_validation():
    assert validate_entropy_array(np.zeros(5, np.float32), 5) == []
    assert validate_entropy_array(np.zeros(5, np.float64), 5)
    assert validate_entropy_array(np.full(5, 8.0, np.float32), 5)  # bits, not nats
    assert validate_entropy_array(np.zeros(4, np.float32), 5)


def test_offsets_multibyte_roundtrip():
    text = "aé€x"  # 1 + 2 + 3 + 1 bytes
    content = text.encode("utf-8")
    assert O.utf8_roundtrip_ok(content) and not O.utf8_roundtrip_ok(b"\xff\xfe")
    assert O.char_to_byte_map(text).tolist() == [0, 1, 3, 6, 7]
    assert O.byte_to_char_map(content).tolist() == [0, 1, -1, 2, -1, -1, 3, 4]
    assert O.bytes_to_chars(content, [0, 3, 7]).tolist() == [0, 2, 4]
    with pytest.raises(ValueError):
        O.bytes_to_chars(content, [2])
    assert O.span_roundtrip_problems(content, [(1, 6, "é€")]) == []
    assert O.span_roundtrip_problems(content, [(2, 6, "€")])
