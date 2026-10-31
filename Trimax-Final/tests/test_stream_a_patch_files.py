"""stream_a.patch_files (boundary extraction for arbitrary files), with the byte-only mock backend."""
import json

import numpy as np
import pandas as pd

from stream_a.extract import MockBackend
from stream_a.patch_files import main, patch_bytes, show
from stream_a.patching import byte_entropies_from_token_entropies, compute_patches

SRC = b"def f(x):\n    return x + 1\n"


def test_patch_bytes_matches_extract_rule():
    backend = MockBackend()
    ent, patches = patch_bytes(backend, SRC, backend.tau)
    ref = byte_entropies_from_token_entropies(backend.token_entropies(SRC), len(SRC))
    assert np.array_equal(ent, ref)
    assert np.array_equal(patches.starts, compute_patches(ref, backend.tau).starts)
    assert show(SRC, patches).replace("|", "") == SRC.decode()


def test_cli_writes_entropies_boundaries_and_summary(tmp_path):
    src = tmp_path / "f.py"
    src.write_bytes(SRC)
    (tmp_path / "empty.txt").write_bytes(b"")
    out = tmp_path / "out"
    main([str(src), str(tmp_path / "empty.txt"), "--mock", "--out", str(out)])
    ent = np.load(out / "f.py.entropies.npy")
    b = pd.read_parquet(out / "f.py.boundaries.parquet")
    s = json.loads((out / "f.py.summary.json").read_text())
    assert ent.dtype == np.float32 and ent.shape == (len(SRC),)
    assert int(b["patch_length"].sum()) == len(SRC) and b["byte_offset"].iloc[0] == 0
    assert s["n_patches"] == len(b) and s["backend"] == "mock"
    assert s["mean_bpp"] == len(SRC) / len(b)
    assert not (out / "empty.txt.summary.json").exists()
