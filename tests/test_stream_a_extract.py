"""Extraction driver tests with the GPU-free mock backend: output contract, resumability,
failure handling, retries, run registry and the no-parse-data guarantee."""

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from escape_common.schema import BOUNDARY_COLUMNS, RUNS_JSON_CONTRACT_KEYS
from stream_a import extract

TAU = 1.335442066192627


@pytest.fixture
def mini_root(tmp_path, repo_root):
    """A throwaway repo root holding 2 py + 2 cpp golden fixtures, a filtered manifest, and
    unreadable decoy structure/ and whitespace/ directories: any attempt by extraction to
    read parse data raises PermissionError."""
    root = tmp_path / "repo"
    (root / "corpus").mkdir(parents=True)
    manifest = pd.read_parquet(repo_root / "corpus" / "manifest.parquet")
    chosen = []
    for domain in ("py", "cpp"):
        ids = sorted(p.stem for p in (repo_root / "golden_fixtures" / domain).glob("*.bin"))[:2]
        dst = root / "golden_fixtures" / domain
        dst.mkdir(parents=True)
        for fid in ids:
            shutil.copy(repo_root / "golden_fixtures" / domain / f"{fid}.bin", dst / f"{fid}.bin")
            chosen.append(fid)
    manifest[manifest["file_id"].isin(chosen)].to_parquet(root / "corpus" / "manifest.parquet", index=False)
    decoys = []
    for name in ("structure", "whitespace"):
        d = root / name
        d.mkdir()
        (d / "poison.parquet").write_bytes(b"not a parquet file")
        os.chmod(d, 0)
        decoys.append(d)
    yield root, chosen
    for d in decoys:
        os.chmod(d, 0o755)


def run(root, *extra):
    return extract.main(["--root", str(root), "--input", "golden", "--domains", "py", "cpp",
                         "--backend", "mock", "--progress-every", "1", *extra])


def check_file(root, run_id, fid, n_bytes):
    ent = np.load(root / "entropies" / run_id / f"{fid}.npy")
    assert ent.dtype == np.float32 and ent.shape == (n_bytes,)
    b = pd.read_parquet(root / "boundaries" / run_id / f"{fid}.parquet")
    assert list(b.columns) == BOUNDARY_COLUMNS
    assert b["file_id"].eq(fid).all()
    assert b["byte_offset"].iloc[0] == 0 and b["trigger"].iloc[0] == "init"
    assert (np.diff(b["byte_offset"].to_numpy()) > 0).all()
    assert int(b["patch_length"].sum()) == n_bytes
    assert b["patch_index"].tolist() == list(range(len(b)))
    np.testing.assert_array_equal(b["entropy"].to_numpy(), ent[b["byte_offset"].to_numpy()])
    entropy_rows = b[b["trigger"] == "entropy"]
    assert (entropy_rows["entropy"].astype(np.float64) > TAU).all()
    expected = np.flatnonzero(ent[1:].astype(np.float64) > TAU) + 1
    assert entropy_rows["byte_offset"].tolist() == expected.tolist()
    assert set(b["trigger"]) <= {"init", "entropy", "length_cap"}


def test_end_to_end_contract_and_no_parse_reads(mini_root):
    root, chosen = mini_root
    run_id = run(root)
    manifest = pd.read_parquet(root / "corpus" / "manifest.parquet").set_index("file_id")
    for fid in chosen:
        check_file(root, run_id, fid, int(manifest.at[fid, "n_bytes"]))
    runs = json.loads((root / "runs.json").read_text())
    entry = runs[run_id]
    for key in RUNS_JSON_CONTRACT_KEYS:
        assert key in entry
    assert entry["backend"] == "mock" and entry["checkpoint"] == "MOCK"
    snap = pd.read_parquet(root / "runs" / run_id / "status.parquet")
    assert sorted(snap["file_id"]) == sorted(chosen) and set(snap["status"]) == {"success"}


def test_resume_skips_done_and_redoes_missing(mini_root, capsys):
    root, chosen = mini_root
    run_id = run(root)
    capsys.readouterr()
    run(root)
    assert "processed=0 skipped_already_done=4" in capsys.readouterr().out
    (root / "boundaries" / run_id / f"{chosen[0]}.parquet").unlink()
    # a stale temp file from a crashed write must not confuse resume
    (root / "entropies" / run_id / f".{chosen[1]}.npy.123.tmp").write_bytes(b"partial")
    run(root)
    assert "processed=1 skipped_already_done=3" in capsys.readouterr().out


def test_length_cap_run_is_a_separate_run_id(mini_root):
    root, chosen = mini_root
    run_id_default = run(root)
    run_id_capped = run(root, "--max-patch-length", "4")
    assert run_id_default != run_id_capped
    b = pd.read_parquet(root / "boundaries" / run_id_capped / f"{chosen[0]}.parquet")
    assert b["patch_length"].max() <= 4
    assert (b["trigger"] == "length_cap").any()


def test_integrity_failure_recorded_not_retried_then_only_failed(mini_root, capsys):
    root, chosen = mini_root
    victim = chosen[0]
    bin_path = root / "golden_fixtures" / "py" / f"{victim}.bin"
    original = bin_path.read_bytes()
    bin_path.write_bytes(original[:-1] + b"X")
    run_id = run(root)
    failure = json.loads((root / "runs" / run_id / "failures" / f"{victim}.json").read_text())
    assert failure["error_type"] == "InputIntegrityError" and failure["attempts"] == 1
    assert "Traceback" in failure["traceback"]
    snap = pd.read_parquet(root / "runs" / run_id / "status.parquet").set_index("file_id")
    assert snap.at[victim, "status"] == "failed"
    assert (snap.drop(index=victim)["status"] == "success").all()
    assert not (root / "boundaries" / run_id / f"{victim}.parquet").exists()

    bin_path.write_bytes(original)
    capsys.readouterr()
    run(root, "--only-failed")
    out = capsys.readouterr().out
    assert "selected=1" in out and "processed=1" in out
    snap = pd.read_parquet(root / "runs" / run_id / "status.parquet").set_index("file_id")
    assert snap.at[victim, "status"] == "success"


def test_transient_error_is_retried(mini_root, monkeypatch):
    root, chosen = mini_root
    calls = {"n": 0}
    original = extract.MockBackend.token_entropies

    def flaky(self, content):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA hiccup")
        return original(self, content)

    monkeypatch.setattr(extract.MockBackend, "token_entropies", flaky)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    run_id = run(root, "--retries", "1")
    events = [json.loads(line) for line in (root / "runs" / run_id / "status.jsonl").read_text().splitlines()]
    first = [e for e in events if e["file_id"] == events[0]["file_id"]]
    assert [e["status"] for e in first] == ["running", "retrying", "running", "success"]


def test_exhausted_retries_marks_failed(mini_root, monkeypatch):
    root, chosen = mini_root

    def always_fail(self, content):
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(extract.MockBackend, "token_entropies", always_fail)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    run_id = run(root, "--retries", "2")
    failure_files = list((root / "runs" / run_id / "failures").glob("*.json"))
    assert len(failure_files) == len(chosen)
    assert all(json.loads(p.read_text())["attempts"] == 3 for p in failure_files)


def test_interrupted_file_is_reprocessed(mini_root, capsys):
    root, chosen = mini_root
    run_id = run(root)
    victim = chosen[2]
    (root / "boundaries" / run_id / f"{victim}.parquet").unlink()
    with open(root / "runs" / run_id / "status.jsonl", "a") as f:
        f.write(json.dumps({"file_id": victim, "status": "running", "attempt": 1}) + "\n")
        f.write('{"file_id": "torn line from a crash"')
    capsys.readouterr()
    run(root)
    assert "processed=1" in capsys.readouterr().out
    snap = pd.read_parquet(root / "runs" / run_id / "status.parquet").set_index("file_id")
    assert snap.at[victim, "status"] == "success"


def test_register_run_rejects_changed_settings(tmp_path):
    entry = {"backend": "mock", "tau": 1.0, "sliding_window": None, "chunk_len": None, "max_patch_length": None,
             "context_mode": "mock", "dtype": "float32", "checkpoint": "MOCK", "commit": "abc", "date": "d",
             "commit_dirty": False, "code_sha256": "x"}
    extract.register_run(tmp_path, "r1", dict(entry))
    extract.register_run(tmp_path, "r1", dict(entry))  # identical settings: fine
    with pytest.raises(SystemExit):
        extract.register_run(tmp_path, "r1", {**entry, "tau": 2.0})


def test_figures_render_from_saved_artifacts(mini_root):
    from stream_a import build_bpp_summary, figures

    root, chosen = mini_root
    run_id = run(root)
    build_bpp_summary.main(["--root", str(root)])
    figures.main(["--root", str(root), "--run-id", run_id, "--heatmap-files", chosen[0]])
    out = root / "results" / "figures" / "stream_a" / run_id
    for name in (f"entropy_heatmap_{chosen[0]}.png", "patch_length_distribution.png", "bpp_distributions.png",
                 "figures.json"):
        assert (out / name).exists(), name
