"""Tiny GPU-free end-to-end runs:
  1. oracle boundaries (every AST start) score precision = recall = 1 at k = 0;
  2. mock Stream A -> bpp summary -> k calibration (SMOKE) -> run_eval (SMOKE) -> result tables."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from escape_eval import data as D
from escape_eval.engine import METHOD_BLT, METHOD_H0, EngineConfig, count_sets, run_engine

TAU = 1.335442066192627


@pytest.fixture
def mini_golden_root(tmp_path, repo_root):
    root = tmp_path / "repo"
    (root / "corpus").mkdir(parents=True)
    manifest = pd.read_parquet(repo_root / "corpus" / "manifest.parquet")
    chosen = []
    for domain in ("py", "cpp", "prose"):
        src = repo_root / "golden_fixtures" / domain
        dst = root / "golden_fixtures" / domain
        dst.mkdir(parents=True)
        for p in sorted(src.glob("*.bin"))[:2]:
            for q in src.glob(f"{p.stem}.*"):
                shutil.copy(q, dst / q.name)
            chosen.append(p.stem)
    manifest[manifest["file_id"].isin(chosen)].to_parquet(root / "corpus" / "manifest.parquet", index=False)
    for name in ("taxonomy.json", "prose_taxonomy.json"):
        shutil.copy(repo_root / name, root / name)
    return root


def write_oracle_run(root):
    """Stream A outputs whose entropy boundaries are exactly the AST start offsets."""
    from escape_common.io import atomic_save_npy, atomic_to_parquet, atomic_write_json
    from escape_common.schema import boundary_path, entropy_path
    from stream_a.extract import boundaries_frame
    from stream_a.patching import compute_patches

    run_id = "oracle"
    table = D.build_file_table(root, "golden", ["py", "cpp"])
    for row in table.itertuples(index=False):
        s = pd.read_parquet(row.structure_path)
        ent = np.zeros(row.n_bytes, dtype=np.float32)
        ent[np.unique(s["start_byte"].to_numpy())[np.unique(s["start_byte"].to_numpy()) < row.n_bytes]] = 5.0
        patches = compute_patches(ent, TAU)
        atomic_save_npy(entropy_path(root, run_id, row.file_id), ent)
        atomic_to_parquet(boundaries_frame(row.file_id, ent, patches), boundary_path(root, run_id, row.file_id))
    atomic_write_json(root / "runs.json", {run_id: {"tau": TAU, "sliding_window": None, "max_patch_length": None,
                                                    "checkpoint": "ORACLE", "commit": "test"}})
    return run_id, table


def test_oracle_boundaries_score_perfectly(mini_golden_root):
    root = mini_golden_root
    run_id, table = write_oracle_run(root)
    entry = D.load_run_entry(root, run_id)
    for domain in ("py", "cpp"):
        files = table[table["domain"] == domain]
        res = run_engine(files, root, run_id, entry, {domain: D.load_taxonomy(root, domain)},
                         EngineConfig(ks=(0,), kinds=("start",), axes=("all",), n_resamples=30), log=None)
        assert not res.problems
        cs = count_sets(res.counts, files["file_id"], "start", 0, [("all", "all")], (METHOD_BLT, METHOD_H0))
        blt = cs[METHOD_BLT].pooled()
        assert blt["precision"][0] == 1.0 and blt["recall"][0] == 1.0
        assert cs[METHOD_H0].pooled()["f1"][0] < 1.0


def test_mock_pipeline_end_to_end(mini_golden_root, capsys):
    from escape_eval import calibrate_k, run_eval
    from stream_a import build_bpp_summary, extract

    root = mini_golden_root
    run_id = extract.main(["--root", str(root), "--input", "golden", "--domains", "py", "cpp", "prose",
                           "--backend", "mock", "--progress-every", "6"])
    build_bpp_summary.main(["--root", str(root)])
    rec = calibrate_k.calibrate(root, root, run_id, "golden", ["py", "cpp"], grid=(0, 1, 2), n_resamples=40,
                                allow_non_calib=True, results_root=root / "results", log=lambda *a: None)
    assert rec["smoke"] and set(rec["selected_k"]) == {"py", "cpp"}
    assert json.loads((root / "results" / "smoke" / "k_calibration" / "selected_k.json").read_text())["smoke"]

    with pytest.raises(SystemExit):  # smoke calibration may not feed non-smoke scoring
        run_eval.main(["--root", str(root), "--run-id", run_id, "--source", "golden",
                       "--k-from", "results/smoke/k_calibration/selected_k.json"])

    out = run_eval.main(["--root", str(root), "--run-id", run_id, "--source", "golden", "--smoke",
                         "--k-override", "py=1", "cpp=2", "--domains", "py", "cpp", "prose",
                         "--n-resamples", "40", "--n-bootstrap", "40", "--iou", "--iou-resamples", "3",
                         "--analysis-id", "e2e"])
    align = pd.read_parquet(out / "alignment.parquet")
    p1 = align[(align["kind"] == "start") & (align["axis"] == "all")]
    assert set(p1["domain"]) == {"py", "cpp", "prose"}
    assert set(p1[p1["domain"] == "prose"]["k"]) == {1, 2}
    for col in ("blt", "blt_lo", "blt_hi", "h0", "ws", "p_perm_h0", "p_rand_ws", "delta_h0_lo", "preliminary", "smoke"):
        assert col in align.columns
    assert align["preliminary"].all()
    assert (p1["p_perm_h0"].between(0, 1)).all()
    p2 = pd.read_parquet(out / "p2_start_end.parquet")
    assert set(p2["metric"]) == {"precision", "recall", "f1"}
    assert set(pd.read_parquet(out / "p3_bpp.parquet")["code_domain"]) == {"py", "cpp"}
    assert len(pd.read_parquet(out / "p3_alignment.parquet")) == 2
    assert set(pd.read_parquet(out / "iou.parquet")["method"]) == {"blt", "whitespace", "h0"}
    params = json.loads((out / "params.json").read_text())
    assert params["smoke"] and params["ks"] == {"py": 1, "cpp": 2}
    manifest = json.loads((root / "results" / "smoke" / "manifest.json").read_text())
    assert manifest["e2e"]["preliminary"] and manifest["e2e"]["smoke"]

    from escape_eval import figures

    figures.main(["--analysis-dir", str(out), "--k-dir", str(root / "results" / "smoke" / "k_calibration")])
    for name in ("p1_alignment_effects.png", "depth_alignment.png", "p2_start_end.png", "k_calibration.png",
                 "alignment_vs_k.png"):  # prose was scored at k=1 and k=2, so the k curve is drawn
        assert (out / "figures" / name).exists(), name
