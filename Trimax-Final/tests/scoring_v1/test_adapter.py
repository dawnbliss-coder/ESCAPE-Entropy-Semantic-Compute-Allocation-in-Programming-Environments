import json
import numpy as np
import pandas as pd
from escape_scoring.adapter import convert
from escape_scoring.contract import Dataset
from escape_scoring.fixtures import create_fixture


def test_adapter_moves_next_patch_start_label_to_previous_end(tmp_path):
    create_fixture(tmp_path / "fixture")
    source = Dataset(tmp_path / "fixture")
    s = source.load(source.manifest[0]["sample_id"])
    root = tmp_path / "legacy"
    for folder in ["corpus/cpp", "structure/cpp", "entropies/run", "boundaries/run"]:
        (root / folder).mkdir(parents=True)
    sid = s.sample_id
    (root / f"corpus/cpp/{sid}.bin").write_bytes(s.text)
    pd.DataFrame([{"file_id": sid, "domain": "cpp", "n_bytes": len(s.text),
                   "sha256": s.manifest["text_sha256"], "split": "main", "licenses": None}]).to_parquet(root / "corpus/manifest.parquet")
    pd.DataFrame([{**r, "file_id": sid} for r in s.nodes]).to_parquet(root / f"structure/cpp/{sid}.parquet")
    np.save(root / f"entropies/run/{sid}.npy", np.zeros(len(s.text)))
    pd.DataFrame([{"file_id": sid, "byte_offset": 0, "trigger": "init", "patch_length": 5},
                  {"file_id": sid, "byte_offset": 5, "trigger": "length_cap", "patch_length": 7},
                  {"file_id": sid, "byte_offset": 12, "trigger": "entropy", "patch_length": len(s.text)-12}]).to_parquet(root / f"boundaries/run/{sid}.parquet")
    (root / "runs.json").write_text(json.dumps({"run": {"backend": "patcher", "entropy_definition": "entropies[i] = -sum_v p(v) ln p(v)",
        "tau": 1.34, "checkpoint": "TEST-ADAPTER", "checkpoint_revision": "test"}}))
    convert(root, tmp_path / "converted", "run")
    result = Dataset(tmp_path / "converted").load(sid)
    assert result.patcher["boundaries"] == [{"byte_offset": 5, "termination_reason": "max_length"},
        {"byte_offset": 12, "termination_reason": "entropy"}, {"byte_offset": len(s.text), "termination_reason": "eof"}]
    assert result.manifest["license_metadata"]["status"] == "unknown"
    assert result.manifest["split"] == "unassigned"
