"""Small SYNTHETIC byte fixtures. Never evidence about real BLT behaviour."""
import hashlib
from pathlib import Path
import numpy as np

from .contract import derived_boundaries, write_jsonl


def create_fixture(root, *, seed=71, files_per_language=5, aligned=True):
    root = Path(root)
    if (root / "sample_manifest.jsonl").exists():
        raise FileExistsError(root)
    (root / "text").mkdir(parents=True, exist_ok=True)
    manifests, patchers, nodes, regions = [], [], [], []
    rng = np.random.default_rng(seed)
    for language, domain in [("cpp", "code"), ("python", "code"), ("english", "prose")]:
        for index in range(files_per_language):
            sid = f"synthetic_{language}_{index}"
            raw = (f"{sid}: " + "abcdefghij " * 8).encode()
            (root / "text" / f"{sid}.bin").write_bytes(raw)
            base = {"schema_version": "1.0.0", "sample_id": sid}
            manifests.append({**base, "split": "unassigned", "domain": domain, "language": language,
                              "source_name": "synthetic_fixture", "text_sha256": hashlib.sha256(raw).hexdigest(),
                              "byte_length": len(raw), "text_path": f"text/{sid}.bin",
                              "license_metadata": {"status": "synthetic", "licenses": []},
                              "source_metadata": {"generator_seed": seed}, "synthetic": True})
            starts = [0, *sorted(rng.choice(np.arange(2, len(raw) - 2), 7, replace=False).tolist())]
            ends = starts[1:] + [len(raw)]
            patches = [{"start_byte": a, "end_byte": b, "termination_reason": "eof" if b == len(raw) else "entropy"}
                       for a, b in zip(starts, ends)]
            patchers.append({**base, "byte_length": len(raw), "model": {
                "checkpoint": "SYNTHETIC-NOT-A-MODEL", "revision": "fixture-v1", "run_id": "synthetic-v1", "configuration": {}},
                "threshold": 1.34, "entropy_definition": "global_next_byte_shannon", "entropy_unit": "nats",
                "entropy_records": [{"byte_offset": i, "entropy": 2.0 if i in starts[1:] else .2} for i in range(len(raw))],
                "patches": patches, "boundaries": derived_boundaries(patches)})
            targets = starts[1:] if aligned else sorted(rng.choice(np.arange(2, len(raw) - 2), 7, replace=False).tolist())
            for j, start in enumerate(targets):
                nodes.append({**base, "node_id": f"n{j}", "node_type": "synthetic_constituent",
                              "start_byte": start, "end_byte": min(start + 2, len(raw)),
                              "depth": 1, "is_primary_alignment_target": True})
            if language == "cpp":
                for j, (a, b) in enumerate([(0, len(raw)//2), (len(raw)//2, len(raw))]):
                    regions.append({**base, "region_id": f"unit{j}", "start_byte": a, "end_byte": b, "region_kind": "analysis_region"})
                regions.extend([{**base, "region_id": "unsafe", "start_byte": 5, "end_byte": 9, "region_kind": "memory_unsafe"},
                                {**base, "region_id": "identifier", "start_byte": 6, "end_byte": 14, "region_kind": "identifier"}])
    for kind, rows in [("sample_manifest", manifests), ("patcher_output", patchers),
                       ("ground_truth_nodes", nodes), ("region_annotations", regions)]:
        write_jsonl(root / f"{kind}.jsonl", rows)
