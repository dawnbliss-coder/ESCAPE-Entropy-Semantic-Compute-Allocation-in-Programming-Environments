"""Read-only conversion of the existing A/B disk contract into C JSONL v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from .contract import Sample, derived_boundaries, validate_sample, write_jsonl


def convert(root, output, run_id, *, source="corpus", sample_ids=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("conversion destination must be new")
    registry = json.loads((root / "runs.json").read_text())
    config = registry[run_id]
    if config.get("backend") != "patcher" or not config.get("entropy_definition", "").startswith("entropies[i] = -sum_v"):
        raise ValueError("A run must declare global predictive Shannon entropy; mock runs require synthetic fixtures")
    manifest = pd.read_parquet(root / "corpus/manifest.parquet").set_index("file_id")
    ids = list(manifest.index) if sample_ids is None else list(sample_ids)
    if source == "golden" and sample_ids is None:
        ids = sorted(p.stem for p in (root / "golden_fixtures").glob("*/*.bin"))
    # Preflight all inputs before creating output; never download or regenerate.
    paths = {}
    missing = []
    for sid in ids:
        domain = manifest.loc[sid, "domain"]
        text = root / (f"golden_fixtures/{domain}/{sid}.bin" if source == "golden" else f"corpus/{domain}/{sid}.bin")
        structure = root / (f"golden_fixtures/{domain}/{sid}.structure.parquet" if source == "golden" else f"structure/{domain}/{sid}.parquet")
        entropy = root / f"entropies/{run_id}/{sid}.npy"
        boundaries = root / f"boundaries/{run_id}/{sid}.parquet"
        paths[sid] = (text, structure, entropy, boundaries)
        missing.extend(str(p) for p in paths[sid] if not p.exists())
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing A/B inputs; first: {missing[:8]}")
    (output / "text").mkdir(parents=True)
    all_manifest, all_nodes, all_regions = [], [], []
    with (output / "patcher_output.jsonl").open("w") as patch_stream:
        for sid in sorted(ids):
            meta = manifest.loc[sid]
            domain = meta["domain"]
            tp, sp, ep, bp = paths[sid]
            raw = tp.read_bytes()
            if len(raw) != int(meta["n_bytes"]) or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError(f"{sid}: B manifest/text mismatch")
            base = {"schema_version": "1.0.0", "sample_id": sid}
            licenses = meta.get("licenses")
            if isinstance(licenses, (list, tuple, np.ndarray)):
                licenses = [str(x) for x in licenses]
            elif isinstance(licenses, str) and licenses:
                licenses = [licenses]
            else:
                licenses = []
            m = {**base, "split": "unassigned", "domain": "prose" if domain == "prose" else "code",
                 "language": {"py": "python", "cpp": "cpp", "prose": "english"}[domain],
                 "source_name": {"py": "CodeSearchNet", "cpp": "bigcode/the-stack-smol", "prose": "WikiText-103"}[domain],
                 "source_metadata": {"legacy_split": str(meta["split"]), "source_path": str(meta.get("source_path", "")),
                                     "selection": source, "parse_ok": None if pd.isna(meta.get("parse_ok")) else bool(meta["parse_ok"])},
                 "license_metadata": {"status": "known" if licenses else "unknown", "licenses": licenses},
                 "text_sha256": meta["sha256"], "byte_length": len(raw), "text_path": f"text/{sid}.bin", "synthetic": False}
            boundary = pd.read_parquet(bp).sort_values("byte_offset").to_dict("records")
            if any(r["file_id"] != sid for r in boundary):
                raise ValueError("A boundary sample mismatch")
            starts = [int(r["byte_offset"]) for r in boundary]
            if raw and (not starts or starts[0] != 0 or boundary[0]["trigger"] != "init"):
                raise ValueError("A needs one initial patch at zero")
            patches = []
            for i, row in enumerate(boundary):
                end = starts[i + 1] if i + 1 < len(starts) else len(raw)
                # A labels the START of the next patch. Move that label onto
                # the END of the current patch; copying the same row is wrong.
                reason = {"entropy": "entropy", "length_cap": "max_length"}[boundary[i + 1]["trigger"]] if i + 1 < len(starts) else "eof"
                if "patch_length" in row and int(row["patch_length"]) != end - starts[i]:
                    raise ValueError("A patch_length disagrees with endpoints")
                patches.append({"start_byte": starts[i], "end_byte": end, "termination_reason": reason})
            entropy = np.load(ep, allow_pickle=False)
            if entropy.shape != (len(raw),):
                raise ValueError("A entropy array shape mismatch")
            stable_config = {k: config.get(k) for k in ("context_mode", "dtype", "sliding_window", "chunk_len", "max_patch_length", "patcher_weights_sha256")}
            p = {**base, "byte_length": len(raw), "model": {"checkpoint": config["checkpoint"],
                 "revision": config["checkpoint_revision"], "run_id": run_id, "configuration": stable_config},
                 "threshold": config["tau"], "entropy_definition": "global_next_byte_shannon", "entropy_unit": "nats",
                 "entropy_records": [{"byte_offset": i, "entropy": float(h)} for i, h in enumerate(entropy)],
                 "patches": patches, "boundaries": derived_boundaries(patches)}
            nodes = []
            for i, row in enumerate(pd.read_parquet(sp).to_dict("records")):
                if row["file_id"] != sid:
                    raise ValueError("B node sample mismatch")
                nodes.append({**base, "node_id": f"legacy-{i}", "node_type": row["node_type"],
                              "start_byte": int(row["start_byte"]), "end_byte": int(row["end_byte"]),
                              "depth": int(row["depth"]), "is_primary_alignment_target": True})
            regions = []
            for folder, kind in [("identifiers", "identifier"), ("memory_regions", "memory_unsafe"),
                                 ("analysis_regions", "analysis_region")]:
                path = root / f"{folder}/{domain}/{sid}.parquet"
                if path.exists():
                    for i, row in enumerate(pd.read_parquet(path).to_dict("records")):
                        if row["file_id"] != sid:
                            raise ValueError("B region sample mismatch")
                        region = {**base, "region_id": f"{folder}-{i}", "region_kind": kind,
                                  "start_byte": int(row["start_byte"]), "end_byte": int(row["end_byte"])}
                        # B's categorical unit descriptors become O5 controls.
                        if kind == "analysis_region" and row.get("statement_type"):
                            region["controls"] = {"statement_type": str(row["statement_type"])}
                        regions.append(region)
            validate_sample(Sample(m, raw, p, nodes, regions))
            shutil.copyfile(tp, output / m["text_path"])
            patch_stream.write(json.dumps(p, allow_nan=False) + "\n")
            all_manifest.append(m)
            all_nodes.extend(nodes)
            all_regions.extend(regions)
    for kind, rows in [("sample_manifest", all_manifest), ("ground_truth_nodes", all_nodes), ("region_annotations", all_regions)]:
        write_jsonl(output / f"{kind}.jsonl", rows)
    return len(all_manifest)
