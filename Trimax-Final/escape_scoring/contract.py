"""JSON Schema and cross-record validation; no A/B imports or mutations."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas" / "v1"
WRAPPERS = {"root", "program", "translation_unit", "module", "document", "paragraph"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line, parse_constant=reject)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def write_json(path, value, *, exclusive=False):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(encoded)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


@lru_cache(maxsize=4)
def validator(kind):
    schema = json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text())
    return Draft202012Validator(schema)


def validate_record(kind, record):
    errors = list(validator(kind).iter_errors(record))
    if errors:
        raise ValueError(f"{kind}: {errors[0].message}")
    # Also rejects nonfinite values passed through the API rather than JSON.
    digest(record)


def derived_boundaries(patches):
    return [{"byte_offset": p["end_byte"], "termination_reason": p["termination_reason"]}
            for p in patches]


@dataclass
class Sample:
    manifest: dict
    text: bytes
    patcher: dict
    nodes: list
    regions: list

    @property
    def sample_id(self):
        return self.manifest["sample_id"]


def validate_sample(sample):
    m, p = sample.manifest, sample.patcher
    validate_record("sample_manifest", m)
    validate_record("patcher_output", p)
    text = sample.text.decode("utf-8", errors="strict")
    n = len(sample.text)
    if n != m["byte_length"] or n != p["byte_length"]:
        raise ValueError("UTF-8 byte length mismatch (character offsets are not byte offsets)")
    if hashlib.sha256(sample.text).hexdigest() != m["text_sha256"]:
        raise ValueError("text_sha256 mismatch")
    if p["sample_id"] != sample.sample_id:
        raise ValueError("patcher sample_id mismatch")
    positions = [r["byte_offset"] for r in p["entropy_records"]]
    if positions != list(range(n)):
        raise ValueError("entropy must predict every byte exactly once, in offset order")
    end = 0
    for patch in p["patches"]:
        if patch["start_byte"] != end or not end < patch["end_byte"] <= n:
            raise ValueError("patches must tile the file without gaps, overlap or empty patches")
        end = patch["end_byte"]
        if (patch["termination_reason"] == "eof") != (end == n):
            raise ValueError("observed EOF label must occur only on the final patch")
    if end != n:
        raise ValueError("patches do not cover byte_length")
    if p["boundaries"] != derived_boundaries(p["patches"]):
        raise ValueError("boundary records disagree with patch endpoints/labels")
    # Model boundaries may split UTF-8 codepoints: it is a BYTE model. Parser
    # spans must align with codepoint edges, a useful char/byte drift check.
    edges = {0}
    cursor = 0
    for char in text:
        cursor += len(char.encode("utf-8"))
        edges.add(cursor)
    for kind, rows, id_key in [("ground_truth_nodes", sample.nodes, "node_id"),
                              ("region_annotations", sample.regions, "region_id")]:
        ids = set()
        for row in rows:
            validate_record(kind, row)
            if row["sample_id"] != sample.sample_id or row[id_key] in ids:
                raise ValueError(f"{kind}: foreign sample or duplicate id")
            ids.add(row[id_key])
            a, b = row["start_byte"], row["end_byte"]
            if not 0 <= a <= b <= n or a not in edges or b not in edges:
                raise ValueError(f"{kind}: invalid UTF-8 span [{a}, {b})")
            if kind == "region_annotations" and a == b:
                raise ValueError("empty region")
    return sample


class Dataset:
    """Index JSONL inputs once; load/validate only requested files.

    Indexing parses records to locate sample IDs but does not inspect outcomes.
    Calibration computations load/validate only calibration samples.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = list(read_jsonl(self.root / "sample_manifest.jsonl"))
        ids = set()
        for row in self.manifest:
            validate_record("sample_manifest", row)
            if row["sample_id"] in ids:
                raise ValueError("duplicate manifest sample_id")
            ids.add(row["sample_id"])
        if not ids:
            raise ValueError("empty dataset")
        if len({row["synthetic"] for row in self.manifest}) != 1:
            raise ValueError("synthetic and real samples must be evaluated in separate datasets")
        self._index = {}
        # Byte indexes avoid holding all per-byte entropy JSON in RAM.
        # sample_id alone is used for indexing, never outcomes.
        for kind in ("patcher_output", "ground_truth_nodes", "region_annotations"):
            index = {}
            path = self.root / f"{kind}.jsonl"
            if not path.exists():
                if kind == "region_annotations":
                    self._index[kind] = index
                    continue
                raise FileNotFoundError(path)
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    sid = json.loads(line)["sample_id"]
                    if sid not in ids:
                        raise ValueError(f"{kind}: unknown sample_id {sid}")
                    index.setdefault(sid, []).append(offset)
            self._index[kind] = index

    def rows(self, kind, sid):
        offsets = self._index[kind].get(sid, [])
        if not offsets:
            return []
        with (self.root / f"{kind}.jsonl").open("rb") as stream:
            rows = []
            for offset in offsets:
                stream.seek(offset)
                rows.append(json.loads(stream.readline()))
        return rows

    def load(self, sid):
        m = next(m for m in self.manifest if m["sample_id"] == sid)
        patchers = self.rows("patcher_output", sid)
        if len(patchers) != 1:
            raise ValueError(f"{sid}: expected exactly one patcher output")
        path = (self.root / m["text_path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("text_path must stay inside dataset directory")
        return validate_sample(Sample(m, path.read_bytes(), patchers[0],
                                     self.rows("ground_truth_nodes", sid),
                                     self.rows("region_annotations", sid)))
