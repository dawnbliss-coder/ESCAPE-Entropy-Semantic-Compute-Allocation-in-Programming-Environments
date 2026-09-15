"""Data-contract constants and path conventions, mirroring docs/SCHEMA.md so no
stream hard-codes a column name or a directory layout.

Conventions every consumer relies on:
  - all offsets are BYTE offsets into the frozen corpus/{domain}/{file_id}.bin
    (never character, codepoint or token offsets);
  - a Stream A boundary row marks the FIRST byte of a patch (start-of-patch
    convention), so it is directly comparable with structure/'s start_byte;
  - per-byte entropy is in nats and entropies[i] belongs to byte i (the
    patcher's predictive distribution for byte i given the bytes before it) -
    the exact derivation is in docs/STREAM_A_FIDELITY.md.
"""

import json
import re
from pathlib import Path

DOMAINS = ("py", "cpp", "prose")
CODE_DOMAINS = ("py", "cpp")
SPLITS = ("calib", "main")

FILE_ID_RE = re.compile(r"^(py|cpp|prose)_[0-9a-f]{12}$")

# --- B -> A, C -------------------------------------------------------------
MANIFEST_PATH = "corpus/manifest.parquet"
MANIFEST_COLUMNS = [
    "file_id", "domain", "n_bytes", "sha256", "split", "source_path", "licenses", "parse_ok",
]
STRUCTURE_COLUMNS = ["file_id", "node_type", "parent_type", "depth", "start_byte", "end_byte"]
WHITESPACE_COLUMNS = ["file_id", "byte_offset", "kind"]
WHITESPACE_KINDS = ("newline", "indent_change")
TAXONOMY_PATH = "taxonomy.json"
PROSE_TAXONOMY_PATH = "prose_taxonomy.json"
GOLDEN_DIR = "golden_fixtures"

# --- A -> C ----------------------------------------------------------------
TRIGGERS = ("init", "entropy", "length_cap")
# Contract columns (docs/SCHEMA.md) followed by additive per-patch metadata.
BOUNDARY_CONTRACT_COLUMNS = ["file_id", "byte_offset", "trigger", "entropy"]
BOUNDARY_EXTRA_COLUMNS = ["patch_index", "patch_length"]
BOUNDARY_COLUMNS = BOUNDARY_CONTRACT_COLUMNS + BOUNDARY_EXTRA_COLUMNS
ENTROPY_DTYPE = "float32"
BPP_SUMMARY_PATH = "bpp_summary.parquet"
BPP_SUMMARY_CONTRACT_COLUMNS = ["file_id", "run_id", "n_bytes", "n_patches", "mean_bpp", "var_bpp"]
RUNS_JSON_PATH = "runs.json"
RUNS_JSON_CONTRACT_KEYS = ("tau", "sliding_window", "max_patch_length", "checkpoint", "commit")


def domain_of(file_id: str) -> str:
    m = FILE_ID_RE.match(file_id)
    if not m:
        raise ValueError(f"not a valid file_id: {file_id!r}")
    return m.group(1)


def corpus_bin_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / "corpus" / domain / f"{file_id}.bin"


def structure_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / "structure" / domain / f"{file_id}.parquet"


def whitespace_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / "whitespace" / domain / f"{file_id}.parquet"


def golden_bin_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / GOLDEN_DIR / domain / f"{file_id}.bin"


def golden_structure_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / GOLDEN_DIR / domain / f"{file_id}.structure.parquet"


def golden_whitespace_path(root, domain: str, file_id: str) -> Path:
    return Path(root) / GOLDEN_DIR / domain / f"{file_id}.whitespace.parquet"


def golden_file_ids(root, domain: str) -> list:
    return sorted(p.stem for p in (Path(root) / GOLDEN_DIR / domain).glob("*.bin"))


def entropy_path(root, run_id: str, file_id: str) -> Path:
    return Path(root) / "entropies" / run_id / f"{file_id}.npy"


def boundary_path(root, run_id: str, file_id: str) -> Path:
    return Path(root) / "boundaries" / run_id / f"{file_id}.parquet"


def load_taxonomy(root=".") -> dict:
    with open(Path(root) / TAXONOMY_PATH) as f:
        return json.load(f)


def load_prose_taxonomy(root=".") -> dict:
    with open(Path(root) / PROSE_TAXONOMY_PATH) as f:
        return json.load(f)
