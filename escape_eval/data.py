"""File tables and loaders for Stream C.

A file table is the one input every Stream C entry point takes: a DataFrame with
one row per source file and the columns

  file_id, domain, split, n_bytes, bin_path, structure_path, whitespace_path, source

whitespace_path is None where no whitespace baseline applies (prose, in both
layouts). The table is built from either the corpus layout
(corpus/manifest.parquet, corpus/, structure/, whitespace/) or golden_fixtures/.
It is always sorted by (domain in schema.DOMAINS order, file_id). Within a
domain that is the canonical order bootstrap and randomisation weights use, so
results never depend on directory listing or processing order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from escape_common import schema
from escape_common.io import sha256_bytes, sha256_file

FILE_TABLE_COLUMNS = [
    "file_id", "domain", "split", "n_bytes", "bin_path", "structure_path", "whitespace_path", "source",
]
SOURCES = ("corpus", "golden")


# --- manifest and file lists ------------------------------------------------


def load_manifest(root) -> pd.DataFrame | None:
    path = Path(root) / schema.MANIFEST_PATH
    return pd.read_parquet(path) if path.exists() else None


def manifest_sha256(root) -> str | None:
    path = Path(root) / schema.MANIFEST_PATH
    return sha256_file(path) if path.exists() else None


def file_ids_sha256(file_ids) -> str:
    """sha256 of the newline-joined sorted file_id list."""
    return sha256_bytes("\n".join(sorted(str(f) for f in file_ids)).encode("utf-8"))


def _check_domain(domain: str) -> None:
    if domain not in schema.DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {schema.DOMAINS}")


def _finish(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FILE_TABLE_COLUMNS)
    df["n_bytes"] = df["n_bytes"].astype("int64")
    dup = df.loc[df["file_id"].duplicated(), "file_id"].tolist()
    if dup:
        raise ValueError(f"duplicate file_ids in file table: {dup[:5]}")
    rank = {d: i for i, d in enumerate(schema.DOMAINS)}
    df = df.assign(_rank=df["domain"].map(rank)).sort_values(["_rank", "file_id"]).drop(columns="_rank")
    return df.reset_index(drop=True)


def file_table_from_golden(root, domains=schema.DOMAINS, *, split=None, prose_split=None) -> pd.DataFrame:
    """Golden fixtures as a file table. split comes from the manifest when the
    manifest lists the file (None otherwise); n_bytes is the .bin size and must
    agree with the manifest. `split` / `prose_split` filter like the corpus
    builder does (default: no filter)."""
    root = Path(root)
    manifest = load_manifest(root)
    info = {} if manifest is None else manifest.set_index("file_id")[["split", "n_bytes"]].to_dict("index")
    rows = []
    for domain in domains:
        _check_domain(domain)
        want = prose_split if (domain == "prose" and prose_split is not None) else split
        for fid in schema.golden_file_ids(root, domain):
            if schema.domain_of(fid) != domain:
                raise ValueError(f"{fid} found under golden_fixtures/{domain}")
            bin_path = schema.golden_bin_path(root, domain, fid)
            n_bytes = bin_path.stat().st_size
            meta = info.get(fid)
            if meta is not None and int(meta["n_bytes"]) != n_bytes:
                raise ValueError(f"{fid}: golden .bin has {n_bytes} bytes, manifest says {meta['n_bytes']}")
            file_split = None if meta is None else meta["split"]
            if want is not None and file_split != want:
                continue
            spath = schema.golden_structure_path(root, domain, fid)
            if not spath.exists():
                raise FileNotFoundError(spath)
            wpath = None
            if domain in schema.CODE_DOMAINS:
                wp = schema.golden_whitespace_path(root, domain, fid)
                if not wp.exists():
                    raise FileNotFoundError(wp)
                wpath = str(wp)
            rows.append({
                "file_id": fid, "domain": domain, "split": file_split, "n_bytes": n_bytes,
                "bin_path": str(bin_path), "structure_path": str(spath), "whitespace_path": wpath,
                "source": "golden",
            })
    return _finish(rows)


def file_table_from_corpus(
    root, domains=schema.DOMAINS, *, split=None, prose_split=None, require_files: bool = True
) -> pd.DataFrame:
    """Manifest files as a file table. `split` filters py/cpp, `prose_split`
    filters prose (default: same as split; prose has no calib split, so pass
    prose_split='main' to include prose next to a calib code split).
    require_files: raise if a .bin, structure or (code) whitespace file is missing."""
    root = Path(root)
    manifest = load_manifest(root)
    if manifest is None:
        raise FileNotFoundError(root / schema.MANIFEST_PATH)
    rows, missing = [], []
    for domain in domains:
        _check_domain(domain)
        sub = manifest[manifest["domain"] == domain]
        want = prose_split if (domain == "prose" and prose_split is not None) else split
        if want is not None:
            sub = sub[sub["split"] == want]
        for fid, n_bytes, file_split in sub[["file_id", "n_bytes", "split"]].itertuples(index=False, name=None):
            bpath = schema.corpus_bin_path(root, domain, fid)
            spath = schema.structure_path(root, domain, fid)
            wpath = schema.whitespace_path(root, domain, fid) if domain in schema.CODE_DOMAINS else None
            if require_files:
                missing.extend(str(p) for p in (bpath, spath, wpath) if p is not None and not p.exists())
            rows.append({
                "file_id": fid, "domain": domain, "split": file_split, "n_bytes": int(n_bytes),
                "bin_path": str(bpath), "structure_path": str(spath),
                "whitespace_path": None if wpath is None else str(wpath), "source": "corpus",
            })
    if missing:
        raise FileNotFoundError(f"{len(missing)} input files missing, e.g. {missing[:3]}")
    return _finish(rows)


def select_files(table: pd.DataFrame, *, file_ids=None, limit_per_domain=None) -> pd.DataFrame:
    """Restrict to explicit file_ids and/or the first N per domain in canonical
    (file_id) order. file_id is a content-hash prefix, so the first N is a
    deterministic, content-blind subset."""
    out = table
    if file_ids is not None:
        wanted = list(dict.fromkeys(str(f) for f in file_ids))
        unknown = sorted(set(wanted) - set(table["file_id"]))
        if unknown:
            raise KeyError(f"file_ids not in the file table: {unknown[:5]}")
        out = out[out["file_id"].isin(wanted)]
    if limit_per_domain is not None:
        out = out.groupby("domain", sort=False, group_keys=False).head(int(limit_per_domain))
    return out.reset_index(drop=True)


def build_file_table(
    root,
    source: str,
    domains=schema.DOMAINS,
    *,
    split=None,
    prose_split=None,
    file_ids=None,
    limit_per_domain=None,
    require_files: bool = True,
) -> pd.DataFrame:
    if source == "golden":
        table = file_table_from_golden(root, domains, split=split, prose_split=prose_split)
    elif source == "corpus":
        table = file_table_from_corpus(
            root, domains, split=split, prose_split=prose_split, require_files=require_files
        )
    else:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return select_files(table, file_ids=file_ids, limit_per_domain=limit_per_domain)


def manifest_splits(root, file_ids) -> dict:
    """file_id -> manifest split (None when the manifest does not list the file)."""
    manifest = load_manifest(root)
    if manifest is None:
        return {f: None for f in file_ids}
    lut = dict(zip(manifest["file_id"], manifest["split"]))
    return {f: lut.get(f) for f in file_ids}


# --- Stream B loaders -------------------------------------------------------


def load_structure(path, file_id: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in schema.STRUCTURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: structure file missing columns {missing}")
    if file_id is not None and len(df) and not (df["file_id"] == file_id).all():
        raise ValueError(f"{path}: rows with file_id != {file_id!r}")
    out = pd.DataFrame({
        "node_type": df["node_type"].astype(str).to_numpy(dtype=object),
        "parent_type": df["parent_type"].astype(object).to_numpy(),
        "depth": df["depth"].to_numpy(dtype=np.int64),
        "start_byte": df["start_byte"].to_numpy(dtype=np.int64),
        "end_byte": df["end_byte"].to_numpy(dtype=np.int64),
    })
    return out


def check_structure_bounds(structure: pd.DataFrame, n_bytes: int, label: str = "structure") -> None:
    """Raise if any node span is outside [0, n_bytes] or reversed (offset drift)."""
    if structure.empty:
        return
    s = structure["start_byte"].to_numpy()
    e = structure["end_byte"].to_numpy()
    bad = (s < 0) | (e > n_bytes) | (e < s)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{label}: node span [{s[i]}, {e[i]}) invalid for n_bytes={n_bytes}")


def load_newline_offsets(path, file_id: str | None = None) -> np.ndarray:
    """Unique sorted byte_offset of kind == 'newline' rows (the P1 whitespace baseline)."""
    df = pd.read_parquet(path)
    missing = [c for c in schema.WHITESPACE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: whitespace file missing columns {missing}")
    if file_id is not None and len(df) and not (df["file_id"] == file_id).all():
        raise ValueError(f"{path}: rows with file_id != {file_id!r}")
    return np.unique(df.loc[df["kind"] == "newline", "byte_offset"].to_numpy(dtype=np.int64))


def load_taxonomy(root, domain: str) -> dict:
    """taxonomy.json for py/cpp, prose_taxonomy.json for prose."""
    _check_domain(domain)
    return schema.load_prose_taxonomy(root) if domain == "prose" else schema.load_taxonomy(root)


# --- Stream A loaders -------------------------------------------------------


def load_runs(stream_a_root) -> dict:
    path = Path(stream_a_root) / schema.RUNS_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_run_entry(stream_a_root, run_id: str) -> dict:
    runs = load_runs(stream_a_root)
    if run_id not in runs:
        raise KeyError(f"run_id {run_id!r} not in {Path(stream_a_root) / schema.RUNS_JSON_PATH}")
    return runs[run_id]


def load_boundaries(stream_a_root, run_id: str, file_id: str) -> pd.DataFrame:
    path = schema.boundary_path(stream_a_root, run_id, file_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_entropies(stream_a_root, run_id: str, file_id: str) -> np.ndarray:
    path = schema.entropy_path(stream_a_root, run_id, file_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def stream_a_outputs_exist(stream_a_root, run_id: str, file_id: str) -> bool:
    return schema.boundary_path(stream_a_root, run_id, file_id).exists()


def load_bpp_summary(stream_a_root, run_id: str | None = None) -> pd.DataFrame | None:
    path = Path(stream_a_root) / schema.BPP_SUMMARY_PATH
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    return df.reset_index(drop=True)
