"""Alignment targets from Stream B structure rows.

Start targets T_f: the unique start_byte values of a file's structure rows,
minus the forced (init) boundary offsets X_f. Nested nodes that share a start
are one target, never double counted. Each unique start is attributed to its
OUTERMOST node, which supplies its depth, node_type and category:
  min depth, then larger end_byte, then lexicographically smallest node_type
(stream_b/ast_walker.py: outermost = smallest depth among rows sharing a start).

End targets E_f: the unique end_byte values minus X_f, attributed to the
outermost node sharing that end:
  min depth, then smaller start_byte, then lexicographically smallest node_type.

category comes from taxonomy.json for py/cpp and prose_taxonomy.json for prose.
A label missing from the taxonomy falls back to its outermost '+'-joined label
(the rule stream_b/build_prose_taxonomy.py classifies by), else 'unmapped'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AXES = ("all", "depth", "node_type", "category")
TARGET_KINDS = ("start", "end")
UNMAPPED = "unmapped"


def category_of(node_type: str, taxonomy: dict) -> str:
    if node_type in taxonomy:
        return taxonomy[node_type]
    outer = node_type.split("+", 1)[0]
    if outer in taxonomy:
        return taxonomy[outer]
    return UNMAPPED


@dataclass(frozen=True)
class Targets:
    kind: str                 # "start" | "end"
    offsets: np.ndarray       # int64, strictly increasing
    depth: np.ndarray         # int64, depth of the attributed (outermost) node
    node_type: np.ndarray     # object (str)
    category: np.ndarray      # object (str)
    n_forced_excluded: int = 0  # unique offsets dropped because they are init offsets
    n_eof: int = 0              # kept offsets equal to n_bytes (possible for ends only)
    n_eof_excluded: int = 0     # offsets == n_bytes dropped by exclude_eof_end

    def __len__(self) -> int:
        return int(self.offsets.size)

    def stratum_values(self, axis: str) -> np.ndarray:
        if axis == "all":
            return np.full(len(self), "all", dtype=object)
        if axis == "depth":
            return np.array([str(int(d)) for d in self.depth], dtype=object)
        if axis == "node_type":
            return self.node_type
        if axis == "category":
            return self.category
        raise ValueError(f"unknown stratum axis {axis!r}")


def attribute(structure: pd.DataFrame, kind: str) -> pd.DataFrame:
    """One row per unique offset of `kind` with its outermost node, sorted by
    offset. Columns: offset, node_type, depth, start_byte, end_byte."""
    df = structure[["node_type", "depth", "start_byte", "end_byte"]]
    if kind == "start":
        df = df.sort_values(
            ["start_byte", "depth", "end_byte", "node_type"], ascending=[True, True, False, True]
        ).drop_duplicates("start_byte", keep="first")
        offsets = df["start_byte"].to_numpy(dtype=np.int64)
    elif kind == "end":
        df = df.sort_values(
            ["end_byte", "depth", "start_byte", "node_type"], ascending=[True, True, True, True]
        ).drop_duplicates("end_byte", keep="first")
        offsets = df["end_byte"].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"kind must be one of {TARGET_KINDS}, got {kind!r}")
    out = df.reset_index(drop=True)
    out.insert(0, "offset", offsets)
    return out


def build_targets(
    structure: pd.DataFrame,
    kind: str,
    *,
    taxonomy: dict,
    exclude_offsets=(),
    n_bytes: int | None = None,
    exclude_eof_end: bool = False,
) -> Targets:
    """T_f (kind='start') or E_f (kind='end') for one file.

    exclude_offsets: X_f, the init offsets read from the boundaries file.
    exclude_eof_end: drop end targets at offset == n_bytes (no patch can start
    there). Off by default: the definition keeps them; see STREAM_C_METHODS.md."""
    att = attribute(structure, kind)
    offsets = att["offset"].to_numpy(dtype=np.int64)
    excl = np.unique(np.asarray(list(exclude_offsets), dtype=np.int64))
    keep = ~np.isin(offsets, excl) if excl.size else np.ones(offsets.size, dtype=bool)
    n_forced = int(np.count_nonzero(~keep))
    n_eof = n_eof_excl = 0
    if n_bytes is not None:
        at_eof = offsets == int(n_bytes)
        n_eof = int(np.count_nonzero(at_eof & keep))
        if exclude_eof_end and kind == "end":
            keep &= ~at_eof
            n_eof_excl, n_eof = n_eof, 0
    att = att[keep]
    node_type = att["node_type"].astype(str).to_numpy(dtype=object)
    category = np.array([category_of(t, taxonomy) for t in node_type], dtype=object)
    return Targets(
        kind=kind,
        offsets=offsets[keep],
        depth=att["depth"].to_numpy(dtype=np.int64),
        node_type=node_type,
        category=category,
        n_forced_excluded=n_forced,
        n_eof=n_eof,
        n_eof_excluded=n_eof_excl,
    )


def unique_spans(structure: pd.DataFrame) -> tuple:
    """Unique non-empty node spans [s, e) (for the secondary IoU metric)."""
    if structure.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    spans = structure[["start_byte", "end_byte"]].drop_duplicates()
    spans = spans[spans["end_byte"] > spans["start_byte"]].sort_values(["start_byte", "end_byte"])
    return spans["start_byte"].to_numpy(dtype=np.int64), spans["end_byte"].to_numpy(dtype=np.int64)
