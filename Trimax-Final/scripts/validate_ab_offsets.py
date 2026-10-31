#!/usr/bin/env python3
"""Phase 8: cross-validate Stream A and Stream B byte offsets on the same frozen bytes
(the proposal's "parser offset drift" risk).

Checks, per file (golden fixtures, plus any corpus file for which the run has outputs):
  bytes     golden .bin (and corpus .bin when both exist) == manifest n_bytes + sha256
  stream A  full contract validation incl. completeness (escape_common.validate)
  stream B  structure spans inside [0, n], start <= end, both offsets on UTF-8 character
            boundaries, and every span decodes as UTF-8;
            whitespace 'newline' offsets == positions right after each '\\n' except a final
            one, recomputed independently from the bytes;
            'indent_change' offsets == the first non-blank byte after a run of spaces/tabs at
            the start of a line
  joins     - shared starts: number of rows per unique start, and outermost attribution
              == minimum depth (brute force);
            - BLT entropy boundaries that are not UTF-8 character boundaries (byte-level
              patching may split a multi-byte character; AST offsets never do);
            - coincidence rates used when reading alignment numbers: AST starts at
              indent_change offsets (R2), BLT entropy boundaries at newline /
              indent_change / exact AST-start offsets.

Writes results/stream_ab_offsets/{run_id}/offset_validation.json. Exits 1 on any hard failure.

Usage: .venv_a/bin/python scripts/validate_ab_offsets.py --run-id RUN
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from escape_common import schema  # noqa: E402
from escape_common.io import atomic_write_json  # noqa: E402
from escape_common.offsets import char_start_mask, utf8_roundtrip_ok  # noqa: E402
from escape_common.validate import validate_file  # noqa: E402
from escape_eval.targets import attribute  # noqa: E402


def recompute_newlines(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    pos = np.flatnonzero(arr == 0x0A) + 1
    return pos[pos < len(content)]


def indent_change_ok(content: bytes, offsets) -> int:
    """Number of indent_change offsets that do NOT point at the first non-blank byte of a line."""
    bad = 0
    for o in offsets:
        o = int(o)
        line_start = content.rfind(b"\n", 0, o) + 1
        prefix = content[line_start:o]
        if o >= len(content) or content[o:o + 1] in (b" ", b"\t", b"\n") or prefix.strip(b" \t") != b"":
            bad += 1
    return bad


def check_file(root: Path, run_id: str, entry: dict, row, source: str) -> dict:
    fid, domain, n = row.file_id, row.domain, int(row.n_bytes)
    out = {"file_id": fid, "domain": domain, "source": source, "n_bytes": n, "problems": []}
    P = out["problems"]
    gb = schema.golden_bin_path(root, domain, fid)
    cb = schema.corpus_bin_path(root, domain, fid)
    bin_path = gb if source == "golden" else cb
    content = bin_path.read_bytes()
    if len(content) != n or hashlib.sha256(content).hexdigest() != row.sha256:
        P.append("bytes do not match manifest n_bytes/sha256")
    if gb.exists() and cb.exists() and gb.read_bytes() != cb.read_bytes():
        P.append("golden .bin differs from corpus .bin")
    valid_utf8 = utf8_roundtrip_ok(content)
    out["valid_utf8"] = valid_utf8
    out["non_ascii_bytes"] = int(sum(1 for b in content if b >= 0x80))
    char_start = np.append(char_start_mask(content), True) if valid_utf8 else None

    problems_a, bdf = validate_file(root, run_id, fid, n, tau=entry["tau"], max_patch_length=entry["max_patch_length"],
                                    check_entropies=True, check_completeness=True)
    P.extend(f"stream A: {p}" for p in problems_a)
    blt = np.empty(0, dtype=np.int64)
    if bdf is not None:
        blt = bdf.loc[bdf["trigger"] == "entropy", "byte_offset"].to_numpy(dtype=np.int64)
        out["n_blt_entropy_boundaries"] = int(blt.size)
        if char_start is not None:
            out["blt_boundaries_inside_multibyte_char"] = int((~char_start[blt]).sum())

    spath = schema.golden_structure_path(root, domain, fid) if source == "golden" else schema.structure_path(root, domain, fid)
    starts = np.empty(0, dtype=np.int64)
    if spath.exists():
        s = pd.read_parquet(spath)
        out["structure_rows"] = int(len(s))
        if len(s):
            sb = s["start_byte"].to_numpy(dtype=np.int64)
            eb = s["end_byte"].to_numpy(dtype=np.int64)
            if ((sb < 0) | (eb > n) | (eb < sb)).any():
                P.append("structure span outside [0, n] or reversed")
            elif char_start is not None:
                n_bad = int((~char_start[sb]).sum() + (~char_start[eb]).sum())
                if n_bad:
                    P.append(f"{n_bad} structure offsets inside a multi-byte UTF-8 character")
                undecodable = 0
                for a, b in zip(sb, eb):
                    try:
                        content[a:b].decode("utf-8")
                    except UnicodeDecodeError:
                        undecodable += 1
                if undecodable:
                    P.append(f"{undecodable} structure spans do not decode as UTF-8")
            starts = np.unique(sb)
            groups = s.groupby("start_byte")
            out["unique_starts"] = int(starts.size)
            out["starts_shared_by_several_rows"] = int((groups.size() > 1).sum())
            att = attribute(s, "start").set_index("offset")["depth"]
            min_depth = groups["depth"].min()
            if not att.sort_index().equals(min_depth.sort_index().rename("depth").rename_axis("offset")):
                P.append("outermost attribution != minimum depth for some shared start")
            out["blt_at_exact_ast_start"] = int(np.isin(blt, starts).sum())
    else:
        out["structure_missing"] = True

    wpath = schema.golden_whitespace_path(root, domain, fid) if source == "golden" else schema.whitespace_path(root, domain, fid)
    if wpath.exists():
        w = pd.read_parquet(wpath)
        nl = np.sort(w.loc[w["kind"] == "newline", "byte_offset"].to_numpy(dtype=np.int64))
        ic = np.sort(w.loc[w["kind"] == "indent_change", "byte_offset"].to_numpy(dtype=np.int64))
        if not np.array_equal(nl, recompute_newlines(content)):
            P.append("whitespace newline offsets != independent recomputation")
        n_bad_ic = indent_change_ok(content, ic)
        if n_bad_ic:
            P.append(f"{n_bad_ic} indent_change offsets not at the first non-blank byte of a line")
        out["n_newline"], out["n_indent_change"] = int(nl.size), int(ic.size)
        out["blt_at_newline"] = int(np.isin(blt, nl).sum())
        out["blt_at_indent_change"] = int(np.isin(blt, ic).sum())
        if starts.size:
            out["ast_starts_at_indent_change"] = int(np.isin(starts, ic).sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    entry = json.loads((root / schema.RUNS_JSON_PATH).read_text())[args.run_id]
    manifest = pd.read_parquet(root / schema.MANIFEST_PATH)
    rows = []
    for domain in schema.DOMAINS:
        golden = set(schema.golden_file_ids(root, domain))
        m = manifest[manifest["domain"] == domain]
        for r in m[m["file_id"].isin(golden)].itertuples(index=False):
            rows.append(check_file(root, args.run_id, entry, r, "golden"))
        for r in m[~m["file_id"].isin(golden)].itertuples(index=False):
            if schema.boundary_path(root, args.run_id, r.file_id).exists() and schema.corpus_bin_path(root, domain, r.file_id).exists():
                rows.append(check_file(root, args.run_id, entry, r, "corpus"))
    df = pd.DataFrame(rows)
    summary = {}
    for (domain, source), g in df.groupby(["domain", "source"]):
        agg = {"n_files": int(len(g)), "n_files_with_problems": int((g["problems"].str.len() > 0).sum()),
               "n_files_non_ascii": int((g["non_ascii_bytes"] > 0).sum())}
        for col in ("n_blt_entropy_boundaries", "blt_boundaries_inside_multibyte_char", "unique_starts",
                    "starts_shared_by_several_rows", "blt_at_exact_ast_start", "n_newline", "n_indent_change",
                    "blt_at_newline", "blt_at_indent_change", "ast_starts_at_indent_change"):
            if col in g:
                agg[col] = int(g[col].fillna(0).sum())
        if "structure_missing" in g:
            agg["n_files_structure_missing"] = int(g["structure_missing"].fillna(False).astype(bool).sum())
        b = agg.get("n_blt_entropy_boundaries", 0)
        if b:
            for col in ("blt_at_exact_ast_start", "blt_at_newline", "blt_at_indent_change", "blt_boundaries_inside_multibyte_char"):
                if col in agg:
                    agg[f"frac_{col}"] = agg[col] / b
        if agg.get("unique_starts") and "ast_starts_at_indent_change" in agg:
            agg["frac_ast_starts_at_indent_change"] = agg["ast_starts_at_indent_change"] / agg["unique_starts"]
        summary[f"{domain}/{source}"] = agg
    problems = [f"{r['file_id']}: {p}" for r in rows for p in r["problems"]]
    report = {"run_id": args.run_id, "summary": summary, "n_files": len(rows), "problems": problems,
              "per_file": rows}
    out = root / "results" / "stream_ab_offsets" / args.run_id / "offset_validation.json"
    atomic_write_json(out, report)
    for key, agg in summary.items():
        print(key, json.dumps(agg))
    print(f"{'OK' if not problems else 'FAIL'}: {len(rows)} files, {len(problems)} problems -> {out.relative_to(root)}")
    for p in problems[:20]:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
