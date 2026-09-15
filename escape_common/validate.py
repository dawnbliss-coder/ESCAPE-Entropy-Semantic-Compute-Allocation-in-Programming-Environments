"""Validators for the Stream A -> Stream C data contract.

Contract: docs/SCHEMA.md and escape_common/schema.py, plus the boundary-file
invariants Stream C relies on (listed in validate_boundaries). Every validator
returns a list of problem strings, and an empty list means valid. Nothing here
raises on bad data, so a whole run is checked in one pass.

CLI (exit status 1 if any problem is found):

  python -m escape_common.validate --root . --run-id RUN --golden
  python -m escape_common.validate --root . --run-id RUN --split calib [--domain py cpp]

--root holds corpus/manifest.parquet and golden_fixtures/. --stream-a-root holds
entropies/, boundaries/, runs.json and bpp_summary.parquet (default: --root).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from escape_common import schema

BOUNDARY_NUMERIC_DTYPES = {
    "byte_offset": np.dtype("int64"),
    "entropy": np.dtype("float32"),
    "patch_index": np.dtype("int32"),
    "patch_length": np.dtype("int32"),
}
BOUNDARY_STRING_COLUMNS = ("file_id", "trigger")
# A distribution over the patcher's 260 output ids has entropy <= ln(260) nats.
MAX_ENTROPY_NATS = math.log(260)
ENTROPY_TOLERANCE = 1e-3
BPP_REL_TOLERANCE = 1e-6
BPP_ABS_TOLERANCE = 1e-9


def _examples(values, limit: int = 5) -> str:
    values = list(values)
    shown = [int(v) if isinstance(v, (int, np.integer)) else v for v in values[:limit]]
    more = len(values) - len(shown)
    return f"{shown}" + (f" (+{more} more)" if more > 0 else "")


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def _is_string_column(series: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(series.dtype) and series.dtype != object:
        return not series.isna().any()
    if series.dtype == object:
        return all(isinstance(v, str) for v in series.tolist())
    return False


def _integral_array(series: pd.Series):
    dt = series.dtype
    if pd.api.types.is_bool_dtype(dt):
        return None
    if pd.api.types.is_integer_dtype(dt):
        if series.isna().any():
            return None
        return series.to_numpy(dtype=np.int64)
    if pd.api.types.is_float_dtype(dt):
        v = series.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(v)) or not np.all(v == np.round(v)):
            return None
        return v.astype(np.int64)
    return None


def _close(a, b, rel=BPP_REL_TOLERANCE, abs_tol=BPP_ABS_TOLERANCE) -> bool:
    try:
        a = float(a)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isclose(a, float(b), rel_tol=rel, abs_tol=abs_tol)


# --- per-file validators ----------------------------------------------------


def validate_entropy_array(arr, n_bytes: int, label: str = "entropies") -> list:
    """float32, 1-D, length n_bytes, finite, within [0, ln 260] nats (+tolerance)."""
    if not isinstance(arr, np.ndarray):
        return [f"{label}: expected a numpy array, got {type(arr).__name__}"]
    problems = []
    if arr.dtype != np.float32:
        problems.append(f"{label}: dtype {arr.dtype}, expected float32")
    if arr.ndim != 1:
        problems.append(f"{label}: shape {arr.shape}, expected 1-D")
        return problems
    if arr.shape[0] != n_bytes:
        problems.append(f"{label}: length {arr.shape[0]} != n_bytes {n_bytes}")
    if not np.issubdtype(arr.dtype, np.floating):
        return problems
    v = arr.astype(np.float64)
    bad = ~np.isfinite(v)
    if bad.any():
        problems.append(f"{label}: {int(bad.sum())} non-finite values")
    fin = v[~bad]
    if fin.size:
        if fin.min() < -ENTROPY_TOLERANCE:
            problems.append(f"{label}: negative entropy (min {fin.min():.6g})")
        if fin.max() > MAX_ENTROPY_NATS + ENTROPY_TOLERANCE:
            problems.append(
                f"{label}: entropy above ln(260) = {MAX_ENTROPY_NATS:.4f} nats "
                f"(max {fin.max():.6g}); bits instead of nats?"
            )
    return problems


def validate_boundaries(
    df,
    *,
    n_bytes: int,
    tau,
    max_patch_length,
    file_id: str | None = None,
    entropies=None,
    check_completeness: bool = False,
) -> list:
    """Checks one boundaries/{run_id}/{file_id}.parquet frame.

    Invariants (docs/SCHEMA.md + the Stream A -> C contract):
      - columns schema.BOUNDARY_COLUMNS; byte_offset int64, entropy float32,
        patch_index int32, patch_length int32; file_id and trigger strings
      - n_bytes == 0  =>  zero rows
      - byte_offset strictly increasing, all in [0, n_bytes)
      - first row: byte_offset 0 with trigger 'init'
      - trigger in {init, entropy, length_cap}; file_id equals the expected id
      - patch_index == 0..n_patches-1
      - patch_length == next start - start (last patch runs to n_bytes), and
        sum(patch_length) == n_bytes
      - entropy finite
      - trigger == 'entropy'    => float64(entropy) > tau
      - trigger == 'length_cap' => max_patch_length is set, the previous patch
        has length max_patch_length, and float64(entropy) <= tau
      - max_patch_length is not None => every patch_length <= max_patch_length
      - entropies given: a valid entropy array, and the entropy column equals
        entropies[byte_offset] bit for bit
      - check_completeness (optional, NOT in the written contract): every byte
        i that is not an init offset and has float64(entropies[i]) > tau is an
        'entropy' row
    """
    lab = f"boundaries[{file_id}]" if file_id else "boundaries"
    if not isinstance(df, pd.DataFrame):
        return [f"{lab}: expected a pandas DataFrame, got {type(df).__name__}"]
    missing = [c for c in schema.BOUNDARY_COLUMNS if c not in df.columns]
    if missing:
        return [f"{lab}: missing columns {missing}"]

    problems = []
    for col, dt in BOUNDARY_NUMERIC_DTYPES.items():
        if df[col].dtype != dt:
            problems.append(f"{lab}: column {col!r} has dtype {df[col].dtype}, expected {dt}")
    for col in BOUNDARY_STRING_COLUMNS:
        if not _is_string_column(df[col]):
            problems.append(f"{lab}: column {col!r} must hold non-null strings (dtype {df[col].dtype})")

    if max_patch_length is not None and not (_is_int(max_patch_length) and max_patch_length >= 1):
        problems.append(f"{lab}: max_patch_length must be None or a positive integer, got {max_patch_length!r}")
        max_patch_length = None
    if not (_is_number(tau) and math.isfinite(float(tau))):
        problems.append(f"{lab}: tau must be a finite number, got {tau!r}")
        return problems
    tau_f = float(tau)

    n = len(df)
    if n_bytes == 0:
        if n:
            problems.append(f"{lab}: n_bytes == 0 but the file has {n} rows (expected zero rows)")
        return problems
    if n == 0:
        problems.append(f"{lab}: zero rows for n_bytes={n_bytes} (expected an 'init' row at offset 0)")
        return problems

    off = _integral_array(df["byte_offset"])
    plen = _integral_array(df["patch_length"])
    pidx = _integral_array(df["patch_index"])
    if off is None or plen is None or pidx is None:
        problems.append(f"{lab}: byte_offset, patch_length and patch_index must hold integers without nulls")
        return problems
    if not pd.api.types.is_numeric_dtype(df["entropy"].dtype) or pd.api.types.is_bool_dtype(df["entropy"].dtype):
        problems.append(f"{lab}: entropy column is not numeric")
        return problems
    ent = df["entropy"].to_numpy(dtype=np.float64, na_value=np.nan)
    trig = np.array(
        [t if isinstance(t, str) else repr(t) for t in df["trigger"].astype(object).tolist()], dtype=object
    )

    unknown = sorted(set(trig.tolist()) - set(schema.TRIGGERS))
    if unknown:
        problems.append(f"{lab}: unknown trigger values {unknown}")
    if file_id is not None:
        n_bad = sum(1 for f in df["file_id"].astype(object).tolist() if f != file_id)
        if n_bad:
            problems.append(f"{lab}: {n_bad} rows with file_id != {file_id!r}")

    if off[0] != 0 or trig[0] != "init":
        problems.append(
            f"{lab}: first row must be byte_offset 0 with trigger 'init' "
            f"(got offset {int(off[0])}, trigger {trig[0]!r})"
        )
    bad = np.flatnonzero(np.diff(off) <= 0)
    if bad.size:
        problems.append(f"{lab}: byte_offset not strictly increasing at rows {_examples(bad + 1)}")
    out_of_range = np.flatnonzero((off < 0) | (off >= n_bytes))
    if out_of_range.size:
        problems.append(f"{lab}: byte_offset outside [0, {n_bytes}) at rows {_examples(out_of_range)}")
    if not np.array_equal(pidx, np.arange(n, dtype=np.int64)):
        problems.append(f"{lab}: patch_index is not 0..{n - 1}")
    bad = np.flatnonzero(plen <= 0)
    if bad.size:
        problems.append(f"{lab}: non-positive patch_length at rows {_examples(bad)}")
    total = int(plen.sum())
    if total != n_bytes:
        problems.append(f"{lab}: sum(patch_length) = {total} != n_bytes {n_bytes}")
    expected = np.diff(np.append(off, n_bytes))
    bad = np.flatnonzero(plen != expected)
    if bad.size:
        problems.append(f"{lab}: patch_length != next start - start at rows {_examples(bad)}")
    nonfinite = np.flatnonzero(~np.isfinite(ent))
    if nonfinite.size:
        problems.append(f"{lab}: non-finite entropy at rows {_examples(nonfinite)}")

    is_ent = trig == "entropy"
    bad = np.flatnonzero(is_ent & ~(ent > tau_f))
    if bad.size:
        problems.append(
            f"{lab}: {bad.size} 'entropy' rows with entropy <= tau={tau_f!r} at offsets {_examples(off[bad])}"
        )
    is_cap = trig == "length_cap"
    if is_cap.any():
        if max_patch_length is None:
            problems.append(f"{lab}: {int(is_cap.sum())} 'length_cap' rows but max_patch_length is None")
        else:
            prev = np.empty(n, dtype=np.int64)
            prev[0] = -1
            prev[1:] = plen[:-1]
            bad = np.flatnonzero(is_cap & (prev != int(max_patch_length)))
            if bad.size:
                problems.append(
                    f"{lab}: 'length_cap' rows whose previous patch length != max_patch_length="
                    f"{max_patch_length} at offsets {_examples(off[bad])}"
                )
        bad = np.flatnonzero(is_cap & ~(ent <= tau_f))
        if bad.size:
            problems.append(
                f"{lab}: 'length_cap' rows with entropy > tau={tau_f!r} at offsets {_examples(off[bad])}"
            )
    if max_patch_length is not None:
        bad = np.flatnonzero(plen > int(max_patch_length))
        if bad.size:
            problems.append(
                f"{lab}: patch_length > max_patch_length={max_patch_length} at rows {_examples(bad)}"
            )

    if entropies is not None:
        problems.extend(validate_entropy_array(entropies, n_bytes, label=f"entropies[{file_id}]"))
        usable = (
            isinstance(entropies, np.ndarray)
            and entropies.ndim == 1
            and entropies.shape[0] == n_bytes
            and np.issubdtype(entropies.dtype, np.floating)
            and not out_of_range.size
        )
        if usable:
            expect = np.ascontiguousarray(entropies[off].astype(np.float32))
            got = np.ascontiguousarray(df["entropy"].to_numpy(dtype=np.float32, na_value=np.nan))
            mism = np.flatnonzero(expect.view(np.uint32) != got.view(np.uint32))
            if mism.size:
                problems.append(
                    f"{lab}: entropy column != entropies[byte_offset] at offsets {_examples(off[mism])}"
                )
            if check_completeness:
                high = np.flatnonzero(entropies.astype(np.float64) > tau_f)
                high = high[~np.isin(high, off[trig == "init"])]
                missed = high[~np.isin(high, off[is_ent])]
                if missed.size:
                    problems.append(
                        f"{lab}: {missed.size} bytes with entropy > tau are not 'entropy' boundaries, "
                        f"e.g. offsets {_examples(missed)}"
                    )
    elif check_completeness:
        problems.append(f"{lab}: check_completeness requires the entropy array")
    return problems


# --- run-level validators ---------------------------------------------------


def validate_runs_json(runs, run_id: str | None = None) -> list:
    """runs.json: run_id -> {tau, sliding_window, max_patch_length, checkpoint, commit, ...}."""
    if not isinstance(runs, dict):
        return ["runs.json: top level must be an object mapping run_id -> entry"]
    problems = []
    for rid, entry in runs.items():
        lab = f"runs.json[{rid!r}]"
        if not isinstance(entry, dict):
            problems.append(f"{lab}: entry must be an object")
            continue
        missing = [k for k in schema.RUNS_JSON_CONTRACT_KEYS if k not in entry]
        if missing:
            problems.append(f"{lab}: missing contract keys {missing}")
        if "tau" in entry and not (_is_number(entry["tau"]) and math.isfinite(float(entry["tau"]))):
            problems.append(f"{lab}: tau must be a finite number, got {entry['tau']!r}")
        for key in ("sliding_window", "max_patch_length"):
            value = entry.get(key)
            if key in entry and value is not None and not (_is_int(value) and value >= 1):
                problems.append(f"{lab}: {key} must be null or a positive integer, got {value!r}")
        for key in ("checkpoint", "commit"):
            if key in entry and not (isinstance(entry[key], str) and entry[key]):
                problems.append(f"{lab}: {key} must be a non-empty string, got {entry[key]!r}")
    if run_id is not None and run_id not in runs:
        problems.append(f"runs.json: no entry for run_id {run_id!r}")
    return problems


def validate_bpp_summary(df, *, runs=None, label: str = "bpp_summary") -> list:
    """bpp_summary.parquet contract columns and internal consistency."""
    if not isinstance(df, pd.DataFrame):
        return [f"{label}: expected a pandas DataFrame"]
    missing = [c for c in schema.BPP_SUMMARY_CONTRACT_COLUMNS if c not in df.columns]
    if missing:
        return [f"{label}: missing columns {missing}"]
    problems = []
    for col in ("file_id", "run_id"):
        if not _is_string_column(df[col]):
            problems.append(f"{label}: column {col!r} must hold non-null strings")
    nb = _integral_array(df["n_bytes"])
    npch = _integral_array(df["n_patches"])
    if nb is None or npch is None or not pd.api.types.is_integer_dtype(df["n_bytes"].dtype) or not pd.api.types.is_integer_dtype(df["n_patches"].dtype):
        problems.append(f"{label}: n_bytes and n_patches must be integer columns without nulls")
        return problems
    for col in ("mean_bpp", "var_bpp"):
        if not pd.api.types.is_float_dtype(df[col].dtype):
            problems.append(f"{label}: column {col!r} must be floating point (dtype {df[col].dtype})")
    if problems:
        return problems
    dup = df.duplicated(["file_id", "run_id"])
    if dup.any():
        problems.append(f"{label}: {int(dup.sum())} duplicate (file_id, run_id) rows")
    if (nb < 0).any() or (npch < 0).any():
        problems.append(f"{label}: negative n_bytes or n_patches")
    bad = np.flatnonzero(((nb == 0) & (npch != 0)) | ((nb > 0) & (npch < 1)))
    if bad.size:
        problems.append(f"{label}: n_patches inconsistent with n_bytes at rows {_examples(bad)}")
    mean = df["mean_bpp"].to_numpy(dtype=np.float64)
    var = df["var_bpp"].to_numpy(dtype=np.float64)
    pos = npch > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_mean = np.where(pos, nb / np.maximum(npch, 1), np.nan)
    ok = np.isclose(mean, expected_mean, rtol=BPP_REL_TOLERANCE, atol=BPP_ABS_TOLERANCE)
    bad = np.flatnonzero(pos & ~ok)
    if bad.size:
        problems.append(f"{label}: mean_bpp != n_bytes / n_patches at rows {_examples(bad)}")
    bad = np.flatnonzero(pos & (~np.isfinite(var) | (var < -BPP_ABS_TOLERANCE)))
    if bad.size:
        problems.append(f"{label}: var_bpp missing or negative at rows {_examples(bad)}")
    if runs is not None and isinstance(runs, dict):
        unknown = sorted(set(df["run_id"].astype(str)) - set(runs))
        if unknown:
            problems.append(f"{label}: run_ids not in runs.json: {unknown[:5]}")
    return problems


def validate_bpp_row(row, boundaries: pd.DataFrame, *, n_bytes: int, file_id: str) -> list:
    """One bpp_summary row against the file's boundaries (population variance)."""
    lab = f"bpp_summary[{file_id}]"
    problems = []
    plen = boundaries["patch_length"].to_numpy(dtype=np.int64)
    n_p = int(plen.size)
    if int(row["n_bytes"]) != n_bytes:
        problems.append(f"{lab}: n_bytes {int(row['n_bytes'])} != {n_bytes}")
    if int(row["n_patches"]) != n_p:
        problems.append(f"{lab}: n_patches {int(row['n_patches'])} != {n_p} boundary rows")
    if n_p:
        if not _close(row["mean_bpp"], n_bytes / n_p):
            problems.append(f"{lab}: mean_bpp {row['mean_bpp']!r} != {n_bytes / n_p!r}")
        var = float(np.var(plen.astype(np.float64)))
        if not _close(row["var_bpp"], var):
            problems.append(f"{lab}: var_bpp {row['var_bpp']!r} != population variance {var!r}")
    return problems


def validate_file(
    stream_a_root,
    run_id: str,
    file_id: str,
    n_bytes: int,
    *,
    tau,
    max_patch_length,
    check_entropies: bool = True,
    check_completeness: bool = False,
):
    """Loads and validates one file's Stream A outputs. Returns (problems, boundaries DataFrame or None)."""
    problems = []
    bpath = schema.boundary_path(stream_a_root, run_id, file_id)
    epath = schema.entropy_path(stream_a_root, run_id, file_id)
    entropies = None
    if check_entropies or check_completeness:
        if not epath.exists():
            problems.append(f"{file_id}: missing entropy file {epath}")
        else:
            try:
                entropies = np.load(epath, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001 - any unreadable file is a problem
                problems.append(f"{file_id}: unreadable entropy file {epath} ({exc})")
    if not bpath.exists():
        problems.append(f"{file_id}: missing boundaries file {bpath}")
        return problems, None
    try:
        df = pd.read_parquet(bpath)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{file_id}: unreadable boundaries file {bpath} ({exc})")
        return problems, None
    problems.extend(
        validate_boundaries(
            df,
            n_bytes=n_bytes,
            tau=tau,
            max_patch_length=max_patch_length,
            file_id=file_id,
            entropies=entropies,
            check_completeness=check_completeness and entropies is not None,
        )
    )
    return problems, df


def validate_run(
    stream_a_root,
    run_id: str,
    files: pd.DataFrame,
    *,
    check_entropies: bool = True,
    check_bpp: bool = True,
    check_completeness: bool = False,
) -> list:
    """All Stream A outputs of `run_id` for `files` (columns file_id, n_bytes)."""
    root = Path(stream_a_root)
    runs_path = root / schema.RUNS_JSON_PATH
    if not runs_path.exists():
        return [f"missing {runs_path}"]
    try:
        runs = json.loads(runs_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{runs_path}: unreadable ({exc})"]
    problems = validate_runs_json(runs, run_id=run_id)
    entry = runs.get(run_id) if isinstance(runs, dict) else None
    if not isinstance(entry, dict) or "tau" not in entry or "max_patch_length" not in entry:
        return problems

    bpp = None
    if check_bpp:
        bpath = root / schema.BPP_SUMMARY_PATH
        if not bpath.exists():
            problems.append(f"missing {bpath}")
        else:
            full = pd.read_parquet(bpath)
            problems.extend(validate_bpp_summary(full, runs=runs))
            if {"file_id", "run_id"} <= set(full.columns):
                bpp = full[full["run_id"] == run_id].drop_duplicates("file_id").set_index("file_id")

    for rec in files.itertuples(index=False):
        fid, n = str(rec.file_id), int(rec.n_bytes)
        file_problems, df = validate_file(
            root,
            run_id,
            fid,
            n,
            tau=entry["tau"],
            max_patch_length=entry["max_patch_length"],
            check_entropies=check_entropies,
            check_completeness=check_completeness,
        )
        problems.extend(file_problems)
        if bpp is not None:
            if fid not in bpp.index:
                problems.append(f"bpp_summary: no row for run {run_id!r}, file {fid}")
            elif df is not None and not file_problems:
                problems.extend(validate_bpp_row(bpp.loc[fid], df, n_bytes=n, file_id=fid))
    return problems


# --- CLI --------------------------------------------------------------------


def _file_list(root, *, golden, split, domains, file_ids, limit_per_domain) -> pd.DataFrame:
    rows = []
    if golden:
        for domain in domains:
            for fid in schema.golden_file_ids(root, domain):
                rows.append((fid, domain, schema.golden_bin_path(root, domain, fid).stat().st_size))
    else:
        m = pd.read_parquet(Path(root) / schema.MANIFEST_PATH, columns=["file_id", "domain", "split", "n_bytes"])
        m = m[m["domain"].isin(domains) & (m["split"] == split)]
        rows = list(m[["file_id", "domain", "n_bytes"]].itertuples(index=False, name=None))
    df = pd.DataFrame(rows, columns=["file_id", "domain", "n_bytes"]).sort_values(["domain", "file_id"])
    if file_ids:
        df = df[df["file_id"].isin(file_ids)]
    if limit_per_domain is not None:
        df = df.groupby("domain", group_keys=False).head(limit_per_domain)
    return df.reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate Stream A outputs against the Stream C contract.")
    ap.add_argument("--root", default=".", help="repo root with corpus/manifest.parquet and golden_fixtures/")
    ap.add_argument("--stream-a-root", default=None, help="root holding entropies/, boundaries/, runs.json (default: --root)")
    ap.add_argument("--run-id", required=True)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--golden", action="store_true", help="validate the golden fixtures")
    sel.add_argument("--split", choices=schema.SPLITS, help="validate manifest files of this split")
    ap.add_argument("--domain", nargs="+", default=list(schema.DOMAINS), choices=schema.DOMAINS)
    ap.add_argument("--file-id", nargs="+", default=None)
    ap.add_argument("--limit-per-domain", type=int, default=None)
    ap.add_argument("--no-entropies", action="store_true", help="skip entropies/*.npy checks")
    ap.add_argument("--no-bpp", action="store_true", help="skip bpp_summary.parquet checks")
    ap.add_argument("--strict-completeness", action="store_true",
                    help="also require every byte with entropy > tau to be an entropy boundary")
    ap.add_argument("--max-print", type=int, default=100)
    args = ap.parse_args(argv)

    files = _file_list(
        args.root,
        golden=args.golden,
        split=args.split,
        domains=args.domain,
        file_ids=args.file_id,
        limit_per_domain=args.limit_per_domain,
    )
    if files.empty:
        print("FAIL: no files selected", file=sys.stderr)
        return 1
    problems = validate_run(
        args.stream_a_root or args.root,
        args.run_id,
        files,
        check_entropies=not args.no_entropies,
        check_bpp=not args.no_bpp,
        check_completeness=args.strict_completeness,
    )
    for p in problems[: args.max_print]:
        print(p)
    if len(problems) > args.max_print:
        print(f"... and {len(problems) - args.max_print} more problems")
    status = "FAIL" if problems else "OK"
    print(f"{status}: run {args.run_id!r}, {len(files)} files, {len(problems)} problems", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
