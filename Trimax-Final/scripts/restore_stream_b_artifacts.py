#!/usr/bin/env python3
"""Restore Stream B's gitignored, regenerable data artifacts without writing
corpus/manifest.parquet (or any other tracked file).

stream_b/build_corpus.py, build_prose_corpus.py and build_structure.py
regenerate this data but also rewrite the committed manifest. This script
reuses their functions unchanged (stream_b/ is put on sys.path) and treats the
committed manifest as the read-only source of truth.

Modes (combinable; they run in this order):
  --code       corpus/{py,cpp}/{file_id}.bin
               py:  build_py_corpus.pull_pool -> build_py_corpus.select_split,
                    exactly as build_py_corpus.main (CodeSearchNet; not gated,
                    no per-example license field, so the manifest licenses=""
                    convention is mirrored and no license check applies).
               cpp: corpus_lib.pull_pool(domain, 8000) -> build_corpus.select_split,
                    exactly as build_corpus.main. bigcode/the-stack-smol is gated:
                    HF_TOKEN must belong to an account that accepted its terms.
  --prose      corpus/prose/{file_id}.bin
               prose_lib.pull_paragraphs(300)[:300], as build_prose_corpus.main.
  --structure  structure/{py,cpp}/{file_id}.parquet
               ast_walker.walk_file + build_structure.STRUCTURE_COLUMNS, as
               build_structure.main; parsed_ok is compared with manifest parse_ok
               (mismatches are reported, not fatal).
  --whitespace DOMAIN [DOMAIN ...]
               whitespace/{DOMAIN}/{file_id}.parquet with whitespace_extractor's
               build_rows + WHITESPACE_COLUMNS, as whitespace_extractor.main but
               limited to the given domains. That script as-is needs every
               manifest .bin on disk; prefer it once all three domains exist.
  --verify     read-only: golden fixtures vs restored corpus/structure/whitespace,
               n_bytes + sha256 of every manifest .bin, presence and schema of
               structure/ and whitespace/, prose node_types vs prose_taxonomy.json.

Before anything is written, every pulled record must match its manifest row
(n_bytes, sha256, split, source_path, plus licenses for code) and the file_id
set of each (domain, split) must equal the manifest's; otherwise the run aborts
with exit code 1 and the full offender list. Files are written atomically (temp
file in the target directory, fsync, os.replace); .bin files are re-read and
re-hashed after writing, and derived parquet is only built from .bin files
whose n_bytes + sha256 match the manifest. Existing outputs that verify are
left untouched, so every mode can be re-run or resumed.

structure/prose/ comes from stream_b/build_prose_structure.py as-is (it only
reads the manifest).

Usage (from any directory; paths resolve against the repo root):
  HF_TOKEN=... .venv/bin/python scripts/restore_stream_b_artifacts.py --code
  .venv/bin/python scripts/restore_stream_b_artifacts.py --prose
  .venv/bin/python scripts/restore_stream_b_artifacts.py --structure
  .venv/bin/python scripts/restore_stream_b_artifacts.py --whitespace prose
  .venv/bin/python scripts/restore_stream_b_artifacts.py --verify
"""

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "stream_b"))

import pandas as pd  # noqa: E402

CODE_DOMAINS = ("py", "cpp")
ALL_DOMAINS = ("py", "cpp", "prose")
# Leftovers of an interrupted atomic write carry this prefix; removed on the next run.
TMP_PREFIX = ".restore-tmp-"


def log(*args) -> None:
    print(*args, flush=True)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_open_file_mode() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


# mkstemp creates 0600 files; give outputs the mode a plain open(path, "wb") would.
FILE_MODE = _plain_open_file_mode()


def _unlink_quiet(path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def remove_stale_temp_files(directory: Path) -> int:
    stale = list(directory.glob(TMP_PREFIX + "*"))
    for p in stale:
        p.unlink()
    return len(stale)


def fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{TMP_PREFIX}{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        _unlink_quiet(tmp)
        raise


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{TMP_PREFIX}{path.name}.")
    os.close(fd)
    try:
        # The same call build_structure.main / whitespace_extractor.main make,
        # aimed at the temp file.
        df.to_parquet(tmp, index=False)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        _unlink_quiet(tmp)
        raise


def _parquet_holds(path: Path, df: pd.DataFrame) -> bool:
    try:
        existing = pd.read_parquet(path)
    except Exception:
        return False
    # Compare after the same parquet round trip, so dtype inference on an
    # in-memory frame (e.g. an empty one) cannot cause a false mismatch.
    return existing.equals(pd.read_parquet(io.BytesIO(df.to_parquet(index=False))))


def write_parquet_if_changed(df: pd.DataFrame, out: Path, counts: Counter) -> None:
    if not out.exists():
        atomic_write_parquet(df, out)
        counts["written"] += 1
    elif _parquet_holds(out, df):
        counts["skipped_existing_verified"] += 1
    else:
        atomic_write_parquet(df, out)
        counts["rewrote_unverified_existing"] += 1


def read_verified_bin(root: Path, domain: str, row: dict, missing: list, bad: list):
    """The frozen bytes of one manifest row, or None (recorded in missing/bad)
    if the file is absent or does not match the manifest's n_bytes/sha256."""
    path = root / "corpus" / domain / f"{row['file_id']}.bin"
    if not path.exists():
        missing.append(row["file_id"])
        return None
    with open(path, "rb") as f:
        content = f.read()
    if len(content) != int(row["n_bytes"]) or sha256_hex(content) != row["sha256"]:
        bad.append(row["file_id"])
        return None
    return content


def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def same_value(manifest_value, pulled_value) -> bool:
    """Exact equality, except that a manifest null (NaN/None/NA after the
    parquet round trip) matches a pulled None."""
    if is_null(manifest_value) or is_null(pulled_value):
        return is_null(manifest_value) and is_null(pulled_value)
    return bool(manifest_value == pulled_value)


def abort(title: str, offenders: list) -> None:
    log(f"\nABORT: {title}: {len(offenders)} offender(s); nothing further written")
    for o in offenders:
        log(f"  OFFENDER {o}")
    sys.exit(1)


# --------------------------------------------------------------------------- pulls


def check_pull_against_manifest(domain: str, pulled: list, manifest: pd.DataFrame) -> list:
    """Offender strings; empty means the pull matches the manifest exactly.
    Each pulled entry has file_id, split, n_bytes, sha256, source_path, data
    (the bytes that would be written) and, for code, licenses."""
    mdom = manifest[manifest["domain"] == domain]
    rows = {r["file_id"]: r for r in mdom.to_dict("records")}
    offenders = []
    if len(rows) != len(mdom):
        offenders.append(f"{domain}: manifest has {len(mdom) - len(rows)} duplicate file_id rows")

    split_of = {}
    for p in pulled:
        fid = p["file_id"]
        if fid in split_of:
            offenders.append(f"{fid}: produced twice by the pull ({split_of[fid]}, {p['split']})")
            continue
        split_of[fid] = p["split"]
        if len(p["data"]) != p["n_bytes"] or sha256_hex(p["data"]) != p["sha256"]:
            offenders.append(f"{fid}: pulled bytes do not match their own n_bytes/sha256")
        row = rows.get(fid)
        if row is None:
            offenders.append(f"{fid}: pulled into split={p['split']} but has no manifest row")
            continue
        diffs = []
        for field in ("n_bytes", "sha256", "split", "source_path", "licenses"):
            if field not in p:
                continue
            want, got = row[field], p[field]
            if field == "n_bytes":
                ok = not is_null(want) and int(want) == int(got)
            else:
                ok = same_value(want, got)
            if not ok:
                diffs.append(f"{field}: manifest={want!r} pulled={got!r}")
        if diffs:
            offenders.append(f"{fid}: " + "; ".join(diffs))

    manifest_only = sorted(set(rows) - set(split_of))
    if manifest_only:
        offenders.append(
            f"{domain}: {len(manifest_only)} manifest file_id(s) not produced by the pull: {manifest_only}"
        )
    for split in sorted(set(mdom["split"].dropna()) | set(split_of.values())):
        want = set(mdom.loc[mdom["split"] == split, "file_id"])
        got = {fid for fid, s in split_of.items() if s == split}
        if want != got:
            offenders.append(
                f"{domain}/{split}: file_id set differs from the manifest (manifest {len(want)}, "
                f"pull {len(got)}, manifest-only {len(want - got)}, pull-only {len(got - want)}; "
                "individual ids listed above)"
            )
    return offenders


def write_bins(domain: str, pulled: list, manifest: pd.DataFrame, root: Path):
    out_dir = root / "corpus" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = remove_stale_temp_files(out_dir)
    expected = (
        manifest[manifest["domain"] == domain]
        .set_index("file_id")[["n_bytes", "sha256"]]
        .to_dict("index")
    )
    counts = Counter()
    failed = []
    for p in pulled:
        fid = p["file_id"]
        want_n, want_sha = int(expected[fid]["n_bytes"]), expected[fid]["sha256"]
        path = out_dir / f"{fid}.bin"
        existed = path.exists()
        if existed:
            current = path.read_bytes()
            if len(current) == want_n and sha256_hex(current) == want_sha:
                counts["skipped_existing_verified"] += 1
                continue
        atomic_write_bytes(path, p["data"])
        back = path.read_bytes()
        if len(back) != want_n or sha256_hex(back) != want_sha:
            failed.append(fid)
        else:
            counts["rewrote_unverified_existing" if existed else "written"] += 1
    fsync_dir(out_dir)
    on_disk = {q.name[: -len(".bin")] for q in out_dir.glob("*.bin")}
    orphans = sorted(on_disk - set(expected))
    return counts, failed, stale, orphans


def verify_and_write(domain: str, pulled: list, manifest: pd.DataFrame, root: Path, fields: str) -> None:
    t0 = time.monotonic()
    offenders = check_pull_against_manifest(domain, pulled, manifest)
    if offenders:
        abort(f"{domain} pull does not match corpus/manifest.parquet", offenders)
    log(f"{domain}: all {len(pulled)} pulled records match the manifest ({fields}; file_id sets per split)")
    counts, failed, stale, orphans = write_bins(domain, pulled, manifest, root)
    if failed:
        abort(f"{domain}: re-read after atomic write does not hash to the manifest", failed)
    log(
        f"{domain}: corpus/{domain}/*.bin written={counts['written']} "
        f"skipped_existing_verified={counts['skipped_existing_verified']} "
        f"rewrote_unverified_existing={counts['rewrote_unverified_existing']} "
        f"re-hash_after_write_failures=0 stale_temp_removed={stale} "
        f"orphan_bins_not_in_manifest={len(orphans)} ({time.monotonic() - t0:.1f}s)"
    )
    if orphans:
        log(f"{domain}: orphan .bin files (left untouched): {orphans}")


def restore_code(manifest: pd.DataFrame, root: Path, domains=CODE_DOMAINS) -> None:
    import build_corpus
    import build_py_corpus
    import corpus_lib

    n_needed = build_corpus.CALIB_PER_DOMAIN + build_corpus.MAIN_PER_DOMAIN
    for domain in domains:
        t0 = time.monotonic()
        if domain == "py":
            # CodeSearchNet (proposal §6): its own pull/dedup, no license field,
            # manifest licenses="" - see stream_b/build_py_corpus.py.
            log(f"\n=== --code {domain}: build_py_corpus.pull_pool(n_needed={n_needed}) ===")
            pool = build_py_corpus.pull_pool(n_needed=n_needed)
            log(
                f"{domain}: pulled={pool['pulled']} kept={pool['kept']} dropped_dup={pool['dropped_dup']} "
                f"(license check N/A: CodeSearchNet has no per-example license field; "
                f"{time.monotonic() - t0:.1f}s)"
            )
            if pool["kept"] < n_needed:  # same guard as build_py_corpus.main
                abort(f"{domain} pool", [f"only {pool['kept']} usable candidates, need {n_needed}"])
            calib, main_ = build_py_corpus.select_split(pool["records"])
            pulled = [
                {
                    "file_id": f"{domain}_{rec['sha256'][:12]}",
                    "split": split_name,
                    "n_bytes": rec["n_bytes"],
                    "sha256": rec["sha256"],
                    "source_path": rec["path"],
                    "licenses": "",  # matches build_py_corpus.write_domain
                    "data": rec["content"].encode("utf-8"),
                }
                for split_name, recs in (("calib", calib), ("main", main_))
                for rec in recs
            ]
        else:
            log(f"\n=== --code {domain}: corpus_lib.pull_pool({domain!r}, n_needed={n_needed}) ===")
            pool = corpus_lib.pull_pool(domain, n_needed=n_needed)
            log(
                f"{domain}: pulled={pool['pulled']} kept={pool['kept']} dropped_dup={pool['dropped_dup']} "
                f"dropped_no_license={pool['dropped_no_license']} ({time.monotonic() - t0:.1f}s)"
            )
            if pool["kept"] < n_needed:  # same guard as build_corpus.main
                abort(f"{domain} pool", [f"only {pool['kept']} usable candidates, need {n_needed}"])
            calib, main_ = build_corpus.select_split(pool["records"])
            pulled = [
                {
                    "file_id": f"{domain}_{rec['sha256'][:12]}",
                    "split": split_name,
                    "n_bytes": rec["n_bytes"],
                    "sha256": rec["sha256"],
                    "source_path": rec["path"],
                    "licenses": ",".join(rec["licenses"]),
                    "data": rec["content"].encode("utf-8"),
                }
                for split_name, recs in (("calib", calib), ("main", main_))
                for rec in recs
            ]
        del pool, calib, main_
        gc.collect()
        verify_and_write(domain, pulled, manifest, root, "n_bytes, sha256, split, source_path, licenses")
        log(f"{domain}: done in {time.monotonic() - t0:.1f}s")
        del pulled
        gc.collect()


def restore_prose(manifest: pd.DataFrame, root: Path) -> None:
    import build_prose_corpus
    import prose_lib

    # Compatibility shim (2026-09-15). prose_lib calls load_dataset("wikitext", ...), but the
    # pinned huggingface_hub (1.27.0) rejects the legacy canonical id:
    # "HfUriError: Repository id must be 'namespace/name', got 'wikitext'". The same dataset
    # now lives at Salesforce/wikitext, and its current sha equals prose_lib.DATASET_REVISION
    # (b08601e0..., checked via the HF API), so the pinned revision still selects identical data.
    # stream_b/ is not edited: only the name lookup is redirected here, and every pulled
    # paragraph is still checked against the manifest's n_bytes + sha256 before anything is written.
    _original_load_dataset = prose_lib.load_dataset

    def _load_dataset_compat(path, *args, **kwargs):
        if path == "wikitext":
            path = "Salesforce/wikitext"
        return _original_load_dataset(path, *args, **kwargs)

    prose_lib.load_dataset = _load_dataset_compat
    n_needed = build_prose_corpus.N_NEEDED
    t0 = time.monotonic()
    log(f"\n=== --prose: prose_lib.pull_paragraphs({n_needed})[:{n_needed}] ===")
    paras = prose_lib.pull_paragraphs(n_needed)
    log(f"prose: {len(paras)} candidate paragraphs ({time.monotonic() - t0:.1f}s)")
    if len(paras) < n_needed:  # same guard as build_prose_corpus.main
        abort("prose pool", [f"only {len(paras)} usable paragraphs, need {n_needed}"])
    pulled = [
        {
            "file_id": f"prose_{p['sha256'][:12]}",
            "split": "main",  # build_prose_corpus.main writes split="main" for every prose row
            "n_bytes": p["n_bytes"],
            "sha256": p["sha256"],
            "source_path": p["article_title"],
            "data": p["content"].encode("utf-8"),
        }
        for p in paras[:n_needed]
    ]
    del paras
    verify_and_write("prose", pulled, manifest, root, "n_bytes, sha256, split, source_path=article_title")
    log(f"prose: done in {time.monotonic() - t0:.1f}s")


# ------------------------------------------------------------- derived parquet


def _report_unusable_bins(missing_bin: list, bad_bin: list) -> None:
    if missing_bin:
        more = " ..." if len(missing_bin) > 20 else ""
        log(f"  missing .bin (nothing written for these): {missing_bin[:20]}{more}")
    if bad_bin:
        log(f"  .bin not matching manifest n_bytes/sha256 (nothing written for these): {bad_bin}")


def restore_structure(manifest: pd.DataFrame, root: Path) -> bool:
    import ast_walker
    import build_structure

    all_ok = True
    code = manifest[manifest["domain"].isin(list(CODE_DOMAINS))]
    for domain, group in code.groupby("domain"):  # same iteration as build_structure.main
        t0 = time.monotonic()
        log(f"\n=== --structure {domain}: {len(group)} files ===")
        out_dir = root / "structure" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        stale = remove_stale_temp_files(out_dir)
        counts = Counter()
        node_type_counts = Counter()
        missing_bin, bad_bin, mismatches = [], [], []
        n_errors = 0
        for row in group[["file_id", "n_bytes", "sha256", "parse_ok"]].to_dict("records"):
            content = read_verified_bin(root, domain, row, missing_bin, bad_bin)
            if content is None:
                continue
            file_id = row["file_id"]
            rows, parsed_ok = ast_walker.walk_file(domain, content)
            n_errors += 0 if parsed_ok else 1
            node_type_counts.update(r["node_type"] for r in rows)
            df = pd.DataFrame(rows, columns=build_structure.STRUCTURE_COLUMNS)
            df.insert(0, "file_id", file_id)
            write_parquet_if_changed(df, out_dir / f"{file_id}.parquet", counts)

            want = row["parse_ok"]
            if is_null(want) or bool(want) != bool(parsed_ok):
                mismatches.append((file_id, want, parsed_ok))
        fsync_dir(out_dir)

        n_parsed = len(group) - len(missing_bin) - len(bad_bin)
        pct = n_errors / n_parsed * 100 if n_parsed else 0.0
        log(
            f"{domain}: structure written={counts['written']} "
            f"skipped_existing_verified={counts['skipped_existing_verified']} "
            f"rewrote_unverified_existing={counts['rewrote_unverified_existing']} "
            f"missing_bin={len(missing_bin)} bad_bin={len(bad_bin)} stale_temp_removed={stale} "
            f"({time.monotonic() - t0:.1f}s)"
        )
        log(
            f"{domain}: {n_errors}/{n_parsed} with tree-sitter error-recovery triggered ({pct:.1f}%); "
            f"parse_ok mismatches vs manifest: {len(mismatches)}"
        )
        for file_id, want, got in mismatches:
            log(f"  PARSE_OK_MISMATCH {file_id}: manifest={want!r} walk_file={got!r}")
        _report_unusable_bins(missing_bin, bad_bin)
        log(f"{domain}: node_type counts: {dict(sorted(node_type_counts.items()))}")
        if missing_bin or bad_bin:
            all_ok = False
    return all_ok


def restore_whitespace(manifest: pd.DataFrame, root: Path, domains) -> bool:
    import whitespace_extractor

    all_ok = True
    selected = manifest[manifest["domain"].isin(list(domains))]
    for domain, group in selected.groupby("domain"):  # same iteration as whitespace_extractor.main
        t0 = time.monotonic()
        log(f"\n=== --whitespace {domain}: {len(group)} files ===")
        out_dir = root / "whitespace" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        stale = remove_stale_temp_files(out_dir)
        counts = Counter()
        missing_bin, bad_bin = [], []
        n_newline = n_indent = 0
        for row in group[["file_id", "n_bytes", "sha256"]].to_dict("records"):
            content = read_verified_bin(root, domain, row, missing_bin, bad_bin)
            if content is None:
                continue
            rows = whitespace_extractor.build_rows(content)
            n_newline += sum(1 for r in rows if r["kind"] == "newline")
            n_indent += sum(1 for r in rows if r["kind"] == "indent_change")
            df = pd.DataFrame(rows, columns=whitespace_extractor.WHITESPACE_COLUMNS)
            df.insert(0, "file_id", row["file_id"])
            write_parquet_if_changed(df, out_dir / f"{row['file_id']}.parquet", counts)
        fsync_dir(out_dir)
        log(
            f"{domain}: whitespace written={counts['written']} "
            f"skipped_existing_verified={counts['skipped_existing_verified']} "
            f"rewrote_unverified_existing={counts['rewrote_unverified_existing']} "
            f"missing_bin={len(missing_bin)} bad_bin={len(bad_bin)} stale_temp_removed={stale} "
            f"({n_newline} newline rows, {n_indent} indent_change rows; {time.monotonic() - t0:.1f}s)"
        )
        _report_unusable_bins(missing_bin, bad_bin)
        if missing_bin or bad_bin:
            all_ok = False
    return all_ok


# ------------------------------------------------------------------------- verify


def describe_frame_diff(restored: pd.DataFrame, golden: pd.DataFrame) -> list:
    out = []
    if list(restored.columns) != list(golden.columns):
        out.append(f"columns restored={list(restored.columns)} golden={list(golden.columns)}")
    rd = restored.dtypes.astype(str).to_dict()
    gd = golden.dtypes.astype(str).to_dict()
    if rd != gd:
        out.append(f"dtypes restored={rd} golden={gd}")
    out.append(f"rows restored={len(restored)} golden={len(golden)}")
    common = [c for c in golden.columns if c in restored.columns]

    def as_tuples(df):
        return [tuple(map(str, t)) for t in df[common].itertuples(index=False, name=None)]

    r_rows, g_rows = as_tuples(restored), as_tuples(golden)
    only_r = Counter(r_rows) - Counter(g_rows)
    only_g = Counter(g_rows) - Counter(r_rows)
    if not only_r and not only_g:
        first = next((i for i, (a, b) in enumerate(zip(r_rows, g_rows)) if a != b), None)
        out.append(f"same rows as a multiset (values compared as str); first row-order difference at index {first}")
    else:
        out.append(f"{sum(only_r.values())} row(s) only in restored, {sum(only_g.values())} only in golden; columns {common}")
        for t, c in sorted(only_r.items()):
            out.append(f"+ restored {t}" + (f" x{c}" if c > 1 else ""))
        for t, c in sorted(only_g.items()):
            out.append(f"- golden   {t}" + (f" x{c}" if c > 1 else ""))
    return out


def verify(manifest: pd.DataFrame, root: Path) -> bool:
    import pyarrow.parquet as pq
    from build_golden_fixtures import SELECTED
    from build_structure import STRUCTURE_COLUMNS
    from whitespace_extractor import WHITESPACE_COLUMNS

    failures = []
    golden_dir = root / "golden_fixtures"

    log("\n=== --verify: golden fixtures vs restored artifacts ===")
    for domain, file_ids in SELECTED.items():
        for file_id in file_ids:
            parts, details = [], []
            corpus_bin = root / "corpus" / domain / f"{file_id}.bin"
            if corpus_bin.exists():
                same = corpus_bin.read_bytes() == (golden_dir / domain / f"{file_id}.bin").read_bytes()
                parts.append("bin=" + ("EQUAL" if same else "DIFF"))
            else:
                parts.append("bin=MISSING")
            kinds = ("structure", "whitespace") if domain in CODE_DOMAINS else ("structure",)
            for kind in kinds:
                restored_path = root / kind / domain / f"{file_id}.parquet"
                if not restored_path.exists():
                    parts.append(f"{kind}=MISSING")
                    continue
                restored = pd.read_parquet(restored_path)
                golden = pd.read_parquet(golden_dir / domain / f"{file_id}.{kind}.parquet")
                if restored.equals(golden):
                    parts.append(f"{kind}=EQUAL")
                else:
                    parts.append(f"{kind}=DIFF")
                    details.extend(f"{kind}: {d}" for d in describe_frame_diff(restored, golden))
            line = f"{domain}/{file_id}: " + " ".join(parts)
            log(line)
            for d in details:
                log(f"    {d}")
            if not all(p.endswith("=EQUAL") for p in parts):
                failures.append(line)

    log("\n=== --verify: corpus/*.bin vs manifest (n_bytes + sha256) ===")
    for domain in ALL_DOMAINS:
        mdom = manifest[manifest["domain"] == domain]
        d = root / "corpus" / domain
        present = verified = 0
        bad = []
        for file_id, n_bytes, sha in mdom[["file_id", "n_bytes", "sha256"]].itertuples(index=False, name=None):
            path = d / f"{file_id}.bin"
            if not path.exists():
                continue
            present += 1
            data = path.read_bytes()
            if len(data) == int(n_bytes) and sha256_hex(data) == sha:
                verified += 1
            else:
                bad.append(file_id)
        on_disk = {p.name[: -len(".bin")] for p in d.glob("*.bin")} if d.is_dir() else set()
        orphans = len(on_disk - set(mdom["file_id"]))
        line = (
            f"corpus/{domain}: manifest={len(mdom)} present={present} verified={verified} "
            f"bad={len(bad)} missing={len(mdom) - present} orphans={orphans}"
        )
        log(line)
        if bad:
            log(f"    bad: {bad}")
        if bad or present != len(mdom) or orphans:
            failures.append(line)

    log("\n=== --verify: structure/ and whitespace/ presence + parquet schema ===")
    expected_columns = {
        "structure": ["file_id"] + list(STRUCTURE_COLUMNS),
        "whitespace": ["file_id"] + list(WHITESPACE_COLUMNS),
    }
    for kind in ("structure", "whitespace"):
        for domain in ALL_DOMAINS:
            ids = set(manifest.loc[manifest["domain"] == domain, "file_id"])
            d = root / kind / domain
            on_disk = {p.name[: -len(".parquet")] for p in d.glob("*.parquet")} if d.is_dir() else set()
            present = sorted(ids & on_disk)
            bad_schema = [
                fid for fid in present
                if pq.read_schema(d / f"{fid}.parquet").names != expected_columns[kind]
            ]
            line = (
                f"{kind}/{domain}: manifest={len(ids)} present={len(present)} "
                f"missing={len(ids - on_disk)} orphans={len(on_disk - ids)} bad_schema={len(bad_schema)}"
            )
            log(line)
            if bad_schema:
                log(f"    bad schema: {bad_schema}")
            if len(present) != len(ids) or on_disk - ids or bad_schema:
                failures.append(line)

    log("\n=== --verify: structure/prose node_type vs prose_taxonomy.json ===")
    taxonomy = json.loads((root / "prose_taxonomy.json").read_text())
    prose_dir = root / "structure" / "prose"
    prose_files = sorted(prose_dir.glob("*.parquet")) if prose_dir.is_dir() else []
    if prose_files:
        allp = pd.concat(
            [pd.read_parquet(f, columns=["node_type", "parent_type", "depth"]) for f in prose_files],
            ignore_index=True,
        )
        observed = set(allp["node_type"])
        unknown = sorted(observed - taxonomy.keys())
        unused = sorted(taxonomy.keys() - observed)
        roots = int(((allp["parent_type"] == "paragraph") & (allp["depth"] == 1)).sum())
        line = (
            f"structure/prose: files={len(prose_files)} rows={len(allp)} observed_node_types={len(observed)} "
            f"taxonomy_keys={len(taxonomy)} observed_not_in_taxonomy={unknown} taxonomy_keys_not_observed={unused}"
        )
        log(line)
        log(f"structure/prose: rows with parent_type='paragraph' at depth 1 (labeled sentence roots): {roots}")
        if unknown:
            failures.append(line)
    else:
        log("structure/prose: no files")
        failures.append("structure/prose: no files")

    log(f"\n--verify: {'PASS' if not failures else 'FAIL'} ({len(failures)} failing check(s))")
    for f in failures:
        log(f"  FAIL {f}")
    return not failures


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Restore Stream B's regenerable data artifacts without writing the manifest."
    )
    ap.add_argument("--code", action="store_true", help="corpus/{py,cpp}/*.bin (py: CodeSearchNet, no token; cpp: the-stack-smol, needs HF_TOKEN)")
    ap.add_argument("--code-domains", nargs="+", choices=CODE_DOMAINS, default=list(CODE_DOMAINS),
                    help="with --code: restore only these (py needs no token; cpp is gated)")
    ap.add_argument("--prose", action="store_true", help="corpus/prose/*.bin from WikiText-103")
    ap.add_argument("--structure", action="store_true", help="structure/{py,cpp}/*.parquet via tree-sitter")
    ap.add_argument(
        "--whitespace", nargs="+", choices=ALL_DOMAINS, metavar="DOMAIN",
        help="whitespace/{DOMAIN}/*.parquet for the given domains (py, cpp, prose)",
    )
    ap.add_argument("--verify", action="store_true", help="read-only checks incl. golden fixtures")
    args = ap.parse_args(argv)
    if not (args.code or args.prose or args.structure or args.whitespace or args.verify):
        ap.error("choose at least one of --code, --prose, --structure, --whitespace, --verify")

    # The committed build scripts are run from the repo root (relative paths,
    # and load_dataset() would prefer a same-named local directory).
    os.chdir(REPO_ROOT)
    manifest_path = REPO_ROOT / "corpus" / "manifest.parquet"
    manifest_sha = sha256_file(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    log(f"repo: {REPO_ROOT}")
    log(f"manifest: {len(manifest)} rows, file sha256 {manifest_sha}")

    t0 = time.monotonic()
    status = 0
    if args.code:
        restore_code(manifest, REPO_ROOT, args.code_domains)
    if args.prose:
        restore_prose(manifest, REPO_ROOT)
    if args.structure and not restore_structure(manifest, REPO_ROOT):
        status = 1
    if args.whitespace and not restore_whitespace(manifest, REPO_ROOT, args.whitespace):
        status = 1
    if args.verify and not verify(manifest, REPO_ROOT):
        status = 1

    if sha256_file(manifest_path) != manifest_sha:
        log("ERROR: corpus/manifest.parquet changed during this run")
        status = 1
    log(f"\ntotal wall time {time.monotonic() - t0:.1f}s, exit status {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
