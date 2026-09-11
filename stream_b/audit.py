"""Consolidated sanity audit across everything built so far: manifest
consistency, on-disk file presence/absence in both directions, dedup
correctness, and a content spot-check for the new prose corpus.
"""

import hashlib

import pandas as pd

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"
WHITESPACE_DIR = "whitespace"


def check_manifest_basic(m: pd.DataFrame):
    print("=== manifest basic checks ===")
    print("total rows:", len(m))
    print(m.groupby(["domain", "split"]).size())
    dup_ids = m["file_id"].duplicated().sum()
    print("duplicate file_ids:", dup_ids)
    dup_sha_within_domain = (
        m.groupby("domain")["sha256"].apply(lambda s: s.duplicated().sum()).to_dict()
    )
    print("duplicate sha256 within domain:", dup_sha_within_domain)


def check_files_on_disk(m: pd.DataFrame):
    print("\n=== corpus/*.bin presence (manifest -> disk) ===")
    missing = []
    for _, row in m.iterrows():
        import os

        path = f"{CORPUS_DIR}/{row['domain']}/{row['file_id']}.bin"
        if not os.path.exists(path):
            missing.append(path)
    print("manifest rows with no .bin on disk:", len(missing))
    if missing:
        print(missing[:5])

    print("\n=== corpus/*.bin presence (disk -> manifest, orphans) ===")
    import glob

    on_disk = set()
    for domain in ("py", "cpp", "prose"):
        for p in glob.glob(f"{CORPUS_DIR}/{domain}/*.bin"):
            file_id = p.split("/")[-1].removesuffix(".bin")
            on_disk.add(file_id)
    in_manifest = set(m["file_id"])
    orphans = on_disk - in_manifest
    print(".bin files on disk with no manifest row:", len(orphans))
    if orphans:
        print(list(orphans)[:5])


def check_structure_whitespace_coverage(m: pd.DataFrame):
    print("\n=== structure/ + whitespace/ coverage (py, cpp only) ===")
    code = m[m["domain"].isin(["py", "cpp"])]
    missing_struct = []
    missing_ws = []
    for _, row in code.iterrows():
        import os

        if not os.path.exists(f"{STRUCTURE_DIR}/{row['domain']}/{row['file_id']}.parquet"):
            missing_struct.append(row["file_id"])
        if not os.path.exists(f"{WHITESPACE_DIR}/{row['domain']}/{row['file_id']}.parquet"):
            missing_ws.append(row["file_id"])
    print("code files missing structure/:", len(missing_struct))
    print("code files missing whitespace/:", len(missing_ws))


def check_taxonomy():
    print("\n=== taxonomy.json vs ast_walker.NODE_TYPES ===")
    import json

    import sys

    sys.path.insert(0, "stream_b")
    from ast_walker import NODE_TYPES

    with open("taxonomy.json") as f:
        tax = json.load(f)
    all_tracked = NODE_TYPES["py"] | NODE_TYPES["cpp"]
    missing = all_tracked - tax.keys()
    stale = tax.keys() - all_tracked
    print("node types tracked but not in taxonomy.json:", missing)
    print("node types in taxonomy.json but not tracked:", stale)


def check_prose_content(m: pd.DataFrame):
    print("\n=== prose content spot-check ===")
    prose = m[m["domain"] == "prose"]
    bad_utf8 = 0
    size_mismatch = 0
    empty_after_strip = 0
    for _, row in prose.iterrows():
        with open(f"{CORPUS_DIR}/prose/{row['file_id']}.bin", "rb") as f:
            content = f.read()
        if len(content) != row["n_bytes"]:
            size_mismatch += 1
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            bad_utf8 += 1
            continue
        if not text.strip():
            empty_after_strip += 1
    print(f"total prose files checked: {len(prose)}")
    print(f"invalid UTF-8: {bad_utf8}")
    print(f"n_bytes mismatch vs actual file size: {size_mismatch}")
    print(f"empty/whitespace-only content: {empty_after_strip}")
    print(f"n_bytes range: min={prose['n_bytes'].min()}, max={prose['n_bytes'].max()}, mean={prose['n_bytes'].mean():.0f}")

    # show 3 real samples for a manual eyeball check
    print("\n--- 3 random samples ---")
    for _, row in prose.sample(3, random_state=1).iterrows():
        with open(f"{CORPUS_DIR}/prose/{row['file_id']}.bin", "rb") as f:
            content = f.read()
        print(f"[{row['file_id']}] title={row['source_path']!r} n_bytes={row['n_bytes']}")
        print(" ", content.decode("utf-8")[:180])
        print()


def main():
    m = pd.read_parquet(f"{CORPUS_DIR}/manifest.parquet")
    check_manifest_basic(m)
    check_files_on_disk(m)
    check_structure_whitespace_coverage(m)
    check_taxonomy()
    check_prose_content(m)


if __name__ == "__main__":
    main()
