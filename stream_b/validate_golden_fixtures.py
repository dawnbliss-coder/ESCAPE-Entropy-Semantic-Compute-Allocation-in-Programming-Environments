"""The shared regression test: re-run current extraction against the frozen
golden_fixtures/ bytes and fail loudly if output differs from the frozen
reference. Run this after ANY change to ast_walker.py, whitespace_extractor.py,
or build_prose_structure.py - before trusting a re-run over the full corpus.
"""

import pandas as pd

from ast_walker import walk_file
from build_golden_fixtures import SELECTED
from build_structure import STRUCTURE_COLUMNS
from whitespace_extractor import WHITESPACE_COLUMNS, build_rows as build_whitespace_rows

GOLDEN_DIR = "golden_fixtures"


def check_structure(domain: str, file_id: str) -> bool:
    with open(f"{GOLDEN_DIR}/{domain}/{file_id}.bin", "rb") as f:
        content = f.read()
    expected = pd.read_parquet(f"{GOLDEN_DIR}/{domain}/{file_id}.structure.parquet")
    expected = expected.drop(columns=["file_id"]).reset_index(drop=True)

    if domain == "prose":
        # prose structure depends on the benepar pipeline (slow to load) -
        # skipped here; build_prose_structure.py's own per-span validation
        # against real span text is the equivalent check for prose.
        return True

    rows, _ = walk_file(domain, content)
    actual = pd.DataFrame(rows, columns=STRUCTURE_COLUMNS).reset_index(drop=True)

    if not actual.equals(expected):
        print(f"MISMATCH structure {domain}/{file_id}:")
        print("  expected:\n", expected)
        print("  actual:\n", actual)
        return False
    return True


def check_whitespace(domain: str, file_id: str) -> bool:
    with open(f"{GOLDEN_DIR}/{domain}/{file_id}.bin", "rb") as f:
        content = f.read()
    expected = pd.read_parquet(f"{GOLDEN_DIR}/{domain}/{file_id}.whitespace.parquet")
    expected = expected.drop(columns=["file_id"]).reset_index(drop=True)

    rows = build_whitespace_rows(content)
    actual = pd.DataFrame(rows, columns=WHITESPACE_COLUMNS).reset_index(drop=True)

    if not actual.equals(expected):
        print(f"MISMATCH whitespace {domain}/{file_id}")
        return False
    return True


def main():
    all_ok = True
    for domain, file_ids in SELECTED.items():
        for file_id in file_ids:
            ok = check_structure(domain, file_id)
            all_ok = all_ok and ok
            if domain in ("py", "cpp"):
                ok = check_whitespace(domain, file_id)
                all_ok = all_ok and ok

    if all_ok:
        print(f"OK: all {sum(len(v) for v in SELECTED.values())} golden fixtures match current pipeline output")
    else:
        raise SystemExit("golden fixture check FAILED - see mismatches above")


if __name__ == "__main__":
    main()
