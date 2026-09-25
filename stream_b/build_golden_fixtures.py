"""Golden fixtures: 18 small files (6 py, 6 cpp, 6 prose), picked by reading
their actual content first, not just size. Freezes both the source bytes and
the current pipeline's output as the "known correct" reference -
validate_golden_fixtures.py re-runs extraction against these frozen bytes and
fails loudly if output ever drifts ("if your output disagrees with the golden
fixtures, your code is wrong, not the fixtures"). A sample of extracted rows is
spot-checked below against actual byte content as an explicit hand-verification
step, not just "the code produced it so it must be right."
"""

import os
import shutil

import pandas as pd

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"
WHITESPACE_DIR = "whitespace"
GOLDEN_DIR = "golden_fixtures"

SELECTED = {
    # Re-picked after the CodeSearchNet switch (old the-stack-smol-derived file_ids
    # no longer exist) - same discipline: read actual content first, spread across
    # a size range, all clean unambiguous Python 3 syntax, parse_ok=True.
    "py": [
        "py_182f41b1a962",
        "py_ca2ef22485b6",
        "py_dd8332bfcc3f",
        "py_61c03226d22f",
        "py_a0132d6203e8",
        "py_44f23ec7455e",
    ],
    "cpp": [
        "cpp_d337aa88b1c3",
        "cpp_2b2c690055bf",
        "cpp_11562d1c691d",
        "cpp_aa561f7c0078",
        "cpp_aa3a0337370b",
        "cpp_34e849a078b1",
    ],
    "prose": [
        "prose_ec3316e574b3",
        "prose_dc4e423d093d",
        "prose_f3f2891305fb",
        "prose_e8bc068f77d1",
        "prose_f5c8c2c9ad96",
        "prose_e2e92c7ca331",
    ],
}


def main():
    for domain, file_ids in SELECTED.items():
        os.makedirs(f"{GOLDEN_DIR}/{domain}", exist_ok=True)
        for file_id in file_ids:
            shutil.copy(
                f"{CORPUS_DIR}/{domain}/{file_id}.bin",
                f"{GOLDEN_DIR}/{domain}/{file_id}.bin",
            )
            shutil.copy(
                f"{STRUCTURE_DIR}/{domain}/{file_id}.parquet",
                f"{GOLDEN_DIR}/{domain}/{file_id}.structure.parquet",
            )
            if domain in ("py", "cpp"):
                shutil.copy(
                    f"{WHITESPACE_DIR}/{domain}/{file_id}.parquet",
                    f"{GOLDEN_DIR}/{domain}/{file_id}.whitespace.parquet",
                )
        print(f"{domain}: froze {len(file_ids)} fixtures")

    print("\n--- hand-verification spot-check (a sample, not exhaustive) ---")
    for domain, file_ids in SELECTED.items():
        file_id = file_ids[0]
        with open(f"{GOLDEN_DIR}/{domain}/{file_id}.bin", "rb") as f:
            content = f.read()
        df = pd.read_parquet(f"{GOLDEN_DIR}/{domain}/{file_id}.structure.parquet")
        print(f"\n{file_id}:")
        for _, row in df.head(4).iterrows():
            text = content[row["start_byte"] : row["end_byte"]].decode("utf-8")
            print(f"  {row['node_type']:<15} [{row['start_byte']}:{row['end_byte']}] {text[:50]!r}")


if __name__ == "__main__":
    main()
