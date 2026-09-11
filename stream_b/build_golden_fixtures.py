"""Day 5: golden fixtures. ~20 small files (picked by reading their actual
content first, not just size - excluded at least one candidate for looking like
ambiguous Python 2 syntax that could confuse hand-verification), freezing both
the source bytes and the CURRENT pipeline's output as the "known correct"
reference. stream_b/validate_golden_fixtures.py re-runs extraction against
these frozen bytes and fails loudly if output ever drifts from this reference -
the regression test this project's data contract calls for ("if your output
disagrees with the golden fixtures, your code is wrong, not the fixtures").

Selected files (6 py, 6 cpp, 6 prose) were read by eye before selection, and a
sample of their extracted rows is spot-checked below against the actual byte
content as an explicit hand-verification step, not just "the code produced it
so it must be right."
"""

import os
import shutil

import pandas as pd

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"
WHITESPACE_DIR = "whitespace"
GOLDEN_DIR = "golden_fixtures"

SELECTED = {
    "py": [
        "py_8bf2cc2f468d",
        "py_a30150e6da73",
        "py_07fbe9f688ef",
        "py_330c5ac1b62f",
        "py_ecbef3331744",
        "py_70c461e16fe6",
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
