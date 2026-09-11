"""R3, part 1: freeze a WikiText-103 paragraph sample into corpus/prose/*.bin and
extend corpus/manifest.parquet, same discipline as the code corpus (raw UTF-8
bytes, never re-encoded; content-addressed file_id; append-only, growth-safe).

N=300: prose serves P3 (code/prose contrast), which doesn't need the scale the
code corpus needed - and prose has no calib split at all, since tolerance k is
calibrated per CODE language (per the proposal), not for prose.
"""

import pandas as pd

from prose_lib import pull_paragraphs

N_NEEDED = 300
CORPUS_DIR = "corpus"


def main():
    import os

    paras = pull_paragraphs(N_NEEDED)
    if len(paras) < N_NEEDED:
        raise SystemExit(f"only {len(paras)} usable paragraphs, need {N_NEEDED}")
    paras = paras[:N_NEEDED]  # stable prefix, same growth-safety as build_corpus.py

    os.makedirs(f"{CORPUS_DIR}/prose", exist_ok=True)
    manifest_rows = []
    for p in paras:
        file_id = f"prose_{p['sha256'][:12]}"
        path = f"{CORPUS_DIR}/prose/{file_id}.bin"
        with open(path, "wb") as f:
            f.write(p["content"].encode("utf-8"))
        manifest_rows.append(
            {
                "file_id": file_id,
                "domain": "prose",
                "n_bytes": p["n_bytes"],
                "sha256": p["sha256"],
                "split": "main",
                "source_path": p["article_title"],
                "licenses": "Wikipedia (CC BY-SA)",
            }
        )

    new_rows = pd.DataFrame(manifest_rows)

    manifest_path = f"{CORPUS_DIR}/manifest.parquet"
    existing = pd.read_parquet(manifest_path)
    # Drop any prior prose rows before appending, so re-running this script
    # replaces the prose portion cleanly instead of duplicating it. py/cpp rows
    # (built independently by build_corpus.py) are untouched either way.
    existing = existing[existing["domain"] != "prose"]
    # prose rows have no parse_ok (that column is tree-sitter-specific); leave NaN
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_parquet(manifest_path, index=False)

    print(f"wrote {len(new_rows)} prose files")
    print(combined.groupby(["domain", "split"]).size())


if __name__ == "__main__":
    main()
