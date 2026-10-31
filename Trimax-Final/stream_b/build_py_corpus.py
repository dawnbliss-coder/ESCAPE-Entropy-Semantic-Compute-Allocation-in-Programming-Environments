"""Python corpus, from CodeSearchNet (matches proposal §6: "Python code from
CodeSearchNet"). Reverts the earlier the-stack-smol substitution for Python only;
C++ stays on the-stack-smol (see corpus_lib.py) since CodeSearchNet has no C++ split.

Mirrors build_prose_corpus.py's pattern: its own dedicated pull+freeze script,
not shoehorned into corpus_lib.pull_pool(), because CodeSearchNet's schema is
genuinely different from the-stack-smol's:
  - no per-example license field at all (see the LICENSE NOTE below)
  - each row is one isolated function (whole_func_string), not a whole source
    file - a known granularity mismatch against C++'s whole-file corpus, already
    discussed and accepted as the cost of proposal-compliance for Python's source

LICENSE NOTE: CodeSearchNet provides no license metadata per example (verified
directly - dataset fields are repository_name, func_path_in_repository, func_name,
whole_func_string, language, func_code_string, func_code_tokens,
func_documentation_string, func_documentation_tokens, split_name, func_code_url;
no "licenses" field). The license sanity-check that runs for C++ (via
corpus_lib.pull_pool) therefore cannot run for Python - this is a genuine gap, not
"0 missing licenses", and is called out explicitly in the write-up rather than
silently glossed over. `func_code_url` is kept in the manifest per file so the
originating repo/lines can be traced by hand if needed.

file_id = "py_{sha256[:12]}" of whole_func_string bytes - same content-addressed
convention as the rest of the corpus, naturally disjoint from the old
the-stack-smol-derived py file_ids since the content differs.

GROWTH-SAFETY: same fixed-position-prefix discipline as build_corpus.py - CodeSearchNet's
python config is pulled in its native row order (never shuffled), so raising
CALIB_PER_DOMAIN/MAIN_PER_DOMAIN later and re-running only appends further down the
list rather than reassigning existing file_ids' splits.
"""

import hashlib
import os
from collections import Counter

import pandas as pd
from datasets import load_dataset

# Pinned the same way corpus_lib.py pins the-stack-smol, for reproducibility.
# HfApi().dataset_info("code_search_net").sha
DATASET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"

CALIB_PER_DOMAIN = 100
MAIN_PER_DOMAIN = 7_900
NEEDED = CALIB_PER_DOMAIN + MAIN_PER_DOMAIN
PULL_MULTIPLIER = 1.05  # small buffer for exact-content dedup, same as corpus_lib.py

CORPUS_DIR = "corpus"


def pull_pool(n_needed: int = NEEDED):
    n_pull = int(n_needed * PULL_MULTIPLIER)
    ds = load_dataset(
        "code_search_net",
        "python",
        split="train",
        revision=DATASET_REVISION,
    )
    ds = ds.select(range(min(n_pull, len(ds))))

    seen_hashes = set()
    kept = []
    dropped_dup = 0

    for ex in ds:
        content = ex["whole_func_string"]
        if not content or not content.strip():
            continue
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes.add(h)

        kept.append(
            {
                "sha256": h,
                "n_bytes": len(content.encode("utf-8")),
                "path": ex.get("func_code_url"),
                "content": content,
            }
        )

    return {
        "pulled": len(ds),
        "kept": len(kept),
        "dropped_dup": dropped_dup,
        "records": kept,
    }


def select_split(records: list):
    if len(records) < NEEDED:
        raise ValueError(f"only {len(records)} candidates, need {NEEDED}")
    calib = records[:CALIB_PER_DOMAIN]
    main = records[CALIB_PER_DOMAIN : CALIB_PER_DOMAIN + MAIN_PER_DOMAIN]
    return calib, main


def write_domain(calib: list, main: list) -> list:
    os.makedirs(f"{CORPUS_DIR}/py", exist_ok=True)
    manifest_rows = []
    for split_name, records in [("calib", calib), ("main", main)]:
        for rec in records:
            file_id = f"py_{rec['sha256'][:12]}"
            path = f"{CORPUS_DIR}/py/{file_id}.bin"
            with open(path, "wb") as f:
                f.write(rec["content"].encode("utf-8"))
            manifest_rows.append(
                {
                    "file_id": file_id,
                    "domain": "py",
                    "n_bytes": rec["n_bytes"],
                    "sha256": rec["sha256"],
                    "split": split_name,
                    "source_path": rec["path"],
                    # empty, not "unknown": CodeSearchNet has no license field to
                    # check at all, distinct from a checked-and-missing license.
                    "licenses": "",
                }
            )
    return manifest_rows


def main():
    print("--- py (CodeSearchNet) ---")
    r = pull_pool()
    print(f"pulled: {r['pulled']}")
    print(f"kept (post dedup; license check: N/A, CodeSearchNet has no per-example "
          f"license field): {r['kept']}  (need {NEEDED})")
    print(f"dropped as exact-content duplicate: {r['dropped_dup']}")
    if r["kept"] < NEEDED:
        raise SystemExit(f"py: only {r['kept']} usable candidates, need {NEEDED}")

    calib, main_ = select_split(r["records"])
    rows = write_domain(calib, main_)
    print(f"wrote {len(rows)} files ({len(calib)} calib + {len(main_)} main)")

    new_rows = pd.DataFrame(rows)
    manifest_path = f"{CORPUS_DIR}/manifest.parquet"
    existing = pd.read_parquet(manifest_path)
    # Replace only domain=="py" - cpp/prose rows and any columns added later
    # (e.g. parse_ok) must survive this re-run untouched.
    existing = existing[existing["domain"] != "py"]
    manifest = pd.concat([existing, new_rows], ignore_index=True)
    manifest.to_parquet(manifest_path, index=False)
    print(f"\nmanifest.parquet: {len(manifest)} rows total")
    print(manifest.groupby(["domain", "split"]).size())


if __name__ == "__main__":
    main()
