"""Day 2 step 2: from the validated py/cpp pools, select calib/main splits, freeze raw
bytes to corpus/{domain}/{file_id}.bin, and write corpus/manifest.parquet.

file_id convention: "{domain}_{sha256[:12]}" — content-addressed, stable across reruns,
unique across domains. Once written, corpus/*.bin are never re-encoded downstream.

GROWTH-SAFETY: calib/main are assigned as fixed-position PREFIXES of corpus_lib's
stably-ordered `records` (first CALIB_PER_DOMAIN -> calib, next MAIN_PER_DOMAIN -> main),
never a reshuffle. `the-stack-smol` is already a random per-language sample, so there's
nothing to gain from reshuffling and real cost to it: a shuffle's permutation depends on
the list's length, so reshuffling after pulling a bigger pool would silently move files
between calib and main. With a fixed prefix split, raising CALIB_PER_DOMAIN/MAIN_PER_DOMAIN
later and re-running this script only APPENDS new files — every file_id already assigned
to calib or main (and anything Stream A/C built on top of it) stays exactly as it was.
"""

import pandas as pd

from corpus_lib import DOMAIN_TO_DATA_DIR, pull_pool

CALIB_PER_DOMAIN = 100
MAIN_PER_DOMAIN = 7_900

CORPUS_DIR = "corpus"


def select_split(records: list):
    n_needed = CALIB_PER_DOMAIN + MAIN_PER_DOMAIN
    if len(records) < n_needed:
        raise ValueError(f"only {len(records)} candidates, need {n_needed}")
    calib = records[:CALIB_PER_DOMAIN]
    main = records[CALIB_PER_DOMAIN : CALIB_PER_DOMAIN + MAIN_PER_DOMAIN]
    return calib, main


def write_domain(domain: str, calib: list, main: list) -> list:
    import os

    os.makedirs(f"{CORPUS_DIR}/{domain}", exist_ok=True)
    manifest_rows = []
    for split_name, records in [("calib", calib), ("main", main)]:
        for rec in records:
            file_id = f"{domain}_{rec['sha256'][:12]}"
            path = f"{CORPUS_DIR}/{domain}/{file_id}.bin"
            with open(path, "wb") as f:
                f.write(rec["content"].encode("utf-8"))
            manifest_rows.append(
                {
                    "file_id": file_id,
                    "domain": domain,
                    "n_bytes": rec["n_bytes"],
                    "sha256": rec["sha256"],
                    "split": split_name,
                    "source_path": rec["path"],
                    "licenses": ",".join(rec["licenses"]),
                }
            )
    return manifest_rows


def main():
    n_needed = CALIB_PER_DOMAIN + MAIN_PER_DOMAIN
    all_rows = []
    for domain in DOMAIN_TO_DATA_DIR:
        print(f"--- {domain} ---")
        r = pull_pool(domain, n_needed=n_needed)
        if r["kept"] < n_needed:
            raise SystemExit(
                f"{domain}: only {r['kept']} usable candidates, need {n_needed}"
            )
        calib, main_ = select_split(r["records"])
        rows = write_domain(domain, calib, main_)
        all_rows.extend(rows)
        print(f"wrote {len(rows)} files ({len(calib)} calib + {len(main_)} main)")

    manifest = pd.DataFrame(all_rows)
    manifest.to_parquet(f"{CORPUS_DIR}/manifest.parquet", index=False)
    print(f"\nmanifest.parquet: {len(manifest)} rows total")
    print(manifest.groupby(["domain", "split"]).size())


if __name__ == "__main__":
    main()
