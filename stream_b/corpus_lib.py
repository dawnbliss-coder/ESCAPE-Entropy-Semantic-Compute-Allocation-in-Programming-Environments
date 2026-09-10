"""Shared corpus-building logic for Stream B: pulling from the-stack-smol, dedup,
license sanity-checking. Used by pull_and_check.py (report-only) and build_corpus.py
(writes corpus/ + manifest.parquet).
"""

import hashlib
from collections import Counter

from datasets import load_dataset

# domain -> the-stack-smol data_dir. domain names follow docs/SCHEMA.md ("py", "cpp",
# "prose"), not the-stack-smol's own folder names (which is why py maps to data/python).
DOMAIN_TO_DATA_DIR = {
    "py": "data/python",
    "cpp": "data/c++",
}

# Pinned so re-running this script (anyone, anytime) pulls byte-identical rows even if
# the dataset is later revised on the Hub. Get the current sha via:
#   HfApi().dataset_info("bigcode/the-stack-smol").sha
DATASET_REVISION = "4a6938ce94446f324c6629e7de00ac591710044b"

# the-stack-smol has exactly 10,000 files per language (verified directly). N is set
# well under that (see build_corpus.py) to leave a deliberate buffer, not because the
# pool is close to exhausted.
POOL_SIZE_PER_DOMAIN = 10_000

NEEDED_PER_DOMAIN = 8_000  # see STREAM-B-PLAN.md for the sizing rationale
# Empirically ~0% dropped to dedup/license at 900-sample scale (see pull_and_check.py's
# original run), so a small buffer is enough — NOT the earlier 3x, which at N=8000 would
# ask for 24,000 and silently clamp to the entire 10,000-file pool, leaving no buffer at
# all (the opposite of the intent).
PULL_MULTIPLIER = 1.05

# NOTE on growth-safety: `records` below preserves the dataset's own row order (never
# shuffled). That means the first K entries of `records` are IDENTICAL no matter how
# large n_needed grows later — pulling more only appends further down the list. This is
# what lets build_corpus.py assign calib/main as fixed-position prefixes that never
# change when N increases (see build_corpus.py's module docstring).


def pull_pool(domain: str, n_needed: int = NEEDED_PER_DOMAIN):
    data_dir = DOMAIN_TO_DATA_DIR[domain]
    n_pull = int(n_needed * PULL_MULTIPLIER)
    ds = load_dataset(
        "bigcode/the-stack-smol",
        data_dir=data_dir,
        split="train",
        revision=DATASET_REVISION,
    )
    ds = ds.select(range(min(n_pull, len(ds))))

    seen_hashes = set()
    license_counts = Counter()
    kept = []
    dropped_dup = 0
    dropped_no_license = 0

    for ex in ds:
        content = ex["content"]
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes.add(h)

        licenses = ex.get("licenses") or []
        license_counts.update(licenses)
        if not licenses:
            dropped_no_license += 1
            continue

        kept.append(
            {
                "sha256": h,
                "n_bytes": len(content.encode("utf-8")),
                "path": ex.get("path") or ex.get("max_stars_repo_path"),
                "licenses": licenses,
                "content": content,
            }
        )

    return {
        "domain": domain,
        "pulled": len(ds),
        "kept": len(kept),
        "dropped_dup": dropped_dup,
        "dropped_no_license": dropped_no_license,
        "license_counts": license_counts,
        "records": kept,
    }
