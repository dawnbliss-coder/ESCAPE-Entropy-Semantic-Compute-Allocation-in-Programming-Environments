"""Day 2 step 1: pull candidate pools for py/cpp from bigcode/the-stack-smol,
dedup by content hash, and sanity-check license metadata. Writes nothing to corpus/
yet — this is the report-before-you-commit step; corpus/ + manifest.parquet come next
once the candidate pools look sane (see build_corpus.py).
"""

from corpus_lib import DOMAIN_TO_DATA_DIR, NEEDED_PER_DOMAIN, pull_pool


def main():
    results = {}
    for domain, data_dir in DOMAIN_TO_DATA_DIR.items():
        print(f"--- {domain} ({data_dir}) ---")
        r = pull_pool(domain)
        results[domain] = r
        print(f"pulled: {r['pulled']}")
        print(f"kept (post dedup + license check): {r['kept']}  (need {NEEDED_PER_DOMAIN})")
        print(f"dropped as exact-content duplicate: {r['dropped_dup']}")
        print(f"dropped for missing license metadata: {r['dropped_no_license']}")
        top_licenses = r["license_counts"].most_common(10)
        print(f"top licenses seen: {top_licenses}")
        print()

    for domain, r in results.items():
        status = "OK" if r["kept"] >= NEEDED_PER_DOMAIN else "SHORT"
        print(f"{domain}: {status} ({r['kept']}/{NEEDED_PER_DOMAIN})")


if __name__ == "__main__":
    main()
