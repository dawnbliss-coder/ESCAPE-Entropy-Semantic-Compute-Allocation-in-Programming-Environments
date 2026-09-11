"""R2: what fraction of AST node starts sit immediately after a newline+indent
position, per language. Answers the proposal's own confound question directly -
this is the number that says how dangerous the whitespace baseline actually is.

An AST node start "sits immediately after a newline+indent" when its start_byte
exactly matches one of whitespace/'s indent_change positions (not newline
positions - indent_change already skips leading whitespace to the first real
content byte, the same convention structure/ uses for node starts).
"""

import pandas as pd


def main():
    manifest = pd.read_parquet("corpus/manifest.parquet")
    results = {}

    for domain in ("py", "cpp"):
        total_starts = 0
        matched_starts = 0

        for file_id in manifest[manifest["domain"] == domain]["file_id"]:
            struct = pd.read_parquet(f"structure/{domain}/{file_id}.parquet")
            if struct.empty:  # a file can have zero tracked nodes (e.g. imports-only)
                continue
            ws = pd.read_parquet(f"whitespace/{domain}/{file_id}.parquet")
            indent_change_offsets = set(
                ws[ws["kind"] == "indent_change"]["byte_offset"]
            )

            starts = struct["start_byte"]
            total_starts += len(starts)
            matched_starts += starts.isin(indent_change_offsets).sum()

        results[domain] = (matched_starts, total_starts)

    print("R2: fraction of AST node starts immediately after newline+indent\n")
    for domain, (matched, total) in results.items():
        pct = matched / total * 100
        print(f"  {domain}: {matched}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
