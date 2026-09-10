"""Day 3, part 2: whitespace baseline extraction. Two tagged kinds of byte offset,
per the proposal's own text and the R2 open item (not one blended signal):

  - "newline": byte offset right after every `\\n`. This is the literal required
    baseline ("Second baseline: whitespace... a trivial 'split at every newline'
    rule"). Deliberately does NOT skip past leading indentation — the proposal's
    argument is that boundary-tolerance k can absorb the small gap from raw
    newline to real content, which is exactly the confound this baseline exists to
    expose. Skipping the indentation here would test a different, stricter claim.

  - "indent_change": byte offset of the first non-whitespace byte on any non-blank
    line whose leading-whitespace length differs from the previous non-blank
    line's. This is what R2 asks directly ("what fraction of AST node starts sit
    immediately after a newline + indent") and is additional to, not a replacement
    for, the newline baseline above.

Indentation is compared by raw leading-whitespace BYTE LENGTH, not by interpreting
tabs/spaces semantically — sufficient for a baseline signal; real indent/dedent
correctness is already handled structurally by tree-sitter in structure/.

Byte offsets use the exact same convention as structure/ (raw byte positions into
corpus/{domain}/{file_id}.bin, no re-encoding).
"""

import os

import pandas as pd

CORPUS_DIR = "corpus"
WHITESPACE_DIR = "whitespace"


def extract_newlines(content: bytes):
    offsets = []
    for i, b in enumerate(content):
        if b == 0x0A and i + 1 < len(content):  # '\n', skip if it's the last byte
            offsets.append(i + 1)
    return offsets


def _leading_ws_len(line: bytes) -> int:
    n = 0
    for b in line:
        if b in (0x20, 0x09):  # space, tab
            n += 1
        else:
            break
    return n


def extract_indent_changes(content: bytes):
    offsets = []
    prev_indent = 0
    pos = 0
    for line in content.split(b"\n"):
        stripped_len = len(line.rstrip(b" \t\r"))
        is_blank = stripped_len == 0
        if not is_blank:
            indent = _leading_ws_len(line)
            if indent != prev_indent:
                offsets.append(pos + indent)
            prev_indent = indent
        pos += len(line) + 1  # +1 for the '\n' consumed by split
    return offsets


def build_rows(content: bytes):
    rows = [{"byte_offset": o, "kind": "newline"} for o in extract_newlines(content)]
    rows += [
        {"byte_offset": o, "kind": "indent_change"}
        for o in extract_indent_changes(content)
    ]
    return rows


def main():
    manifest = pd.read_parquet(f"{CORPUS_DIR}/manifest.parquet")
    counts = {}

    for domain, group in manifest.groupby("domain"):
        os.makedirs(f"{WHITESPACE_DIR}/{domain}", exist_ok=True)
        n_newline = n_indent = 0
        for file_id in group["file_id"]:
            with open(f"{CORPUS_DIR}/{domain}/{file_id}.bin", "rb") as f:
                content = f.read()
            rows = build_rows(content)
            n_newline += sum(1 for r in rows if r["kind"] == "newline")
            n_indent += sum(1 for r in rows if r["kind"] == "indent_change")

            df = pd.DataFrame(rows)
            df.insert(0, "file_id", file_id)
            df.to_parquet(
                f"{WHITESPACE_DIR}/{domain}/{file_id}.parquet", index=False
            )
        counts[domain] = (n_newline, n_indent)
        print(
            f"{domain}: wrote whitespace for {len(group)} files "
            f"({n_newline} newline rows, {n_indent} indent_change rows)"
        )


if __name__ == "__main__":
    main()
