"""Whitespace baseline extraction. Two tagged kinds, not one blended signal:

  - "newline": byte offset right after every `\\n` - the literal required P1
    baseline. Deliberately does NOT skip leading indentation: the proposal's
    argument is that tolerance k absorbing that small gap IS the confound this
    baseline exists to expose; skipping it would test a different, stricter claim.
  - "indent_change": byte offset of the first non-whitespace byte on lines whose
    leading-whitespace length differs from the previous non-blank line's - what
    R2 asks directly, additional to (not a replacement for) the newline baseline.

Indentation compared by raw leading-whitespace byte length, not tab/space
semantics - sufficient for a baseline signal; real indent/dedent correctness is
already handled structurally by tree-sitter in structure/.
"""

import os

import pandas as pd

CORPUS_DIR = "corpus"
WHITESPACE_DIR = "whitespace"
# Explicit columns so a file with zero rows (e.g. a single line, no trailing
# newline) still gets the right schema - see the same fix in build_structure.py.
WHITESPACE_COLUMNS = ["byte_offset", "kind"]


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

            df = pd.DataFrame(rows, columns=WHITESPACE_COLUMNS)
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
