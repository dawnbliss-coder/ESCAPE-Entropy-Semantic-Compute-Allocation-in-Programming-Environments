"""Research check before writing taxonomy.json: for each node type we extract,
empirically measure how constrained its "opener" actually is in our own corpus,
rather than only relying on the proposal's stated examples (if/for/while/def
deterministic; expression_statement/return_statement/assignment open-ended).
This is needed because several tracked node types (call, binary op, C++
declaration, augmented_assignment) aren't explicitly classified in the proposal's
text at all.

Method: for every occurrence of a node type, take the "opener" = the first
contiguous run of word characters (alnum/underscore) starting at start_byte
(e.g. "if_statement" -> "if", "call" -> the callee name). Then measure, across
ALL occurrences of that node type in the whole corpus: how many DISTINCT openers
appear, and what fraction of occurrences the single most common opener covers.

A node type with one opener covering ~100% of occurrences (e.g. always
literally "if") has branching factor ~1 at that position - a real deterministic
opener. A node type with thousands of distinct openers, no single one dominant,
has an effectively open vocabulary at that position.
"""

import re
from collections import Counter

import pandas as pd

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"

WORD_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def opener_token(content: bytes, start_byte: int) -> str:
    m = WORD_RE.match(content, start_byte)
    return m.group().decode("utf-8", errors="replace") if m else "<non-word>"


def main():
    manifest = pd.read_parquet(f"{CORPUS_DIR}/manifest.parquet")
    code_manifest = manifest[manifest["domain"].isin(["py", "cpp"])]
    openers = {}  # (domain, node_type) -> Counter of opener strings

    for domain, group in code_manifest.groupby("domain"):
        for file_id in group["file_id"]:
            struct_path = f"{STRUCTURE_DIR}/{domain}/{file_id}.parquet"
            df = pd.read_parquet(struct_path)
            if df.empty:
                continue
            with open(f"{CORPUS_DIR}/{domain}/{file_id}.bin", "rb") as f:
                content = f.read()
            for _, row in df.iterrows():
                key = (domain, row["node_type"])
                openers.setdefault(key, Counter())
                openers[key][opener_token(content, row["start_byte"])] += 1

    print(f"{'domain':>5} {'node_type':<22} {'n':>9} {'distinct':>9} {'top1_share':>11}  top openers")
    for (domain, node_type), counter in sorted(openers.items()):
        total = sum(counter.values())
        distinct = len(counter)
        top = counter.most_common(5)
        top1_share = top[0][1] / total
        top_str = ", ".join(f"{tok!r}:{n}" for tok, n in top)
        print(
            f"{domain:>5} {node_type:<22} {total:>9} {distinct:>9} {top1_share:>10.1%}  {top_str}"
        )


if __name__ == "__main__":
    main()
