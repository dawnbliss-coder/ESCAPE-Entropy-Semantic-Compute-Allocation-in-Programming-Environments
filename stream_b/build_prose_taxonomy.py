"""prose_taxonomy.json - the prose-side mirror of taxonomy.json, for P3
(code/prose contrast). Verified against the real label set produced by
build_prose_structure.py (32 distinct types across all 300 files).

WHY THIS AXIS, NOT deterministic_opener/open_ended: that split is specifically
about whether a keyword forces a narrow continuation, and prose constituents
have no keyword-driven openers. The genuine mirror is code's OTHER split -
statement-level vs expression-level - expressed in constituency grammar's own
vocabulary: CLAUSE-level (has its own subject/predicate structure - S/SBAR/
SINV/RRC, mirrors statement-level) vs PHRASE-level (single-category, no
independent clause structure - NP/VP/PP/etc., mirrors expression-level).
Stacked labels (unary-chain collapses, e.g. "S+VP") are classified by their
OUTERMOST label.

A third bucket, structural_other (FRAG/LST/PRN/UCP/X), is kept rather than
forced into the other two - these are genuinely miscellaneous Penn Treebank
categories (fragment, list marker, parenthetical, unlike-coordination, unknown)
that don't carry the clause-vs-phrase distinction the rest do.
"""

import glob
import json

import pandas as pd

CLAUSE_LEVEL = {"S", "SBAR", "SINV", "RRC"}
PHRASE_LEVEL = {
    "ADJP", "ADVP", "CONJP", "NML", "NP", "PP", "PRT", "QP", "VP",
    "WHADJP", "WHADVP", "WHNP", "WHPP",
}
STRUCTURAL_OTHER = {"FRAG", "LST", "PRN", "UCP", "X"}

ALL_BASE_TYPES = CLAUSE_LEVEL | PHRASE_LEVEL | STRUCTURAL_OTHER


def classify(node_type: str) -> str:
    outer = node_type.split("+")[0]
    if outer in CLAUSE_LEVEL:
        return "clause_level"
    if outer in PHRASE_LEVEL:
        return "phrase_level"
    if outer in STRUCTURAL_OTHER:
        return "structural_other"
    raise ValueError(f"unclassified prose label: {node_type!r} (outer={outer!r})")


def main():
    observed = set()
    for p in glob.glob("structure/prose/*.parquet"):
        df = pd.read_parquet(p)
        observed.update(df["node_type"].unique())

    observed_outer = {t.split("+")[0] for t in observed}
    missing = observed_outer - ALL_BASE_TYPES
    if missing:
        raise SystemExit(f"prose_taxonomy.py doesn't cover observed labels: {missing}")

    taxonomy = {t: classify(t) for t in sorted(observed)}
    with open("prose_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2, sort_keys=True)

    print(f"wrote prose_taxonomy.json ({len(taxonomy)} node types observed)")
    for k, v in sorted(taxonomy.items()):
        print(f"  {k:<15} {v}")


if __name__ == "__main__":
    main()
