"""Day 5: prose_taxonomy.json - the prose-side mirror of taxonomy.json, for P3
(code/prose contrast). Verified against the real label set actually produced by
build_prose_structure.py (32 distinct types across all 300 files), not guessed.

WHY THIS AXIS, NOT deterministic_opener/open_ended: code's taxonomy.json splits
on whether what's grammatically forced after a keyword is narrow (if/for/while)
or open (return, assignment) - a mechanism specific to code having fixed
keywords at all. Prose constituents have no keyword-driven openers, so that
axis doesn't transfer. The genuine structural mirror is code's OTHER split -
statement-level vs expression-level (see ast_walker.py's NODE_TYPES) - expressed
in constituency-grammar's own vocabulary: CLAUSE-level (a constituent with its
own subject/predicate structure, standing in for a full clause - S/SBAR/SINV,
mirrors code's statement-level: if_statement/for_statement/etc.) vs
PHRASE-level (a single-category constituent with no independent clause
structure - NP/VP/PP/etc., mirrors code's expression-level: call/binary_operator/
etc.). Stacked labels (unary-chain collapses, e.g. "S+VP") are classified by
their OUTERMOST label - "S+VP" is clause-level because it IS an S, just one
that happens to unary-dominate a VP.

A third bucket is kept, not forced into the other two: FRAG/LST/PRN/UCP/X are
genuinely structural/miscellaneous categories in the Penn Treebank tagset
(fragment, list marker, parenthetical, unlike-coordinated-phrase, unknown) -
none of them carry the clause-vs-phrase distinction the other 27 do, and
mislabeling them either way would be worse than an honest third category.
"""

import json

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
    import glob

    import pandas as pd

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
