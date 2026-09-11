"""taxonomy.json - classify every node type ast_walker.py extracts as
deterministic_opener or open_ended, for Stream C's P4 stratification.

Classification test: NOT "is the opening keyword fixed" - C++'s return_statement
opener is "return" 100% of the time (see taxonomy_analysis.py), exactly as fixed
as if/for/while, yet the proposal classifies it open-ended (P4, S5.3). The real
test is what's grammatically forced AFTER the keyword: if/for/while/def commit
to a rigid template (S5.2: "followed by a forced token" / "must be followed by
a name"); return can be followed by nothing or by an arbitrarily-shaped
expression, exactly as open as expression_statement.

Types the proposal never classifies (call/call_expression, binary_operator/
binary_expression, C++ declaration) are decided by the same test, backed by
taxonomy_analysis.py's opener-diversity data: call/binary-op have no keyword at
all (43K-105K distinct openers, none above ~18%); declaration's primitive-type
keywords (const/auto/int/static) cover only ~32% of real occurrences, so the
empirical majority is a custom type, not a keyword. All open_ended.

Single flat mapping (matches docs/SCHEMA.md) since py/cpp node-type names never
collide (call vs call_expression, etc).
"""

import json

from ast_walker import NODE_TYPES

TAXONOMY = {
    # deterministic_opener: mandatory keyword + forced/rigid continuation
    "if_statement": "deterministic_opener",
    "for_statement": "deterministic_opener",
    "while_statement": "deterministic_opener",
    "function_definition": "deterministic_opener",
    # open_ended: either no keyword at all (arbitrary content from the first
    # byte), or a keyword whose continuation is grammatically unconstrained
    "call": "open_ended",
    "call_expression": "open_ended",
    "binary_operator": "open_ended",
    "binary_expression": "open_ended",
    "expression_statement": "open_ended",
    "return_statement": "open_ended",
    "assignment": "open_ended",
    "augmented_assignment": "open_ended",
    "assignment_expression": "open_ended",
    "declaration": "open_ended",
}


def main():
    all_tracked = NODE_TYPES["py"] | NODE_TYPES["cpp"]
    missing = all_tracked - TAXONOMY.keys()
    stale = TAXONOMY.keys() - all_tracked
    if missing:
        raise SystemExit(f"taxonomy.json missing classification for: {missing}")
    if stale:
        raise SystemExit(f"taxonomy.json has stale entries no longer tracked: {stale}")

    with open("taxonomy.json", "w") as f:
        json.dump(TAXONOMY, f, indent=2, sort_keys=True)
    print(f"wrote taxonomy.json ({len(TAXONOMY)} node types)")
    for k, v in sorted(TAXONOMY.items()):
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
