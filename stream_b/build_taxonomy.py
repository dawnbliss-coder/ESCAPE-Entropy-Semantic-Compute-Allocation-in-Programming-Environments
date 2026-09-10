"""Day 4, part 1: taxonomy.json - classify every node type ast_walker.py extracts
as deterministic_opener or open_ended, for Stream C's P4 stratification.

THE TEST THAT ACTUALLY MATTERS (found via stream_b/taxonomy_analysis.py, an
empirical check run before finalizing this): it is NOT "is the opening keyword
drawn from a small/fixed set." That test fails on the proposal's own examples -
C++ return_statement's opener is literally "return" in 100% of 119,627
occurrences (distinct=1), exactly as fixed as if/for/while's openers, yet the
proposal explicitly classifies return_statement as open-ended (P4, S5.3):
"...than for open-ended ones (expression_statement, return_statement, assignment
right-hand sides)."

The real distinction is what's grammatically forced AFTER the keyword, not the
keyword's own uniqueness:
  - if/for/while commit to a rigid template once entered (a forced condition
    shape, in C++ literally a forced '(' next) - branching factor near one for
    what follows, per S5.2: "if, for and while are followed by a forced token."
  - def/function_definition commits to "must be followed by a name" (S5.2) -
    the token CLASS is forced even though the specific name varies.
  - return can be followed by nothing at all, or by an expression of completely
    unbounded shape - the keyword is fixed but what follows it is exactly as
    open as expression_statement's content. Same fixed opener, opposite
    classification - this is why "opener uniqueness" is the wrong test.

So: deterministic_opener = types with a mandatory keyword that additionally
constrains what comes next to a narrow/rigid continuation. open_ended = every
other type extracted, including ones with no keyword at all (call, binary op)
and ones with a keyword but unconstrained continuation (return).

Types the proposal never explicitly classifies (call/call_expression,
binary_operator/binary_expression, C++ declaration) are decided by the same
principle, using taxonomy_analysis.py's empirical opener-diversity numbers as
supporting evidence:
  - call/call_expression, binary_operator/binary_expression: no keyword at all,
    start directly with arbitrary content (a callee expression / left operand).
    Empirically: 43,169-105,338 distinct openers, no single opener above ~18%
    of occurrences (cpp binary_expression's top opener, '<non-word>', covers
    17.9%; most others are far lower). Open-ended.
  - declaration (C++ only, no Python equivalent): sometimes begins with a
    primitive-type keyword (const/auto/int/static), but empirically these
    together cover only ~32% of occurrences (top1_share alone is 11.2%,
    18,715 distinct openers total) - the empirical majority begins with a
    custom class/template/namespaced type, not a fixed keyword. Open-ended.

Node-type name differences between languages (py's call/binary_operator vs
cpp's call_expression/binary_expression) mean no string collisions occur, so
this is a single flat mapping, matching docs/SCHEMA.md's stated shape
("taxonomy.json  node_type -> {deterministic_opener | open_ended}").
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
