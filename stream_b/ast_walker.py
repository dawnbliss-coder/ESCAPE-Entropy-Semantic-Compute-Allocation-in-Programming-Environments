"""Walks the AST and emits node_type, parent_type, depth, start_byte, end_byte
for the statement/expression-level nodes the proposal's hypothesis (§5.2) and
predictions (P4) need - a superset of the 5 example types in Methodology
(also: expression_statement/return_statement/assignment for P4's open-ended
contrast set, since P4 can't run without them).

Node-type names are verified empirically per language, not copied from the
proposal's prose (which uses C++'s names even when describing Python - the real
Python types are `call`/`binary_operator`, not `call_expression`/
`binary_expression`). `declaration` is cpp-only; Python has no equivalent node.

DEPTH: full-tree depth from the parse root (root=0), counting every ancestor,
not just whitelisted types - required for cross-language depth comparison (O3)
to mean anything, since Python and C++ carry very different amounts of grammar
scaffolding per statement. Stream C does the depth NORMALIZATION; this only
produces the raw depth it's computed from.

TIE-BREAK for Stream C: "credit the outermost matching node when types share a
start offset" is resolvable from (start_byte, depth) alone - outermost = smallest
depth among rows sharing a start_byte - no parent-chain walk needed.
"""

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language, Parser

NODE_TYPES = {
    "py": {
        "if_statement",
        "for_statement",
        "while_statement",
        "function_definition",
        "call",
        "binary_operator",
        "expression_statement",
        "return_statement",
        "assignment",
        "augmented_assignment",
    },
    "cpp": {
        "if_statement",
        "for_statement",
        "while_statement",
        "function_definition",
        "declaration",
        "call_expression",
        "binary_expression",
        "expression_statement",
        "return_statement",
        "assignment_expression",
    },
}

_LANGUAGES = {
    "py": Language(tree_sitter_python.language()),
    "cpp": Language(tree_sitter_cpp.language()),
}


def _parser(domain: str) -> Parser:
    return Parser(_LANGUAGES[domain])


def walk_file(domain: str, content: bytes):
    """Returns (rows, parsed_ok). rows is a list of dicts matching structure/'s
    schema (minus file_id, added by the caller). parsed_ok is False if tree-sitter's
    error-recovery kicked in anywhere in the tree (still returns whatever rows it
    found — tree-sitter is error-tolerant, but the caller should track/report this
    rather than silently trust a possibly-corrupted tree).
    """
    parser = _parser(domain)
    tree = parser.parse(content)
    wanted = NODE_TYPES[domain]
    rows = []

    # Iterative (explicit-stack) DFS, not recursive: real files in this corpus have
    # been observed with 1000+ levels of nesting (deeply nested/generated code), which
    # blows Python's recursion limit on a recursive walker. This has no depth limit.
    stack = [(tree.root_node, None, 0)]
    while stack:
        node, parent_type, depth = stack.pop()
        if node.type in wanted:
            rows.append(
                {
                    "node_type": node.type,
                    "parent_type": parent_type,
                    "depth": depth,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                }
            )
        for child in node.children:
            stack.append((child, node.type, depth + 1))

    return rows, not tree.root_node.has_error
