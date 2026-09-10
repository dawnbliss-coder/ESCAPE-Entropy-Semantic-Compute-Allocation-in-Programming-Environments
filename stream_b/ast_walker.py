"""Real tree-sitter extraction (Day 3, part 1): walk the AST and emit node_type,
parent_type, depth, start_byte, end_byte for the statement/expression-level nodes the
proposal's hypothesis (§5.2) and predictions (P4) actually need.

Node-type names are verified empirically per language (see STREAM-B-PLAN.md Day 3
notes) rather than copied from the proposal's prose, which uses C++'s names
(call_expression, binary_expression) even when describing Python, where the real
node types are `call` and `binary_operator`.

Coverage, and why each is here (not just the proposal's 5 example types):
  - if/for/while/function-def: named as deterministic openers in §5.2 ("if, for and
    while are followed by a forced token, and def must be followed by a name")
  - expression_statement/return_statement/assignment: named as the open-ended
    contrast set in P4 ("...than for open-ended ones (expression_statement,
    return_statement, assignment right-hand sides)")
  - call/binary-op: the proposal's own Methodology example types
  - declaration (cpp only; Python has no equivalent node - its declarations are just
    `assignment`): kept since Methodology names it explicitly for C++

DEPTH CONVENTION: full-tree depth from the parse root (root = 0), counting every
ancestor node — not just whitelisted ones. This is required for cross-language
depth comparison (O3) to mean anything, since Python and C++ have very different
amounts of grammar scaffolding per statement. Stream C does the depth NORMALIZATION
(relative depth, quantiles, etc. — still an open item per the proposal); this module
only produces the raw depth those normalizations are computed from.

TIE-BREAK NOTE for Stream C: the Methodology's "credit the outermost matching node
once, when node types share a start offset" rule is resolvable from (start_byte, depth)
alone — outermost = smallest depth among rows sharing a start_byte — no parent-chain
walk needed, since `depth` here is already full-tree depth.
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
