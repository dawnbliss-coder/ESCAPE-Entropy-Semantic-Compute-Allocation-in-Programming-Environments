"""Day-1 sanity check: confirm tree-sitter parses Python and C++ before
building the real extraction pipeline on top of it."""

import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language, Parser

PY_TOY = b"""def add(a, b):
    if a > 0:
        return a + b
    return b
"""

CPP_TOY = b"""int add(int a, int b) {
    if (a > 0) {
        return a + b;
    }
    return b;
}
"""


def print_tree(node, depth: int = 0) -> None:
    print(f"{'  ' * depth}{node.type} [{node.start_byte}:{node.end_byte}]")
    for child in node.children:
        print_tree(child, depth + 1)


def parse_and_show(label: str, language: Language, source: bytes) -> None:
    parser = Parser(language)
    tree = parser.parse(source)
    print(f"--- {label} ---")
    print_tree(tree.root_node)
    assert not tree.root_node.has_error, f"{label}: parse produced an error node"


def main() -> None:
    parse_and_show("python", Language(tree_sitter_python.language()), PY_TOY)
    parse_and_show("cpp", Language(tree_sitter_cpp.language()), CPP_TOY)
    print("\nOK: both grammars load and parse without errors.")


if __name__ == "__main__":
    main()
