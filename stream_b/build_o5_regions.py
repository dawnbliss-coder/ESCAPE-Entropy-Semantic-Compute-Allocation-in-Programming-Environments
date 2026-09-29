"""O5 inputs for C++: analysis regions, memory-unsafe masks and identifier spans.

Writes, for each selected C++ file,
  analysis_regions/cpp/{file_id}.parquet  file_id, start_byte, end_byte, statement_type
  memory_regions/cpp/{file_id}.parquet    file_id, start_byte, end_byte, unsafe_kind
  identifiers/cpp/{file_id}.parquet       file_id, start_byte, end_byte
plus analysis_regions/cpp/_selection.json recording the rule, seed and counts.
escape_scoring's convert-legacy reads these into region_annotations.jsonl.

Operationalisation (a default, recorded so the team can revise it):
  - Analysis units are simple statements directly inside a block
    (`expression_statement`, `declaration`, `return_statement` whose parent is a
    `compound_statement`), so unsafe and comparison units are the same kind of thing.
  - A unit is memory-unsafe if it contains one of: `new`/`delete` expressions;
    unary `*` dereference or `&` address-of (`pointer_expression`); calls to the C
    allocation/raw-memory/string functions in UNSAFE_CALLS; `reinterpret_cast` or
    `const_cast`; a C-style cast to a pointer type.
  - Only files whose tree-sitter parse needed no error recovery (manifest
    parse_ok) and that contain at least one unsafe and one comparison unit are
    eligible: O5 uses file fixed effects, so a file contributes only through its
    within-file contrast.
  - From each eligible file up to PER_FILE unsafe and PER_FILE comparison units are
    drawn, and at most MAX_FILES eligible files are kept, all by seeded SHA256
    rank. Identifier and unsafe spans are written only where they overlap a
    selected unit, since O5 reads nothing else.

This labels syntax, not program safety: no types, aliasing or bounds are known.
Reads only corpus bytes and the manifest; never BLT outputs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import tree_sitter_cpp
from tree_sitter import Language, Parser

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260927
PER_FILE = 5
MAX_FILES = 1500
STATEMENTS = {"expression_statement", "declaration", "return_statement"}
IDENTIFIERS = {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}
UNSAFE_CALLS = {"malloc", "calloc", "realloc", "free", "alloca", "memcpy", "memmove", "memset",
                "strcpy", "strncpy", "strcat", "strncat", "sprintf", "vsprintf", "gets", "scanf", "sscanf"}
UNSAFE_CASTS = {"reinterpret_cast", "const_cast"}
PARSER = Parser(Language(tree_sitter_cpp.language()))


def rank(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).hexdigest()


def callee_name(node, content: bytes):
    fn = node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "template_function":
        fn = fn.child_by_field_name("name") or fn
    if fn.type in {"qualified_identifier"}:
        fn = fn.child_by_field_name("name") or fn
    if fn.type == "identifier":
        return content[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
    return None


def unsafe_kind(node, content: bytes):
    if node.type in {"new_expression", "delete_expression"}:
        return node.type
    if node.type == "pointer_expression":
        return "pointer_expression"
    if node.type == "call_expression":
        name = callee_name(node, content)
        if name in UNSAFE_CALLS:
            return f"call:{name}"
        if name in UNSAFE_CASTS:
            return name
    if node.type == "cast_expression":
        type_node = node.child_by_field_name("type")
        if type_node is not None and b"*" in content[type_node.start_byte:type_node.end_byte]:
            return "pointer_cast"
    return None


def scan(content: bytes):
    tree = PARSER.parse(content)
    units, unsafe, idents = [], [], []
    stack = [(tree.root_node, None)]
    while stack:
        node, parent_type = stack.pop()
        if node.type in STATEMENTS and parent_type == "compound_statement" and node.end_byte > node.start_byte:
            units.append((node.start_byte, node.end_byte, node.type))
        kind = unsafe_kind(node, content)
        if kind:
            unsafe.append((node.start_byte, node.end_byte, kind))
        if node.type in IDENTIFIERS and node.end_byte > node.start_byte:
            idents.append((node.start_byte, node.end_byte))
        for child in node.children:
            stack.append((child, node.type))
    return units, unsafe, idents


def overlaps(a, b, spans):
    return [s for s in spans if max(a, s[0]) < min(b, s[1])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    manifest = pd.read_parquet(root / "corpus/manifest.parquet")
    cpp = manifest[(manifest["domain"] == "cpp") & (manifest["parse_ok"] == True)]  # noqa: E712
    eligible = {}
    for file_id in cpp["file_id"]:
        content = (root / f"corpus/cpp/{file_id}.bin").read_bytes()
        units, unsafe, idents = scan(content)
        flagged = [u for u in units if overlaps(u[0], u[1], unsafe)]
        clean = [u for u in units if not overlaps(u[0], u[1], unsafe)]
        if flagged and clean:
            eligible[file_id] = (flagged, clean, unsafe, idents)
    chosen = sorted(eligible, key=lambda f: rank("o5-file", f))[:MAX_FILES]
    out = {name: root / f"{name}/cpp" for name in ("analysis_regions", "memory_regions", "identifiers")}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    n_units = {"unsafe": 0, "comparison": 0}
    for file_id in sorted(chosen):
        flagged, clean, unsafe, idents = eligible[file_id]
        pick_u = sorted(flagged, key=lambda u: rank("o5-unit", file_id, *u))[:PER_FILE]
        pick_c = sorted(clean, key=lambda u: rank("o5-unit", file_id, *u))[:PER_FILE]
        n_units["unsafe"] += len(pick_u)
        n_units["comparison"] += len(pick_c)
        units = sorted(pick_u + pick_c)
        keep_unsafe = sorted({s for u in pick_u for s in overlaps(u[0], u[1], unsafe)})
        keep_idents = sorted({s for u in units for s in overlaps(u[0], u[1], idents)})
        pd.DataFrame([{"file_id": file_id, "start_byte": a, "end_byte": b, "statement_type": t} for a, b, t in units],
                     columns=["file_id", "start_byte", "end_byte", "statement_type"]).to_parquet(
            out["analysis_regions"] / f"{file_id}.parquet", index=False)
        pd.DataFrame([{"file_id": file_id, "start_byte": a, "end_byte": b, "unsafe_kind": k} for a, b, k in keep_unsafe],
                     columns=["file_id", "start_byte", "end_byte", "unsafe_kind"]).to_parquet(
            out["memory_regions"] / f"{file_id}.parquet", index=False)
        pd.DataFrame([{"file_id": file_id, "start_byte": a, "end_byte": b} for a, b in keep_idents],
                     columns=["file_id", "start_byte", "end_byte"]).to_parquet(
            out["identifiers"] / f"{file_id}.parquet", index=False)
    record = {"seed": SEED, "per_file": PER_FILE, "max_files": MAX_FILES,
              "statement_types": sorted(STATEMENTS), "unsafe_calls": sorted(UNSAFE_CALLS),
              "unsafe_casts": sorted(UNSAFE_CASTS), "n_cpp_parse_ok": int(len(cpp)),
              "n_eligible_files": len(eligible), "n_selected_files": len(chosen), "n_units": n_units}
    (out["analysis_regions"] / "_selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
