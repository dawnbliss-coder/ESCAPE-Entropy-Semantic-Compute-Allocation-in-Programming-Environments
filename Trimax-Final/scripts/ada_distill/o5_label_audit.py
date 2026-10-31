"""Full-run Step 8 (docs/ADA_FULL_RUN_GATES.md): audit the O5 memory-unsafe labels.

Reads corpus bytes and the frozen results/final/c-v1-2721954/o5/o5_regions.csv; never BLT
outputs, never the scorer. Two parts:

  reproduction  re-run stream_b/build_o5_regions.py's tree-sitter scan on the restored
                bytes of all O5 files: does every frozen region exist with the same span,
                statement type and is_memory_unsafe flag?
  validity      for every frozen region of SAMPLE files (seeded SHA256 rank among the O5
                files), compare the tree-sitter labels (*, &, new, delete, unsafe call,
                unsafe cast) with an independent lexical labeller; every disagreement is
                listed with its source text for manual adjudication.

Writes results/final/o5_label_audit/{summary.json,units.csv}. Run with the Stream-B env
(tree_sitter): b_py scripts/ada_distill/o5_label_audit.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stream_b"))
from build_o5_regions import (PARSER, STATEMENTS, UNSAFE_CALLS, UNSAFE_CASTS,  # noqa: E402
                              overlaps, scan)

FROZEN = ROOT / "results/final/c-v1-2721954/o5/o5_regions.csv"
OUT = ROOT / "results/final/o5_label_audit"
SEED = 20260927
SAMPLE = 50
CATS = ("*", "&", "new", "delete", "call", "cast")

TOKEN = re.compile(r"""
    (?P<ws>\s+) | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
  | (?P<id>[A-Za-z_]\w*) | (?P<num>\.?\d[\w.']*)
  | (?P<op>->\*|->|::|<<=|>>=|\+\+|--|&&|\|\||<=|>=|==|!=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\.\.\.|[-+*/%&|^~!<>=?:;,.()\[\]{}\#])
""", re.S | re.X)
# A '*' or '&' is unary (dereference / address-of) when the token before it cannot end an
# operand: nothing, an operator or opening bracket, or a keyword that precedes an expression.
OPERAND_END_OPS = {")", "]", "++", "--"}
EXPR_KEYWORDS = {"return", "throw", "case", "sizeof", "co_return", "co_yield", "delete", "new"}
TYPE_WORDS = {"const", "volatile", "unsigned", "signed", "struct", "class", "enum", "typename"}


def tokens(text: str):
    return [(m.lastgroup, m.group()) for m in TOKEN.finditer(text)
            if m.lastgroup not in ("ws", "comment")]


def lexical_labels(text: str) -> set:
    toks = tokens(text)
    labels = set()
    for i, (kind, tok) in enumerate(toks):
        prev = toks[i - 1] if i else (None, None)
        nxt = toks[i + 1] if i + 1 < len(toks) else (None, None)
        if kind == "id":
            if tok == "new" and prev[1] != "operator":
                labels.add("new")
            elif tok == "delete" and prev[1] not in ("operator", "="):
                labels.add("delete")
            elif tok in UNSAFE_CALLS and nxt[1] == "(" and prev[1] not in (".", "->"):
                labels.add("call")
            elif tok in UNSAFE_CASTS and nxt[1] == "<":
                labels.add("cast")
        elif kind == "op" and tok in ("*", "&"):
            unary = (prev[0] is None or (prev[0] == "op" and prev[1] not in OPERAND_END_OPS)
                     or (prev[0] == "id" and prev[1] in EXPR_KEYWORDS))
            if unary and not c_style_pointer_cast_star(toks, i):
                labels.add(tok)
        if kind == "op" and tok == "(" and c_style_pointer_cast_at(toks, i):
            labels.add("cast")
    return labels


def c_style_pointer_cast_at(toks, i) -> bool:
    """'(' type-words... '*'+ ')' followed by an operand start: a C-style pointer cast."""
    j, saw_type = i + 1, False
    while j < len(toks) and (toks[j][0] == "id" or toks[j][1] in ("::", "<", ">", ",")):
        saw_type, j = True, j + 1
    stars = 0
    while j < len(toks) and toks[j][1] == "*":
        stars, j = stars + 1, j + 1
    if not (saw_type and stars and j < len(toks) and toks[j][1] == ")"):
        return False
    after = toks[j + 1] if j + 1 < len(toks) else (None, None)
    return after[0] in ("id", "num", "str") or after[1] in ("(", "*", "&", "-", "!", "~")


def c_style_pointer_cast_star(toks, i) -> bool:
    """The '*' at i sits inside a C-style pointer cast's type, e.g. (char *) p."""
    j = i
    while j >= 0 and toks[j][1] == "*":
        j -= 1
    while j >= 0 and (toks[j][0] == "id" or toks[j][1] in ("::", "<", ">", ",")):
        j -= 1
    return j >= 0 and toks[j][1] == "(" and j + 1 < i and c_style_pointer_cast_at(toks, j)


def tree_sitter_labels(content: bytes, root, a: int, b: int, unsafe_spans) -> set:
    labels = set()
    for s, e, kind in overlaps(a, b, unsafe_spans):
        if kind == "pointer_expression":
            node = root.descendant_for_byte_range(s, e)
            while node is not None and not (node.type == "pointer_expression"
                                             and node.start_byte == s and node.end_byte == e):
                node = node.parent
            op = node.child_by_field_name("operator") if node is not None else None
            labels.add(content[op.start_byte:op.end_byte].decode() if op is not None else "?")
        elif kind == "new_expression":
            labels.add("new")
        elif kind == "delete_expression":
            labels.add("delete")
        elif kind.startswith("call:"):
            labels.add("call")
        else:  # reinterpret_cast, const_cast, pointer_cast
            labels.add("cast")
    return labels


def main():
    frozen = pd.read_csv(FROZEN)
    files = sorted(frozen["sample_id"].unique())
    rep = {"files": len(files), "regions": len(frozen), "regions_reproduced": 0,
           "unsafe_flag_agree": 0, "statement_type_agree": 0, "files_fully_reproduced": 0}
    sample = set(sorted(files, key=lambda f: hashlib.sha256(f"{SEED}\0o5-audit\0{f}".encode()).hexdigest())[:SAMPLE])
    rows = []
    for fid in files:
        content = (ROOT / f"corpus/cpp/{fid}.bin").read_bytes()
        units, unsafe, _ = scan(content)
        unit_type = {(a, b): t for a, b, t in units}
        regions = frozen[frozen["sample_id"] == fid]
        ok_file = True
        root = PARSER.parse(content).root_node if fid in sample else None
        for r in regions.itertuples():
            a, b = int(r.start_byte), int(r.end_byte)
            found = (a, b) in unit_type
            flag = int(bool(overlaps(a, b, unsafe)))
            rep["regions_reproduced"] += found
            rep["unsafe_flag_agree"] += flag == int(r.is_memory_unsafe)
            rep["statement_type_agree"] += found and unit_type[(a, b)] == r.control_statement_type
            ok_file &= found and flag == int(r.is_memory_unsafe) and unit_type[(a, b)] == r.control_statement_type
            if fid in sample:
                text = content[a:b].decode("utf-8", "replace")
                ts, lex = tree_sitter_labels(content, root, a, b, unsafe), lexical_labels(text)
                rows.append({"sample_id": fid, "start_byte": a, "end_byte": b,
                             "statement_type": r.control_statement_type, "is_memory_unsafe": int(r.is_memory_unsafe),
                             "tree_sitter": " ".join(sorted(ts)), "lexical": " ".join(sorted(lex)),
                             "agree": ts == lex, "text": " ".join(text.split())[:240]})
        rep["files_fully_reproduced"] += ok_file
    units_df = pd.DataFrame(rows)
    validity = {"sample_files": SAMPLE, "units": len(units_df),
                "unsafe_units": int(units_df["is_memory_unsafe"].sum()),
                "units_all_labels_agree": int(units_df["agree"].sum()), "per_label": {}}
    for c in CATS:
        ts = units_df["tree_sitter"].str.split().apply(lambda s, c=c: c in s)
        lx = units_df["lexical"].str.split().apply(lambda s, c=c: c in s)
        validity["per_label"][c] = {"both": int((ts & lx).sum()), "tree_sitter_only": int((ts & ~lx).sum()),
                                    "lexical_only": int((~ts & lx).sum()), "neither": int((~ts & ~lx).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    units_df.to_csv(OUT / "units.csv", index=False)
    summary = {"frozen": str(FROZEN.relative_to(ROOT)), "seed": SEED, "statement_types": sorted(STATEMENTS),
               "reproduction": rep, "validity": validity}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nDISAGREEMENTS (tree_sitter | lexical | text):")
    for r in units_df[~units_df["agree"]].itertuples():
        print(f"  {r.sample_id}:{r.start_byte} [{r.tree_sitter}] | [{r.lexical}] | {r.text[:160]}")


if __name__ == "__main__":
    main()
