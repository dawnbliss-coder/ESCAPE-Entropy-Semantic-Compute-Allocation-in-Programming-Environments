"""Objective 4 inputs: noised variants of held-out code files, with mapped targets.

Proposal (Trimax-Proposal, "Adversarial Data"): inject missing semicolons, localized
indentation changes, camelCase<->snake_case inversions and typos in syntax keywords.
Each noise type is applied on its own to the same files, so every condition is
compared with the clean condition file by file.

  missing_semicolon  (C++ only) delete RATE of the `;` that terminate a statement
  indentation        on RATE of non-blank lines, add or remove one leading space
  case_inversion     rename CASE_RATE of the distinct identifier names that have a
                     camelCase or snake_case form, consistently at every occurrence
  keyword_typo       swap two adjacent letters in RATE of keyword tokens (if -> fi)

Targets are the CLEAN tree-sitter nodes mapped through the edits to the noised byte
offsets, i.e. the structure the author intended. Parsing the noised bytes would score
BLT against tree-sitter's error recovery instead. A node whose start or end falls
strictly inside an edited range is dropped and counted.

Files: FILES_PER_LANGUAGE held-out TEST files per code language from the frozen
calibration record (never calibration files), by seeded SHA256 rank.

Writes o4/manifest.parquet and o4/{condition}/{text,structure}/{sample_id}.* .
Reads only corpus bytes, structure/ and the calibration record; never BLT outputs.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import tree_sitter_cpp
import tree_sitter_python
from tree_sitter import Language, Parser

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260927
FILES_PER_LANGUAGE = 1000
RATE = 0.2
CASE_RATE = 0.5
DOMAIN = {"python": "py", "cpp": "cpp"}
CONDITIONS = {"python": ("indentation", "case_inversion", "keyword_typo"),
              "cpp": ("missing_semicolon", "indentation", "case_inversion", "keyword_typo")}
KEYWORDS = {
    "python": {"if", "elif", "else", "for", "while", "return", "def", "class", "import", "from", "in", "not",
               "and", "or", "try", "except", "finally", "with", "lambda", "yield", "pass", "break", "continue",
               "raise", "assert", "del", "global", "is", "as"},
    "cpp": {"if", "else", "for", "while", "do", "return", "switch", "case", "default", "break", "continue",
            "class", "struct", "new", "delete", "const", "static", "template", "typename", "namespace", "using",
            "public", "private", "protected", "virtual", "void", "int", "char", "bool", "auto", "unsigned"},
}
IDENTIFIERS = {"python": {"identifier"}, "cpp": {"identifier", "field_identifier", "type_identifier"}}
STATEMENT_PARENTS = {"expression_statement", "declaration", "return_statement", "field_declaration"}
PARSERS = {"python": Parser(Language(tree_sitter_python.language())), "cpp": Parser(Language(tree_sitter_cpp.language()))}


def rank(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).hexdigest()


def rng_for(*parts) -> random.Random:
    return random.Random(int(rank(*parts)[:16], 16))


def walk(tree):
    stack = [(tree.root_node, None)]
    while stack:
        node, parent = stack.pop()
        yield node, parent
        for child in node.children:
            stack.append((child, node))


def invert_case(name: str):
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", name):          # snake_case -> camelCase
        head, *rest = name.split("_")
        return head + "".join(p[:1].upper() + p[1:] for p in rest)
    if re.fullmatch(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+", name):        # camelCase -> snake_case
        return re.sub(r"(?<!^)([A-Z])", lambda m: "_" + m.group(1).lower(), name)
    return None


def edits_for(condition, language, content: bytes, sid):
    """Non-overlapping (start, end, replacement) byte edits for one condition."""
    rng = rng_for("o4-edit", condition, sid)
    tree = PARSERS[language].parse(content)
    edits = []
    if condition == "missing_semicolon":
        sites = sorted(n.start_byte for n, p in walk(tree)
                       if n.type == ";" and p is not None and p.type in STATEMENT_PARENTS)
        edits = [(s, s + 1, b"") for s in sites if rng.random() < RATE]
    elif condition == "keyword_typo":
        sites = sorted((n.start_byte, n.end_byte) for n, _ in walk(tree)
                       if not n.is_named and n.type in KEYWORDS[language] and n.end_byte - n.start_byte >= 2)
        for a, b in sites:
            if rng.random() >= RATE:
                continue
            word = content[a:b]
            j = rng.randrange(len(word) - 1)
            typo = word[:j] + word[j + 1:j + 2] + word[j:j + 1] + word[j + 2:]
            if typo != word:
                edits.append((a, b, typo))
    elif condition == "case_inversion":
        occurrences = {}
        for n, _ in walk(tree):
            if n.type in IDENTIFIERS[language]:
                occurrences.setdefault(content[n.start_byte:n.end_byte].decode("utf-8", "replace"), []).append(
                    (n.start_byte, n.end_byte))
        for name in sorted(occurrences):
            new = invert_case(name)
            if new and rng.random() < CASE_RATE:
                edits.extend((a, b, new.encode()) for a, b in occurrences[name])
    elif condition == "indentation":
        pos = 0
        for line in content.split(b"\n"):
            body = line.lstrip(b" \t")
            indent = len(line) - len(body)
            if body.strip() and rng.random() < RATE:
                if indent > 0 and rng.random() < 0.5:
                    edits.append((pos, pos + 1, b""))           # remove one leading whitespace byte
                else:
                    edits.append((pos, pos, b" "))              # add one leading space
            pos += len(line) + 1
    else:
        raise ValueError(condition)
    edits.sort()
    for (a1, b1, _), (a2, _, _) in zip(edits, edits[1:]):
        if a2 < b1:
            raise ValueError(f"{sid}/{condition}: overlapping edits")
    return edits


def apply(content: bytes, edits):
    out, cursor = [], 0
    for a, b, r in edits:
        out += [content[cursor:a], r]
        cursor = b
    out.append(content[cursor:])
    return b"".join(out)


def mapper(edits):
    """Old offset -> new offset, or None if the offset lies strictly inside an edit."""
    import bisect
    ends = [b for _, b, _ in edits]
    cum = [0]
    for a, b, r in edits:
        cum.append(cum[-1] + len(r) - (b - a))

    def f(o):
        i = bisect.bisect_right(ends, o)   # edits[:i] end at or before o (edits are sorted, disjoint)
        if i < len(edits) and edits[i][0] < o:
            return None                    # strictly inside edits[i]
        return o + cum[i]
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--calibration", required=True, help="frozen C-v1 calibration.json (for the test split)")
    ap.add_argument("--files-per-language", type=int, default=FILES_PER_LANGUAGE)
    args = ap.parse_args(argv)
    root = Path(args.root)
    record = json.loads(Path(args.calibration).read_text())
    out = root / "o4"
    if (out / "manifest.parquet").exists():
        raise SystemExit("o4/manifest.parquet exists; refusing to overwrite")
    rows = []
    for language, domain in DOMAIN.items():
        tests = sorted((m["sample_id"] for m in record["split_manifest"]
                        if m["language"] == language and m["split"] == "test"), key=lambda s: rank("o4-file", s))
        chosen = sorted(tests[:args.files_per_language])
        for sid in chosen:
            content = (root / f"corpus/{domain}/{sid}.bin").read_bytes()
            nodes = pd.read_parquet(root / f"structure/{domain}/{sid}.parquet")
            for condition in ("clean", *CONDITIONS[language]):
                edits = [] if condition == "clean" else edits_for(condition, language, content, sid)
                new = apply(content, edits)
                f = mapper(edits)
                mapped = nodes.copy()
                mapped["start_byte"] = [f(int(x)) for x in nodes["start_byte"]]
                mapped["end_byte"] = [f(int(x)) for x in nodes["end_byte"]]
                keep = mapped["start_byte"].notna() & mapped["end_byte"].notna()
                mapped = mapped[keep].astype({"start_byte": "int64", "end_byte": "int64"})
                mapped = mapped[mapped["end_byte"] > mapped["start_byte"]]
                for sub in ("text", "structure"):
                    (out / condition / sub).mkdir(parents=True, exist_ok=True)
                (out / condition / "text" / f"{sid}.bin").write_bytes(new)
                mapped.to_parquet(out / condition / "structure" / f"{sid}.parquet", index=False)
                rows.append({"sample_id": sid, "language": language, "condition": condition,
                             "n_bytes": len(new), "sha256": hashlib.sha256(new).hexdigest(),
                             "clean_n_bytes": len(content), "n_edits": len(edits),
                             "n_nodes": int(len(nodes)), "n_nodes_dropped": int(len(nodes) - len(mapped))})
    pd.DataFrame(rows).to_parquet(out / "manifest.parquet", index=False)
    summary = (pd.DataFrame(rows).groupby(["language", "condition"])
               .agg(files=("sample_id", "count"), edits=("n_edits", "sum"), dropped=("n_nodes_dropped", "sum"),
                    nodes=("n_nodes", "sum")).reset_index())
    (out / "build_summary.json").write_text(json.dumps({
        "seed": SEED, "rate": RATE, "case_rate": CASE_RATE, "files_per_language": args.files_per_language,
        "calibration_record_sha256": record["record_sha256"], "summary": summary.to_dict("records")}, indent=2) + "\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
