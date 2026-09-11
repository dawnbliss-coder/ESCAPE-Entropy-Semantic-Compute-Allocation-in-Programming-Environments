"""R3, part 2: parse every prose file with benepar, convert its character offsets
to byte offsets, and write structure/prose/{file_id}.parquet - same schema as
code's structure/ (file_id, node_type, parent_type, depth, start_byte, end_byte)
so Stream C's scoring code can treat all three domains uniformly.

Real structural differences from code, decided before writing the walker (see
STREAM-B-PLAN.md for the full reasoning, verified against real benepar output on
an actual corpus file, not a toy sentence):

  - benepar/NLTK collapses unary chains (a constituent whose only child is
    itself, structurally, e.g. S dominating nothing but VP) into ONE span with
    a STACKED label tuple, e.g. ('S', 'VP'). Represented here as node_type
    "S+VP" (labels joined with '+') rather than picking one and discarding the
    rest - lossless and simple to split back out later if needed. NOTE for
    whoever does O3-style depth comparison: this means prose depth is not
    perfectly comparable to code depth in an absolute sense (a collapsed
    "S+VP" counts as ONE depth level here, where tree-sitter would give S and
    VP separate levels if code had an equivalent unary chain) - relative/
    normalized depth comparisons (already an open item per the proposal) matter
    more than raw depth counts for this reason.
  - Leaf tokens (individual words/punctuation) have an EMPTY label tuple - not
    real constituents, skipped, same principle as not tracking every AST leaf.
  - A paragraph is multiple independent sentences (multiple parse-tree roots),
    unlike a code file (one root: module/translation_unit). To keep depth
    comparable across domains, each sentence's root constituent is placed at
    depth=1 (matching how a code file's top-level statement is depth 1 under
    the module root at depth 0) rather than depth=0; parent_type for a
    sentence root is the synthetic string "paragraph" (there is no real
    benepar node for "the whole file", analogous to how code's top-level
    parent_type is a real grammar node like "module").

Byte-offset conversion is validated per span, not assumed correct: after
converting start_char/end_char to start_byte/end_byte, the corresponding slice
of the RAW BYTES is decoded and compared against benepar's own span text. Any
mismatch raises immediately rather than silently writing a wrong offset.
"""

import os

import pandas as pd

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"


def char_to_byte_offsets(text: str):
    """offsets[i] = byte offset corresponding to character index i. Length is
    len(text)+1 so end-of-span offsets (one past the last character) work too."""
    offsets = [0] * (len(text) + 1)
    byte_pos = 0
    for i, ch in enumerate(text):
        offsets[i] = byte_pos
        byte_pos += len(ch.encode("utf-8"))
    offsets[len(text)] = byte_pos
    return offsets


def walk_sentence(sent, char_to_byte, content: bytes):
    """Iterative (stack-based) walk of one sentence's constituent tree.
    Returns a list of row dicts. Root is placed at depth=1, parent_type
    "paragraph" - see module docstring."""
    rows = []
    # stack entries: (span, parent_node_type, depth)
    stack = [(sent, "paragraph", 1)]
    while stack:
        span, parent_type, depth = stack.pop()
        labels = span._.labels
        if labels:
            node_type = "+".join(labels)
            start_byte = char_to_byte[span.start_char]
            end_byte = char_to_byte[span.end_char]
            # Validate the conversion against benepar's own span text - don't
            # trust it silently.
            recovered = content[start_byte:end_byte].decode("utf-8")
            if recovered != span.text:
                raise ValueError(
                    f"offset mismatch: expected {span.text!r}, got {recovered!r} "
                    f"(chars {span.start_char}:{span.end_char}, "
                    f"bytes {start_byte}:{end_byte})"
                )
            rows.append(
                {
                    "node_type": node_type,
                    "parent_type": parent_type,
                    "depth": depth,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                }
            )
            next_parent_type = node_type
        else:
            # leaf token: not emitted as a row, but its children (there are
            # none for a true leaf) would inherit its parent_type unchanged
            next_parent_type = parent_type

        for child in span._.children:
            stack.append((child, next_parent_type, depth + (1 if labels else 0)))
    return rows


def main():
    import spacy
    import benepar

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    manifest = pd.read_parquet(f"{CORPUS_DIR}/manifest.parquet")
    prose = manifest[manifest["domain"] == "prose"]

    os.makedirs(f"{STRUCTURE_DIR}/prose", exist_ok=True)
    node_type_counts = {}
    n_sentences = 0

    for _, row in prose.iterrows():
        file_id = row["file_id"]
        with open(f"{CORPUS_DIR}/prose/{file_id}.bin", "rb") as f:
            content = f.read()
        text = content.decode("utf-8")
        char_to_byte = char_to_byte_offsets(text)

        doc = nlp(text)
        all_rows = []
        for sent in doc.sents:
            n_sentences += 1
            sent_rows = walk_sentence(sent, char_to_byte, content)
            all_rows.extend(sent_rows)
            for r in sent_rows:
                node_type_counts[r["node_type"]] = (
                    node_type_counts.get(r["node_type"], 0) + 1
                )

        df = pd.DataFrame(all_rows)
        df.insert(0, "file_id", file_id)
        df.to_parquet(f"{STRUCTURE_DIR}/prose/{file_id}.parquet", index=False)

    print(f"processed {len(prose)} prose files, {n_sentences} sentences total")
    print("\ntop node_type counts:")
    for k, v in sorted(node_type_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
