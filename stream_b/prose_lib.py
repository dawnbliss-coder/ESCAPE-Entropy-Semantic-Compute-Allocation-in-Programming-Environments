"""Prose corpus pulling logic (WikiText-103), mirroring corpus_lib.py's discipline
for code but adapted to a real structural difference: WikiText-103 is not
organized as discrete per-document rows like the-stack-smol. It is a single long
stream of LINES that together reconstruct articles:

  - a line like " = Article Title = "        (one '=' each side) -> article title
  - a line like " = = Section heading = = "  (two+ '=' each side) -> section heading
  - a blank line                              -> separator
  - anything else                             -> one paragraph of real prose

There is no ready-made "one row = one document" unit here (unlike the code
corpus), so the unit used for corpus/prose/{file_id}.bin is one PARAGRAPH (one
non-heading, non-blank row) - comparable in size to the code corpus's files,
and small enough to keep benepar's per-file parsing cost bounded (see
STREAM-B-PLAN.md's benepar throughput measurement). Headings are excluded on
purpose: they are not prose sentences and would not parse as one under a
constituency grammar.
"""

import hashlib

from datasets import load_dataset

DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CONFIG = "wikitext-103-raw-v1"

MIN_BYTES = 300
MAX_BYTES = 3000


def heading_level_and_title(line: str):
    """WikiText's heading markup nests by wrapping: level 1 is "= Title =",
    level 2 is "= = Title = =", etc - each level adds a SEPARATE,
    space-delimited '=' token on each side, not an adjacent run of '=' chars.
    That nesting means a level-2 heading contains a valid-looking level-1
    match as a substring ("= = X = =" contains "= X ="), so a single
    non-recursive regex can't safely tell levels apart. Iteratively strip one
    level's markers and count how many times that succeeds instead - correct
    at any nesting depth by construction. Returns (level, title), or
    (None, None) if the line isn't a heading.
    """
    s = line.strip()
    level = 0
    while len(s) >= 4 and s[0] == "=" and s[1] == " " and s[-1] == "=" and s[-2] == " ":
        s = s[2:-2]
        level += 1
    if level == 0:
        return None, None
    return level, s.strip()


def pull_paragraphs(n_needed: int):
    """Streams WikiText-103, reconstructs paragraph units, dedups by content
    hash, filters by length. Returns up to n_needed*PULL_MULTIPLIER candidates
    in stable stream order (never shuffled - same growth-safety reasoning as
    corpus_lib.py: re-running with a bigger n_needed only appends further down
    an otherwise-identical prefix)."""
    PULL_MULTIPLIER = 2  # headings/blanks/too-short/too-long/dupes all get filtered
    n_pull_target = n_needed * PULL_MULTIPLIER

    ds = load_dataset(
        "wikitext", CONFIG, split="train", streaming=True, revision=DATASET_REVISION
    )

    seen_hashes = set()
    kept = []
    current_title = None

    for row in ds:
        line = row["text"]
        if not line.strip():
            continue
        level, title = heading_level_and_title(line)
        if level == 1:
            current_title = title
            continue
        if level is not None:  # a deeper heading (section/subsection) - not content
            continue

        content = line.strip("\n")
        n_bytes = len(content.encode("utf-8"))
        if n_bytes < MIN_BYTES or n_bytes > MAX_BYTES:
            continue

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        kept.append(
            {
                "sha256": h,
                "n_bytes": n_bytes,
                "content": content,
                "article_title": current_title,
            }
        )
        if len(kept) >= n_pull_target:
            break

    return kept
