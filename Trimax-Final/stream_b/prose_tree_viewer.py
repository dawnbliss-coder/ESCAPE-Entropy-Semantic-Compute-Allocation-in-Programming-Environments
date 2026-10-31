"""Dev tool: render an interactive HTML view of one prose file's full benepar
constituency tree next to its source, mirroring tree_viewer.py's design for
code. Local, self-contained (no server, no network) - just open the output file.

Usage:
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    python stream_b/prose_tree_viewer.py <file_id>
"""

import json
import os
import sys

import benepar  # noqa: F401 - registers the "benepar" spaCy pipeline factory
import spacy

CORPUS_DIR = "corpus"
OUT_DIR = "stream_b/tree_views"


def char_to_byte_offsets(text: str):
    offsets = [0] * (len(text) + 1)
    byte_pos = 0
    for i, ch in enumerate(text):
        offsets[i] = byte_pos
        byte_pos += len(ch.encode("utf-8"))
    offsets[len(text)] = byte_pos
    return offsets


def collect_nodes(doc, char_to_byte, content: bytes):
    """Flat list of {id, parent_id, type, depth, start_byte, end_byte,
    extracted} across every sentence in the paragraph.

    `span._.children` can yield UNLABELED single-token children even under a
    labeled parent (e.g. NP -> "New", "gods", each with empty labels) - this
    must be checked at every call, not just at the entry point, or those bare
    tokens get miscounted as real constituents (extracted=True). This mirrors
    build_prose_structure.py's validated walk_sentence exactly: extracted=True
    only when `labels` is non-empty; recursion into `span._.children` happens
    unconditionally either way, same as there. The one addition here (this is
    a viewer, not the real extractor) is showing an unlabeled leaf's own POS
    tag as its node_type, so the tree looks complete rather than silently
    stopping.
    """
    nodes = []
    next_id = [0]

    def recurse(span, parent_id, depth):
        my_id = next_id[0]
        next_id[0] += 1
        labels = span._.labels
        start_char, end_char = span.start_char, span.end_char
        nodes.append(
            {
                "id": my_id,
                "parent_id": parent_id,
                "type": "+".join(labels) if labels else (span[0].tag_ or span[0].pos_ or "TOKEN"),
                "depth": depth,
                "start_byte": char_to_byte[start_char],
                "end_byte": char_to_byte[end_char],
                "start_char": start_char,
                "end_char": end_char,
                "extracted": bool(labels),
                "is_error": False,
            }
        )
        for child in span._.children:
            recurse(child, my_id, depth + 1)
        return my_id

    for sent in doc.sents:
        recurse(sent, None, 1)

    return nodes


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Prose parse tree — {file_id}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1a1a1a; --muted: #888; --border: #ddd;
    --extracted: #1a6b3c; --extracted-bg: #e6f4ea;
    --highlight: #fff3a3; --panel-bg: #f7f7f8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a1a; --fg: #e8e8e8; --muted: #999; --border: #3a3a3a;
      --extracted: #6fd99a; --extracted-bg: #123322;
      --highlight: #5a4a00; --panel-bg: #232323;
    }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,Segoe UI,sans-serif; }}
  .layout {{ display:flex; height:100vh; }}
  .panel {{ overflow:auto; padding:12px; box-sizing:border-box; }}
  .src {{ width:45%; border-right:1px solid var(--border); background:var(--panel-bg); }}
  .tree {{ width:55%; }}
  pre.src-text {{ white-space:pre-wrap; word-break:break-word; font:13px/1.6 -apple-system,sans-serif; margin:0; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 8px; }}
  .node {{ font:12px ui-monospace,Menlo,monospace; cursor:pointer; padding:1px 4px; border-radius:3px; white-space:nowrap; }}
  .node:hover {{ background:var(--panel-bg); }}
  .node.extracted {{ color:var(--extracted); font-weight:600; background:var(--extracted-bg); }}
  .node .type {{ }}
  .node .range {{ color:var(--muted); font-weight:400; }}
  .toggle {{ display:inline-block; width:14px; color:var(--muted); user-select:none; }}
  .children {{ margin-left:16px; }}
  .hl {{ background:var(--highlight); }}
  .legend {{ font:11px sans-serif; color:var(--muted); margin-bottom:10px; }}
  .legend span {{ padding:1px 6px; border-radius:3px; margin-right:8px; }}
</style>
</head>
<body>
<div class="layout">
  <div class="panel src">
    <h2>source — {file_id}</h2>
    <pre class="src-text" id="src"></pre>
  </div>
  <div class="panel tree">
    <h2>benepar constituency tree</h2>
    <div class="legend">
      <span class="node extracted">extracted</span> = real constituent, kept in
      structure/prose/{file_id}.parquet (plain rows are individual tokens/POS
      tags, shown for completeness, not extracted) &nbsp;·&nbsp;
      click any node to highlight its span in the source
    </div>
    <div id="tree"></div>
  </div>
</div>
<script>
const SOURCE = {source_json};
const NODES = {nodes_json};

const srcEl = document.getElementById('src');
srcEl.textContent = SOURCE;

const byId = {{}};
const children = {{}};
for (const n of NODES) {{
  byId[n.id] = n;
  children[n.id] = children[n.id] || [];
  if (n.parent_id !== null) {{
    children[n.parent_id] = children[n.parent_id] || [];
    children[n.parent_id].push(n.id);
  }}
}}
const roots = NODES.filter(n => n.parent_id === null).map(n => n.id);

function renderNode(id, depthLimit) {{
  const n = byId[id];
  const kids = children[id] || [];
  const wrap = document.createElement('div');
  const row = document.createElement('div');
  row.className = 'node' + (n.extracted ? ' extracted' : '');

  const toggle = document.createElement('span');
  toggle.className = 'toggle';
  row.appendChild(toggle);

  const label = document.createElement('span');
  label.innerHTML = `<span class="type">${{n.type}}</span> <span class="range">[${{n.start_byte}}:${{n.end_byte}}]</span>`;
  row.appendChild(label);

  row.onclick = (e) => {{
    e.stopPropagation();
    highlightSource(n.start_char, n.end_char);
    if (kids.length) {{
      childContainer.style.display = childContainer.style.display === 'none' ? 'block' : 'none';
      toggle.textContent = childContainer.style.display === 'none' ? '▸' : '▾';
    }}
  }};

  wrap.appendChild(row);

  const childContainer = document.createElement('div');
  childContainer.className = 'children';
  childContainer.style.display = (n.depth >= depthLimit) ? 'none' : 'block';
  toggle.textContent = kids.length ? (childContainer.style.display === 'none' ? '▸' : '▾') : ' ';
  for (const cid of kids) {{
    childContainer.appendChild(renderNode(cid, depthLimit));
  }}
  wrap.appendChild(childContainer);
  return wrap;
}}

const treeEl = document.getElementById('tree');
for (const rid of roots) {{
  treeEl.appendChild(renderNode(rid, 4));
}}

function highlightSource(startChar, endChar) {{
  // Each node carries BOTH byte offsets (shown in the label, matching
  // structure/prose/'s actual stored convention) and char offsets (used here
  // for slicing, since a JS string is indexed by UTF-16 code unit, not bytes -
  // using byte offsets directly would misalign on any multi-byte character).
  const text = SOURCE;
  srcEl.innerHTML =
    escapeHtml(text.slice(0, startChar)) +
    '<span class="hl" id="hlspan">' + escapeHtml(text.slice(startChar, endChar)) + '</span>' +
    escapeHtml(text.slice(endChar));
  const span = document.getElementById('hlspan');
  if (span) span.scrollIntoView({{block: 'center'}});
}}
function escapeHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) != 2:
        print("usage: python stream_b/prose_tree_viewer.py <file_id>")
        sys.exit(1)
    file_id = sys.argv[1]

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    with open(f"{CORPUS_DIR}/prose/{file_id}.bin", "rb") as f:
        content = f.read()
    text = content.decode("utf-8")
    char_to_byte = char_to_byte_offsets(text)

    doc = nlp(text)
    nodes = collect_nodes(doc, char_to_byte, content)

    html = HTML_TEMPLATE.format(
        file_id=file_id,
        source_json=json.dumps(text),
        nodes_json=json.dumps(nodes),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{file_id}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(nodes)} nodes)")
    return out_path


if __name__ == "__main__":
    main()
