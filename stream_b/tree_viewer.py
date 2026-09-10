"""Dev tool: render an interactive HTML view of one corpus file's full tree-sitter
parse tree, with the nodes ast_walker.py actually extracts highlighted, next to the
source. Local, self-contained (no server, no network) — just open the output file.

Usage:
    python stream_b/tree_viewer.py <file_id>
    python stream_b/tree_viewer.py py_a1567fe2419b
"""

import json
import sys

from tree_sitter import Language, Parser
import tree_sitter_cpp
import tree_sitter_python

from ast_walker import NODE_TYPES

CORPUS_DIR = "corpus"
OUT_DIR = "stream_b/tree_views"

_LANGUAGES = {
    "py": Language(tree_sitter_python.language()),
    "cpp": Language(tree_sitter_cpp.language()),
}


def domain_from_file_id(file_id: str) -> str:
    return file_id.split("_", 1)[0]


def byte_to_char_offsets(content: bytes):
    """Precompute byte-offset -> char-offset for every byte position, so the browser
    (which indexes JS strings, not raw bytes) highlights the exact right span even
    with multi-byte UTF-8 characters in the source."""
    text = content.decode("utf-8", errors="replace")
    offsets = [0] * (len(content) + 1)
    char_i = 0
    byte_i = 0
    for ch in text:
        ch_bytes = len(ch.encode("utf-8", errors="replace"))
        for _ in range(ch_bytes):
            if byte_i < len(offsets):
                offsets[byte_i] = char_i
            byte_i += 1
        char_i += 1
    offsets[len(content)] = char_i
    return text, offsets


def collect_nodes(domain: str, content: bytes):
    lang = _LANGUAGES[domain]
    tree = Parser(lang).parse(content)
    wanted = NODE_TYPES[domain]

    nodes = []
    next_id = [0]

    def recurse(node, parent_id, depth):
        my_id = next_id[0]
        next_id[0] += 1
        nodes.append(
            {
                "id": my_id,
                "parent_id": parent_id,
                "type": node.type,
                "depth": depth,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "extracted": node.type in wanted,
                "is_error": node.type == "ERROR" or node.is_missing,
            }
        )
        for child in node.children:
            recurse(child, my_id, depth + 1)

    recurse(tree.root_node, None, 0)
    return nodes, not tree.root_node.has_error


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AST tree — {file_id}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1a1a1a; --muted: #888; --border: #ddd;
    --extracted: #1a6b3c; --extracted-bg: #e6f4ea;
    --error: #b3261e; --error-bg: #fbeaea;
    --highlight: #fff3a3; --panel-bg: #f7f7f8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a1a; --fg: #e8e8e8; --muted: #999; --border: #3a3a3a;
      --extracted: #6fd99a; --extracted-bg: #123322;
      --error: #ff8a80; --error-bg: #3a1414;
      --highlight: #5a4a00; --panel-bg: #232323;
    }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,Segoe UI,sans-serif; }}
  .layout {{ display:flex; height:100vh; }}
  .panel {{ overflow:auto; padding:12px; box-sizing:border-box; }}
  .src {{ width:45%; border-right:1px solid var(--border); background:var(--panel-bg); }}
  .tree {{ width:55%; }}
  pre.src-text {{ white-space:pre-wrap; word-break:break-all; font:12px/1.5 ui-monospace,Menlo,monospace; margin:0; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 8px; }}
  .node {{ font:12px ui-monospace,Menlo,monospace; cursor:pointer; padding:1px 4px; border-radius:3px; white-space:nowrap; }}
  .node:hover {{ background:var(--panel-bg); }}
  .node.extracted {{ color:var(--extracted); font-weight:600; background:var(--extracted-bg); }}
  .node.err {{ color:var(--error); background:var(--error-bg); }}
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
    <h2>full parse tree (parse_ok: {parse_ok})</h2>
    <div class="legend">
      <span class="node extracted">extracted</span> = kept in structure/{domain}/{file_id}.parquet &nbsp;·&nbsp;
      <span class="node err">error</span> = tree-sitter error-recovery node &nbsp;·&nbsp;
      click any node to highlight its span in the source
    </div>
    <div id="tree"></div>
  </div>
</div>
<script>
const SOURCE = {source_json};
const NODES = {nodes_json};
const CHAR_OFFSETS = {char_offsets_json};

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

function labelFor(n) {{
  const cls = ['node'];
  if (n.extracted) cls.push('extracted');
  if (n.is_error) cls.push('err');
  return {{cls: cls.join(' ')}};
}}

function renderNode(id, depthLimit) {{
  const n = byId[id];
  const kids = children[id] || [];
  const wrap = document.createElement('div');
  const row = document.createElement('div');
  const {{cls}} = labelFor(n);
  row.className = cls;

  const toggle = document.createElement('span');
  toggle.className = 'toggle';
  toggle.textContent = kids.length ? '▸' : ' ';
  row.appendChild(toggle);

  const label = document.createElement('span');
  label.innerHTML = `<span class="type">${{n.type}}</span> <span class="range">[${{n.start_byte}}:${{n.end_byte}}]</span>`;
  row.appendChild(label);

  row.onclick = (e) => {{
    e.stopPropagation();
    highlightSource(n.start_byte, n.end_byte);
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
  treeEl.appendChild(renderNode(rid, 3)); // auto-expand first 3 levels
}}

let lastMark = null;
function highlightSource(startByte, endByte) {{
  const startChar = CHAR_OFFSETS[startByte];
  const endChar = CHAR_OFFSETS[endByte];
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
        print("usage: python stream_b/tree_viewer.py <file_id>")
        sys.exit(1)

    file_id = sys.argv[1]
    domain = domain_from_file_id(file_id)
    if domain not in NODE_TYPES:
        print(f"unrecognized domain prefix for file_id {file_id!r}")
        sys.exit(1)

    with open(f"{CORPUS_DIR}/{domain}/{file_id}.bin", "rb") as f:
        content = f.read()

    nodes, parsed_ok = collect_nodes(domain, content)
    text, char_offsets = byte_to_char_offsets(content)

    html = HTML_TEMPLATE.format(
        file_id=file_id,
        domain=domain,
        parse_ok=parsed_ok,
        source_json=json.dumps(text),
        nodes_json=json.dumps(nodes),
        char_offsets_json=json.dumps(char_offsets),
    )

    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{file_id}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(nodes)} nodes, parse_ok={parsed_ok})")
    return out_path


if __name__ == "__main__":
    main()
