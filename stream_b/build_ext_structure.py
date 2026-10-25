"""tree-sitter structure and whitespace baselines for the extension corpus (code only).

Same schema and same walker as the evaluation corpus (build_structure.py,
whitespace_extractor.py), reading corpus_ext/{domain}/{file_id}.bin and writing
corpus_ext/structure/{domain}/{file_id}.parquet and corpus_ext/whitespace/{domain}/...
Adds parse_ok to corpus_ext/manifest_ext.parquet. English structure (benepar) is not
built here.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ast_walker import walk_file  # noqa: E402
from whitespace_extractor import build_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "corpus_ext"
STRUCTURE_COLUMNS = ["node_type", "parent_type", "depth", "start_byte", "end_byte"]


def main():
    mpath = EXT / "manifest_ext.parquet"
    manifest = pd.read_parquet(mpath)
    parse_ok = {}
    for domain in ("py", "cpp"):
        (EXT / "structure" / domain).mkdir(parents=True, exist_ok=True)
        (EXT / "whitespace" / domain).mkdir(parents=True, exist_ok=True)
        ids = manifest.loc[manifest["domain"] == domain, "file_id"]
        for n, file_id in enumerate(ids, 1):
            content = (EXT / domain / f"{file_id}.bin").read_bytes()
            rows, ok = walk_file(domain, content)
            parse_ok[file_id] = ok
            df = pd.DataFrame(rows, columns=STRUCTURE_COLUMNS)
            df.insert(0, "file_id", file_id)
            df.to_parquet(EXT / "structure" / domain / f"{file_id}.parquet", index=False)
            ws = pd.DataFrame(build_rows(content), columns=["byte_offset", "kind"])
            ws.insert(0, "file_id", file_id)
            ws.to_parquet(EXT / "whitespace" / domain / f"{file_id}.parquet", index=False)
            if n % 5000 == 0:
                print(domain, n, "of", len(ids), flush=True)
        bad = sum(1 for f in ids if not parse_ok[f])
        print(f"{domain}: {len(ids)} files, {bad} with tree-sitter error recovery ({bad / len(ids) * 100:.1f}%)", flush=True)
    manifest["parse_ok"] = manifest["file_id"].map(parse_ok)
    manifest.to_parquet(mpath, index=False)
    print("added parse_ok to manifest_ext.parquet")


if __name__ == "__main__":
    main()
