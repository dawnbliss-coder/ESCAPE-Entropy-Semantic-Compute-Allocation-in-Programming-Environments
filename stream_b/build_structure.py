"""Day 3, part 1 driver: run ast_walker over every file in corpus/manifest.parquet,
write structure/{domain}/{file_id}.parquet (schema: file_id, node_type, parent_type,
depth, start_byte, end_byte). Reads corpus/{domain}/{file_id}.bin directly — never
re-decodes or re-encodes B's frozen bytes.
"""

import os

import pandas as pd

from ast_walker import walk_file

CORPUS_DIR = "corpus"
STRUCTURE_DIR = "structure"
# Explicit columns so a file with zero tracked nodes (e.g. imports-only) still
# gets the right schema - pd.DataFrame([]) with no columns= would otherwise
# produce a parquet file missing every column but file_id.
STRUCTURE_COLUMNS = ["node_type", "parent_type", "depth", "start_byte", "end_byte"]


def main():
    manifest = pd.read_parquet(f"{CORPUS_DIR}/manifest.parquet")
    # Only py/cpp - ast_walker.py is tree-sitter-based and has no "prose" grammar.
    # Prose structure is built separately by build_prose_structure.py (benepar).
    code_manifest = manifest[manifest["domain"].isin(["py", "cpp"])]

    parse_ok_by_file_id = {}
    node_type_counts = {}

    for domain, group in code_manifest.groupby("domain"):
        os.makedirs(f"{STRUCTURE_DIR}/{domain}", exist_ok=True)
        for file_id in group["file_id"]:
            bin_path = f"{CORPUS_DIR}/{domain}/{file_id}.bin"
            with open(bin_path, "rb") as f:
                content = f.read()

            rows, parsed_ok = walk_file(domain, content)
            parse_ok_by_file_id[file_id] = parsed_ok

            for r in rows:
                node_type_counts[(domain, r["node_type"])] = (
                    node_type_counts.get((domain, r["node_type"]), 0) + 1
                )

            df = pd.DataFrame(rows, columns=STRUCTURE_COLUMNS)
            df.insert(0, "file_id", file_id)
            df.to_parquet(f"{STRUCTURE_DIR}/{domain}/{file_id}.parquet", index=False)

        n_errors = sum(
            1 for fid in group["file_id"] if not parse_ok_by_file_id[fid]
        )
        print(
            f"{domain}: wrote structure for {len(group)} files "
            f"({n_errors} with tree-sitter error-recovery triggered, "
            f"{n_errors / len(group) * 100:.1f}%)"
        )

    # Record parse_ok on the manifest itself — not a separate report — so anyone
    # reading corpus/manifest.parquet sees it without joining another file. See
    # STREAM-B-PLAN.md for why files with recovery triggered are kept, not dropped:
    # tree-sitter's error recovery only affects the region right around the issue
    # (typically an unresolved macro/preprocessor construct), the rest of the file's
    # nodes are still trustworthy, and ~28% of the entire C++ pool has this property,
    # so excluding them would both fail to reach N=8000 clean and bias the sample
    # toward simpler, less representative C++ files.
    manifest["parse_ok"] = manifest["file_id"].map(parse_ok_by_file_id)
    manifest.to_parquet(f"{CORPUS_DIR}/manifest.parquet", index=False)
    print("\nadded parse_ok column to corpus/manifest.parquet")

    print("\nnode_type counts:")
    for (domain, nt), count in sorted(node_type_counts.items()):
        print(f"  {domain:>4}  {nt:<22} {count}")


if __name__ == "__main__":
    main()
