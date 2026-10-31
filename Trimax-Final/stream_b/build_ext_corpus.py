"""Extension corpus for teacher-student training: raw files disjoint from the frozen
evaluation corpus (corpus/manifest.parquet is only read, never written).

Writes corpus_ext/{domain}/{file_id}.bin and corpus_ext/manifest_ext.parquet, plus
corpus_ext/DATA_NOTE.json (sources, revisions, counts, drops, overlap checks).

Disjointness from the evaluation set, enforced here for every candidate:
  - exact content hash (sha256 of the UTF-8 bytes) not in the evaluation manifest
  - source repository (code) or article (prose) not used by any evaluation file
  - hash-deduplicated against everything already taken from the other sources

Splits are train/val only (there is no test split; the evaluation files are the test).
A file's split is decided by its repository / article, so one repo never spans both.
`rank` is a deterministic pseudo-random key (sha256 prefix), so any prefix of the
manifest sorted by rank is a reproducible random subset.

Sources (revisions pinned):
  cpp   the-stack-smol rows not used by the evaluation corpus + the-stack-smol-xl
  py    CodeSearchNet python train, native order
  prose WikiText-103 train paragraphs, stream order
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
EVAL_MANIFEST = ROOT / "corpus" / "manifest.parquet"
OUT = ROOT / "corpus_ext"

SMOL_REV = "4a6938ce94446f324c6629e7de00ac591710044b"
# the-stack-smol-xl was loaded UNPINNED (HEAD) by the distillation run (Ada job 34241,
# 2026-10-22). This is its HEAD as checked on 2026-10-29 via the HF API; the API reported
# lastModified 2023-02-10T17:22:38Z, i.e. before that run, so it should be the same commit.
SMOL_XL_REV = "e782ebf35c7e4cafccb08ca680b0a76706533067"
CSN_REV = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"

CPP_MAX_BYTES = 100_000  # a few huge files would dominate labelling cost
VAL_EVERY = 20  # about 5% of groups go to val
PROSE_MIN, PROSE_MAX = 300, 3000  # same filter as the evaluation prose corpus


def sha(content: str):
    try:
        b = content.encode("utf-8")
    except UnicodeEncodeError:
        return None, None
    return hashlib.sha256(b).hexdigest(), b


def split_of(group: str) -> str:
    return "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % VAL_EVERY == 0 else "train"


def usable(b: bytes) -> bool:
    return len(b) > 0 and b"\x00" not in b


def eval_info(domain):
    m = pd.read_parquet(EVAL_MANIFEST)
    m = m[m["domain"] == domain]
    return set(m["sha256"]), m


def collect_cpp(limit):
    ev_hashes, _ = eval_info("cpp")
    smol = load_dataset("bigcode/the-stack-smol", data_dir="data/c++", split="train", revision=SMOL_REV)
    s_content, s_repo = smol["content"], smol["repository_name"]
    s_path, s_lic = smol["path"], smol["licenses"]
    smol_hash = [sha(c)[0] for c in s_content]
    ev_idx = [i for i, h in enumerate(smol_hash) if h in ev_hashes]
    if len(ev_idx) != len(ev_hashes):
        raise SystemExit(f"only {len(ev_idx)} of {len(ev_hashes)} evaluation C++ files found in smol")
    ev_repos = {s_repo[i] for i in ev_idx}

    cands = []
    for i in range(len(s_content)):
        if smol_hash[i] in ev_hashes:
            continue
        cands.append(("the-stack-smol", i, s_content[i], s_repo[i], s_path[i], s_lic[i] or []))
    xl = load_dataset("bigcode/the-stack-smol-xl", data_dir="data/c++", split="train", revision=SMOL_XL_REV)
    x_content, x_repo = xl["content"], xl["max_stars_repo_name"]
    x_path, x_lic = xl["max_stars_repo_path"], xl["max_stars_repo_licenses"]
    for i in range(len(x_content)):
        cands.append(("the-stack-smol-xl", i, x_content[i], x_repo[i], x_path[i], x_lic[i] or []))

    drops, kept, seen = Counter(), [], set(ev_hashes)
    for src, i, content, repo, path, lic in cands:
        h, b = sha(content)
        if b is None or not usable(b):
            drops["unusable_bytes"] += 1
            continue
        if h in seen:
            drops["duplicate_hash"] += 1
            continue
        if repo in ev_repos:
            drops["evaluation_repo"] += 1
            continue
        if not lic:
            drops["no_license"] += 1
            continue
        if len(b) > CPP_MAX_BYTES:
            drops["too_large"] += 1
            continue
        seen.add(h)
        kept.append(dict(sha256=h, content=content, n_bytes=len(b), group=repo, path=path,
                         licenses=",".join(lic), source=src, source_row=i))
        if limit and len(kept) >= limit:
            break
    return kept, drops, {"candidates": len(cands), "evaluation_repos": len(ev_repos)}


def collect_py(limit):
    ev_hashes, _ = eval_info("py")
    ds = load_dataset("code-search-net/code_search_net", "python", split="train", revision=CSN_REV)
    ds = ds.select_columns(["whole_func_string", "repository_name", "func_code_url"])
    # CodeSearchNet's native order is clustered by repository, so a prefix would cover very
    # few repositories. Pass 1 hashes every row; the subset is then the lowest sha256 ranks,
    # a deterministic random sample across the whole split.
    hashes, repos, bad = [], [], []
    for batch in ds.iter(batch_size=5000):
        for content, repo in zip(batch["whole_func_string"], batch["repository_name"]):
            h, b = sha(content) if content and content.strip() else (None, None)
            hashes.append(h if b is not None and usable(b) else None)
            repos.append(repo)
    ev_repos = {repos[i] for i, h in enumerate(hashes) if h in ev_hashes}
    if sum(h in ev_hashes for h in hashes) < len(ev_hashes):
        raise SystemExit("not all evaluation python functions were found")

    drops, seen, picked = Counter(), set(ev_hashes), []
    for i, h in enumerate(hashes):
        if h is None:
            drops["empty_or_unusable"] += 1
        elif h in ev_hashes or h in seen:
            drops["duplicate_hash"] += 1
        elif repos[i] in ev_repos:
            drops["evaluation_repo"] += 1
        else:
            seen.add(h)
            picked.append((int(h[:8], 16), i, h))
    picked.sort()
    if limit:
        picked = picked[:limit]
    idx = sorted(i for _, i, _ in picked)
    sub = ds.select(idx)
    kept = []
    for i, row in zip(idx, sub):
        content = row["whole_func_string"]
        h, b = sha(content)
        kept.append(dict(sha256=h, content=content, n_bytes=len(b), group=row["repository_name"],
                         path=row["func_code_url"], licenses="", source="code_search_net", source_row=i))
    return kept, drops, {"evaluation_repos": len(ev_repos), "pool_after_filters": len(picked) if not limit else None}


def collect_prose(limit):
    sys.path.insert(0, str(Path(__file__).parent))
    from prose_lib import CONFIG, heading_level_and_title

    ev_hashes, ev = eval_info("prose")
    ev_titles = set(ev["source_path"])
    ds = load_dataset("Salesforce/wikitext", CONFIG, split="train", streaming=True, revision=WIKI_REV)
    drops, kept, seen = Counter(), [], set(ev_hashes)
    title = None
    for n, row in enumerate(ds):
        line = row["text"]
        if not line.strip():
            continue
        level, t = heading_level_and_title(line)
        if level == 1:
            title = t
            continue
        if level is not None:
            continue
        content = line.strip("\n")
        h, b = sha(content)
        if b is None or not (PROSE_MIN <= len(b) <= PROSE_MAX):
            drops["length_or_bytes"] += 1
            continue
        if title in ev_titles:
            drops["evaluation_article"] += 1
            continue
        if h in seen:
            drops["duplicate_hash"] += 1
            continue
        seen.add(h)
        kept.append(dict(sha256=h, content=content, n_bytes=len(b), group=title or "",
                         path=title, licenses="Wikipedia (CC BY-SA)", source="wikitext-103", source_row=n))
        if limit and len(kept) >= limit:
            break
    return kept, drops, {"evaluation_articles": len(ev_titles)}


COLLECT = {"cpp": collect_cpp, "py": collect_py, "prose": collect_prose}
DEFAULT_LIMIT = {"cpp": 0, "py": 40_000, "prose": 20_000}  # 0 = take everything that passes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("domain", choices=sorted(COLLECT))
    ap.add_argument("--limit", type=int, default=None, help="max files to keep (default per domain)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true", help="collect and report, write nothing")
    args = ap.parse_args(argv)
    limit = DEFAULT_LIMIT[args.domain] if args.limit is None else args.limit
    out = Path(args.out)

    kept, drops, extra = COLLECT[args.domain](limit)
    ev_hashes, _ = eval_info(args.domain)
    assert not any(r["sha256"] in ev_hashes for r in kept), "overlap with evaluation corpus"
    assert len({r["sha256"] for r in kept}) == len(kept), "duplicate hashes in extension pool"

    rows = []
    for r in kept:
        rows.append({"file_id": f"{args.domain}_{r['sha256'][:12]}", "domain": args.domain,
                     "n_bytes": r["n_bytes"], "sha256": r["sha256"], "split": split_of(r["group"]),
                     "group": r["group"], "source_path": r["path"], "licenses": r["licenses"],
                     "source": r["source"], "source_row": r["source_row"],
                     "rank": int(r["sha256"][:8], 16)})
    df = pd.DataFrame(rows)
    summary = {"domain": args.domain, "files": len(df), "bytes": int(df["n_bytes"].sum()) if len(df) else 0,
               "by_split": df["split"].value_counts().to_dict() if len(df) else {},
               "by_source": df["source"].value_counts().to_dict() if len(df) else {},
               "groups": int(df["group"].nunique()) if len(df) else 0,
               "dropped": dict(drops), "limit": limit, **extra}
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    (out / args.domain).mkdir(parents=True, exist_ok=True)
    for r in kept:
        (out / args.domain / f"{args.domain}_{r['sha256'][:12]}.bin").write_bytes(r["content"].encode("utf-8"))
    mpath = out / "manifest_ext.parquet"
    if mpath.exists():
        old = pd.read_parquet(mpath)
        df = pd.concat([old[old["domain"] != args.domain], df], ignore_index=True)
    df.to_parquet(mpath, index=False)
    npath = out / "DATA_NOTE.json"
    note = json.loads(npath.read_text()) if npath.exists() else {}
    note[args.domain] = summary
    note["revisions"] = {"the-stack-smol": SMOL_REV, "the-stack-smol-xl": SMOL_XL_REV,
                         "code_search_net": CSN_REV, "wikitext-103": WIKI_REV}
    npath.write_text(json.dumps(note, indent=2) + "\n")
    print(f"wrote {len(kept)} files to {out / args.domain}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
