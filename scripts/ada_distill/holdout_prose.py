"""Full-run Step 7 (docs/ADA_CLAUDE_HANDOFF.md): English holdout materialization.

DATA ONLY: nothing here computes or reads a boundary score. The holdout is not scored
until team sign-off (pre-registration: reports/ESCAPE_STUDENT_CV1_EVALUATION.md 8.6).

  select       the 120 lowest-rank prose articles of corpus_ext/manifest_ext.parquet that
               the distillation never used -> corpus_ext/holdout_manifest.parquet
  materialize  copy those paragraphs from a fresh `build_ext_corpus prose --out SRC` build,
               after checking that the build reproduces the tracked prose manifest rows
               exactly; every file is sha256-verified -> corpus_ext/holdout/text/
  structure    benepar constituents with the same per-span byte round-trip validation as
               stream_b/build_prose_structure.py -> corpus_ext/holdout/structure/
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

HOLDOUT_MANIFEST = Path("corpus_ext/holdout_manifest.parquet")
TEXT_DIR = Path("corpus_ext/holdout/text")
STRUCTURE_DIR = Path("corpus_ext/holdout/structure")
N_ARTICLES = 120
KEY = ["file_id", "domain", "n_bytes", "sha256", "split", "group", "source_row", "rank"]


def select():
    full = pd.read_parquet("corpus_ext/manifest_ext.parquet")
    dist = pd.read_parquet("results/distill/full/manifest_ext.parquet")
    prose_full = full[full["domain"] == "prose"]
    used = set(dist[dist["domain"] == "prose"]["group"])
    unused = prose_full[~prose_full["group"].isin(used)].copy()
    unused["article_rank"] = unused.groupby("group")["rank"].transform("min")
    arts = unused.groupby("group").agg(article_rank=("article_rank", "min")).sort_values("article_rank")
    holdout_arts = set(arts.head(N_ARTICLES).index)
    holdout = unused[unused["group"].isin(holdout_arts)]
    holdout[["file_id", "domain", "n_bytes", "sha256", "group"]].to_parquet(HOLDOUT_MANIFEST)
    print("holdout:", len(holdout), "files,", len(holdout_arts), "articles,",
          int(holdout["n_bytes"].sum()), "bytes")


def materialize(src: Path):
    tracked = pd.read_parquet("corpus_ext/manifest_ext.parquet")
    tracked = tracked[tracked["domain"] == "prose"][KEY].sort_values("file_id").reset_index(drop=True)
    built = pd.read_parquet(src / "manifest_ext.parquet")
    built = built[built["domain"] == "prose"][KEY].sort_values("file_id").reset_index(drop=True)
    if not tracked.astype(str).equals(built.astype(str)):  # values only; int width may differ
        sys.exit(f"rebuilt prose manifest differs from the tracked one ({len(built)} vs {len(tracked)} rows); "
                 "the holdout bytes would not be the pre-registered ones")
    print(f"rebuilt prose manifest matches the tracked one ({len(built)} rows)")
    m = pd.read_parquet(HOLDOUT_MANIFEST)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for _, r in m.iterrows():
        content = (src / r.domain / f"{r.file_id}.bin").read_bytes()
        if len(content) != r.n_bytes or hashlib.sha256(content).hexdigest() != r.sha256:
            sys.exit(f"{r.file_id}: bytes do not match the manifest")
        (TEXT_DIR / f"{r.file_id}.bin").write_bytes(content)
    print(f"materialized {len(m)} holdout files to {TEXT_DIR} (all sha256-verified)")


def structure():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stream_b"))
    import spacy
    from build_prose_structure import STRUCTURE_COLUMNS, char_to_byte_offsets, walk_sentence

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    m = pd.read_parquet(HOLDOUT_MANIFEST)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    done = skipped = n_sentences = 0
    for i, fid in enumerate(m["file_id"]):
        out = STRUCTURE_DIR / f"{fid}.parquet"
        if out.exists():
            skipped += 1
            continue
        content = (TEXT_DIR / f"{fid}.bin").read_bytes()
        text = content.decode("utf-8")
        char_to_byte = char_to_byte_offsets(text)
        rows = []
        for sent in nlp(text).sents:
            n_sentences += 1
            rows.extend(walk_sentence(sent, char_to_byte, content))
        df = pd.DataFrame(rows, columns=STRUCTURE_COLUMNS)
        df.insert(0, "file_id", fid)
        df.to_parquet(out, index=False)
        done += 1
        if done % 250 == 0:
            print(f"[structure] {i + 1}/{len(m)}", flush=True)
    missing = [f for f in m["file_id"] if not (STRUCTURE_DIR / f"{f}.parquet").exists()]
    print(f"structure: {done} parsed ({n_sentences} sentences), {skipped} already present, "
          f"{len(missing)} missing")
    if missing:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    p = sub.add_parser("materialize")
    p.add_argument("--src", type=Path, required=True, help="--out directory of build_ext_corpus prose")
    sub.add_parser("structure")
    args = ap.parse_args()
    if args.cmd == "select":
        select()
    elif args.cmd == "materialize":
        materialize(args.src)
    else:
        structure()


if __name__ == "__main__":
    main()
