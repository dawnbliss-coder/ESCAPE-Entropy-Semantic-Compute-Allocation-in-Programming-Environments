"""Final-submission analyses that score boundary RATES at chosen byte positions against
the frozen fixed-EOF null: the P4 post-keyword follow-up and the O3 depth normalisation.

Both reuse the frozen C-v1 machinery unchanged: test files from the frozen calibration
record, primary targets from `escape_scoring.metrics.target_nodes` semantics (unique
starts, outermost attribution), predictions = interior entropy boundaries, k = 0, and the
null draws of `FixedEofNull` from the SAME seeded generator as the frozen scorer
(rng_for(seed, sample_id, "test_permutation")). The `start` rows therefore reproduce the
frozen node_type recall exactly, which the summary checks against scores.csv.

At k = 0 a target set's recall is the fraction of its byte positions that are entropy
boundaries, so every row below is "boundary rate at these positions" vs the null's rate.

P4 (code only). For each primary target start s of a taxonomy type, its first token is
the run of identifier bytes [A-Za-z0-9_] starting at s (the keyword for if/for/while/def;
an identifier or type for open-ended constructs; one byte if s is punctuation):
  start          s itself (the frozen P4 measure)
  tok_interior   s+1 .. s+L-1, the bytes already forced once the token has begun
  post_tok       s+L, the byte right after the token (space, '(' or ':')
  next_choice    first non-whitespace byte at or after s+L, where the next choice is made
  offset_j       s+j for j = 1..6 inside the node, a distance-from-start curve
Families: deterministic_opener vs open_ended from taxonomy.json (the frozen P4 classes).

O3 (all languages). Scaffold-free constituent depth = number of stored taxonomy nodes that
strictly enclose the target's node (raw tree-sitter depth also counts grammar scaffolding
such as block / compound_statement / argument_list, which differs by language), reported
as cdepth_c and as within-file relative depth decile reldepth_d = floor(10 c / (max c + 1)).

Usage (Ada, c_py):
  python scripts/final_analysis/position_scores.py --run-id swa512-float32-t1.3354-mplnone-91aa6b8e \
      --out results/final/analysis/positions_teacher.json --workers 24
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from escape_scoring.contract import WRAPPERS  # noqa: E402
from escape_scoring.fast import FixedEofNull  # noqa: E402
from escape_scoring.metrics import rng_for  # noqa: E402

DOMAIN = {"python": "py", "cpp": "cpp", "english": "prose"}
IDENT = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
SPACE = frozenset(b" \t\r\n\f\v")
MAX_OFFSET = 6
TAXONOMY = json.loads((REPO_ROOT / "taxonomy.json").read_text())


def load_file(root, run_id, sid, language):
    dom = DOMAIN[language]
    raw = (root / f"corpus/{dom}/{sid}.bin").read_bytes()
    nodes = pd.read_parquet(root / f"structure/{dom}/{sid}.parquet").reset_index(drop=True)
    b = pd.read_parquet(root / f"boundaries/{run_id}/{sid}.parquet").sort_values("byte_offset")
    starts = b["byte_offset"].astype(int).tolist()
    trig = b["trigger"].astype(str).tolist()
    n = len(raw)
    # Same patch / reason construction as escape_scoring.adapter.convert.
    patches = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n
        reason = {"entropy": "entropy", "length_cap": "max_length"}[trig[i + 1]] if i + 1 < len(starts) else "eof"
        patches.append({"start_byte": s, "end_byte": end, "termination_reason": reason})
    pred = np.array(sorted(s for s, t in zip(starts, trig) if t == "entropy" and 0 < s < n), dtype=np.int64)
    return raw, nodes, patches, pred


def primary_targets(nodes, n):
    """escape_scoring.metrics.target_nodes(kind='start') on the legacy-converted nodes."""
    rows = []
    for i, r in enumerate(nodes.itertuples(index=False)):
        s, e, d, t = int(r.start_byte), int(r.end_byte), int(r.depth), str(r.node_type)
        if t.lower() in WRAPPERS or d <= 0 or e <= s or not 0 < s < n:
            continue
        rows.append((d, -(e - s), t, f"legacy-{i}", s, e, i))
    rows.sort(key=lambda x: x[:4])
    unique = {}
    for d, negw, t, nid, s, e, i in rows:
        unique.setdefault(s, (t, s, e, d, i))
    return unique


def constituent_depth(nodes):
    """Number of stored nodes strictly enclosing each node (span containment, shallower raw depth)."""
    s = nodes["start_byte"].to_numpy(np.int64)
    e = nodes["end_byte"].to_numpy(np.int64)
    d = nodes["depth"].to_numpy(np.int64)
    keep = ~nodes["node_type"].str.lower().isin(WRAPPERS).to_numpy() & (d > 0) & (e > s)
    out = np.zeros(len(nodes), dtype=np.int64)
    idx = np.flatnonzero(keep)
    for i in idx:
        enc = keep & (s <= s[i]) & (e >= e[i]) & (d < d[i])
        out[i] = int(enc.sum())
    return out


def first_token_len(raw, s, e):
    j = s
    while j < e and raw[j] in IDENT:
        j += 1
    return max(1, j - s)


def target_sets(raw, nodes, language):
    n = len(raw)
    targets = primary_targets(nodes, n)
    sets = {}

    def add(key, pos):
        if 0 < pos < n:
            sets.setdefault(key, set()).add(pos)

    if language in ("python", "cpp"):
        for s, (t, _, e, _, _) in targets.items():
            fam = TAXONOMY.get(t)
            if fam is None:
                continue
            L = first_token_len(raw, s, e)
            labels = [("start", [s]), ("tok_interior", range(s + 1, s + L)), ("post_tok", [s + L])]
            j = s + L
            while j < n and raw[j] in SPACE:
                j += 1
            labels.append(("next_choice", [j]))
            labels += [(f"offset_{o}", [s + o]) for o in range(1, MAX_OFFSET + 1) if s + o < e]
            for label, positions in labels:
                for p in positions:
                    add(("p4", label, t), p)
                    add(("p4", label, fam), p)
    cd = constituent_depth(nodes)
    max_c = max((int(cd[i]) for (_, _, _, _, i) in targets.values()), default=0)
    for s, (_, _, _, _, i) in targets.items():
        c = int(cd[i])
        add(("o3", "cdepth", str(min(c, 8))), s)
        add(("o3", "reldepth", str(min(9, (10 * c) // (max_c + 1)))), s)
        add(("o3", "all", "all"), s)
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in sets.items()}


def score_file(task):
    root, run_id, sid, language, n_bytes, perms, seed = task
    raw, nodes, patches, pred = load_file(Path(root), run_id, sid, language)
    if len(raw) != n_bytes:
        raise ValueError(f"{sid}: byte length {len(raw)} != manifest {n_bytes}")
    sets = target_sets(raw, nodes, language)
    keys = sorted(sets)
    if not keys:
        return sid, language, {}
    # Position x set membership; a draw's per-set hit counts are then one row-matrix product.
    pos_all = np.unique(np.concatenate([sets[k] for k in keys]))
    member = np.zeros((len(pos_all), len(keys)), dtype=np.float32)
    for j, k in enumerate(keys):
        member[np.searchsorted(pos_all, sets[k]), j] = 1
    pos_index = np.full(len(raw) + 1, -1, dtype=np.int64)
    pos_index[pos_all] = np.arange(len(pos_all))
    hit = np.zeros((1, len(pos_all)), dtype=np.float32)
    hi = pos_index[pred]
    hit[0, hi[hi >= 0]] = 1
    obs = (hit @ member)[0]
    null = np.zeros((perms, len(keys)), dtype=np.int64)
    sampler = FixedEofNull(patches, len(raw))
    rng = rng_for(seed, sid, "test_permutation")
    block = 1000
    for a in range(0, perms, block):
        b = min(perms, a + block)
        H = np.zeros((b - a, len(pos_all)), dtype=np.float32)
        for r in range(b - a):
            idx = pos_index[sampler.draw(rng)]
            H[r, idx[idx >= 0]] = 1
        null[a:b] = np.rint(H @ member).astype(np.int64)
    return sid, language, {k: (len(sets[k]), int(round(obs[j])), null[:, j]) for j, k in enumerate(keys)}


def score_chunk(tasks):
    pooled, per_file = {}, []
    for task in tasks:
        sid, language, res = score_file(task)
        rows = {}
        for k, (n, o, nl) in res.items():
            key = (language,) + k
            if key in pooled:
                pooled[key] += nl.astype(np.int64)
            else:
                pooled[key] = nl.astype(np.int64)
            rows[key] = (n, o, float(nl.mean()))
        per_file.append((sid, rows))
    return pooled, per_file


def summarize(pooled, per_file, boot, seed):
    files_by_key = {}
    for sid, rows in per_file:
        for key, v in rows.items():
            files_by_key.setdefault(key, []).append(v)
    out = []
    for key in sorted(files_by_key):
        arr = np.asarray(files_by_key[key], dtype=float)  # [files, (n, obs, null_mean)]
        n, o = arr[:, 0].sum(), arr[:, 1].sum()
        null = pooled[key].astype(float) / n
        rate = o / n
        rng = rng_for(seed, "|".join(key), "position_bootstrap")
        idx = rng.integers(0, len(arr), (boot, len(arr)))
        bn = arr[idx, 0].sum(1)
        bm = (arr[idx, 1].sum(1) - arr[idx, 2].sum(1)) / bn
        brate = arr[idx, 1].sum(1) / bn
        out.append({"language": key[0], "analysis": key[1], "label": key[2], "group": key[3],
                    "n_files": len(arr), "n_positions": int(n), "boundaries": int(o),
                    "rate": rate, "rate_h0_mean": float(null.mean()),
                    "rate_h0_p025": float(np.quantile(null, .025)), "rate_h0_p975": float(np.quantile(null, .975)),
                    "margin": rate - float(null.mean()),
                    "ratio": rate / float(null.mean()) if null.mean() > 0 else None,
                    "p_above": float((1 + np.sum(null >= rate)) / (len(null) + 1)),
                    "p_below": float((1 + np.sum(null <= rate)) / (len(null) + 1)),
                    "rate_ci_low": float(np.quantile(brate, .025)), "rate_ci_high": float(np.quantile(brate, .975)),
                    "margin_ci_low": float(np.quantile(bm, .025)), "margin_ci_high": float(np.quantile(bm, .975))})
    return out


def check_against_frozen(rows, scores_csv):
    """The `start` rows per type must equal the frozen node_type recall (same targets, same draws)."""
    frozen = pd.read_csv(scores_csv)
    frozen = frozen[(frozen.kind == "start") & (frozen.axis == "node_type")].set_index(["language", "stratum"])
    report = []
    for r in rows:
        if r["analysis"] != "p4" or r["label"] != "start" or r["group"] in ("deterministic_opener", "open_ended"):
            continue
        f = frozen.loc[(r["language"], r["group"])]
        report.append({"language": r["language"], "type": r["group"],
                       "recall": r["rate"], "frozen_recall": float(f.recall),
                       "h0": r["rate_h0_mean"], "frozen_h0": float(f.recall_h0_mean),
                       "match": bool(abs(r["rate"] - f.recall) < 1e-12 and abs(r["rate_h0_mean"] - f.recall_h0_mean) < 1e-9)})
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--calibration", default="results/final/c-v1-2721954/calibration.json")
    ap.add_argument("--frozen-scores", default="results/final/c-v1-2721954/evaluation/scores.csv",
                    help="checked against the start rows; empty to skip")
    ap.add_argument("--languages", nargs="+", default=["python", "cpp", "english"])
    ap.add_argument("--file-ids", nargs="*", help="restrict to these test files (smoke runs)")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    rec = json.loads((root / args.calibration).read_text())
    seed = rec["seed"]
    tests = [m for m in rec["split_manifest"] if m["split"] == "test" and m["language"] in args.languages]
    if args.file_ids:
        keep = set(args.file_ids)
        tests = [m for m in tests if m["sample_id"] in keep]
    tasks = [(str(root), args.run_id, m["sample_id"], m["language"], m["byte_length"], args.perms, seed)
             for m in sorted(tests, key=lambda m: m["sample_id"])]
    n_chunks = max(1, min(len(tasks), 16 * args.workers))
    bounds = np.linspace(0, len(tasks), n_chunks + 1).astype(int)
    chunks = [tasks[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    pooled, per_file = {}, []
    if args.workers > 1:
        with get_context("fork").Pool(args.workers) as pool:
            results = pool.imap(score_chunk, chunks)
            for i, (p, f) in enumerate(results, 1):
                for k, v in p.items():
                    pooled[k] = pooled[k] + v if k in pooled else v
                per_file.extend(f)
                if i % 50 == 0:
                    print(f"[positions] {i}/{len(chunks)} chunks", flush=True)
    else:
        for p, f in map(score_chunk, chunks):
            for k, v in p.items():
                pooled[k] = pooled[k] + v if k in pooled else v
            per_file.extend(f)
    rows = summarize(pooled, per_file, args.boot, seed)
    result = {"run_id": args.run_id, "seed": seed, "perms": args.perms, "boot": args.boot, "k": 0,
              "n_files": len(tasks), "calibration_record": args.calibration,
              "null": "FixedEofNull draws from rng_for(seed, sample_id, 'test_permutation') (frozen scorer's draws)",
              "rows": rows}
    if args.frozen_scores and not args.file_ids:
        result["frozen_check"] = check_against_frozen(rows, root / args.frozen_scores)
        result["frozen_check_all_match"] = all(r["match"] for r in result["frozen_check"])
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[wrote] {out}")
    for r in rows:
        if r["analysis"] == "p4" and r["group"] in ("deterministic_opener", "open_ended") and not r["label"].startswith("offset"):
            print(f"  {r['language']:<7} {r['label']:<13} {r['group']:<21} n={r['n_positions']:>8} "
                  f"rate={r['rate']:.4f} h0={r['rate_h0_mean']:.4f} margin={r['margin']:+.4f} "
                  f"[{r['margin_ci_low']:+.4f},{r['margin_ci_high']:+.4f}]")
    if "frozen_check_all_match" in result:
        print("frozen start-recall check:", "ALL MATCH" if result["frozen_check_all_match"] else "MISMATCH")


if __name__ == "__main__":
    main()
