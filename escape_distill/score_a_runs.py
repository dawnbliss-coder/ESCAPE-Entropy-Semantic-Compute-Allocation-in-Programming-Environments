"""Score any Stream-A run_id (or several) against structure targets at the FROZEN k=0.

For a given file list (the held-out test split from the frozen calibration record, a
pre-registered subset, the calibration split, or the golden fixtures), reads:
  - boundaries from boundaries/{run_id}/{file_id}.parquet (A format, start-of-patch)
  - targets from structure/{domain}/{file_id}.parquet (tree-sitter / benepar; or
    golden_fixtures/{domain}/... with --golden)
and computes, per domain and pooled:
  - one-to-one matching at k=0 (escape_scoring.metrics.match)
  - micro-pooled P / R / F1
  - fixed-EOF density-matched null (R permutations, sequential PCG64 draws per file)
    with pooled permutation p-value
  - newline / indentation-change / word-start baselines
  - paired file percentile bootstrap CIs (B) and BLT-minus-best-baseline intervals

No k selection happens here: test files must never select k. Use the frozen k=0
(results/final/c-v1-2721954/calibration.json). Primary student numbers go through the
frozen `escape_scoring` CLI; this harness serves the teacher-512 subset control, the
tau-sweep curves and the golden gates.

Usage:
  python -m escape_distill.score_a_runs --run-ids student-v1-14326k teacher-512-ctl \
      --file-ids <ids> [--seed 20260927 --perms 10000 --boot 2000] [--out dir]
  python -m escape_distill.score_a_runs --run-ids student-golden-gate --golden
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from escape_scoring.baselines import indent_change, newline, word_start  # noqa: E402
from escape_scoring.contract import WRAPPERS  # noqa: E402
from escape_scoring.metrics import match, metrics, rng_for, shuffled_boundaries  # noqa: E402

SEED = 20260927
TAU = 1.335442066192627


def interior_starts_from_a(boundaries: pd.DataFrame, n_bytes: int) -> list:
    b = boundaries.sort_values("byte_offset")
    starts, triggers = [int(s) for s in b["byte_offset"]], [str(t) for t in b["trigger"]]
    # A format: one row per patch; byte_offset = first byte; trigger labels the patch's own start.
    return sorted({s for s, t in zip(starts, triggers) if t == "entropy" and 0 < s < n_bytes})


def c_v1_patches_from_a(boundaries: pd.DataFrame, n_bytes: int) -> list:
    """Reconstruct C-v1 (start,end,termination_reason) patches from A-format rows."""
    b = boundaries.sort_values("byte_offset")
    starts = [int(s) for s in b["byte_offset"]]
    triggers = [str(t) for t in b["trigger"]]
    if not starts or starts[0] != 0:
        raise ValueError("A-format boundaries need an initial patch at byte 0")
    patches = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_bytes
        reason = {"entropy": "entropy", "length_cap": "max_length"}[triggers[i + 1]] if i + 1 < len(starts) else "eof"
        patches.append({"start_byte": s, "end_byte": e, "termination_reason": reason})
    return patches


def targets_from_parquet(path: Path, n_bytes: int) -> list:
    df = pd.read_parquet(path)
    df = df[(df["depth"] > 0) & (~df["node_type"].str.lower().isin(WRAPPERS))
            & (df["end_byte"] > df["start_byte"])
            & (df["start_byte"] > 0) & (df["start_byte"] < n_bytes)]
    df = df.sort_values(["depth", "end_byte", "node_type"], ascending=[True, False, True])
    seen, out = set(), []
    for s in df["start_byte"]:
        s = int(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--file-ids", nargs="*", help="default: all test files from the frozen calibration record")
    ap.add_argument("--calibration", default="results/final/c-v1-2721954/calibration.json",
                    help="frozen record; only its split_manifest is used to select test files")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--golden", action="store_true",
                    help="score the golden fixtures (golden_fixtures/{domain}/ paths)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.golden:
        file_ids = sorted(p.stem for d in ("py", "cpp", "prose")
                          for p in (root / "golden_fixtures" / d).glob("*.bin"))
        domains = ("py", "cpp", "prose")
        dom_of = {fid: fid.split("_")[0] for fid in file_ids}
        n_bytes_of = {fid: (root / "golden_fixtures" / dom_of[fid] / f"{fid}.bin").stat().st_size
                      for fid in file_ids}
        def bin_path(fid):
            return root / "golden_fixtures" / dom_of[fid] / f"{fid}.bin"
        def struct_path(fid):
            return root / "golden_fixtures" / dom_of[fid] / f"{fid}.structure.parquet"
    else:
        if args.file_ids:
            file_ids = sorted(set(args.file_ids))
        else:
            rec = json.loads((root / args.calibration).read_text())
            file_ids = sorted(m["sample_id"] for m in rec["split_manifest"] if m["split"] == "test")
        manifest = pd.read_parquet(root / "corpus/manifest.parquet").set_index("file_id")
        file_ids = [f for f in file_ids if f in manifest.index]
        domains = sorted({manifest.loc[f, "domain"] for f in file_ids})
        n_bytes_of = {fid: int(manifest.loc[fid, "n_bytes"]) for fid in file_ids}
        def bin_path(fid):
            return root / "corpus" / manifest.loc[fid, "domain"] / f"{fid}.bin"
        def struct_path(fid):
            return root / "structure" / manifest.loc[fid, "domain"] / f"{fid}.parquet"

    results = {"run_ids": args.run_ids, "seed": args.seed, "perms": args.perms,
               "boot": args.boot, "k": 0, "tau": TAU, "n_files": len(file_ids),
               "calibration_record": args.calibration, "golden": args.golden, "domains": {}}
    for d in domains:
        fids = [f for f in file_ids if (dom_of[f] if args.golden else manifest.loc[f, "domain"]) == d]
        if not fids:
            continue
        tgt = {}
        base = {b: {} for b in ("newline", "indent_change", "word_start")}
        for fid in fids:
            n = n_bytes_of[fid]
            content = bin_path(fid).read_bytes()
            tgt[fid] = targets_from_parquet(struct_path(fid), n)
            base["newline"][fid] = sorted({o for o in newline(content) if 0 < o < n})
            base["indent_change"][fid] = sorted({o for o in indent_change(content) if 0 < o < n})
            base["word_start"][fid] = sorted({int(o) for o in word_start(content) if 0 < o < n})
        rows = {}
        for b in base:
            per = []
            for fid in fids:
                tp = len(match(base[b][fid], tgt[fid], 0))
                per.append([tp, len(base[b][fid]) - tp, len(tgt[fid]) - tp])
            arr = np.asarray(per, dtype=float)
            m = metrics(arr.sum(axis=0))
            rows[f"baseline_{b}"] = {"tp": int(arr[:, 0].sum()), "fp": int(arr[:, 1].sum()),
                                     "fn": int(arr[:, 2].sum()), "f1": float(m["f1"]),
                                     "precision": float(m["precision"]), "recall": float(m["recall"]),
                                     "per_file": per}
        for run_id in args.run_ids:
            per = []
            patches_by_fid = {}
            for fid in fids:
                bpath = root / "boundaries" / run_id / f"{fid}.parquet"
                if not bpath.exists():
                    raise FileNotFoundError(f"{run_id}/{fid}: missing boundary parquet")
                b = pd.read_parquet(bpath)
                n = n_bytes_of[fid]
                pred = interior_starts_from_a(b, n)
                tp = len(match(pred, tgt[fid], 0))
                per.append([tp, len(pred) - tp, len(tgt[fid]) - tp])
                patches_by_fid[fid] = c_v1_patches_from_a(b, n)
            arr = np.asarray(per, dtype=float)
            m = metrics(arr.sum(axis=0))
            obs_f1 = float(m["f1"])
            rngs = {fid: rng_for(args.seed, fid, f"score_a_runs_{run_id}") for fid in fids}
            null_f1 = np.empty(args.perms)
            for r in range(args.perms):
                t_all = f_all = n_all = 0
                for fid in fids:
                    nb = shuffled_boundaries(patches_by_fid[fid], n_bytes_of[fid],
                                             rngs[fid], eof_policy="fixed_eof")
                    t = len(match(nb, tgt[fid], 0))
                    t_all += t; f_all += len(nb) - t; n_all += len(tgt[fid]) - t
                null_f1[r] = 2 * t_all / max(1e-12, 2 * t_all + f_all + n_all)
            rows[run_id] = {"tp": int(arr[:, 0].sum()), "fp": int(arr[:, 1].sum()),
                            "fn": int(arr[:, 2].sum()), "f1": obs_f1,
                            "precision": float(m["precision"]), "recall": float(m["recall"]),
                            "f1_h0_mean": float(null_f1.mean()),
                            "f1_margin": float(obs_f1 - null_f1.mean()),
                            "f1_permutation_p": float((1 + np.sum(null_f1 >= obs_f1)) / (args.perms + 1)),
                            "per_file": per}
        rng_b = rng_for(args.seed, f"{d}\0score_a_runs", "file_bootstrap")
        for name in rows:
            arr = np.asarray(rows[name]["per_file"], dtype=float)
            draws = []
            for _ in range(args.boot):
                idx = rng_b.integers(0, len(arr), len(arr))
                draws.append(float(metrics(arr[idx].sum(axis=0))["f1"]))
            rows[name]["f1_ci_low"], rows[name]["f1_ci_high"] = (
                float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))
        best_base = max(("newline", "indent_change", "word_start"),
                        key=lambda b: rows[f"baseline_{b}"]["f1"])
        for name in rows:
            if name.startswith("baseline_"):
                continue
            arr = np.asarray(rows[name]["per_file"], dtype=float)
            bb = np.asarray(rows[f"baseline_{best_base}"]["per_file"], dtype=float)
            draws = []
            for _ in range(args.boot):
                idx = rng_b.integers(0, len(arr), len(arr))
                draws.append(float(metrics(arr[idx].sum(axis=0))["f1"]
                                   - metrics(bb[idx].sum(axis=0))["f1"]))
            rows[name]["strongest_baseline"] = best_base
            rows[name]["f1_minus_best_baseline"] = float(
                metrics(arr.sum(axis=0))["f1"] - metrics(bb.sum(axis=0))["f1"])
            rows[name]["f1_minus_best_baseline_ci"] = [float(np.quantile(draws, .025)),
                                                       float(np.quantile(draws, .975))]
        results["domains"][d] = {"n_files": len(fids), "rows": rows,
                                 "n_targets": int(rows[args.run_ids[0]]["tp"] + rows[args.run_ids[0]]["fn"])}

    out_path = Path(args.out) if args.out else root / f"results/score_a_runs_{args.run_ids[0]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[wrote] {out_path}")
    for d, dres in results["domains"].items():
        print(f"\n=== {d} (n={dres['n_files']}, targets={dres['n_targets']}) ===")
        for name, r in dres["rows"].items():
            extra = ""
            if not name.startswith("baseline_"):
                extra = (f" | margin={r['f1_margin']:+.4f} p={r['f1_permutation_p']:.3g}"
                         f" | vs {r['strongest_baseline']}={r['f1_minus_best_baseline']:+.4f}")
            print(f"  {name:18s} F1={r['f1']:.4f} [{r['f1_ci_low']:.4f},{r['f1_ci_high']:.4f}] "
                  f"P={r['precision']:.4f} R={r['recall']:.4f} tp={r['tp']}{extra}")


if __name__ == "__main__":
    main()
