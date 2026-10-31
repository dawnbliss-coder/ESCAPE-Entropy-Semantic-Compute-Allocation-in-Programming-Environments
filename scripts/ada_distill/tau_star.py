"""Full-run Steps 5-6 (docs/ADA_FULL_RUN_GATES.md): pick tau* from a calibration-split
tau sweep written by escape_distill.score_a_runs.

Pre-registered rule: per language, tau* = argmax over the grid of the median per-file
(F1_observed - mean_F1_H0) at k=0 on CALIBRATION files; ties -> smaller tau. The handoff
gives two readings of mean_F1_H0, and both are reported:
  snippet  its snippet: each file's F1 minus the run's pooled null mean
  frozen   the frozen k-criterion's comparator (escape_scoring/calibration.py): each
           file's F1 minus that file's own null mean (score_a_runs per_file_f1_h0_mean)
When they pick different tau*, the single confirmatory test run waits for a decision.

Usage:
  python scripts/ada_distill/tau_star.py results/final/teacher_tau_sweep_calib.json \
      --runs teacher-tau08 ... --taus 0.8 ... --out results/final/teacher_tau_star.json
"""
import argparse
import json

import numpy as np

LANGS = (("python", "py"), ("cpp", "cpp"), ("english", "prose"))
TIE = 1e-12  # tie_tolerance of the frozen calibration record


def pick(table, key):
    best = max(r[key] for r in table)
    return min(r["tau"] for r in table if r[key] >= best - TIE)


def grid(sweep, runs, taus, dom):
    """Per-tau median per-file margins under both comparators, plus pooled F1 and margin."""
    table = []
    for run, tau in zip(runs, taus):
        row = sweep["domains"][dom]["rows"][run]
        per = np.asarray(row["per_file"], dtype=float)
        f1_obs = 2 * per[:, 0] / np.maximum(1e-12, 2 * per[:, 0] + per[:, 1] + per[:, 2])
        table.append({"tau": tau, "run_id": run, "n_files": len(per),
                      "median_margin_snippet": float(np.median(f1_obs - row["f1_h0_mean"])),
                      "median_margin_frozen": float(np.median(f1_obs - np.asarray(row["per_file_f1_h0_mean"]))),
                      "pooled_f1": row["f1"], "pooled_margin": row["f1_margin"]})
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--taus", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.runs) != len(args.taus):
        ap.error("--runs and --taus must pair up")
    sweep = json.load(open(args.sweep))
    out = {"sweep": args.sweep, "k": 0, "tie_tolerance": TIE,
           "rule": "per language: argmax over tau of median per-file (F1_observed - mean_F1_H0), "
                   "calibration files only, k=0; ties -> smaller tau",
           "languages": {}}
    for lang, dom in LANGS:
        table = grid(sweep, args.runs, args.taus, dom)
        ts, tf = pick(table, "median_margin_snippet"), pick(table, "median_margin_frozen")
        out["languages"][lang] = {"tau_star_snippet": ts, "tau_star_frozen": tf, "agree": ts == tf, "grid": table}
        print(lang)
        for r in table:
            print(f"  tau={r['tau']:<18} snippet={r['median_margin_snippet']:+.4f} "
                  f"frozen={r['median_margin_frozen']:+.4f} pooled F1={r['pooled_f1']:.4f}")
        print(f"  tau* snippet={ts} frozen={tf} {'AGREE' if ts == tf else 'DISAGREE'}")
    out["all_agree"] = all(v["agree"] for v in out["languages"].values())
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out}; " + ("comparators agree" if out["all_agree"] else
                                     "comparators DISAGREE: decide before the confirmatory run"))


if __name__ == "__main__":
    main()
