"""Exploratory: summarise the tau grid scored on TEST files (tau_sweep_test.sbatch).

Run on request (2026-10-30) after tau* had been fixed on calibration files and each model's
single confirmatory test run had been scored. Descriptive only: nothing here selects tau or
k, and tau* stays the calibration record's. Same per-file median margins as tau_star.py, at
the taus of the calibration record, plus whether the calibration tau* is also the argmax on
test files. The test grid runs have their own run ids (teacher-test-tau08, ...) and are
paired with the calibration grid by position.

Usage:
  python scripts/ada_distill/tau_sweep_test.py results/final/analysis/teacher_tau_sweep_test.json \
      --tau-star results/final/teacher_tau_star.json \
      --out results/final/analysis/teacher_tau_sweep_test_summary.json
"""
import argparse
import json

from tau_star import LANGS, grid, pick


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep")
    ap.add_argument("--tau-star", required=True, help="the calibration tau* record (run ids, taus, tau*)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sweep, star = json.load(open(args.sweep)), json.load(open(args.tau_star))
    out = {"sweep": args.sweep, "tau_star_record": args.tau_star, "split": "test", "k": sweep["k"],
           "perms": sweep["perms"], "boot": sweep["boot"],
           "status": "exploratory, scored after tau* was fixed and confirmed; not used to select tau or k",
           "languages": {}}
    for lang, dom in LANGS:
        cal = star["languages"][lang]
        runs = sweep["run_ids"]
        if len(runs) != len(cal["grid"]) or any(r != c["run_id"] and not r.endswith(c["run_id"].rsplit("-", 1)[-1])
                                                for r, c in zip(runs, cal["grid"])):
            raise SystemExit(f"test runs {runs} do not line up with the calibration grid")
        table = grid(sweep, runs, [c["tau"] for c in cal["grid"]], dom)
        ts, tf = pick(table, "median_margin_snippet"), pick(table, "median_margin_frozen")
        out["languages"][lang] = {"tau_star_calibration": cal["tau_star_frozen"], "test_argmax_snippet": ts,
                                  "test_argmax_frozen": tf, "same_as_calibration": tf == cal["tau_star_frozen"],
                                  "grid": table}
        print(lang)
        for r in table:
            print(f"  tau={r['tau']:<18} snippet={r['median_margin_snippet']:+.4f} "
                  f"frozen={r['median_margin_frozen']:+.4f} pooled F1={r['pooled_f1']:.4f}")
        print(f"  test argmax={tf} (calibration tau*={cal['tau_star_frozen']})")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
