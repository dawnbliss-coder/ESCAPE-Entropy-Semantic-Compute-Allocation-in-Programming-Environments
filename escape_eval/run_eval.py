"""P1 / P2 / P3 / P4 alignment evaluation for one Stream A run (PRELIMINARY).

Tables written to results/{analysis_id}/ (parquet + csv, each with preliminary=True):
  alignment      one row per (domain, target kind, k, stratum axis, stratum, metric):
                 - BLT, E[H0], whitespace and word-boundary metrics with file-bootstrap CIs
                 - permutation test vs H0 (p, null mean/sd/quantiles, z, ratio)
                 - BLT-H0, BLT-whitespace and BLT-word deltas with CIs, and the paired
                   randomisation p-values vs whitespace and vs the word baseline
                 - length-capped boundaries scored separately (never mixed into BLT)
                 P1 = kind start, axis all; per depth = axis depth; P4 detail = axis category/node_type
  p2_start_end   start vs end alignment (BLT and margin over H0) with CIs; precision > recall at starts
  p4_openers     recall lift over H0 for deterministic openers vs open-ended constructs, and their contrast
  p3_bpp         BPP variance and mean, code vs prose (median differences, bootstrap CIs, Mann-Whitney U)
  p3_alignment   F1 margin over H0, code vs prose at the same k
  iou            secondary span-IoU metric (BLT, H0, whitespace)
Plus params.json and a results/manifest.json entry.

k comes only from a non-smoke results/k_calibration/selected_k.json. --k-override needs --smoke.
Prose is scored at every calibrated code k, because prose has no calib split.

Usage:
  python -m escape_eval.run_eval --run-id RUN --source corpus --domains py cpp prose --split main
  python -m escape_eval.run_eval --run-id RUN --source golden --smoke --k-override py=2 cpp=2   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_common.io import sha256_file  # noqa: E402
from escape_eval import data as D  # noqa: E402
from escape_eval import results_io as RIO  # noqa: E402
from escape_eval import stats as S  # noqa: E402
from escape_eval.engine import (  # noqa: E402
    METHOD_BLT,
    METHOD_CAP,
    METHOD_H0,
    METHOD_WS,
    METHOD_WORD,
    EngineConfig,
    count_sets,
    run_engine,
    stratum_keys,
)
from escape_eval.targets import AXES  # noqa: E402

METRICS = S.METRICS
CI_LEVEL = 0.95
DEFAULT_SEED = 20260915
QUANTITIES = ("blt", "h0", "ws", "delta_h0", "delta_ws", "word", "delta_word")
COLS_PER_BOOT_CHUNK = 4


# ---------------------------------------------------------------------------------------
# alignment table
# ---------------------------------------------------------------------------------------
def _alignment_draws(cs: dict, w: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(b, len(QUANTITIES), len(METRICS), len(cols)) bootstrap values for one weight chunk."""
    blt = cs[METHOD_BLT].select(cols).pooled(w)
    h0 = cs[METHOD_H0].select(cols).pooled(w)
    ws = cs[METHOD_WS].select(cols).pooled(w) if METHOD_WS in cs else None
    word = cs[METHOD_WORD].select(cols).pooled(w) if METHOD_WORD in cs else None
    out = np.full((w.shape[0], len(QUANTITIES), len(METRICS), len(cols)), np.nan)
    for mi, m in enumerate(METRICS):
        out[:, 0, mi] = blt[m]
        out[:, 1, mi] = h0[m]
        out[:, 3, mi] = blt[m] - h0[m]
        if ws is not None:
            out[:, 2, mi] = ws[m]
            out[:, 4, mi] = blt[m] - ws[m]
        if word is not None:
            out[:, 5, mi] = word[m]
            out[:, 6, mi] = blt[m] - word[m]
    return out


def boot_entropy(seed: int, domain: str, k: int) -> list:
    """Shared by every kind/axis/contrast of one (domain, k): identical file weights, so all
    bootstrap contrasts within a domain are paired."""
    return S.seed_entropy(seed, f"{domain}|k={k}", S.PURPOSE_BOOTSTRAP)


def alignment_rows(domain, res, file_ids, kind, k, axis, n_boot, n_rand, seed) -> list:
    keys = stratum_keys(res.counts, kind, k, axis)
    if not keys:
        return []
    cs = count_sets(res.counts, file_ids, kind, k, keys,
                    (METHOD_BLT, METHOD_WS, METHOD_WORD, METHOD_H0, METHOD_CAP))
    n_pred_total = float(cs[METHOD_BLT].n_pred.sum())
    n_t_total = cs[METHOD_BLT].n_targets.sum(0)
    obs = {m: cs[m].pooled() for m in cs}
    null = res.null.null_metrics(kind, k, keys, n_pred_total, n_t_total)
    perm = {m: S.permutation_summary(obs[METHOD_BLT][m], null[m]) for m in METRICS}
    C = len(keys)
    lo = np.full((len(QUANTITIES), len(METRICS), C), np.nan)
    hi = np.full_like(lo, np.nan)
    ent = boot_entropy(seed, domain, k)
    for c0 in range(0, C, COLS_PER_BOOT_CHUNK):
        cols = np.arange(c0, min(C, c0 + COLS_PER_BOOT_CHUNK))
        draws = S.bootstrap_draws(ent, len(file_ids), n_boot, lambda w, cols=cols: _alignment_draws(cs, w, cols))
        l_, h_ = S.percentile_ci(draws, CI_LEVEL)
        lo[..., cols], hi[..., cols] = l_, h_
    rand = None
    rand_word = None
    if METHOD_WS in cs:
        rand = S.paired_randomisation(cs[METHOD_BLT], cs[METHOD_WS],
                                      S.seed_entropy(seed, f"{domain}|{kind}|k={k}|{axis}", S.PURPOSE_RANDOMISATION_WS),
                                      n_rand)
    if METHOD_WORD in cs:
        rand_word = S.paired_randomisation(cs[METHOD_BLT], cs[METHOD_WORD],
                                           S.seed_entropy(seed, f"{domain}|{kind}|k={k}|{axis}",
                                                          S.PURPOSE_RANDOMISATION_WORD),
                                           n_rand)
    rows = []
    nan = float("nan")
    for j, (ax, st) in enumerate(keys):
        for mi, m in enumerate(METRICS):
            b = float(obs[METHOD_BLT][m][j])
            h = float(obs[METHOD_H0][m][j])
            w = float(obs[METHOD_WS][m][j]) if METHOD_WS in cs else nan
            wd = float(obs[METHOD_WORD][m][j]) if METHOD_WORD in cs else nan
            rows.append({
                "domain": domain, "kind": kind, "k": int(k), "axis": ax, "stratum": st, "metric": m,
                "n_files": len(file_ids), "n_targets": int(n_t_total[j]), "n_pred_blt": int(n_pred_total),
                "n_pred_ws": int(cs[METHOD_WS].n_pred.sum()) if METHOD_WS in cs else None,
                "n_pred_word": int(cs[METHOD_WORD].n_pred.sum()) if METHOD_WORD in cs else None,
                "blt": b, "blt_lo": lo[0, mi, j], "blt_hi": hi[0, mi, j],
                "h0": h, "h0_lo": lo[1, mi, j], "h0_hi": hi[1, mi, j],
                "ws": w, "ws_lo": lo[2, mi, j], "ws_hi": hi[2, mi, j],
                "word": wd, "word_lo": lo[5, mi, j], "word_hi": hi[5, mi, j],
                "delta_h0": b - h, "delta_h0_lo": lo[3, mi, j], "delta_h0_hi": hi[3, mi, j],
                "p_perm_h0": float(perm[m]["p_value"][j]), "h0_null_mean": float(perm[m]["null_mean"][j]),
                "h0_null_sd": float(perm[m]["null_sd"][j]), "h0_null_q025": float(perm[m]["null_q025"][j]),
                "h0_null_q975": float(perm[m]["null_q975"][j]), "z_h0": float(perm[m]["z"][j]),
                "ratio_h0": float(perm[m]["ratio"][j]),
                "delta_ws": b - w, "delta_ws_lo": lo[4, mi, j], "delta_ws_hi": hi[4, mi, j],
                "p_rand_ws": float(rand[m]["p_value"][j]) if rand is not None else nan,
                "delta_word": b - wd, "delta_word_lo": lo[6, mi, j], "delta_word_hi": hi[6, mi, j],
                "p_rand_word": float(rand_word[m]["p_value"][j]) if rand_word is not None else nan,
                "length_cap": float(obs[METHOD_CAP][m][j]) if METHOD_CAP in cs else nan,
                "n_pred_length_cap": int(cs[METHOD_CAP].n_pred.sum()) if METHOD_CAP in cs else 0,
            })
    return rows


# ---------------------------------------------------------------------------------------
# P2, P4, IoU, P3
# ---------------------------------------------------------------------------------------
def p2_rows(domain, res, file_ids, k, n_boot, seed) -> list:
    keys = [("all", "all")]
    cs_s = count_sets(res.counts, file_ids, "start", k, keys, (METHOD_BLT, METHOD_H0))
    cs_e = count_sets(res.counts, file_ids, "end", k, keys, (METHOD_BLT, METHOD_H0))
    if METHOD_BLT not in cs_s or METHOD_BLT not in cs_e:
        return []

    def quantities(w):
        sb, sh, eb, eh = (c.pooled(w) for c in (cs_s[METHOD_BLT], cs_s[METHOD_H0], cs_e[METHOD_BLT], cs_e[METHOD_H0]))
        cols = []
        for m in METRICS:
            cols.append(sb[m][..., 0] - eb[m][..., 0])
            cols.append((sb[m][..., 0] - sh[m][..., 0]) - (eb[m][..., 0] - eh[m][..., 0]))
        cols.append((sb["precision"][..., 0] > sb["recall"][..., 0]).astype(np.float64))
        return np.stack(cols, axis=-1)

    observed = quantities(None)
    draws = S.bootstrap_draws(boot_entropy(seed, domain, k), len(file_ids), n_boot, quantities)
    lo, hi = S.percentile_ci(draws, CI_LEVEL)
    sb, sh, eb, eh = (c.pooled() for c in (cs_s[METHOD_BLT], cs_s[METHOD_H0], cs_e[METHOD_BLT], cs_e[METHOD_H0]))
    rows = []
    for mi, m in enumerate(METRICS):
        i_diff, i_margin = 2 * mi, 2 * mi + 1
        rows.append({
            "domain": domain, "k": int(k), "metric": m, "n_files": len(file_ids),
            "blt_start": float(sb[m][0]), "blt_end": float(eb[m][0]),
            "blt_start_minus_end": float(observed[i_diff]), "blt_start_minus_end_lo": lo[i_diff], "blt_start_minus_end_hi": hi[i_diff],
            "margin_h0_start": float(sb[m][0] - sh[m][0]), "margin_h0_end": float(eb[m][0] - eh[m][0]),
            "margin_start_minus_end": float(observed[i_margin]), "margin_start_minus_end_lo": lo[i_margin],
            "margin_start_minus_end_hi": hi[i_margin],
            "precision_gt_recall_at_starts": bool(observed[-1] == 1.0),
            "bootstrap_share_precision_gt_recall": float(np.nanmean(draws[:, -1])),
        })
    return rows


def p4_rows(domain, res, file_ids, k, n_boot, seed) -> list:
    keys = [("category", c) for c in ("deterministic_opener", "open_ended")]
    present = {s for _, s in stratum_keys(res.counts, "start", k, "category")}
    if not all(c in present for _, c in keys):
        return []
    cs = count_sets(res.counts, file_ids, "start", k, keys, (METHOD_BLT, METHOD_H0))

    def quantities(w):
        b, h = cs[METHOD_BLT].pooled(w)["recall"], cs[METHOD_H0].pooled(w)["recall"]
        lift = b - h
        return np.stack([b[..., 0], b[..., 1], h[..., 0], h[..., 1], lift[..., 0], lift[..., 1],
                         lift[..., 1] - lift[..., 0]], axis=-1)

    obs = quantities(None)
    draws = S.bootstrap_draws(boot_entropy(seed, domain, k), len(file_ids), n_boot, quantities)
    lo, hi = S.percentile_ci(draws, CI_LEVEL)
    names = ["recall_blt_deterministic", "recall_blt_open_ended", "recall_h0_deterministic", "recall_h0_open_ended",
             "lift_deterministic", "lift_open_ended", "contrast_open_minus_deterministic"]
    n_t = cs[METHOD_BLT].n_targets.sum(0)
    row = {"domain": domain, "k": int(k), "n_files": len(file_ids),
           "n_targets_deterministic": int(n_t[0]), "n_targets_open_ended": int(n_t[1])}
    for i, name in enumerate(names):
        row[name], row[f"{name}_lo"], row[f"{name}_hi"] = float(obs[i]), float(lo[i]), float(hi[i])
    row["p4_direction_as_predicted"] = bool(obs[-1] > 0)
    row["bootstrap_share_contrast_positive"] = float(np.nanmean(draws[:, -1] > 0))
    return [row]


def iou_rows(domain, iou: pd.DataFrame, n_boot, seed) -> list:
    if iou.empty:
        return []
    n = iou["n_spans"].to_numpy(dtype=np.float64)
    sums = iou[["iou_sum_blt", "iou_sum_whitespace", "iou_sum_word", "iou_sum_h0"]].to_numpy(dtype=np.float64)

    def quantities(w):
        if w is None:
            return sums.sum(0) / n.sum()
        return (w @ sums) / (w @ n)[:, None]

    obs = quantities(None)
    draws = S.bootstrap_draws(S.seed_entropy(seed, f"{domain}|iou", S.PURPOSE_BOOTSTRAP), len(n), n_boot, quantities)
    lo, hi = S.percentile_ci(draws, CI_LEVEL)
    return [{"domain": domain, "method": m, "mean_best_iou": float(obs[i]), "lo": float(lo[i]), "hi": float(hi[i]),
             "n_files": int(len(n)), "n_spans": int(n.sum()), "secondary_metric": True}
            for i, m in enumerate(("blt", "whitespace", "word", "h0"))]


def f1_margin_draws(domain, res, file_ids, k, n_boot, seed):
    keys = [("all", "all")]
    cs = count_sets(res.counts, file_ids, "start", k, keys, (METHOD_BLT, METHOD_H0))

    def q(w):
        return cs[METHOD_BLT].pooled(w)["f1"][..., 0] - cs[METHOD_H0].pooled(w)["f1"][..., 0]

    return float(q(None)), S.bootstrap_draws(boot_entropy(seed, domain, k), len(file_ids), n_boot, q)


def p3_alignment_rows(results: dict, ks: dict, n_boot, seed) -> list:
    rows = []
    if "prose" not in results:
        return rows
    for code in ("py", "cpp"):
        if code not in results or code not in ks:
            continue
        k = ks[code]
        obs_c, d_c = f1_margin_draws(code, results[code]["res"], results[code]["ids"], k, n_boot, seed)
        obs_p, d_p = f1_margin_draws("prose", results["prose"]["res"], results["prose"]["ids"], k, n_boot, seed)
        diff = d_c.reshape(-1) - d_p.reshape(-1)
        lo, hi = S.percentile_ci(diff, CI_LEVEL)
        rows.append({"code_domain": code, "k": int(k), "f1_margin_code": obs_c, "f1_margin_prose": obs_p,
                     "code_minus_prose": obs_c - obs_p, "lo": float(lo), "hi": float(hi),
                     "n_files_code": len(results[code]["ids"]), "n_files_prose": len(results["prose"]["ids"])})
    return rows


def p3_bpp_rows(bpp: pd.DataFrame, ids_by_domain: dict, n_boot, seed) -> list:
    from scipy.stats import mannwhitneyu

    rows = []
    if bpp is None or "prose" not in ids_by_domain:
        return rows
    prose = bpp[bpp["file_id"].isin(ids_by_domain["prose"])]
    for code in ("py", "cpp"):
        if code not in ids_by_domain:
            continue
        code_rows = bpp[bpp["file_id"].isin(ids_by_domain[code])]
        lo_b, hi_b = prose["n_bytes"].min(), prose["n_bytes"].max()
        for subset, df in (("all_code_files", code_rows),
                           ("code_n_bytes_within_prose_range", code_rows[code_rows["n_bytes"].between(lo_b, hi_b)])):
            for stat in ("var_bpp", "mean_bpp"):
                x = df[stat].dropna().to_numpy(dtype=np.float64)
                y = prose[stat].dropna().to_numpy(dtype=np.float64)
                if x.size == 0 or y.size == 0:
                    continue
                diff, draws = S.bootstrap_median_difference(
                    x, y, S.seed_entropy(seed, f"{code}|{subset}|{stat}", S.PURPOSE_BOOTSTRAP_BPP),
                    S.seed_entropy(seed, f"prose|{subset}|{stat}", S.PURPOSE_BOOTSTRAP_BPP), n_boot)
                lo, hi = S.percentile_ci(draws, CI_LEVEL)
                u = mannwhitneyu(x, y, alternative="two-sided")
                rows.append({"code_domain": code, "subset": subset, "statistic": stat, "n_code": int(x.size),
                             "n_prose": int(y.size), "median_code": float(np.median(x)), "median_prose": float(np.median(y)),
                             "median_diff": diff, "lo": float(lo), "hi": float(hi),
                             "mannwhitney_u": float(u.statistic), "mannwhitney_p": float(u.pvalue)})
    return rows


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def resolve_ks(root: Path, results_root: Path, k_from, k_override, smoke: bool, domains) -> tuple:
    if k_override:
        if not smoke:
            raise SystemExit("--k-override is only allowed with --smoke (k must come from the calib split)")
        ks = {}
        for item in k_override:
            dom, val = item.split("=")
            ks[dom] = int(val)
        return ks, {"k_source": "override (SMOKE)", "k_override": ks}
    path = Path(k_from) if k_from else results_root / "k_calibration" / "selected_k.json"
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise SystemExit(f"{path} not found: run escape_eval.calibrate_k on the calib split first")
    rec = json.loads(path.read_text())
    if rec.get("smoke") and not smoke:
        raise SystemExit(f"{path} is a SMOKE calibration and cannot be used for non-smoke scoring")
    ks = {d: int(v["k"]) for d, v in rec["selected_k"].items()}
    shown = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return ks, {"k_source": shown, "k_source_sha256": sha256_file(path), "k_calibration_run_id": rec.get("run_id")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream C alignment evaluation (PRELIMINARY)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--stream-a-root", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", choices=D.SOURCES, default="corpus")
    ap.add_argument("--domains", nargs="+", default=["py", "cpp", "prose"])
    ap.add_argument("--split", choices=("calib", "main"), default="main", help="code split")
    ap.add_argument("--prose-split", default="main")
    ap.add_argument("--k-from")
    ap.add_argument("--k-override", nargs="+", help="SMOKE ONLY, e.g. py=2 cpp=3")
    ap.add_argument("--k-grid", nargs="+", type=int,
                    help="report every k in the grid and select none; allowed for prose (it has no calib split) "
                         "or with --smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--kinds", nargs="+", default=["start", "end"])
    ap.add_argument("--axes", nargs="+", default=list(AXES))
    ap.add_argument("--iou", action="store_true")
    ap.add_argument("--iou-resamples", type=int, default=100)
    ap.add_argument("--limit-per-domain", type=int)
    ap.add_argument("--parse-ok", choices=("all", "clean", "recovered"), default="all",
                    help="restrict CODE files by the manifest parse_ok flag (tree-sitter error-recovery). "
                         "prose is unaffected (no parse_ok). clean = parse_ok True, recovered = parse_ok False. "
                         "Recorded in params.json.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--analysis-id")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    stream_a_root = Path(args.stream_a_root).resolve() if args.stream_a_root else root
    if args.source == "golden" and not args.smoke:
        raise SystemExit("golden fixtures are a regression set, not an evaluation set: add --smoke")
    base_results = Path(args.results_root) if Path(args.results_root).is_absolute() else root / args.results_root
    if args.k_grid:
        if set(args.domains) - {"prose"} and not args.smoke:
            raise SystemExit("--k-grid is only allowed for prose (no calib split exists) or with --smoke")
        ks, k_meta = {}, {"k_source": "grid (uncalibrated: every k reported, none selected)",
                          "k_grid": sorted(set(args.k_grid))}
    else:
        ks, k_meta = resolve_ks(root, base_results, args.k_from, args.k_override, args.smoke, args.domains)
    entry = D.load_run_entry(stream_a_root, args.run_id)
    out_root = RIO.effective_results_root(base_results, args.smoke)
    analysis_id = args.analysis_id or f"eval_{args.source}_{args.split}_{args.run_id}"
    out_dir = out_root / analysis_id

    results, tables = {}, {"alignment": [], "p2_start_end": [], "p4_openers": [], "iou": []}
    problems, file_ids_by_domain, engine_seconds = [], {}, {}
    for domain in args.domains:
        files = D.build_file_table(root, args.source, [domain],
                                   split=None if args.source == "golden" else args.split,
                                   prose_split=None if args.source == "golden" else args.prose_split,
                                   limit_per_domain=args.limit_per_domain)
        if domain in ("py", "cpp") and args.parse_ok != "all":
            mask = files["parse_ok"].astype(object) == (args.parse_ok == "clean")
            files = files.loc[mask].reset_index(drop=True)
            if files.empty:
                print(f"{domain}: no files with parse_ok == {args.parse_ok == 'clean'} - skipped")
                continue
        if files.empty:
            print(f"{domain}: no files - skipped")
            continue
        if args.k_grid:
            dom_ks = sorted(set(args.k_grid))
        elif domain == "prose":
            dom_ks = sorted({ks[c] for c in ("py", "cpp") if c in ks})
        else:
            if domain not in ks:
                raise SystemExit(f"no calibrated k for {domain}")
            dom_ks = [ks[domain]]
        if not dom_ks:
            raise SystemExit("prose needs at least one calibrated code k")
        missing = [f for f in files["file_id"] if not D.stream_a_outputs_exist(stream_a_root, args.run_id, f)]
        if missing:
            raise SystemExit(f"{domain}: {len(missing)} files lack Stream A outputs, e.g. {missing[:5]}")
        cfg = EngineConfig(ks=tuple(dom_ks), kinds=tuple(args.kinds), axes=tuple(args.axes),
                           n_resamples=args.n_resamples, master_seed=args.seed, iou=args.iou,
                           iou_resamples=args.iou_resamples)
        print(f"{domain}: {len(files)} files, k={dom_ks}, R={args.n_resamples}, B={args.n_bootstrap}")
        res = run_engine(files, stream_a_root, args.run_id, entry, {domain: D.load_taxonomy(root, domain)}, cfg)
        problems.extend(res.problems)
        if res.problems:
            raise SystemExit(f"{domain}: engine problems: {res.problems[:5]}")
        ids = files["file_id"].tolist()
        results[domain] = {"res": res, "ids": ids}
        file_ids_by_domain[domain] = ids
        engine_seconds[domain] = res.seconds
        for k in dom_ks:
            for kind in args.kinds:
                for axis in args.axes:
                    tables["alignment"].extend(alignment_rows(domain, res, ids, kind, k, axis, args.n_bootstrap,
                                                              args.n_resamples, args.seed))
            if "start" in args.kinds and "end" in args.kinds:
                tables["p2_start_end"].extend(p2_rows(domain, res, ids, k, args.n_bootstrap, args.seed))
            if domain != "prose":
                tables["p4_openers"].extend(p4_rows(domain, res, ids, k, args.n_bootstrap, args.seed))
        if args.iou:
            tables["iou"].extend(iou_rows(domain, res.iou, args.n_bootstrap, args.seed))

    bpp = D.load_bpp_summary(stream_a_root, args.run_id)
    tables["p3_bpp"] = p3_bpp_rows(bpp, file_ids_by_domain, args.n_bootstrap, args.seed)
    tables["p3_alignment"] = p3_alignment_rows(results, ks, args.n_bootstrap, args.seed)

    written = []
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        df["k_mode"] = "grid_uncalibrated" if args.k_grid else ("override_smoke" if args.k_override else "calibrated")
        if args.smoke:
            df["smoke"] = True
        written += RIO.write_table(df, out_dir, name)
    params = {
        "analysis_id": analysis_id, "run_id": args.run_id, "source": args.source, "domains": args.domains,
        "split": args.split, "prose_split": args.prose_split, "ks": ks, **k_meta,
        "n_resamples": args.n_resamples, "n_bootstrap": args.n_bootstrap, "master_seed": args.seed,
        "ci_level": CI_LEVEL, "ci_method": "percentile, paired cluster (file) bootstrap",
        "kinds": args.kinds, "axes": args.axes, "iou": args.iou, "iou_resamples": args.iou_resamples,
        "parse_ok": args.parse_ok,
        "smoke": args.smoke, "preliminary": True,
        "run_entry": entry, "file_ids": file_ids_by_domain,
        "file_ids_sha256": {d: D.file_ids_sha256(v) for d, v in file_ids_by_domain.items()},
        "manifest_sha256": D.manifest_sha256(root), "engine_seconds": engine_seconds, "problems": problems,
        "created": RIO.utc_now_iso(), **RIO.git_info(root), "packages": RIO.package_versions(),
        "outputs": written,
    }
    RIO.write_json(params, out_dir / "params.json")
    RIO.register_analysis(out_root, analysis_id, out_dir, "alignment_eval", smoke=args.smoke,
                          extra={"run_id": args.run_id, "source": args.source, "split": args.split})
    align = pd.DataFrame(tables["alignment"])
    if not align.empty:
        p1 = align[(align["kind"] == "start") & (align["axis"] == "all") & (align["metric"] == "f1")]
        for r in p1.itertuples(index=False):
            print(f"P1 {r.domain} k={r.k}: F1 BLT {r.blt:.4f} [{r.blt_lo:.4f}, {r.blt_hi:.4f}]  "
                  f"E[H0] {r.h0:.4f}  WS {r.ws:.4f}  WORD {r.word:.4f}  "
                  f"p_perm(H0)={r.p_perm_h0:.4g}  p_rand(WS)={r.p_rand_ws:.4g}  p_rand(WORD)={r.p_rand_word:.4g}")
    print(f"wrote {out_dir}" + (" (SMOKE)" if args.smoke else ""))
    return out_dir


if __name__ == "__main__":
    main()
