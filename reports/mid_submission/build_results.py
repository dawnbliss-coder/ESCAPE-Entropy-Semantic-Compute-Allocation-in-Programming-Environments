"""Generate the mid-submission report's numbers, tables and figure from a completed Ada C-v1 run.

Reads results/final/c-v1-<job>/ (30_score.sbatch output) and the fidelity rerun, and
writes reports/mid_submission/generated/{results.tex, evidence.json, depth_margin.pdf}.
Every number and every data-dependent sentence in the paper comes from here; the
verdict rules are fixed below, before results exist, and are the same for every run.

Refuses synthetic inputs. `--allow-synthetic` exists only to test the pipeline: it
watermarks every generated table, and check_report.py rejects the watermark.

  python3 reports/mid_submission/build_results.py                    # newest run
  python3 reports/mid_submission/build_results.py --run results/final/c-v1-123456
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 9})

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "generated"
LANGS = [("python", "Python"), ("cpp", "C++"), ("english", "English")]
CODE = [("python", "Python"), ("cpp", "C++")]
WATERMARK = "SYNTHETIC PIPELINE TEST - NOT A RESULT"
MISSING = "\\textbf{MISSING-RESULT}"
ALPHA = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(x, d=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def signed(x, d=3):
    return "--" if x is None else f"{x:+.{d}f}"


def ci(lo, hi, d=3):
    return "--" if lo is None or hi is None else f"[{lo:.{d}f}, {hi:.{d}f}]"


def pval(p, permutations):
    if p is None:
        return "--"
    floor = 1 / (permutations + 1)
    return f"$\\leq${floor:.0e}".replace("e-0", "e-") if p <= floor * (1 + 1e-9) else f"{p:.3f}"


def texescape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def newest_run() -> Path:
    runs = sorted((ROOT / "results/final").glob("c-v1-*"), key=lambda p: p.stat().st_mtime)
    runs = [r for r in runs if (r / "evaluation/scores.json").exists()]
    if not runs:
        raise SystemExit("no completed results/final/c-v1-* run; fetch it from Ada first")
    return runs[-1]


def verdict_vs_zero(lo, hi):
    """Fixed rule: a difference is reported as positive/negative only if its 95%
    bootstrap interval excludes zero."""
    if lo is None or hi is None:
        return "unresolved"
    return "above" if lo > 0 else "below" if hi < 0 else "indistinguishable"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--fidelity", type=Path, default=ROOT / "results/final/stream_a_fidelity_ada/fidelity.json")
    ap.add_argument("--allow-synthetic", action="store_true", help="pipeline test only; watermarks the output")
    ap.add_argument("--report", type=Path, default=Path(__file__).resolve().parent,
                    help="report directory: its <body>.tex decides which macros must exist; output goes to <report>/generated")
    ap.add_argument("--body", default="paper-body.tex")
    args = ap.parse_args(argv)
    global OUT
    OUT = args.report.resolve() / "generated"
    run = (args.run if args.run else newest_run()).resolve()
    scores = json.loads((run / "evaluation/scores.json").read_text())
    calibration = json.loads((run / "calibration.json").read_text())
    baselines = json.loads((run / "baselines.json").read_text())
    o5_path, o5b_path = run / "o5/o5.json", run / "o5_inverse_bpp/o5.json"
    o5 = json.loads(o5_path.read_text()) if o5_path.exists() else None
    o5b = json.loads(o5b_path.read_text()) if o5b_path.exists() else None
    synthetic = any(r.get("synthetic") for r in (scores, calibration, baselines, o5 or {}))
    if synthetic and not args.allow_synthetic:
        raise SystemExit("refusing synthetic inputs for the final report")
    perms = scores["configuration"]["permutations"]
    if not synthetic and (perms != 10000 or calibration["permutations"] != 10000
                          or scores["configuration"]["bootstrap"] != 2000):
        raise SystemExit("final report requires the frozen 10,000-permutation / 2,000-bootstrap protocol")
    if scores["configuration"]["calibration_record_sha256"] != calibration["record_sha256"]:
        raise SystemExit("scores were not produced from this calibration record")

    rows = scores["scores"]

    def row(kind, axis, language, stratum):
        hit = [r for r in rows if (r["kind"], r["axis"], r["language"], r["stratum"]) == (kind, axis, language, stratum)]
        return hit[0] if hit else None

    def base(language, kind, proposer):
        hit = [r for r in baselines["rows"] if (r["language"], r["kind"], r["proposer"]) == (language, kind, proposer)]
        return hit[0] if hit else None

    present = [(k, name) for k, name in LANGS if row("start", "language", "all", k)]
    macros, evidence = {}, {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
                            "synthetic": synthetic, "inputs": {}}
    for p in [run / "evaluation/scores.json", run / "calibration.json", run / "baselines.json", o5_path, o5b_path, args.fidelity]:
        if p.exists():
            evidence["inputs"][p.name if p.parent == run else str(p.relative_to(run)) if p.is_relative_to(run) else p.name] = sha(p)

    def macro(name, value):
        assert name.isalpha(), name
        macros[name] = value

    split = calibration["split_manifest"]
    tag = {"python": "Py", "cpp": "Cpp", "english": "En"}
    for lang, _ in LANGS:
        t = tag[lang]
        macro(f"k{t}", str(calibration["selected_k"].get(lang, "--")))
        macro(f"nCal{t}", f"{sum(m['language'] == lang and m['split'] == 'calibration' for m in split):,}")
        macro(f"nTest{t}", f"{sum(m['language'] == lang and m['split'] == 'test' for m in split):,}")
    macro("nSamples", f"{len(split):,}")
    macro("nTestAll", f"{sum(m['split'] == 'test' for m in split):,}")
    macro("nPerm", f"{perms:,}")
    macro("nBoot", f"{scores['configuration']['bootstrap']:,}")

    # ---- Table 1: P1/P3 primary scores against H0 -------------------------------------
    t1 = []
    for lang, name in present:
        r = row("start", "language", "all", lang)
        t = tag[lang]
        for key in ("precision", "recall", "f1", "f1_h0_mean", "f1_margin", "f1_margin_ci_low", "f1_margin_ci_high",
                    "f1_ratio", "precision_h0_mean", "recall_h0_mean"):
            evidence.setdefault("primary", {}).setdefault(lang, {})[key] = r[key]
        evidence["primary"][lang]["f1_permutation_p"] = r["f1_permutation_p"]
        macro(f"Fone{t}", num(r["f1"]))
        macro(f"FoneNull{t}", num(r["f1_h0_mean"]))
        macro(f"Margin{t}", signed(r["f1_margin"]))
        macro(f"MarginCI{t}", ci(r["f1_margin_ci_low"], r["f1_margin_ci_high"]))
        macro(f"Prec{t}", num(r["precision"]))
        macro(f"Rec{t}", num(r["recall"]))
        macro(f"Pval{t}", pval(r["f1_permutation_p"], perms))
        macro(f"PrecRec{t}", "above" if r["precision"] > r["recall"] else "below")
        macro(f"POne{t}", {"above": "exceeds", "below": "falls below",
                             "indistinguishable": "is statistically indistinguishable from",
                             "unresolved": "cannot be compared with"}[verdict_vs_zero(r["f1_margin_ci_low"], r["f1_margin_ci_high"])])
        t1.append(f"{name} & {calibration['selected_k'][lang]} & {r['n_files']:,} & {num(r['precision'])} & {num(r['recall'])} & "
                  f"{num(r['f1'])} & {num(r['f1_h0_mean'])} & {signed(r['f1_margin'])} {ci(r['f1_margin_ci_low'], r['f1_margin_ci_high'])} & "
                  f"{pval(r['f1_permutation_p'], perms)} \\\\")

    # ---- Table 2: structure-blind baselines (R2 / P3 confound) -------------------------
    t2 = []
    for lang, name in present:
        t = tag[lang]
        b = {p: base(lang, "start", p) for p in ("blt", "newline", "indent_change", "word_start")}
        cells = [num(b[p]["f1"]) if b[p] else "--" for p in ("blt", "newline", "indent_change", "word_start")]
        strongest = max((p for p in ("newline", "indent_change", "word_start") if b[p]), key=lambda p: b[p]["f1"])
        s = b[strongest]
        evidence.setdefault("baselines", {})[lang] = {p: b[p] for p in b}
        evidence["baselines"][lang]["strongest"] = strongest
        label = {"newline": "newline", "indent_change": "indentation", "word_start": "word-start"}[strongest]
        macro(f"Strongest{t}", label)
        macro(f"StrongestFone{t}", num(s["f1"]))
        macro(f"VsBase{t}", signed(s["blt_minus_baseline_f1"]))
        macro(f"VsBaseCI{t}", ci(s["ci_low"], s["ci_high"]))
        macro(f"RTwo{t}", {"above": "outperforms", "below": "underperforms",
                             "indistinguishable": "is statistically indistinguishable from",
                             "unresolved": "cannot be compared with"}[verdict_vs_zero(s["ci_low"], s["ci_high"])])
        t2.append(f"{name} & " + " & ".join(cells) + f" & {signed(s['blt_minus_baseline_f1'])} {ci(s['ci_low'], s['ci_high'])} \\\\")

    # ---- Table 3: P2 starts vs ends -------------------------------------------------
    t3 = []
    for lang, name in present:
        t = tag[lang]
        s, e = row("start", "language", "all", lang), row("end", "language", "all", lang)
        evidence.setdefault("p2", {})[lang] = {"start": {k: s[k] for k in ("precision", "recall", "f1", "f1_margin")},
                                               "end": {k: e[k] for k in ("precision", "recall", "f1", "f1_margin")}}
        macro(f"EndMargin{t}", signed(e["f1_margin"]))
        macro(f"PTwo{t}", "larger" if s["f1_margin"] > e["f1_margin"] else "not larger")
        t3.append(f"{name} & {num(s['recall'])} & {num(s['f1'])} & {signed(s['f1_margin'])} & "
                  f"{num(e['recall'])} & {num(e['f1'])} & {signed(e['f1_margin'])} {ci(e['f1_margin_ci_low'], e['f1_margin_ci_high'])} \\\\")

    # ---- Table 4: P4 by constituent type (recall, since predictions are shared) -------
    taxonomy = json.loads((ROOT / "taxonomy.json").read_text())
    t4, p4 = [], {}
    for lang, name in CODE:
        types = sorted(r["stratum"] for r in rows if r["kind"] == "start" and r["axis"] == "node_type" and r["language"] == lang)
        groups = {"deterministic_opener": [], "open_ended": []}
        for nt in types:
            r = row("start", "node_type", lang, nt)
            cls = taxonomy.get(nt)
            if cls not in groups or r["tp"] + r["fn"] == 0:
                continue
            groups[cls].append(r["recall_margin"])
            t4.append(f"{name} & \\texttt{{{texescape(nt)}}} & {'det.' if cls == 'deterministic_opener' else 'open'} & "
                      f"{r['tp'] + r['fn']:,} & {num(r['recall'])} & {num(r['recall_h0_mean'])} & {signed(r['recall_margin'])} \\\\")
        means = {c: (sum(v) / len(v) if v else None) for c, v in groups.items()}
        p4[lang] = {"mean_recall_margin": means, "n_types": {c: len(v) for c, v in groups.items()}}
        t = tag[lang]
        macro(f"DetMargin{t}", signed(means["deterministic_opener"]))
        macro(f"OpenMargin{t}", signed(means["open_ended"]))
        ok = None not in means.values() and means["deterministic_opener"] < means["open_ended"]
        macro(f"PFour{t}", "smaller" if ok else "not smaller")
    evidence["p4"] = p4

    # ---- P3 BPP variability ---------------------------------------------------------
    with (run / "evaluation/diagnostics.csv").open() as f:
        diag = list(csv.DictReader(f))
    bpp = {}
    for lang, name in present:
        means = [float(d["mean_bpp"]) for d in diag if d["language"] == lang and d["mean_bpp"]]
        within = [float(d["bpp_variance"]) for d in diag if d["language"] == lang and d["bpp_variance"]]
        mu = sum(means) / len(means)
        bpp[lang] = {"mean_bpp": mu, "sd_file_mean_bpp": (sum((x - mu) ** 2 for x in means) / (len(means) - 1)) ** .5,
                     "mean_within_file_bpp_variance": sum(within) / len(within), "n_files": len(means)}
        t = tag[lang]
        macro(f"Bpp{t}", num(bpp[lang]["mean_bpp"], 2))
        macro(f"BppSd{t}", num(bpp[lang]["sd_file_mean_bpp"], 2))
        macro(f"BppVar{t}", num(bpp[lang]["mean_within_file_bpp_variance"], 1))
    evidence["bpp"] = bpp
    code_dom, prose_dom = row("start", "domain", "all", "code"), row("start", "domain", "all", "prose")
    if code_dom and prose_dom:
        macro("MarginCode", signed(code_dom["f1_margin"]))
        macro("MarginProse", signed(prose_dom["f1_margin"]))
        macro("PThree", "larger" if code_dom["f1_margin"] > prose_dom["f1_margin"] else "not larger")
        evidence["p3_domain"] = {"code": code_dom["f1_margin"], "prose": prose_dom["f1_margin"]}

    # ---- Table 5: O5 ------------------------------------------------------------------
    t5 = []
    if o5:
        sel = json.loads((run / "o5_selection.json").read_text()) if (run / "o5_selection.json").exists() else {}
        macro("OFiveRegions", f"{o5['n_regions']:,}")
        macro("OFiveFiles", f"{o5['n_files']:,}")
        macro("OFiveEligible", f"{sel.get('n_eligible_files', 0):,}")
        macro("OFiveInference", texescape(o5["inference"]))
        for label, res in (("boundary density", o5), ("inverse BPP", o5b)):
            if not res:
                continue
            for term, pretty in (("is_memory_unsafe", "memory-unsafe"), ("mean_identifier_entropy_nats", "identifier entropy (nats)"),
                                 ("identifier_byte_proportion", "identifier byte share"), ("region_length", "region length (bytes)")):
                c = res["coefficients"][term]
                lo, hi = (res.get("confidence_intervals", {}).get(term) or [None, None])
                p = res.get("pvalues", {}).get(term)
                t5.append(f"{label} & {pretty} & {c:+.2e} & {ci(lo, hi, 4) if lo is not None else '--'} & "
                          f"{'--' if p is None else f'{p:.3f}'} \\\\")
        lo, hi = (o5.get("confidence_intervals", {}).get("is_memory_unsafe") or [None, None])
        macro("OFiveUnsafe", f"{o5['coefficients']['is_memory_unsafe']:+.2e}")
        macro("OFiveUnsafeCI", ci(lo, hi, 4) if lo is not None else "unavailable")
        macro("OFive", {"above": "a positive residual", "below": "a negative residual",
                        "indistinguishable": "no detectable residual", "unresolved": "no inferential result"}[
            verdict_vs_zero(lo, hi) if lo is not None else "unresolved"])
        evidence["o5"] = {"boundary_density": o5, "inverse_bpp": o5b, "selection": sel}

    # ---- secondary: span IoU (escape_scoring/iou.py) -------------------------------------
    t6 = []
    iou_path = run / "iou.json"
    if iou_path.exists():
        iou = json.loads(iou_path.read_text())
        if iou["calibration_record_sha256"] != calibration["record_sha256"]:
            raise SystemExit("iou.json was not computed on this calibration record's split")
        evidence["inputs"]["iou.json"] = sha(iou_path)
        evidence["iou"] = iou["rows"]
        for r in iou["rows"]:
            t = tag[r["language"]]
            macro(f"IouBlt{t}", num(r["mean_best_iou_blt"]))
            macro(f"IouNull{t}", num(r["mean_best_iou_h0"]))
            macro(f"IouLine{t}", num(r["mean_best_iou_line"]))
            macro(f"IouWord{t}", num(r["mean_best_iou_word"]))
            macro(f"IouVsNull{t}", signed(r["blt_minus_h0"]))
            macro(f"IouVsNullCI{t}", ci(r["blt_minus_h0_ci_low"], r["blt_minus_h0_ci_high"]))
        order = {k: i for i, (k, _) in enumerate(LANGS)}
        names = dict(LANGS)
        for r in sorted(iou["rows"], key=lambda r: order[r["language"]]):
            t6.append(f"{names[r['language']]} & {r['n_files']:,} & {num(r['mean_best_iou_blt'])} & {num(r['mean_best_iou_h0'])} & "
                      f"{num(r['mean_best_iou_line'])} & {num(r['mean_best_iou_word'])} & "
                      f"{signed(r['blt_minus_h0'])} {ci(r['blt_minus_h0_ci_low'], r['blt_minus_h0_ci_high'])} \\\\")
        macro("IouResamples", f"{iou['resamples']:,}")

    # ---- Objective 4: robustness to code noise (escape_scoring/o4.py) ---------------------
    t7 = []
    o4_path = run / "o4.json"
    if o4_path.exists():
        o4 = json.loads(o4_path.read_text())
        if o4["calibration_record_sha256"] != calibration["record_sha256"]:
            raise SystemExit("o4.json was not computed under this calibration record")
        evidence["inputs"]["o4.json"] = sha(o4_path)
        evidence["o4"] = o4["rows"]
        pretty = {"missing_semicolon": "missing \\texttt{;}", "indentation": "indentation",
                  "case_inversion": "camel$\\leftrightarrow$snake", "keyword_typo": "keyword typo", "clean": "clean"}
        short = {"missing_semicolon": "Semi", "indentation": "Indent", "case_inversion": "Case", "keyword_typo": "Typo"}
        builds = {(b["language"], b["condition"]): b for b in o4["build"]["summary"]}
        macro("OFourFiles", f"{o4['build']['files_per_language']:,}")
        macro("OFourPerm", f"{o4['permutations']:,}")
        macro("OFourRate", f"{int(round(100 * o4['build']['rate']))}\\%")
        macro("OFourCaseRate", f"{int(round(100 * o4['build']['case_rate']))}\\%")
        names = dict(LANGS)
        for lang in ("python", "cpp"):
            rows_l = [r for r in o4["rows"] if r["language"] == lang]
            clean = next(r for r in rows_l if r["condition"] == "clean")
            t = tag[lang]
            macro(f"OFourClean{t}", signed(clean["f1_margin"]))
            noisy = [r for r in rows_l if r["condition"] != "clean"]
            worst = min(noisy, key=lambda r: r["delta_margin"])
            macro(f"OFourWorst{t}", pretty[worst["condition"]])
            macro(f"OFourWorstDm{t}", signed(worst["delta_margin"]))
            macro(f"OFourWorstDmCI{t}", ci(*worst["delta_margin_ci"]))
            macro(f"OFourMinMargin{t}", signed(min(r["f1_margin"] for r in noisy)))
            for r in noisy:
                macro(f"OFourDbpp{t}{short[r['condition']]}", signed(r["delta_bpp"], 2))
            for r in [clean, *sorted(noisy, key=lambda r: r["condition"])]:
                b = builds.get((lang, r["condition"]), {})
                cells = (f"{names[lang]} & {pretty[r['condition']]} & {b.get('edits', 0):,} & {signed(r['f1_margin'])} & ")
                if r["condition"] == "clean":
                    cells += f"-- & {num(r['mean_bpp'], 2)} & -- \\\\"
                else:
                    cells += (f"{signed(r['delta_margin'])} {ci(*r['delta_margin_ci'])} & {num(r['mean_bpp'], 2)} & "
                              f"{signed(r['delta_bpp'], 2)} {ci(*r['delta_bpp_ci'], 2)} \\\\")
                t7.append(cells)

    # ---- fidelity rerun and extraction completeness -------------------------------------
    if args.fidelity.exists():
        fid = json.loads(args.fidelity.read_text())
        macro("FidelityAda", "passed" if fid.get("gates_effective_all_pass") else "did not pass")
        evidence["fidelity_ada"] = {"gates_effective_all_pass": fid.get("gates_effective_all_pass"),
                                    "gates": fid.get("gates"), "gates_effective": fid.get("gates_effective")}
    else:
        macro("FidelityAda", "was not rerun")
    # Completion is judged from artifacts, not the status event log: 30_score.sbatch stops
    # unless every selected file has complete, consistent artifacts, and convert-legacy
    # refuses any missing input, so every sample in the frozen split was extracted.
    # (extraction_status.parquet summarises status.jsonl, whose lines concurrent NFS
    # appends tore; it is kept for the record only.)
    macro("nExtracted", f"{len(split):,}")
    status_path = run / "extraction_status.parquet"
    if status_path.exists():
        import pandas as pd
        st = pd.read_parquet(status_path)
        evidence["extraction_event_log"] = st.status.value_counts().to_dict()

    # ---- Figure: F1 margin by AST depth --------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.03, 2.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    for (lang, name), color, marker in zip(CODE, ("#234f7d", "#b0582a"), ("o", "s")):
        pts = sorted((int(r["stratum"]), r) for r in rows if r["kind"] == "start" and r["axis"] == "ast_depth"
                     and r["language"] == lang and r["tp"] + r["fn"] >= 200)
        if not pts:
            continue
        x = [d for d, _ in pts]
        # Recall, not F1: every stratum shares the full prediction set, so per-stratum
        # precision (and F1) shrinks with the stratum's share of targets by construction.
        y = [r["recall_margin"] for _, r in pts]
        lo = [r["recall"] - r["recall_ci_low"] if r["recall_ci_low"] is not None else 0 for _, r in pts]
        hi = [r["recall_ci_high"] - r["recall"] if r["recall_ci_high"] is not None else 0 for _, r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], marker=marker, markersize=2.5, linewidth=1, capsize=1.5, color=color, label=name)
    ax.axhline(0, color="#888888", linewidth=.6)
    ax.set(xlabel="Raw tree-sitter depth of target start", ylabel="Recall $-$ mean $H_0$ recall")
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.legend(frameon=False, fontsize=8)
    if synthetic:
        ax.set_title(WATERMARK, fontsize=6, color="red")
    fig.savefig(OUT / "depth_margin.pdf", metadata={"Title": "Start-boundary F1 margin over H0 by depth",
                                                   "Subject": "recall margin; strata with >= 200 target starts; file-bootstrap recall intervals"})
    plt.close(fig)

    # ---- cross-language summary phrases (data-driven; fall back to per-language lists) --
    def summarize(verbs, names_):
        verbs = [(v, n) for v, n in zip(verbs, names_) if v is not None]
        if not verbs:
            return None
        if len({v for v, _ in verbs}) == 1:
            scope = {1: f"in {verbs[0][1]}", 2: "in both code languages", 3: "in all three languages"}[len(verbs)]
            return f"{verbs[0][0]} {scope}"
        return "; ".join(f"{v} in {n}" for v, n in verbs)
    langs3 = [(l, n) for l, n in LANGS if row("start", "language", "all", l)]
    prim = {l: row("start", "language", "all", l) for l, _ in langs3}
    macro("POneAll", summarize([{"above": "exceeds the null", "below": "falls below the null",
                                 "indistinguishable": "is indistinguishable from the null", "unresolved": None}[
        verdict_vs_zero(prim[l]["f1_margin_ci_low"], prim[l]["f1_margin_ci_high"])] for l, _ in langs3], [n for _, n in langs3]))
    macro("RTwoAll", summarize([{"above": "outperforms it", "below": "underperforms it", "indistinguishable": "matches it",
                                 "unresolved": None}[verdict_vs_zero(evidence["baselines"][l][evidence["baselines"][l]["strongest"]]["ci_low"],
                                                                     evidence["baselines"][l][evidence["baselines"][l]["strongest"]]["ci_high"])]
                                for l, _ in langs3], [n for _, n in langs3]))
    macro("PrecRecAll", summarize(["is below recall" if prim[l]["precision"] < prim[l]["recall"] else "is above recall" for l, _ in langs3],
                                  [n for _, n in langs3]))
    p2 = evidence["p2"]
    macro("PTwoAll", summarize(["exceeds the end margin" if p2[l]["start"]["f1_margin"] > p2[l]["end"]["f1_margin"] else "does not exceed the end margin"
                                for l, _ in langs3], [n for _, n in langs3]))
    if "p3_domain" in evidence:
        macro("PThreeCmp", "larger than" if evidence["p3_domain"]["code"] > evidence["p3_domain"]["prose"] else "smaller than")
    p4m = evidence["p4"]
    macro("PFourAll", summarize([None if None in p4m[l]["mean_recall_margin"].values() else
                                 ("is larger" if p4m[l]["mean_recall_margin"]["deterministic_opener"] > p4m[l]["mean_recall_margin"]["open_ended"] else "is smaller")
                                 for l, _ in CODE], [n for _, n in CODE]))
    macro("PFourSupported", "support" if all(
        None not in p4m[l]["mean_recall_margin"].values()
        and p4m[l]["mean_recall_margin"]["deterministic_opener"] < p4m[l]["mean_recall_margin"]["open_ended"] for l, _ in CODE)
        else "do not support")

    # Every macro the paper uses gets a definition. Anything this run could not supply is
    # rendered as a visible marker that check_report.py rejects, never silently dropped.
    body = (args.report.resolve() / args.body).read_text()
    table_macros = {"TableMain", "TableBaselines", "TableStartEnd", "TableTypes", "TableOFive", "TableIoU", "TableOFour"}
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", body)) | {m for m in re.findall(r"\\([A-Za-z]+)", body) if m in table_macros}
    missing = sorted(m for m in used - table_macros - set(macros) if m[0].isupper() or m[0] in "nk")
    for name in missing:
        macros[name] = MISSING
    evidence["missing_macros"] = missing
    if missing:
        print("WARNING: no data for", ", ".join(missing))

    # ---- write LaTeX ------------------------------------------------------------------------
    mark = f"\\textcolor{{red}}{{\\textbf{{{WATERMARK}}}}}\\\\" if synthetic else ""
    lines = ["% Generated by reports/mid_submission/build_results.py -- do not edit by hand.",
             f"% Source run: {evidence['run']}", "\\providecommand{\\textcolor}[2]{#2}"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    lines += ["\\newcommand{\\TableMain}{", "\\begin{table*}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lrrrrrrlr}\\toprule",
              "Language & $k$ & Test files & P & R & F1 & $H_0$ F1 & $\\Delta$F1 [95\\% CI] & $p$ \\\\\\midrule",
              *t1, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Held-out entropy-boundary alignment with constituent starts (P1). $k$ was frozen on the "
              "calibration split. $H_0$ F1 is the permutation mean of the fixed-EOF density-matched null "
              f"({perms:,} permutations); intervals are paired file bootstraps; $p$ is the one-sided permutation value.}}",
              "\\label{tab:main}\\end{table*}}",
              "\\newcommand{\\TableBaselines}{", "\\begin{table}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrl}\\toprule",
              "Lang. & BLT & NL & Indent & Word & BLT$-$best [95\\% CI] \\\\\\midrule",
              *t2, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Start F1 of structure-blind baselines under the same targets, $k$ and matching: boundary after every newline (NL), "
              "at indentation changes, and at every word start. The last column compares BLT with the strongest baseline for that language.}",
              "\\label{tab:baselines}\\end{table}}",
              "\\newcommand{\\TableStartEnd}{", "\\begin{table}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrrl}\\toprule",
              " & \\multicolumn{3}{c}{Starts} & \\multicolumn{3}{c}{Ends} \\\\",
              "Lang. & R & F1 & $\\Delta$ & R & F1 & $\\Delta$ [95\\% CI] \\\\\\midrule",
              *t3, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Starts versus ends (P2). $\\Delta$ is F1 minus the mean $H_0$ F1.}",
              "\\label{tab:startend}\\end{table}}",
              "\\newcommand{\\TableTypes}{", "\\begin{table}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{llcrrrr}\\toprule",
              "Lang. & Type & Opener & Starts & R & $H_0$ R & $\\Delta$R \\\\\\midrule",
              *t4, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Start recall by constituent type (P4). Types are rematched separately, so rows are not additive. "
              "A start shared by several nodes is credited to the outermost one, which is why nested types such as Python "
              "\\texttt{assignment} have few starts. Opener class follows the empirically checked \\texttt{taxonomy.json}.}",
              "\\label{tab:types}\\end{table}}",
              "\\newcommand{\\TableIoU}{", "\\begin{table}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrrl}\\toprule",
              "Lang. & Files & BLT & $H_0$ & Line & Word & BLT$-H_0$ [95\\% CI] \\\\\\midrule",
              *t6, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Secondary metric: mean best single-patch IoU over unique constituent spans, held-out split "
              "(files with at least one span). $H_0$ averages "
              + macros.get("IouResamples", "--") + " fixed-EOF permutations; Line and Word are newline and word-start segmentations.}",
              "\\label{tab:iou}\\end{table}}",
              "\\newcommand{\\TableOFour}{", "\\begin{table*}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{llrrlrl}\\toprule",
              "Lang. & Noise & Edits & $\\Delta$F1 vs.\\ $H_0$ & Change vs.\\ clean [95\\% CI] & BPP & Change vs.\\ clean [95\\% CI] \\\\\\midrule",
              *t7, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Objective 4: robustness to code noise on the same held-out files per language. Each noise type is applied alone; "
              "targets are the clean constituents mapped to noised offsets; $k$ is the frozen value. "
              "The null uses " + macros.get("OFourPerm", "--") + " fixed-EOF permutations per file; intervals are paired file bootstraps of noised minus clean.}",
              "\\label{tab:o4}\\end{table*}}",
              "\\newcommand{\\TableOFive}{", "\\begin{table}[t]\\centering\\small", mark,
              "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{llrlr}\\toprule",
              "Outcome & Term & Coef. & 95\\% CI & $p$ \\\\\\midrule",
              *t5, "\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{O5 region regressions with file fixed effects and statement-type controls; file-clustered intervals.}",
              "\\label{tab:o5}\\end{table}}"]
    (OUT / "results.tex").write_text("\n".join(lines) + "\n")
    evidence["macros"] = macros
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {OUT / 'results.tex'} from {evidence['run']}{' (SYNTHETIC TEST)' if synthetic else ''}")


if __name__ == "__main__":
    main()
