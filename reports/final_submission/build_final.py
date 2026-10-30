"""Generate every number, table and figure of the final report from results/final/.

Two stages, both reading only retained result files (no model, corpus or null is re-run):
  1. reports/mid_submission/build_results.py, unchanged, on the frozen C-v1 run
     (results/final/c-v1-2721954) -> generated/results.tex: the P1-P4, IoU, O4 and O5
     macros/tables of the frozen held-out evaluation.
  2. this file -> generated/final.tex (macros named \\x..., tables \\xTable...) and
     generated/*.pdf for the final-submission analyses and the extension, plus
     generated/final_evidence.json listing every input file with its sha256.
An input that does not exist yet renders as a visible MISSING marker (check_final.py
rejects it), never as a silently dropped number.

  python3 reports/final_submission/build_final.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.size": 8})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "generated"
FINAL = ROOT / "results/final"
FROZEN = FINAL / "c-v1-2721954"
ANALYSIS = FINAL / "analysis"
MISSING = "\\textbf{MISSING}"
LANGS = [("python", "Python", "Py"), ("cpp", "C++", "Cpp"), ("english", "English", "En")]
DOM = {"python": "py", "cpp": "cpp", "english": "prose"}
COLORS = {"python": "#234f7d", "cpp": "#b0582a", "english": "#5b8c3a"}
SEED = 20260927
TEACHER = "swa512-float32-t1.3354-mplnone-91aa6b8e"

sys.path.insert(0, str(ROOT / "reports/mid_submission"))
import build_results as mid  # noqa: E402

num, signed, ci, texescape = mid.num, mid.signed, mid.ci, mid.texescape
macros, tables, evidence = {}, {}, {"inputs": {}, "missing_inputs": []}


def load(path: Path):
    if not path.exists():
        evidence["missing_inputs"].append(str(path.relative_to(ROOT)))
        return None
    evidence["inputs"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def read_csv(path: Path):
    if not path.exists():
        evidence["missing_inputs"].append(str(path.relative_to(ROOT)))
        return None
    evidence["inputs"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


def macro(name, value):
    assert re.fullmatch(r"x[A-Za-z]+", name), name
    macros[name] = MISSING if value is None else value


def pct(x, d=0):
    return "--" if x is None else f"{100 * x:.{d}f}\\%"


def rng_for(*parts):
    key = hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "big")))


def ci_pair(lo_hi, d=3):
    return ci(lo_hi[0], lo_hi[1], d) if lo_hi else "--"


# --------------------------------------------------------------------------- C-v1 CSVs
def language_rows(scores_csv):
    rows = read_csv(scores_csv)
    if rows is None:
        return None
    return {r["stratum"]: {k: (float(v) if v not in ("", None) and k not in ("kind", "axis", "language", "stratum", "ci_method") else v)
                           for k, v in r.items()}
            for r in rows if r["kind"] == "start" and r["axis"] == "language"}


def per_file_start(path):
    rows = read_csv(path)
    if rows is None:
        return None
    out = {}
    for r in rows:
        if r["kind"] != "start":
            continue
        tp, fp, fn, h0 = float(r["tp"]), float(r["fp"]), float(r["fn"]), float(r["mean_h0_f1"])
        den = 2 * tp + fp + fn
        out[r["sample_id"]] = (r["language"], tp, h0 * den / 2, den)
    return out


def paired_margin_diff(a, b, lang, key, boot=2000):
    """margin(a) - margin(b) on the files both evaluated, paired file bootstrap; margins are
    pooled 2*(T - E[T_H0]) / sum(2T+FP+FN), exactly the frozen scorer's margin."""
    ids = sorted(s for s in a if s in b and a[s][0] == lang)
    A = np.array([a[s][1:] for s in ids])
    B = np.array([b[s][1:] for s in ids])

    def m(x, idx=None):
        x = x if idx is None else x[idx]
        return 2 * (x[..., 0].sum(-1) - x[..., 1].sum(-1)) / x[..., 2].sum(-1)
    idx = rng_for(key, lang, "paired_margin").integers(0, len(ids), (boot, len(ids)))
    d = m(A, idx) - m(B, idx)
    return float(m(A) - m(B)), [float(np.quantile(d, .025)), float(np.quantile(d, .975))], len(ids)


def baseline_rows(path):
    b = load(path)
    if b is None:
        return None
    return {(r["language"], r["proposer"]): r for r in b["rows"] if r["kind"] == "start"}


def strongest(brows, lang):
    cands = [brows[(lang, p)] for p in ("newline", "indent_change", "word_start") if (lang, p) in brows]
    return max(cands, key=lambda r: r["f1"])


PRETTY_BASE = {"newline": "newline", "indent_change": "indentation", "word_start": "word-start"}


# ====================================================================================
def tau_star_outputs(model):
    """Confirmatory outputs at tau*: one run (<model>-tau-STAR-*) when all languages share tau*,
    else one run per language (<model>-tau-STAR-<lang>-*). Returns (selected_k, rows, baselines)
    merged over languages, or None if any part is missing."""
    single = FINAL / f"{model}-tau-STAR-calibration.json"
    if single.exists():
        cal = load(single)
        rows = language_rows(FINAL / f"{model}-tau-STAR-eval/scores.csv")
        base = baseline_rows(FINAL / f"{model}-tau-STAR-baselines.json")
        return None if None in (cal, rows, base) else (cal["selected_k"], rows, base)
    k, rows, base = {}, {}, {}
    for lang, _, _ in LANGS:
        cal = load(FINAL / f"{model}-tau-STAR-{lang}-calibration.json")
        r = language_rows(FINAL / f"{model}-tau-STAR-{lang}-eval/scores.csv")
        b = baseline_rows(FINAL / f"{model}-tau-STAR-{lang}-baselines.json")
        if None in (cal, r, b) or lang not in r:
            return None
        k[lang], rows[lang] = cal["selected_k"][lang], r[lang]
        base.update({key: v for key, v in b.items() if key[0] == lang})
    return k, rows, base


def student_table():
    """Extension 1: the distilled student on the frozen test split (own calibration record)."""
    t = language_rows(FROZEN / "evaluation/scores.csv")
    s = language_rows(FINAL / "student-v1-eval/scores.csv")
    tb, sb = baseline_rows(FROZEN / "baselines.json"), baseline_rows(FINAL / "student-v1-baselines.json")
    tp, sp = per_file_start(FROZEN / "evaluation/per_file.csv"), per_file_start(FINAL / "student-v1-eval/per_file.csv")
    cal = load(FINAL / "student-v1-calibration.json")
    if None in (t, s, tb, sb, tp, sp, cal):
        tables["xTableStudent"] = MISSING
        return
    rows = []
    ev = {}
    for lang, name, tag in LANGS:
        T, S = t[lang], s[lang]
        d, dci, n = paired_margin_diff(sp, tp, lang, "student_minus_teacher")
        sbest = strongest(sb, lang)
        ev[lang] = {"teacher": T, "student": S, "student_minus_teacher_margin": d, "ci": dci, "n_paired": n,
                    "student_vs_best_baseline": sbest}
        macro(f"xStuFone{tag}", num(S["f1"]))
        macro(f"xStuMargin{tag}", signed(S["f1_margin"]))
        macro(f"xStuMarginCI{tag}", ci(S["f1_margin_ci_low"], S["f1_margin_ci_high"]))
        macro(f"xStuRetain{tag}", pct(S["f1_margin"] / T["f1_margin"]))
        macro(f"xStuDiff{tag}", signed(d))
        macro(f"xStuDiffCI{tag}", ci_pair(dci))
        macro(f"xStuVsBase{tag}", signed(sbest["blt_minus_baseline_f1"]))
        macro(f"xStuVsBaseCI{tag}", ci(sbest["ci_low"], sbest["ci_high"]))
        macro(f"xStuK{tag}", str(cal["selected_k"][lang]))
        rows.append(f"{name} & {num(T['precision'])} & {num(T['recall'])} & {num(T['f1'])} & {signed(T['f1_margin'])} & "
                    f"{num(S['precision'])} & {num(S['recall'])} & {num(S['f1'])} & {signed(S['f1_margin'])} & "
                    f"{signed(d)} {ci_pair(dci)} & {signed(sbest['blt_minus_baseline_f1'])} \\\\")
    evidence["student_test"] = ev
    tables["xTableStudent"] = "\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lrrrrrrrrlr}\\toprule",
        " & \\multicolumn{4}{c}{Teacher (BLT patcher, 99.5M)} & \\multicolumn{4}{c}{Student (distilled, 14.3M)} & & \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Lang. & P & R & F1 & $\\Delta$ & P & R & F1 & $\\Delta$ & Student$-$teacher $\\Delta$ [95\\% CI] & vs.\\ best base. \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{The distilled student on the frozen held-out split, scored by the frozen C-v1 pipeline with its own "
        "calibration record (which also selected $k=0$ in every language). $\\Delta$ is F1 minus the mean $H_0$ F1 "
        "(10{,}000 fixed-EOF permutations); the student$-$teacher interval is a paired file bootstrap; the last column is "
        "the student's F1 minus the strongest structure-blind baseline (indentation for Python, word-start otherwise).}",
        "\\label{tab:student}\\end{table*}"])
    # BPP
    for name, path in (("Tea", FROZEN / "evaluation/diagnostics.csv"), ("Stu", FINAL / "student-v1-eval/diagnostics.csv")):
        diag = read_csv(path)
        for lang, _, tag in LANGS:
            v = [float(d["mean_bpp"]) for d in diag if d["language"] == lang]
            macro(f"x{name}Bpp{tag}", num(sum(v) / len(v), 2))


def t512_table():
    d = load(FINAL / "teacher512_subset_scores.json")
    if d is None:
        tables["xTableTfiveTwelve"] = MISSING
        return
    rows, ev = [], {}
    for dom, name, tag in (("py", "Python", "Py"), ("cpp", "C++", "Cpp")):
        R = d["domains"][dom]["rows"]
        nat, ctl, stu = R[TEACHER], R["teacher-512-ctl"], R["student-v1-14326k"]
        a, b = np.asarray(nat["per_file"], float), np.asarray(ctl["per_file"], float)
        rng = rng_for("t512", dom, "paired_f1")
        idx = rng.integers(0, len(a), (2000, len(a)))

        def f1(x):
            return 2 * x[..., 0].sum(-1) / (2 * x[..., 0].sum(-1) + x[..., 1].sum(-1) + x[..., 2].sum(-1))
        diff = f1(a[idx]) - f1(b[idx])
        dci = [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))]
        ev[dom] = {"native_minus_512": float(f1(a) - f1(b)), "ci": dci, "n_files": d["domains"][dom]["n_files"]}
        macro(f"xTfiveDiff{tag}", signed(float(f1(a) - f1(b))))
        macro(f"xTfiveDiffCI{tag}", ci_pair(dci))
        macro(f"xTfiveN{tag}", f"{d['domains'][dom]['n_files']:,}")
        gap = nat["f1"] - stu["f1"]
        ev[dom]["native_minus_student"] = gap
        macro(f"xTfiveStuGap{tag}", signed(gap))
        macro(f"xTfiveShare{tag}", pct(float(f1(a) - f1(b)) / gap))
        rows.append(f"{name} & {d['domains'][dom]['n_files']} & {num(nat['f1'])} & {signed(nat['f1_margin'])} & "
                    f"{num(ctl['f1'])} & {signed(ctl['f1_margin'])} & {num(stu['f1'])} & {signed(stu['f1_margin'])} & "
                    f"{signed(float(f1(a) - f1(b)))} {ci_pair(dci)} \\\\")
    evidence["teacher512"] = ev
    tables["xTableTfiveTwelve"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrrrrl}\\toprule",
        " & & \\multicolumn{2}{c}{Native} & \\multicolumn{2}{c}{Teacher-512} & \\multicolumn{2}{c}{Student} & \\\\",
        "Lang. & Files & F1 & $\\Delta$ & F1 & $\\Delta$ & F1 & $\\Delta$ & Native$-$512 F1 [95\\% CI] \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{Context control on the pre-registered subset (300 test files per code language, 600--2{,}000 bytes): the "
        "frozen teacher (native 512-byte sliding window), the same teacher restricted to the student's fresh 512-byte windows, "
        "and the student. $k=0$, 10{,}000 permutations.}",
        "\\label{tab:t512}\\end{table}"])


def tau_section():
    taus = [0.8, 1.0, 1.335442066192627, 1.7, 2.0, 2.5]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), constrained_layout=True, sharey=False)
    ev = {}
    for ax, model in zip(axes, ("teacher", "student")):
        rec = load(FINAL / f"{model}_tau_star.json")
        if rec is None:
            ax.text(.5, .5, f"{model}: sweep pending", ha="center", va="center", transform=ax.transAxes)
            for _, _, tag in LANGS:
                macro(f"xTauStar{model.capitalize()}{tag}", None)
            continue
        ev[model] = rec
        for lang, name, tag in LANGS:
            g = rec["languages"][lang]["grid"]
            ax.plot([r["tau"] for r in g], [r["median_margin_frozen"] for r in g], marker="o", markersize=2.5,
                    linewidth=1, color=COLORS[lang], label=name)
            star = rec["languages"][lang]["tau_star_frozen"]
            macro(f"xTauStar{model.capitalize()}{tag}", f"{star:g}")
            ckpt = next(r for r in g if abs(r["tau"] - 1.335442066192627) < 1e-9)
            best = next(r for r in g if r["tau"] == star)
            macro(f"xTauGain{model.capitalize()}{tag}", signed(best["median_margin_frozen"] - ckpt["median_margin_frozen"]))
        ax.axvline(1.335442066192627, color="#888888", linewidth=.6, linestyle=":")
        ax.set(xlabel="threshold $\\tau$ (nats)", title=f"{model}: calibration split",
               ylabel="median per-file F1 $-$ $H_0$" if model == "teacher" else None)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].legend(frameon=False, fontsize=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tau_sweep.pdf")
    plt.close(fig)
    evidence["tau_sweep"] = ev
    # confirmatory test runs at tau*
    rows = []
    for model, src_scores, src_base in (("teacher", FROZEN / "evaluation/scores.csv", FROZEN / "baselines.json"),
                                        ("student", FINAL / "student-v1-eval/scores.csv", FINAL / "student-v1-baselines.json")):
        out = tau_star_outputs(model)
        ref = language_rows(src_scores)
        rec = ev.get(model)
        if out is None or ref is None or rec is None:
            for _, _, tag in LANGS:
                macro(f"xTauConf{model.capitalize()}{tag}", None)
            rows.append(f"{model} & \\multicolumn{{7}}{{c}}{{{MISSING}}} \\\\")
            continue
        sel_k, star, base = out
        cal = {"selected_k": sel_k}
        # paired gain of the tau* run over the checkpoint-tau run on the same test files
        ref_pf = per_file_start(src_scores.parent / "per_file.csv")
        star_pf = {}
        for path in [FINAL / f"{model}-tau-STAR-eval/per_file.csv"] + \
                [FINAL / f"{model}-tau-STAR-{lang}-eval/per_file.csv" for lang, _, _ in LANGS]:
            if path.exists():
                star_pf.update(per_file_start(path))
        for lang, _, tag in LANGS:
            if ref_pf and star_pf:
                g, gci, _ = paired_margin_diff(star_pf, ref_pf, lang, f"{model}_tau_gain")
                macro(f"xTauPairGain{model.capitalize()}{tag}", signed(g))
                macro(f"xTauPairGainCI{model.capitalize()}{tag}", ci_pair(gci))
        for lang, name, tag in LANGS:
            S, R = star[lang], ref[lang]
            b = strongest(base, lang)
            macro(f"xTauConf{model.capitalize()}{tag}", signed(S["f1_margin"]))
            macro(f"xTauConfF{model.capitalize()}{tag}", num(S["f1"]))
            macro(f"xTauConfVsBase{model.capitalize()}{tag}", signed(b["blt_minus_baseline_f1"]))
            macro(f"xTauK{model.capitalize()}{tag}", str(cal["selected_k"][lang]))
            rows.append(f"{model.capitalize()} & {name} & {rec['languages'][lang]['tau_star_frozen']:g} & {cal['selected_k'][lang]} & "
                        f"{num(R['f1'])} & {num(S['f1'])} & {signed(S['f1_margin'])} {ci(S['f1_margin_ci_low'], S['f1_margin_ci_high'])} & "
                        f"{signed(b['blt_minus_baseline_f1'])} {ci(b['ci_low'], b['ci_high'])} \\\\")
    tables["xTableTau"] = "\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{llrrrrll}\\toprule",
        "Model & Lang. & $\\tau^*$ & $k^*$ & F1 at $\\tau{=}1.335$ & F1 at $\\tau^*$ & $\\Delta$ at $\\tau^*$ [95\\% CI] & "
        "vs.\\ best baseline [95\\% CI] \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{The single confirmatory test-split run at the pre-registered $\\tau^*$ (argmax over "
        "$\\{0.8, 1.0, 1.335, 1.7, 2.0, 2.5\\}$ of the median per-file margin on calibration files only, ties to the smaller "
        "$\\tau$), with $k$ re-frozen on the calibration split under $\\tau^*$ by the frozen criterion. The test split was "
        "scored once per model.}",
        "\\label{tab:tau}\\end{table*}"])


def k_sweep():
    rec = load(FROZEN / "calibration.json")
    rows, ev = [], {}
    sens = {k: load(ANALYSIS / f"teacher_test_k{k}.json") for k in (1, 2)}
    test0 = language_rows(FROZEN / "evaluation/scores.csv")
    for lang, name, tag in LANGS:
        cells = []
        ev[lang] = {}
        for k in rec["candidates"]:
            m = [c["margin"] for c in rec["curves"] if c["language"] == lang and c["k"] == k]
            ev[lang][str(k)] = float(np.median(m))
            cells.append(signed(float(np.median(m))))
        test = [signed(test0[lang]["f1_margin"])]
        for k in (1, 2):
            s = sens[k]
            if s is None:
                test.append(MISSING)
                continue
            r = s["domains"][DOM[lang]]["rows"][TEACHER]
            test.append(signed(r["f1_margin"]))
            ev[lang][f"test_k{k}"] = {"f1": r["f1"], "f1_margin": r["f1_margin"],
                                      "f1_minus_best_baseline": r["f1_minus_best_baseline"]}
            macro(f"xTestKMargin{tag}{'I' * k}", signed(r["f1_margin"]))
            macro(f"xTestKF{tag}{'I' * k}", num(r["f1"]))
            macro(f"xTestKVsBase{tag}{'I' * k}", signed(r["f1_minus_best_baseline"]))
        rows.append(f"{name} & " + " & ".join(cells) + " & " + " & ".join(test) + " \\\\")
    evidence["k_sweep"] = ev
    tables["xTableK"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrrrrr}\\toprule",
        " & \\multicolumn{5}{c}{Calibration: median per-file $\\Delta$} & \\multicolumn{3}{c}{Test: pooled $\\Delta$} \\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}",
        "Lang. & $k{=}0$ & 1 & 2 & 4 & 8 & 0 & 1 & 2 \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{Tolerance sweep for the teacher. Calibration columns are the frozen record's selection curve "
        "(the criterion picks $k=0$ everywhere); test columns at $k=1,2$ are sensitivity only (1{,}000 permutations) and "
        "select nothing.}",
        "\\label{tab:k}\\end{table}"])


def efficiency():
    e = load(ANALYSIS / "efficiency.json")
    if e is None:
        tables["xTableEff"] = MISSING
        for m in ("xEffForward", "xEffRolling", "xEffNative"):
            macro(m, None)
        return
    f, r, n = e["forward"], e["rolling"], e["teacher_native_swa512_single_pass"]
    rows = []
    for who, name in (("teacher", "Teacher"), ("student", "Student")):
        b32 = f[who]["B32"]
        rows.append(f"{name} & {f[who]['n_params'] / 1e6:.1f}M & {b32['tokens_per_s']:,.0f} & {b32['peak_mem_mib']:,.0f} & "
                    f"{r[who]['bytes_per_s']:,.0f} & {r[who]['peak_mem_mib']:,.0f} \\\\")
    rows.append(f"Teacher, native single pass & & {n['tokens_per_s_8192_chunk_swa512']:,.0f} & {n['peak_mem_mib']:,.0f} & -- & -- \\\\")
    macro("xEffForward", f"{e['speedups']['forward_B32_tokens_per_s_student_over_teacher']:.1f}")
    macro("xEffRolling", f"{e['speedups']['rolling_bytes_per_s_student_over_teacher']:.1f}")
    macro("xEffNative", f"{n['tokens_per_s_8192_chunk_swa512']:,.0f}")
    macro("xEffGpu", texescape(e["gpu"].replace("NVIDIA GeForce ", "")))
    evidence["efficiency"] = e
    tables["xTableEff"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrrr}\\toprule",
        " & & \\multicolumn{2}{c}{Forward, $32{\\times}513$} & \\multicolumn{2}{c}{Rolling extraction} \\\\",
        "Model & Params & tok/s & MiB & bytes/s & MiB \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        f"\\caption{{Inference cost on one {texescape(e['gpu'].replace('NVIDIA GeForce ', ''))}, float32. Rolling extraction "
        "(one fresh 512-byte window per byte) is how both student runs were extracted; the native teacher processes a file "
        "in one sliding-window pass, which the student architecture does not implement.}",
        "\\label{tab:eff}\\end{table}"])


def alpha1():
    a = load(ROOT / "results/distill/alpha1/eval_alpha1/distill_eval.json")
    d = load(ROOT / "results/distill/alpha1/eval_distilled/distill_eval.json")
    if a is None or d is None:
        tables["xTableAlpha"] = MISSING
        macro("xAlphaFone", None)
        macro("xDistFone", None)
        return
    rows = []
    for label, src, key in (("Distilled ($\\alpha{=}0.1$)", d, "trained"), ("No teacher ($\\alpha{=}1$)", a, "trained"),
                            ("Untrained", d, "untrained_baseline")):
        r = src[key]["all"]
        rows.append(f"{label} & {num(r['boundary_f1'])} & {num(r['entropy_pearson_r'])} & "
                    f"{num(r['kl_teacher_student_nats_per_byte'])} & {num(r['bits_per_byte_student'], 2)} \\\\")
    ws = d["trained"]["all"].get("word_start_vs_teacher_f1")
    rows.append(f"Word starts (no model) & {num(ws)} & -- & -- & -- \\\\")
    macro("xWordVsTeacherFone", num(ws))
    for dom in ("py", "cpp", "prose"):
        macro(f"xAlphaFone{dom.capitalize()}", num(a["trained"][dom]["boundary_f1"]))
        macro(f"xDistFone{dom.capitalize()}", num(d["trained"][dom]["boundary_f1"]))
    macro("xAlphaFone", num(a["trained"]["all"]["boundary_f1"]))
    macro("xDistFone", num(d["trained"]["all"]["boundary_f1"]))
    macro("xAlphaBpb", num(a["trained"]["all"]["bits_per_byte_student"], 2))
    macro("xDistBpb", num(d["trained"]["all"]["bits_per_byte_student"], 2))
    macro("xTeacherBpb", num(d["trained"]["all"]["bits_per_byte_teacher"], 2))
    evidence["alpha1"] = {"alpha1": a, "distilled": d}
    tables["xTableAlpha"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrr}\\toprule",
        "Student & Boundary F1 & $r$ & KL & bits/byte \\\\\\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{adjustbox}",
        f"\\caption{{What distillation adds: agreement with the teacher on held-out extension-corpus windows "
        f"(F1 of the student's boundaries against the teacher's at $\\tau=1.335$; Pearson $r$ and KL of per-byte entropy). "
        f"Same data, architecture, seed and schedule; only the loss differs. Teacher bits/byte: "
        f"{num(d['trained']['all']['bits_per_byte_teacher'], 2)}.}}",
        "\\label{tab:alpha}\\end{table}"])


def holdout():
    h = load(ANALYSIS / "english_holdout_scores.json")
    if h is None:
        tables["xTableHoldout"] = MISSING
        for m in ("xHoldFiles", "xHoldTeaMargin", "xHoldStuMargin", "xHoldTeaVsWord", "xHoldStuVsWord"):
            macro(m, None)
        return
    R = h["domains"]["prose"]["rows"]
    rows = []
    for run, name, tag in (("holdout-teacher-swa512", "Teacher", "Tea"), ("holdout-student-v1", "Student", "Stu")):
        r = R[run]
        rows.append(f"{name} & {num(r['precision'])} & {num(r['recall'])} & {num(r['f1'])} [{num(r['f1_ci_low'])}, {num(r['f1_ci_high'])}] & "
                    f"{num(r['f1_h0_mean'])} & {signed(r['f1_margin'])} & {mid.pval(r['f1_permutation_p'], h['perms'])} & "
                    f"{signed(r['f1_minus_best_baseline'])} {ci_pair(r['f1_minus_best_baseline_ci'])} \\\\")
        macro(f"xHold{tag}Margin", signed(r["f1_margin"]))
        macro(f"xHold{tag}Fone", num(r["f1"]))
        macro(f"xHold{tag}VsWord", signed(r["f1_minus_best_baseline"]))
        macro(f"xHold{tag}VsWordCI", ci_pair(r["f1_minus_best_baseline_ci"]))
    w = R["baseline_word_start"]
    macro("xHoldWord", num(w["f1"]))
    macro("xHoldFiles", f"{h['domains']['prose']['n_files']:,}")
    macro("xHoldTargets", f"{h['domains']['prose']['n_targets']:,}")
    evidence["holdout"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("per_file", "per_file_f1_h0_mean")} for k, v in R.items()}
    tables["xTableHoldout"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrlrrrl}\\toprule",
        "Model & P & R & F1 [95\\% CI] & $H_0$ & $\\Delta$ & $p$ & vs.\\ word-start [95\\% CI] \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        f"\\caption{{The confirmatory English result: the pre-registered holdout of {h['domains']['prose']['n_files']:,} paragraphs "
        f"from 120 WikiText-103 articles unused by evaluation and distillation, scored once at the frozen $k=0$ "
        f"({h['perms']:,} permutations). Word-start F1 = {num(w['f1'])}.}}",
        "\\label{tab:holdout}\\end{table}"])


# =============================================================== proposal analyses
def p4_positions():
    d = load(ANALYSIS / "positions_teacher.json")
    if d is None:
        tables["xTableFour"] = MISSING
        for m in ("xPostKwDetPy", "xPostKwDetCpp", "xPostKwOpenPy", "xPostKwOpenCpp", "xPosCheck"):
            macro(m, None)
        return
    R = {(r["language"], r["analysis"], r["label"], r["group"]): r for r in d["rows"]}
    rows = []
    labels = (("start", "start $s$"), ("tok_interior", "rest of first token"), ("post_tok", "byte after token"),
              ("next_choice", "next non-space byte"))
    for lang, name, tag in LANGS[:2]:
        for lab, pretty in labels:
            cells = []
            for fam in ("deterministic_opener", "open_ended"):
                r = R.get((lang, "p4", lab, fam))
                cells.append("--" if r is None else
                             f"{num(r['rate'])} & {signed(r['margin'])} {ci(r['margin_ci_low'], r['margin_ci_high'])}")
            rows.append(f"{name if lab == 'start' else ''} & {pretty} & " + " & ".join(cells) + " \\\\")
        for lab, short in (("start", "Start"), ("tok_interior", "Kw"), ("post_tok", "PostKw"), ("next_choice", "Next")):
            for fam, fs in (("deterministic_opener", "Det"), ("open_ended", "Open")):
                r = R.get((lang, "p4", lab, fam))
                macro(f"x{short}{fs}{tag}", None if r is None else signed(r["margin"]))
                macro(f"x{short}Rate{fs}{tag}", None if r is None else num(r["rate"]))
                macro(f"x{short}{fs}CI{tag}", None if r is None else ci(r["margin_ci_low"], r["margin_ci_high"]))
        if lang == "python":
            rows.append("\\midrule")
    for lang, _, tag in LANGS[:2]:
        r = R.get((lang, "p4", "next_choice", "if_statement"))
        macro(f"xNextIf{tag}", None if r is None else signed(r["margin"]))
        rt = "return_statement"
        r = R.get((lang, "p4", "post_tok", rt))
        macro(f"xPostKwReturn{tag}", None if r is None else signed(r["margin"]))
        for o in (1, 2, 3, 4):
            r = R.get((lang, "p4", f"offset_{o}", "deterministic_opener"))
            macro(f"xOffDet{tag}{'IV' if o == 4 else 'I' * o}", None if r is None else num(r["rate"]))
        r = R.get((lang, "p4", "offset_1", "deterministic_opener"))
        macro(f"xOffNull{tag}", None if r is None else num(r["rate_h0_mean"]))
    macro("xPosCheck", "reproduce" if d.get("frozen_check_all_match") else "do NOT reproduce")
    evidence["p4_positions"] = {"frozen_check_all_match": d.get("frozen_check_all_match"), "n_files": d["n_files"]}
    tables["xTableFour"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{llrlrl}\\toprule",
        " & & \\multicolumn{2}{c}{Forced opener} & \\multicolumn{2}{c}{Open-ended} \\\\",
        "Lang. & Position & rate & $\\Delta$ [95\\% CI] & rate & $\\Delta$ [95\\% CI] \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{P4 follow-up: boundary rate at fixed positions relative to each constituent start $s$, against the "
        "same 10{,}000 null draws as the frozen scorer ($\\Delta$ = rate $-$ $H_0$ rate). The first token is the identifier "
        "run starting at $s$: the keyword (\\texttt{if}, \\texttt{for}, \\texttt{while}, \\texttt{def}) for forced openers.}",
        "\\label{tab:p4pos}\\end{table}"])
    # offset curve figure
    fig, axes = plt.subplots(1, 2, figsize=(3.1, 1.75), constrained_layout=True, sharey=True)
    for ax, (lang, name, _) in zip(axes, LANGS[:2]):
        for fam, style, lab in (("deterministic_opener", "-", "forced"), ("open_ended", "--", "open")):
            xs, ys, lo, hi = [], [], [], []
            for o in range(0, 7):
                r = R.get((lang, "p4", "start" if o == 0 else f"offset_{o}", fam))
                if r:
                    xs.append(o)
                    ys.append(r["rate"])
                    lo.append(r["rate"] - r["rate_ci_low"])
                    hi.append(r["rate_ci_high"] - r["rate"])
            ax.errorbar(xs, ys, yerr=[lo, hi], linestyle=style, marker="o", markersize=2, linewidth=1, capsize=1,
                        color=COLORS[lang], label=lab)
            h0 = R.get((lang, "p4", "offset_1", fam))
            if h0:
                ax.axhline(h0["rate_h0_mean"], color="#999999", linewidth=.6, linestyle=":")
        ax.set(title=name, xlabel="bytes after start")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].set_ylabel("boundary rate")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.savefig(OUT / "p4_offsets.pdf")
    plt.close(fig)


def o3_depth():
    d = load(ANALYSIS / "positions_teacher.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), constrained_layout=True)
    if d is None:
        for ax in axes:
            ax.text(.5, .5, "pending", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(OUT / "depth_normalised.pdf")
        plt.close(fig)
        macro("xDepthNote", None)
        return
    R = {(r["language"], r["analysis"], r["label"], r["group"]): r for r in d["rows"]}
    ev = {}
    for ax, label, xl in ((axes[0], "cdepth", "constituent depth (enclosing constituents; 8 = 8+)"),
                          (axes[1], "reldepth", "relative depth decile within file")):
        for lang, name, _ in LANGS:
            pts = sorted((int(g), r) for (l, a, lab, g), r in R.items() if l == lang and a == "o3" and lab == label
                         and r["n_positions"] >= 200)
            if not pts:
                continue
            ax.errorbar([x for x, _ in pts], [r["margin"] for _, r in pts],
                        yerr=[[r["margin"] - r["margin_ci_low"] for _, r in pts], [r["margin_ci_high"] - r["margin"] for _, r in pts]],
                        marker="o", markersize=2.5, linewidth=1, capsize=1.2, color=COLORS[lang], label=name)
            ev.setdefault(label, {})[lang] = {x: {"margin": r["margin"], "n": r["n_positions"]} for x, r in pts}
        ax.axhline(0, color="#888888", linewidth=.6)
        ax.set(xlabel=xl)
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].set_ylabel("start recall $-$ $H_0$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "depth_normalised.pdf")
    plt.close(fig)
    evidence["o3"] = ev
    for lang, _, tag in LANGS:
        c = ev.get("cdepth", {}).get(lang, {})
        macro(f"xDepthOne{tag}", signed(c[1]["margin"]) if 1 in c else None)
        macro(f"xDepthSeven{tag}", signed(c[7]["margin"]) if 7 in c else None)


def o4_sweep():
    d = load(ANALYSIS / "o4_rate_sweep.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), constrained_layout=True)
    if d is None:
        for ax in axes:
            ax.text(.5, .5, "pending", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(OUT / "o4_sweep.pdf")
        plt.close(fig)
        macro("xOFourSweepNote", None)
        return
    styles = {"indentation": ("o", "-"), "keyword_typo": ("s", "--"), "missing_semicolon": ("^", "-."),
              "case_inversion": ("D", ":")}
    ev = {}
    for ax, lang, name in ((axes[0], "python", "Python"), (axes[1], "cpp", "C++")):
        series = {}
        for r in d["rows"]:
            if r["language"] != lang or r["condition"] == "clean":
                continue
            kind, rate = r["condition"].rsplit("_r", 1)
            series.setdefault(kind, []).append((float(rate), r))
        for kind, pts in sorted(series.items()):
            pts.sort()
            m, ls = styles[kind]
            ax.errorbar([x for x, _ in pts], [r["delta_margin"] for _, r in pts],
                        yerr=[[r["delta_margin"] - r["delta_margin_ci"][0] for _, r in pts],
                              [r["delta_margin_ci"][1] - r["delta_margin"] for _, r in pts]],
                        marker=m, linestyle=ls, markersize=2.5, linewidth=1, capsize=1.2, color=COLORS[lang],
                        label=kind.replace("_", " "))
            ev.setdefault(lang, {})[kind] = {x: {"delta_margin": r["delta_margin"], "ci": r["delta_margin_ci"],
                                                 "delta_bpp": r["delta_bpp"], "f1_margin": r["f1_margin"]} for x, r in pts}
        ax.axhline(0, color="#888888", linewidth=.6)
        ax.set(title=name, xlabel="noise rate")
        ax.set_xscale("log")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("$\\Delta$ margin vs.\\ clean")
    fig.savefig(OUT / "o4_sweep.pdf")
    plt.close(fig)
    evidence["o4_sweep"] = ev
    worst = min(((l, k, x, v) for l, kk in ev.items() for k, xs in kk.items() for x, v in xs.items()),
                key=lambda t: t[3]["delta_margin"])
    macro("xOFourSweepWorst", f"{worst[1].replace('_', ' ')} at rate {worst[2]:g} in "
                              f"{'Python' if worst[0] == 'python' else 'C++'} ({signed(worst[3]['delta_margin'])})")
    macro("xOFourSweepMinMargin", signed(min(v["f1_margin"] for kk in ev.values() for xs in kk.values() for v in xs.values())))
    macro("xOFourSweepFiles", f"{d['build']['files_per_language']:,}")
    for lang, tag in (("python", "Py"), ("cpp", "Cpp")):
        e = ev.get(lang, {})
        ind, typo, case = e.get("indentation", {}), e.get("keyword_typo", {}), e.get("case_inversion", {})
        macro(f"xOFourIndentLo{tag}", signed(ind[min(ind)]["delta_margin"]) if ind else None)
        macro(f"xOFourIndentHi{tag}", signed(ind[max(ind)]["delta_margin"]) if ind else None)
        macro(f"xOFourTypoHi{tag}", signed(typo[max(typo)]["delta_margin"]) if typo else None)
        macro(f"xOFourTypoBpp{tag}", signed(typo[max(typo)]["delta_bpp"], 2) if typo else None)
        macro(f"xOFourCaseMax{tag}", signed(min(v["delta_margin"] for v in case.values())) if case else None)


def o5_confound():
    d = load(ANALYSIS / "o5_confound.json")
    rows = []
    names = {"A_unsafe_fileFE": "file FE only", "B_plus_statement_type": "+ statement type",
             "B1_plus_identifier_entropy_only": "+ identifier entropy", "B2_plus_identifier_proportion_only": "+ identifier share",
             "C_frozen_full": "full (frozen)"}
    for model, pretty in names.items():
        cells = []
        for outcome in ("all_boundary_density", "inverse_bpp_compute_density"):
            u = d["outcomes"][outcome]["models"][model]["is_memory_unsafe"]
            cells.append(f"{u['coef']:+.4f} {ci(u['ci'][0], u['ci'][1], 4)}")
        rows.append(f"{pretty} & " + " & ".join(cells) + " \\\\")
    o = d["outcomes"]["all_boundary_density"]
    oi = d["outcomes"]["inverse_bpp_compute_density"]
    macro("xOFiveRaw", f"{o['models']['A_unsafe_fileFE']['is_memory_unsafe']['coef']:+.4f}")
    macro("xOFiveRawCI", ci(*o["models"]["A_unsafe_fileFE"]["is_memory_unsafe"]["ci"], 4))
    macro("xOFiveFull", f"{o['models']['C_frozen_full']['is_memory_unsafe']['coef']:+.4f}")
    macro("xOFiveFullCI", ci(*o["models"]["C_frozen_full"]["is_memory_unsafe"]["ci"], 4))
    macro("xOFiveRel", pct(o["C_unsafe_relative_to_safe_mean"], 1))
    macro("xOFiveRelBpp", pct(oi["C_unsafe_relative_to_safe_mean"], 1))
    macro("xOFiveVsIdent", pct(o["C_unsafe_over_1sd_identifier_effect"], 0))
    macro("xOFiveSafeMean", num(o["raw_means"]["safe"]))
    macro("xOFiveUnsafeMean", num(o["raw_means"]["unsafe"]))
    macro("xOFiveCheck", "reproduces" if o["reproduces_frozen"] and oi["reproduces_frozen"] else "does NOT reproduce")
    audit = load(FINAL / "o5_label_audit/summary.json")
    v, rp = audit["validity"], audit["reproduction"]
    macro("xAuditAgree", f"{v['units_all_labels_agree']}/{v['units']}")
    macro("xAuditRepro", f"{rp['regions_reproduced']:,}/{rp['regions']:,}")
    evidence["o5_confound"] = d
    tables["xTableOFive"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lll}\\toprule",
        "Model & Boundary density & Inverse BPP \\\\\\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{adjustbox}",
        f"\\caption{{O5: the memory-unsafe coefficient [95\\% CI] as controls are added (OLS, file fixed effects, "
        f"file-clustered CR1 intervals; {d['n_regions']:,} statements in {d['n_files']:,} clean-parse C++ test files, "
        f"{d['n_unsafe']:,} unsafe). The last row is the frozen model.}}",
        "\\label{tab:o5}\\end{table}"])


def robustness():
    rows = []
    for who, path in (("Teacher", ANALYSIS / "subsets_teacher.json"), ("Student", ANALYSIS / "subsets_student.json")):
        s = load(path)
        if s is None:
            continue
        a = s["analyses"]
        cp, lf, pr = a["cpp_parse"], a["long_files"], a["py_repo_boot"]
        rows.append(f"{who} & C++ clean parse ({cp['clean_parse']['n_files']:,}) & {signed(cp['clean_parse']['f1_margin'])} "
                    f"{ci_pair(cp['clean_parse']['f1_margin_ci'])} \\\\")
        rows.append(f" & C++ error recovery ({cp['error_recovery']['n_files']:,}) & {signed(cp['error_recovery']['f1_margin'])} "
                    f"{ci_pair(cp['error_recovery']['f1_margin_ci'])} \\\\")
        rows.append(f" & C++ $>$8{{,}}191 bytes ({lf['cpp']['gt_8191']['n_files']:,}) & {signed(lf['cpp']['gt_8191']['f1_margin'])} "
                    f"{ci_pair(lf['cpp']['gt_8191']['f1_margin_ci'])} \\\\")
        rows.append(f" & C++ $\\leq$8{{,}}191 bytes ({lf['cpp']['le_8191']['n_files']:,}) & {signed(lf['cpp']['le_8191']['f1_margin'])} "
                    f"{ci_pair(lf['cpp']['le_8191']['f1_margin_ci'])} \\\\")
        rows.append(f" & Python, file bootstrap & {signed(pr['f1_margin'])} {ci_pair(pr['file_bootstrap']['f1_margin_ci'])} \\\\")
        rows.append(f" & Python, repo-cluster bootstrap ({pr['n_repos']} repos) & {signed(pr['f1_margin'])} "
                    f"{ci_pair(pr['repo_cluster_bootstrap']['f1_margin_ci'])} \\\\")
        if who == "Teacher":
            rows.append("\\midrule")
            macro("xErrDiff", signed(cp["margin_difference_err_minus_clean"]))
            macro("xErrDiffCI", ci_pair(cp["margin_difference_ci"]))
            macro("xErrShare", pct(cp["share_error_recovery"], 1))
            macro("xLongShare", pct(lf["cpp"]["share_gt_8191"], 1))
            macro("xLongBytes", pct(lf["cpp"]["share_bytes_gt_8191"], 0))
            macro("xLongDiff", signed(lf["cpp"]["margin_difference_long_minus_short"]))
            macro("xLongDiffCI", ci_pair(lf["cpp"]["margin_difference_ci"]))
            macro("xRepoN", str(pr["n_repos"]))
            macro("xRepoCI", ci_pair(pr["repo_cluster_bootstrap"]["f1_margin_ci"]))
            macro("xFileCI", ci_pair(pr["file_bootstrap"]["f1_margin_ci"]))
            w_file = pr["file_bootstrap"]["f1_margin_ci"][1] - pr["file_bootstrap"]["f1_margin_ci"][0]
            w_repo = pr["repo_cluster_bootstrap"]["f1_margin_ci"][1] - pr["repo_cluster_bootstrap"]["f1_margin_ci"][0]
            macro("xRepoWiden", f"{w_repo / w_file:.1f}")
            evidence["robustness_teacher"] = s
    tables["xTableRobust"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lll}\\toprule",
        "Model & Subset (files) & $\\Delta$ F1 vs.\\ $H_0$ [95\\% CI] \\\\\\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{Held-out start margin on subsets, recomputed exactly from the frozen per-file counts (the null mean is "
        "linear in each file's expected matches). File-level 95\\% bootstrap intervals except where stated.}",
        "\\label{tab:robust}\\end{table}"])


def p3_ratio():
    """P3 on a scale-free axis: observed F1 / mean H0 F1, with the frozen file-bootstrap interval."""
    t = language_rows(FROZEN / "evaluation/scores.csv")
    rows = []
    for lang, name, tag in LANGS:
        r = t[lang]
        macro(f"xRatio{tag}", f"{r['f1_ratio']:.1f}")
        macro(f"xPredPerTarget{tag}", f"{r['recall'] / r['precision']:.1f}")  # recall/precision = #predictions/#targets
        rows.append(f"{name} & {num(r['f1'])} & {num(r['f1_h0_mean'])} & {signed(r['f1_margin'])} & "
                    f"{r['f1_ratio']:.2f} [{r['f1_ratio_ci_low']:.2f}, {r['f1_ratio_ci_high']:.2f}] \\\\")
    evidence["p3_ratio"] = {lang: {k: t[lang][k] for k in ("f1", "f1_h0_mean", "f1_margin", "f1_ratio",
                                                           "f1_ratio_ci_low", "f1_ratio_ci_high")} for lang, _, _ in LANGS}
    tables["xTableRatio"] = "\n".join([
        "\\begin{table}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\columnwidth}\\begin{tabular}{lrrrl}\\toprule",
        "Lang. & F1 & $H_0$ F1 & $\\Delta$ & F1 / $H_0$ [95\\% CI] \\\\\\midrule", *rows,
        "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{P3 on a scale-free axis. The difference $\\Delta$ grows with target density, the ratio does not: "
        "code aligns more strongly than prose relative to its own null. Frozen held-out run; paired file bootstrap.}",
        "\\label{tab:ratio}\\end{table}"])


def golden_gate():
    g = load(ROOT / "results/distill/golden_gate_ada.json")
    if g is None:
        return
    vals = {d: g["domains"][d]["rows"]["student-golden-gate"]["f1"] for d in ("py", "cpp", "prose")}
    ref = {"py": 0.1313, "cpp": 0.2196, "prose": 0.4290}
    macro("xGateMaxDev", f"{max(abs(vals[d] - ref[d]) for d in ref):.4f}")


# --------------------------------------------------------------------------- post-submission arms
ARMS = [(TEACHER, "Teacher (native)"), ("teacher-512-ctl", "Teacher, 512 windows"),
        ("student-v1-14326k", "Student (distilled)"), ("student-bal-v1", "Student, byte-balanced"),
        ("teacher-adapt-v1", "Teacher, adapted (512 windows)")]


def _arm_files(row, lang):
    """score_a_runs per-file counts -> {index: (lang, T, E[T_H0], 2T+FP+FN)} for paired_margin_diff."""
    out = {}
    for i, ((tp, fp, fn), h0) in enumerate(zip(row["per_file"], row["per_file_f1_h0_mean"])):
        den = 2 * tp + fp + fn
        out[str(i)] = (lang, float(tp), h0 * den / 2, float(den))
    return out


def _ci_digits(lo_hi):
    """One more decimal when an endpoint rounds to zero at 3, so the sign stays visible."""
    return 4 if lo_hi and any(x != 0 and round(x, 3) == 0 for x in lo_hi) else 3


def new_arms():
    """Byte-balanced student and adapted teacher (file_toimplement.md optional arms) on the
    pre-registered teacher-512 subset plus the English test split, k = 0, paired with the
    student. Adaptation verdict (STEP 9, applied on this subset): the adapted-minus-student
    paired margin difference exceeds the width of the student's own 95% F1 CI on the same
    files, in at least two of the three domains (rule fixed in code before the runs finished)."""
    code = load(ANALYSIS / "newarms_subset_code.json")
    prose = load(ANALYSIS / "newarms_subset_prose.json")
    win = {k: load(ROOT / p) for k, p in (
        ("bal", "results/distill/balanced/eval_balanced/distill_eval.json"),
        ("bal_ref", "results/distill/balanced/eval_distilled/distill_eval.json"),
        ("adapt", "results/adapt/teacher-finetuned/eval_adapted/distill_eval.json"),
        ("adapt_ref", "results/adapt/teacher-finetuned/eval_distilled/distill_eval.json"))}
    if code is None or prose is None or None in win.values():
        tables["xTableArms"] = MISSING
        for name in ("xBalDiffPy", "xBalDiffCpp", "xBalDiffEn", "xAdaptDiffPy", "xAdaptDiffCpp", "xAdaptDiffEn",
                     "xAdaptVerdict", "xAdaptWins", "xBalWinFone", "xAdaptWinFone"):
            macro(name, None)
        return
    rows, ev, wins = [], {}, 0
    for (lang, name, tag), (src, dom) in zip(LANGS, ((code, "py"), (code, "cpp"), (prose, "prose"))):
        R = src["domains"][dom]["rows"]
        stu = R["student-v1-14326k"]
        cells = []
        for run, _ in ARMS:
            cells.append(f"{num(R[run]['f1'])} & {signed(R[run]['f1_margin'])}" if run in R else "-- & --")
        sf = _arm_files(stu, lang)
        d_bal, ci_bal, n = paired_margin_diff(_arm_files(R["student-bal-v1"], lang), sf, lang, "arms_bal")
        d_ad, ci_ad, _ = paired_margin_diff(_arm_files(R["teacher-adapt-v1"], lang), sf, lang, "arms_adapt")
        width = stu["f1_ci_high"] - stu["f1_ci_low"]
        win_ = d_ad > width
        wins += win_
        ev[lang] = {"n_files": n, "rows": {r: {k: R[r][k] for k in ("f1", "f1_margin", "precision", "recall",
                                                                     "f1_ci_low", "f1_ci_high", "f1_minus_best_baseline")}
                                          for r, _ in ARMS if r in R},
                    "bal_minus_student": d_bal, "bal_ci": ci_bal, "adapt_minus_student": d_ad, "adapt_ci": ci_ad,
                    "student_f1_ci_width": width, "adapt_beats_student_by_ci_width": bool(win_)}
        macro(f"xBalDiff{tag}", signed(d_bal))
        macro(f"xBalDiffCI{tag}", ci_pair(ci_bal, _ci_digits(ci_bal)))
        macro(f"xAdaptDiff{tag}", signed(d_ad))
        macro(f"xAdaptDiffCI{tag}", ci_pair(ci_ad, _ci_digits(ci_ad)))
        macro(f"xAdaptWidth{tag}", num(width))
        macro(f"xArmsN{tag}", f"{n:,}")
        macro(f"xAdaptVsBase{tag}", signed(R["teacher-adapt-v1"]["f1_minus_best_baseline"]))
        macro(f"xBalVsBase{tag}", signed(R["student-bal-v1"]["f1_minus_best_baseline"]))
        rows.append(f"{name} & {n} & " + " & ".join(cells) + " \\\\")
    verdict = wins >= 2
    macro("xAdaptWins", str(wins))
    macro("xAdaptVerdict", "met" if verdict else "not met")
    for key, tag in (("bal", "Bal"), ("adapt", "Adapt")):
        macro(f"x{tag}WinFone", num(win[key]["trained"]["all"]["boundary_f1"]))
        macro(f"x{tag}RefWinFone", num(win[key + "_ref"]["trained"]["all"]["boundary_f1"]))
        macro(f"x{tag}Bpb", num(win[key]["trained"]["all"]["bits_per_byte_student"], 2))
        for dom in ("py", "cpp", "prose"):
            macro(f"x{tag}WinFone{dom.capitalize()}", num(win[key]["trained"][dom]["boundary_f1"]))
            macro(f"x{tag}RefWinFone{dom.capitalize()}", num(win[key + "_ref"]["trained"][dom]["boundary_f1"]))
    ev["adaptation_verdict"] = {"rule": "adapted minus student paired margin difference > width of the student's "
                                        "95% F1 CI on the same files, in >= 2 of 3 domains",
                                "domains_passing": wins, "met": verdict}
    ev["window_eval"] = {k: {d: v["trained"][d] for d in ("all", "py", "cpp", "prose")} for k, v in win.items()}
    evidence["new_arms"] = ev
    tables["xTableArms"] = "\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lr" + "rr" * len(ARMS) + "}\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for _, label in ARMS) + " \\\\",
        "Lang. & Files & " + " & ".join(["F1 & $\\Delta$"] * len(ARMS)) + " \\\\\\midrule",
        *rows, "\\bottomrule\\end{tabular}\\end{adjustbox}",
        "\\caption{Follow-up arms on the pre-registered subset (300 test files per code language, "
        "600--2{,}000 bytes, as in Table~\\ref{tab:t512}) and the English test split, $k=0$, 10{,}000 permutations. "
        "All models except the native teacher see fresh 512-byte windows. The teacher-512 control was not extracted "
        "for English.}",
        "\\label{tab:arms}\\end{table*}"])


def hf_full_sens():
    """hf_full (native HF full attention; sensitivity only) vs the primary swa512 run, k = 0."""
    d = load(ANALYSIS / "hf_full_test_k0.json")
    hf = "hf_full-float32-t1.3354-mplnone-91aa6b8e"
    if d is None:
        for _, _, tag in LANGS:
            macro(f"xHfDiff{tag}", None)
            macro(f"xHfMargin{tag}", None)
        return
    ev = {}
    for lang, name, tag in LANGS:
        R = d["domains"][DOM[lang]]["rows"]
        diff, dci, n = paired_margin_diff(_arm_files(R[hf], lang), _arm_files(R[TEACHER], lang), lang, "hf_full")
        ev[lang] = {"n_files": n, "hf_full": {k: R[hf][k] for k in ("f1", "f1_margin", "f1_minus_best_baseline")},
                    "swa512": {k: R[TEACHER][k] for k in ("f1", "f1_margin", "f1_minus_best_baseline")},
                    "hf_minus_swa512": diff, "ci": dci}
        macro(f"xHfMargin{tag}", signed(R[hf]["f1_margin"]))
        macro(f"xHfSwaMargin{tag}", signed(R[TEACHER]["f1_margin"]))
        macro(f"xHfDiff{tag}", signed(diff))
        macro(f"xHfDiffCI{tag}", ci_pair(dci))
        macro(f"xHfVsBase{tag}", signed(R[hf]["f1_minus_best_baseline"]))
        macro(f"xHfN{tag}", f"{n:,}")
    evidence["hf_full"] = ev


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by reports/final_submission/build_final.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    lines += [f"\\newcommand{{\\{k}}}{{%\n{v}}}" for k, v in sorted(tables.items())]
    (OUT / "final.tex").write_text("\n".join(lines) + "\n")
    evidence["macros"] = macros
    evidence["missing_macros"] = sorted(k for k, v in macros.items() if v == MISSING) + \
        sorted(k for k, v in tables.items() if v == MISSING)
    (OUT / "final_evidence.json").write_text(json.dumps(evidence, indent=1, sort_keys=True, default=str) + "\n")


def main():
    # Stage 1: the frozen C-v1 macros, by the unchanged mid-submission builder.
    mid.main(["--run", str(FROZEN), "--report", str(HERE), "--body", "paper-body.tex"])
    student_table()
    t512_table()
    tau_section()
    k_sweep()
    efficiency()
    alpha1()
    holdout()
    p4_positions()
    o3_depth()
    o4_sweep()
    o5_confound()
    robustness()
    p3_ratio()
    golden_gate()
    new_arms()
    hf_full_sens()
    # Every \x macro the paper uses must exist.
    body = (HERE / "paper-body.tex").read_text() + (HERE / "slides/Trimax-Final-Viva.tex").read_text()
    for name in sorted(set(re.findall(r"\\(x[A-Za-z]+)", body)) - set(macros) - set(tables)):
        macros[name] = MISSING
    write()
    miss = evidence["missing_macros"]
    print(f"wrote {OUT / 'final.tex'}: {len(macros)} macros, {len(tables)} tables; "
          f"missing inputs: {len(evidence['missing_inputs'])}; MISSING macros: {len(miss)}")
    for p in evidence["missing_inputs"]:
        print("  missing input:", p)


if __name__ == "__main__":
    main()
