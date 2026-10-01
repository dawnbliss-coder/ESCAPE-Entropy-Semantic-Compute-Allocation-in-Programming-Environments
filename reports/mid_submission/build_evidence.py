"""Regenerate only the factual ACL table and plot from saved, marked results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.family": "serif", "font.size": 10})

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LEGACY_ID = "prose_kgrid_swa512-float32-t1.3354-mplnone-91aa6b8e"
LEGACY = ROOT / "results" / LEGACY_ID
ADA = ROOT / "results/ada/synthetic-2719654"


def texnum(x, digits=3):
    return f"{float(x):.{digits}f}"


def main():
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "evidence").mkdir(exist_ok=True)
    params_path = LEGACY / "params.json"
    alignment_path = LEGACY / "alignment.csv"
    params = json.loads(params_path.read_text())
    if params.get("domains") != ["prose"] or not params.get("preliminary"):
        raise ValueError("Expected saved preliminary prose-only analysis")
    df = pd.read_csv(alignment_path)
    row = df[(df.domain == "prose") & (df.kind == "start") & (df.axis == "all")
             & (df.stratum == "all") & (df.metric == "f1") & (df.k == 0)]
    if len(row) != 1:
        raise ValueError("Expected exactly one legacy prose start/global F1 row at k=0")
    r = row.iloc[0]
    precision_row = df[(df.domain == "prose") & (df.kind == "start") & (df.axis == "all")
                       & (df.stratum == "all") & (df.metric == "precision") & (df.k == 0)]
    if len(precision_row) != 1:
        raise ValueError("Expected one precision row; never substitute F1 for precision")
    # Preserve estimator provenance in the table caption and attached evidence.
    if int(r.n_files) != 300 or not bool(r.preliminary) or r.k_mode != "grid_uncalibrated":
        raise ValueError("Legacy exploratory result identity changed")

    fig = plt.figure(figsize=(3.03, 2.1), constrained_layout=True)
    ax = fig.add_subplot(111)
    all_rows = df[(df.domain == "prose") & (df.kind == "start") & (df.axis == "all")
                  & (df.stratum == "all") & (df.metric == "f1")].sort_values("k")
    ax.errorbar(all_rows.k, all_rows.blt,
                yerr=[all_rows.blt - all_rows.blt_lo, all_rows.blt_hi - all_rows.blt],
                marker="o", markersize=3, linewidth=1.1, capsize=2, color="#234f7d",
                label="BLT (legacy)")
    ax.errorbar(all_rows.k, all_rows.h0,
                yerr=[all_rows.h0 - all_rows.h0_lo, all_rows.h0_hi - all_rows.h0],
                marker="s", markersize=3, linewidth=1.1, capsize=2, color="#888888",
                label="H0")
    ax.set(xlabel="Tolerance k (grid; not calibrated)", ylabel="Pooled start-boundary F1",
           xticks=all_rows.k, ylim=(0, 1))
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "figures/prose_legacy_f1_by_k.pdf", metadata={
        "Title": "Preliminary prose alignment from saved legacy results",
        "Subject": "Exploratory; legacy any-match scorer; k grid, no selected tolerance"})
    plt.close(fig)

    scores = json.loads((ADA / "evaluation/scores.json").read_text())
    calibration = json.loads((ADA / "calibration.json").read_text())
    o5 = json.loads((ADA / "o5/o5.json").read_text())
    for record in (scores, calibration, o5):
        if record.get("synthetic") is not True:
            raise ValueError("Ada demonstration must remain labelled synthetic")
    if scores["configuration"]["permutations"] != 10000 or calibration["permutations"] != 10000:
        raise ValueError("Ada run did not use 10,000 permutations")
    global_score = next(x for x in scores["scores"]
                        if x["kind"] == "start" and x["axis"] == "global" and x["stratum"] == "all")
    evidence = {
        "legacy_real_prose": {"source": f"results/{LEGACY_ID}/alignment.csv",
                              "params_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
                              "alignment_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
                              "scorer": "legacy escape_eval; any-match; exploratory; k=0 from an unselected grid",
                              "n_files": int(r.n_files), "n_targets": int(r.n_targets),
                              "n_entropy_boundaries": int(r.n_pred_blt), "precision": float(precision_row.iloc[0].blt),
                              "recall": float(df[(df.domain == "prose") & (df.kind == "start") & (df.axis == "all")
                                                  & (df.stratum == "all") & (df.metric == "recall") & (df.k == 0)].iloc[0].blt),
                              "f1": float(r.blt), "h0_mean_f1": float(r.h0_null_mean),
                              "h0_sd_f1": float(r.h0_null_sd), "f1_ci95": [float(r.blt_lo), float(r.blt_hi)],
                              "margin": float(r.delta_h0), "permutation_p": float(r.p_perm_h0),
                              "whitespace_control": "not applicable to saved single-paragraph prose rows",
                              "language_word_control": "absent from this legacy CSV; do not claim controlled"},
        "ada_synthetic_software_check": {"source": "results/ada/synthetic-2719654",
                                         "scores_sha256": hashlib.sha256((ADA / "evaluation/scores.json").read_bytes()).hexdigest(),
                                         "permutations": 10000, "bootstrap": 2000,
                                         "n_test_files": int(global_score["n_files"]),
                                         "tp": int(global_score["tp"]), "fp": int(global_score["fp"]), "fn": int(global_score["fn"]),
                                         "f1": float(global_score["f1"]),
                                         "h0_mean_f1": float(global_score["f1_h0_mean"]),
                                         "permutation_p": float(global_score["f1_permutation_p"]),
                                         "selected_k": calibration["selected_k"],
                                         "n_o5_regions": int(o5["n_regions"]),
                                         "o5_inference": o5["inference"],
                                         "interpretation": "synthetic software validation only; no BLT experiment"},
        "checkpoint_fidelity": {"source": "results/stream_a_fidelity/fidelity.json",
                                 "status": "effective amended gates pass; original G2/G7 declared criteria failed",
                                 "gate_caveat": "G2/G7 thresholds were amended and must be reported transparently"},
        "code_real_data": {"status": "not run",
                           "preflight_missing_files": 65200,
                           "samples_in_manifest": 16300},
    }
    (OUT / "evidence" / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    e = evidence["legacy_real_prose"]
    s = evidence["ada_synthetic_software_check"]
    tex = r"""%% Generated by build_evidence.py from frozen result files.
\begin{table}[t]
\centering
\begin{tabular}{lrr}
\toprule
Quantity & Observed & Legacy $H_0$ \\
\midrule
Precision & %(p).3f & %(hp).3f \\
Recall & %(r).3f & %(hr).3f \\
F1 & %(f).3f & %(h).3f \\
\bottomrule
\end{tabular}
\caption{Real prose, exploratory: 300 paragraphs at $k=0$, with 25,185 target starts and 65,515 predictions. Legacy any-match scoring; no tolerance was selected. The null column reports permutation means.}
\label{tab:prose}
\end{table}
""" % {"p": e["precision"], "r": e["recall"], "f": e["f1"], "h": e["h0_mean_f1"],
       "hp": float(precision_row.iloc[0].h0_null_mean),
       "hr": float(df[(df.domain == "prose") & (df.kind == "start") & (df.axis == "all")
                       & (df.stratum == "all") & (df.metric == "recall") & (df.k == 0)].iloc[0].h0_null_mean)}
    (OUT / "evidence_tables.tex").write_text(tex)
    print(f"Wrote ACL evidence for {e['n_files']} exploratory prose files and {s['n_test_files']} synthetic test files")


if __name__ == "__main__":
    main()
