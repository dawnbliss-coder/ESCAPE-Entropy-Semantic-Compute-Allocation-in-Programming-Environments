"""Stream C figures, drawn only from saved results tables (PRELIMINARY; SMOKE runs are
watermarked in every title).

  p1_alignment_effects.png  precision / recall / F1 of BLT vs E[H0] vs whitespace vs word
                            baseline per domain (start targets, all strata) with 95%
                            bootstrap CIs and the permutation p-value vs H0
  depth_alignment.png       recall by raw AST depth (BLT, E[H0], whitespace, word), strata
                            with >= --min-targets targets
  p2_start_end.png          BLT start vs end alignment per domain
  p4_openers.png            recall lift over H0: deterministic openers vs open-ended (with CIs)
  k_calibration.png         F1_BLT(k), E[F1_H0](k) and margin(k) per language

Usage:
  python -m escape_eval.figures --analysis-dir results/smoke/golden_smoke_realA \
      --k-dir results/smoke/k_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from escape_common.io import atomic_write_json  # noqa: E402

METHOD_STYLE = {"blt": ("BLT (entropy-triggered)", "#1d3557"), "h0": ("E[H0] density-matched", "#adb5bd"),
                "ws": ("whitespace (newline)", "#e76f51"), "word": ("word starts", "#2a9d8f")}
DOMAIN_ORDER = ("py", "cpp", "prose")


def _tag(params: dict) -> str:
    return "SMOKE - not results" if params.get("smoke") else "PRELIMINARY"


def _err(values, lo, hi):
    values, lo, hi = (np.asarray(x, dtype=float) for x in (values, lo, hi))
    return np.vstack([np.clip(values - lo, 0, None), np.clip(hi - values, 0, None)])


def p1_effects(align: pd.DataFrame, params: dict, out: Path) -> None:
    sub = align[(align["kind"] == "start") & (align["axis"] == "all")]
    domains = [d for d in DOMAIN_ORDER if d in set(sub["domain"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 4.2), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        d = sub[sub["domain"] == domain]
        k_values = sorted(d["k"].unique())
        d = d[d["k"] == k_values[0]].set_index("metric").loc[["precision", "recall", "f1"]]
        x = np.arange(3)
        n_methods = len(METHOD_STYLE)
        for i, (key, (label, colour)) in enumerate(METHOD_STYLE.items()):
            vals = d[key].to_numpy(dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.bar(x + (i - (n_methods - 1) / 2) * 0.22, vals, width=0.21, color=colour, label=label,
                   yerr=_err(vals, d[f"{key}_lo"], d[f"{key}_hi"]), capsize=3)
        p = d.loc["f1", "p_perm_h0"]
        ax.set_xticks(x, ["precision", "recall", "F1"])
        ax.set_title(f"{domain} (k={k_values[0]}, n={int(d['n_files'].iloc[0])} files)\n"
                     f"perm p(F1 vs H0) = {p:.2g}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("pooled score vs AST/constituent starts")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"P1 boundary alignment, 95% file-bootstrap CIs [{_tag(params)}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def depth_alignment(align: pd.DataFrame, params: dict, out: Path, min_targets: int) -> None:
    sub = align[(align["kind"] == "start") & (align["axis"] == "depth") & (align["metric"] == "recall")]
    domains = [d for d in DOMAIN_ORDER if d in set(sub["domain"])]
    if not domains:
        return
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 3.8), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        d = sub[(sub["domain"] == domain) & (sub["n_targets"] >= min_targets)].copy()
        d = d[d["k"] == d["k"].min()]
        d["depth"] = d["stratum"].astype(int)
        d = d.sort_values("depth")
        for key, (label, colour) in METHOD_STYLE.items():
            vals = d[key].to_numpy(dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(d["depth"], vals, marker="o", color=colour, label=label, linewidth=1.3)
            ax.fill_between(d["depth"], d[f"{key}_lo"], d[f"{key}_hi"], color=colour, alpha=0.18)
        ax.set_xlabel("raw AST depth of outermost node at the start")
        ax.set_title(f"{domain} (k={int(d['k'].iloc[0]) if len(d) else '-'}; strata with >= {min_targets} targets)", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("recall of starts")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"Depth-stratified alignment (raw depth, not normalised across languages) [{_tag(params)}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def p2_start_end(p2: pd.DataFrame, params: dict, out: Path) -> None:
    if p2.empty:
        return
    domains = [d for d in DOMAIN_ORDER if d in set(p2["domain"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.8), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        d = p2[p2["domain"] == domain]
        d = d[d["k"] == d["k"].min()].set_index("metric").loc[["precision", "recall", "f1"]]
        x = np.arange(3)
        ax.bar(x - 0.2, d["blt_start"], width=0.38, color="#1d3557", label="starts")
        ax.bar(x + 0.2, d["blt_end"], width=0.38, color="#a8dadc", label="ends")
        ax.set_xticks(x, ["precision", "recall", "F1"])
        ax.set_ylim(0, 1)
        f1 = d.loc["f1"]
        ax.set_title(f"{domain}: F1 start-end = {f1['blt_start_minus_end']:.3f}\n"
                     f"[{f1['blt_start_minus_end_lo']:.3f}, {f1['blt_start_minus_end_hi']:.3f}]", fontsize=9)
    axes[0][0].set_ylabel("BLT pooled score")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"P2 start vs end alignment [{_tag(params)}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def p4_openers(p4: pd.DataFrame, params: dict, out: Path) -> None:
    if p4.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    domains = [d for d in DOMAIN_ORDER if d in set(p4["domain"])]
    x = np.arange(len(domains))
    for off, name, colour in ((-0.2, "deterministic", "#e9c46a"), (0.2, "open_ended", "#2a9d8f")):
        rows = p4.set_index("domain").loc[domains]
        vals = rows[f"lift_{name}"].to_numpy(dtype=float)
        ax.bar(x + off, vals, width=0.38, color=colour, label=f"{name} (recall BLT - E[H0])",
               yerr=_err(vals, rows[f"lift_{name}_lo"], rows[f"lift_{name}_hi"]), capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    labels = [f"{d}\ncontrast {p4.set_index('domain').loc[d, 'contrast_open_minus_deterministic']:.3f}" for d in domains]
    ax.set_xticks(x, labels)
    ax.set_ylabel("recall lift over H0")
    ax.legend(fontsize=8)
    ax.set_title(f"P4 deterministic openers vs open-ended constructs [{_tag(params)}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alignment_vs_k(align: pd.DataFrame, params: dict, out: Path) -> bool:
    """Precision / recall / F1 against start targets as a function of k, for domains scored at
    several k (e.g. prose with --k-grid). Returns False if no domain has more than one k."""
    sub = align[(align["kind"] == "start") & (align["axis"] == "all")]
    domains = [d for d in DOMAIN_ORDER if sub.loc[sub["domain"] == d, "k"].nunique() > 1]
    if not domains:
        return False
    fig, axes = plt.subplots(len(domains), 3, figsize=(13, 3.6 * len(domains)), squeeze=False)
    for row, domain in enumerate(domains):
        d = sub[sub["domain"] == domain]
        for col, metric in enumerate(("precision", "recall", "f1")):
            ax = axes[row][col]
            m = d[d["metric"] == metric].sort_values("k")
            for key, (label, colour) in METHOD_STYLE.items():
                vals = m[key].to_numpy(dtype=float)
                if np.all(np.isnan(vals)):
                    continue
                ax.plot(m["k"], vals, marker="o", color=colour, label=label, linewidth=1.3)
                ax.fill_between(m["k"], m[f"{key}_lo"], m[f"{key}_hi"], color=colour, alpha=0.18)
            ax.set_xlabel("tolerance k (bytes)")
            ax.set_title(f"{domain}: {metric} (n={int(m['n_files'].iloc[0])} files)", fontsize=9)
            ax.set_ylim(0, 1)
        axes[row][0].legend(fontsize=7)
    fig.suptitle(f"Alignment vs tolerance k, every k reported, none selected [{_tag(params)}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def k_calibration(curve: pd.DataFrame, record: dict, out: Path) -> None:
    domains = [d for d in DOMAIN_ORDER if d in set(curve["domain"])]
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 3.6), squeeze=False)
    for ax, domain in zip(axes[0], domains):
        d = curve[curve["domain"] == domain].sort_values("k")
        ax.plot(d["k"], d["f1_blt"], marker="o", color="#1d3557", label="F1 BLT")
        ax.plot(d["k"], d["f1_h0_mean"], marker="o", color="#adb5bd", label="E[F1 H0]")
        ax.plot(d["k"], d["margin_f1"], marker="s", color="#2a9d8f", label="margin")
        k_star = record.get("selected_k", {}).get(domain, {}).get("k")
        if k_star is not None:
            ax.axvline(k_star, color="#e76f51", linestyle="--", label=f"k* = {k_star}")
        ax.set_xlabel("tolerance k (bytes)")
        ax.set_title(f"{domain} (n={int(d['n_files'].iloc[0])} files)", fontsize=9)
    axes[0][-1].legend(fontsize=7)
    tag = "SMOKE - not results" if record.get("smoke") else "calib split, PRELIMINARY"
    fig.suptitle(f"Tolerance calibration [{tag}]", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream C figures from results tables")
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--k-dir", default=None)
    ap.add_argument("--min-targets", type=int, default=5)
    args = ap.parse_args(argv)
    adir = Path(args.analysis_dir)
    params = json.loads((adir / "params.json").read_text())
    out_dir = adir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    align = pd.read_parquet(adir / "alignment.parquet")
    for name, fn in (("p1_alignment_effects.png", lambda p: p1_effects(align, params, p)),
                     ("depth_alignment.png", lambda p: depth_alignment(align, params, p, args.min_targets))):
        fn(out_dir / name)
        written.append(name)
    if alignment_vs_k(align, params, out_dir / "alignment_vs_k.png"):
        written.append("alignment_vs_k.png")
    for table, name, fn in (("p2_start_end", "p2_start_end.png", p2_start_end), ("p4_openers", "p4_openers.png", p4_openers)):
        path = adir / f"{table}.parquet"
        if path.exists():
            fn(pd.read_parquet(path), params, out_dir / name)
            written.append(name)
    if args.k_dir:
        kdir = Path(args.k_dir)
        k_calibration(pd.read_parquet(kdir / "curve.parquet"), json.loads((kdir / "selected_k.json").read_text()),
                      out_dir / "k_calibration.png")
        written.append("k_calibration.png")
    written = [w for w in written if (out_dir / w).exists()]
    atomic_write_json(out_dir / "figures.json", {"analysis_dir": str(adir), "k_dir": args.k_dir, "outputs": written,
                                                 "smoke": bool(params.get("smoke")), "preliminary": True})
    for w in written:
        print(f"wrote {out_dir / w}")


if __name__ == "__main__":
    main()
