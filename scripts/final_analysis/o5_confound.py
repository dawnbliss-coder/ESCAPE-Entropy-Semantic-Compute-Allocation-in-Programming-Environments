"""O5 interpretation: how much of the memory-unsafe effect survives the identifier confound.

Reads the frozen region table results/final/c-v1-2721954/o5/o5_regions.csv (no model or
region is recomputed) and refits the frozen O5 regression (escape_scoring/o5.py: OLS with
file fixed effects, file-cluster robust covariance) as a nested sequence:

  A  is_memory_unsafe + file FE
  B  A + statement type
  B1 B + mean identifier entropy;  B2  B + identifier byte proportion (which control matters)
  C  B + mean identifier entropy + identifier byte proportion + region length  (= frozen model)

The frozen coefficient is reproduced by C (checked). Reported: the unsafe coefficient in
each model, the share of A's coefficient that survives in C, the effect relative to the
mean outcome of safe regions, and C's coefficient against one SD of identifier entropy.

File fixed effects are absorbed by within-file demeaning (Frisch-Waugh-Lovell; same point
estimates as the dummy design). Cluster-robust variance is CR1 with the dummy design's
parameter count, as statsmodels computes it; intervals use the normal quantile (1,500
clusters).

Usage: python scripts/final_analysis/o5_confound.py --out results/final/analysis/o5_confound.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = "results/final/c-v1-2721954"
Z = 1.959963984540054


def fit(df, outcome, cols):
    g = df["sample_id"].to_numpy()
    X = df[cols].to_numpy(float)
    y = df[outcome].to_numpy(float)
    # within-file demeaning absorbs the file fixed effects and the intercept
    Xd = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
    yd = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    bread = np.linalg.pinv(Xd.T @ Xd)
    scores = pd.DataFrame(Xd * resid[:, None]).groupby(g).sum().to_numpy()
    meat = scores.T @ scores
    n, n_groups = len(y), len(np.unique(g))
    k = len(cols) + n_groups  # intercept + (G-1) dummies + slopes, as in the frozen dummy design
    cov = bread @ meat @ bread * (n_groups / (n_groups - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(cov))
    return {c: {"coef": float(b), "se": float(s), "ci": [float(b - Z * s), float(b + Z * s)], "z": float(b / s)}
            for c, b, s in zip(cols, beta, se)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", default=f"{FROZEN}/o5/o5_regions.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    df = pd.read_csv(REPO_ROOT / args.regions)
    for level in sorted(df["control_statement_type"].unique())[1:]:
        df[f"stmt={level}"] = (df["control_statement_type"] == level).astype(float)
    stmt = [c for c in df.columns if c.startswith("stmt=")]
    ident = ["mean_identifier_entropy_nats", "identifier_byte_proportion", "region_length"]
    models = {"A_unsafe_fileFE": ["is_memory_unsafe"],
              "B_plus_statement_type": ["is_memory_unsafe", *stmt],
              "B1_plus_identifier_entropy_only": ["is_memory_unsafe", "mean_identifier_entropy_nats", *stmt],
              "B2_plus_identifier_proportion_only": ["is_memory_unsafe", "identifier_byte_proportion", *stmt],
              "C_frozen_full": ["is_memory_unsafe", *ident, *stmt]}
    frozen = {"all_boundary_density": json.loads((REPO_ROOT / f"{FROZEN}/o5/o5.json").read_text()),
              "inverse_bpp_compute_density": json.loads((REPO_ROOT / f"{FROZEN}/o5_inverse_bpp/o5.json").read_text())}
    out = {"regions": args.regions, "n_regions": int(len(df)), "n_files": int(df["sample_id"].nunique()),
           "n_unsafe": int(df["is_memory_unsafe"].sum()), "outcomes": {}}
    for outcome in ("all_boundary_density", "inverse_bpp_compute_density"):
        res = {name: fit(df, outcome, cols) for name, cols in models.items()}
        a, c = res["A_unsafe_fileFE"]["is_memory_unsafe"], res["C_frozen_full"]["is_memory_unsafe"]
        safe_mean = float(df.loc[df["is_memory_unsafe"] == 0, outcome].mean())
        sd_ident = float(df["mean_identifier_entropy_nats"].std(ddof=1))
        b_ident = res["C_frozen_full"]["mean_identifier_entropy_nats"]["coef"]
        frozen_coef = frozen[outcome]["coefficients"]["is_memory_unsafe"]
        out["outcomes"][outcome] = {
            "models": res,
            "raw_means": {"safe": safe_mean, "unsafe": float(df.loc[df["is_memory_unsafe"] == 1, outcome].mean())},
            "share_of_A_surviving_in_C": c["coef"] / a["coef"],
            "C_unsafe_relative_to_safe_mean": c["coef"] / safe_mean,
            "identifier_entropy_sd_nats": sd_ident,
            "identifier_entropy_1sd_effect": b_ident * sd_ident,
            "C_unsafe_over_1sd_identifier_effect": c["coef"] / (b_ident * sd_ident),
            "frozen_coef": frozen_coef, "frozen_ci": frozen[outcome]["confidence_intervals"]["is_memory_unsafe"],
            "reproduces_frozen": bool(abs(c["coef"] - frozen_coef) < 1e-9)}
    path = REPO_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    for outcome, r in out["outcomes"].items():
        print(outcome, "reproduces frozen:", r["reproduces_frozen"])
        for name, m in r["models"].items():
            u = m["is_memory_unsafe"]
            print(f"  {name:<24} unsafe={u['coef']:+.5f} [{u['ci'][0]:+.5f},{u['ci'][1]:+.5f}]")
        print(f"  safe mean={r['raw_means']['safe']:.4f} unsafe mean={r['raw_means']['unsafe']:.4f} "
              f"surviving share={r['share_of_A_surviving_in_C']:.2f} relative={r['C_unsafe_relative_to_safe_mean']:+.3f} "
              f"1sd ident effect={r['identifier_entropy_1sd_effect']:+.4f} ratio={r['C_unsafe_over_1sd_identifier_effect']:.3f}")


if __name__ == "__main__":
    main()
