"""Region-level surface-confound regression; no risk-understanding inference."""
from __future__ import annotations

import numpy as np


def region_table(samples):
    """B supplies analysis_region units, memory_unsafe masks, identifier spans.

    No selection or labeling of safe/unsafe code is performed here. A region
    gets is_memory_unsafe=1 on positive byte overlap with a supplied mask.
    Overlapping identifier spans count each byte once. Missing identifiers
    give entropy=0 alongside proportion=0 (not a missing-value imputation).
    """
    rows = []
    for sample in samples:
        m = sample.manifest
        if m["domain"] != "code" or m["language"] not in {"cpp", "c++"}:
            continue
        identifiers = [r for r in sample.regions if r["region_kind"] == "identifier"]
        unsafe = [r for r in sample.regions if r["region_kind"] == "memory_unsafe"]
        entropy = np.array([r["entropy"] for r in sample.patcher["entropy_records"]])
        # Standardize to nats; never local-context normalize entropy.
        if sample.patcher["entropy_unit"] == "bits":
            entropy *= np.log(2)
        for region in sample.regions:
            if region["region_kind"] != "analysis_region":
                continue
            a, b = region["start_byte"], region["end_byte"]
            mask = np.zeros(b - a, dtype=bool)
            for ident in identifiers:
                lo, hi = max(a, ident["start_byte"]), min(b, ident["end_byte"])
                if lo < hi:
                    mask[lo - a:hi - a] = True
            boundary_count = sum(a <= r["byte_offset"] < b and 0 < r["byte_offset"] < m["byte_length"]
                                 and r["termination_reason"] in {"entropy", "max_length"}
                                 for r in sample.patcher["boundaries"])
            # Fractional patch mass yields a region-local compute proxy:
            # integral over bytes of 1 / containing-patch-length, divided by length.
            patch_mass = sum(max(0, min(b, p["end_byte"]) - max(a, p["start_byte"])) /
                             (p["end_byte"] - p["start_byte"]) for p in sample.patcher["patches"])
            row = {"sample_id": sample.sample_id, "region_id": region["region_id"],
                   "language": m["language"], "start_byte": a, "end_byte": b,
                   "region_length": b - a,
                   "is_memory_unsafe": int(any(max(a, r["start_byte"]) < min(b, r["end_byte"]) for r in unsafe)),
                   "mean_identifier_entropy_nats": float(entropy[a:b][mask].mean()) if mask.any() else 0.0,
                   "identifier_byte_proportion": float(mask.mean()),
                   "all_boundary_density": boundary_count / (b - a),
                   "inverse_bpp_compute_density": patch_mass / (b - a)}
            for name, value in region.get("controls", {}).items():
                row[f"control_{name}"] = value
            rows.append(row)
    return rows


def regress(rows, *, outcome="all_boundary_density", controls=("sample_id",), numeric_controls=()):
    if outcome not in {"all_boundary_density", "inverse_bpp_compute_density"}:
        raise ValueError("unsupported O5 outcome")
    if not rows:
        raise ValueError("O5 requires B-provided analysis_region annotations")
    names = ["intercept", "is_memory_unsafe", "mean_identifier_entropy_nats",
             "identifier_byte_proportion", "region_length", *numeric_controls]
    X = np.array([[1.0, *[float(row[key]) for key in names[1:]]] for row in rows])
    for control in controls:
        levels = sorted({str(row[control]) for row in rows})
        for level in levels[1:]:
            X = np.column_stack([X, [float(str(row[control]) == level) for row in rows]])
            names.append(f"{control}={level}")
    y = np.array([float(row[outcome]) for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite O5 design matrix")
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    output = {"outcome": outcome, "predictors": names, "n_regions": len(rows),
              "n_files": len({r["sample_id"] for r in rows}), "rank": int(rank),
              "identified": bool(rank == X.shape[1]),
              "coefficients": dict(zip(names, map(float, beta))),
              "coefficient_method": "OLS minimum-norm least squares",
              "inference": "unavailable: statsmodels not installed",
              "interpretation": "surface-confound isolation only; no semantic-risk-understanding claim"}
    if rank < X.shape[1]:
        output["inference"] = "unavailable: rank-deficient design; coefficients not uniquely identified"
    elif output["n_files"] < 2 or len(rows) <= X.shape[1]:
        output["inference"] = "unavailable: insufficient independent files or residual degrees of freedom"
    else:
        try:
            import statsmodels.api as sm
        except ImportError:
            pass
        else:
            fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={
                "groups": [row["sample_id"] for row in rows], "use_correction": True}, use_t=True)
            ci = fit.conf_int()
            if np.isfinite(ci).all() and np.isfinite(fit.pvalues).all():
                output.update({"inference": "OLS with file-cluster robust covariance; t intervals",
                               "statsmodels_version": sm.__version__,
                               "confidence_intervals": dict(zip(names, ci.tolist())),
                               "pvalues": dict(zip(names, fit.pvalues.tolist()))})
            else:
                output["inference"] = "unavailable: nonfinite clustered inference"
    design = [{"sample_id": row["sample_id"], "region_id": row["region_id"],
               "outcome": float(yi), **dict(zip(names, map(float, xi)))} for row, xi, yi in zip(rows, X, y)]
    return output, design
