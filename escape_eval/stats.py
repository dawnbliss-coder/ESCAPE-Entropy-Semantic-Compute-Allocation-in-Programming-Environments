"""Pooled metrics and resampling statistics (PRELIMINARY).

Pooled (micro) metrics over files: P = sum TPp / sum |B|, R = sum TPr / sum |T|,
F1 = 2PR / (P + R) = 2 TPp TPr / (TPp |T| + TPr |B|). A zero denominator gives
NaN; P = R = 0 gives F1 = 0. Computing F1 as one division of exactly
represented integers makes equal rationals compare equal, so permutation ties
are counted exactly.

Permutation p-value (one-sided, observed > null):
  p = (1 + #{null >= observed}) / (R + 1).

Paired cluster bootstrap: files are the independent sampling unit, and
boundaries within a file are dependent. Each resample draws n_files files with
replacement (as integer weights), and every metric, method and contrast of a
group is recomputed from the same weights. Percentile CIs use linear-interpolated
quantiles of the non-NaN draws.

Paired randomisation test (BLT vs whitespace): per file and resample, with
probability 1/2 swap the two methods' counts (numerators and denominators);
the statistic is the pooled difference.

All resampling is chunked to a memory budget. Every draw comes from one
Generator in row order, so results do not depend on the chunk size.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

PURPOSE_BOOTSTRAP = 201
PURPOSE_RANDOMISATION_WS = 202
PURPOSE_BOOTSTRAP_BPP = 203
METRICS = ("precision", "recall", "f1")
DEFAULT_BUDGET = 2_000_000


def name_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


def seed_entropy(master_seed: int, name: str, purpose: int) -> list:
    """SeedSequence entropy for group-level resampling: [master_seed, sha256(name)[:8], purpose]."""
    if int(master_seed) < 0:
        raise ValueError("master_seed must be a non-negative integer")
    return [int(master_seed), name_key(name), int(purpose)]


def rng_from_entropy(entropy) -> Generator:
    return Generator(PCG64(SeedSequence(entropy)))


# --- metrics -----------------------------------------------------------------


def prf(tp_p, n_pred, tp_r, n_targets):
    """Broadcasting pooled precision, recall, F1 (float64 arrays)."""
    tp_p = np.asarray(tp_p, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    tp_r = np.asarray(tp_r, dtype=np.float64)
    n_targets = np.asarray(n_targets, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp_p / n_pred, np.nan)
        recall = np.where(n_targets > 0, tp_r / n_targets, np.nan)
        den = tp_p * n_targets + tp_r * n_pred
        f1 = np.where(den > 0, (2.0 * tp_p * tp_r) / den, 0.0)
        f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)
    return precision, recall, f1


def metric_dict(tp_p, n_pred, tp_r, n_targets) -> dict:
    p, r, f = prf(tp_p, n_pred, tp_r, n_targets)
    return {"precision": p, "recall": r, "f1": f}


@dataclass
class CountSet:
    """Per-file counts of one method over C columns, files in canonical order.
    tp_p, tp_r, n_targets: (n_files, C); n_pred: (n_files,)."""

    tp_p: np.ndarray
    tp_r: np.ndarray
    n_pred: np.ndarray
    n_targets: np.ndarray

    def __post_init__(self):
        self.tp_p = np.asarray(self.tp_p, dtype=np.float64)
        self.tp_r = np.asarray(self.tp_r, dtype=np.float64)
        self.n_pred = np.asarray(self.n_pred, dtype=np.float64)
        self.n_targets = np.asarray(self.n_targets, dtype=np.float64)

    @property
    def n_files(self) -> int:
        return int(self.tp_p.shape[0])

    def select(self, cols) -> "CountSet":
        cols = np.asarray(cols, dtype=np.int64)
        return CountSet(self.tp_p[:, cols], self.tp_r[:, cols], self.n_pred, self.n_targets[:, cols])

    def pooled(self, weights=None) -> dict:
        """metric -> (C,) without weights, (b, C) with (b x n_files) weights."""
        if weights is None:
            return metric_dict(self.tp_p.sum(0), self.n_pred.sum(), self.tp_r.sum(0), self.n_targets.sum(0))
        w = np.asarray(weights, dtype=np.float64)
        return metric_dict(w @ self.tp_p, (w @ self.n_pred)[:, None], w @ self.tp_r, w @ self.n_targets)

    def per_file(self) -> dict:
        return metric_dict(self.tp_p, self.n_pred[:, None], self.tp_r, self.n_targets)


def nanmean_quiet(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


# --- permutation test --------------------------------------------------------


def permutation_summary(observed, null) -> dict:
    """observed (C,), null (R, C) -> dict of (C,) arrays: p_value, null_mean,
    null_sd (ddof=1), null_q025, null_q975 (linear), delta = obs - null_mean,
    z = delta / null_sd, ratio = obs / null_mean, n_null (non-NaN draws)."""
    obs = np.atleast_1d(np.asarray(observed, dtype=np.float64))
    null = np.asarray(null, dtype=np.float64)
    if null.ndim == 1:
        null = null[:, None]
    if null.shape[1] != obs.shape[0]:
        raise ValueError("observed and null column counts differ")
    n_valid = np.count_nonzero(~np.isnan(null), axis=0)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        ge = np.count_nonzero(null >= obs[None, :], axis=0)
        p = (1.0 + ge) / (n_valid + 1.0)
        mean = np.nanmean(null, axis=0)
        sd = np.nanstd(null, axis=0, ddof=1)
        if null.shape[0]:
            q025, q975 = np.nanquantile(null, [0.025, 0.975], axis=0)
        else:
            q025 = q975 = np.full(obs.shape, np.nan)
        delta = obs - mean
        z = np.where(sd > 0, delta / sd, np.nan)
        ratio = np.where(mean != 0, obs / mean, np.nan)
    undefined = np.isnan(obs) | (n_valid == 0)
    p = np.where(undefined, np.nan, p)
    return {
        "p_value": p, "null_mean": mean, "null_sd": sd, "null_q025": q025, "null_q975": q975,
        "delta": delta, "z": z, "ratio": ratio, "n_null": n_valid,
    }


def permutation_p_value(observed: float, null) -> float:
    return float(permutation_summary([observed], np.asarray(null, dtype=np.float64)[:, None])["p_value"][0])


# --- bootstrap ---------------------------------------------------------------


def iter_bootstrap_weights(entropy, n_items: int, n_resamples: int, budget: int = DEFAULT_BUDGET):
    """Yields (b x n_items) float64 multiplicity matrices; each row is one
    resample of n_items items drawn with replacement."""
    if n_items <= 0:
        raise ValueError("cannot bootstrap zero items")
    rng = rng_from_entropy(entropy)
    chunk = max(1, int(budget) // n_items)
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        idx = rng.integers(0, n_items, size=(b, n_items))
        idx += (np.arange(b, dtype=np.int64) * n_items)[:, None]
        w = np.bincount(idx.ravel(), minlength=b * n_items).reshape(b, n_items)
        yield w.astype(np.float64)
        done += b


def bootstrap_draws(entropy, n_items: int, n_resamples: int, fn, budget: int = DEFAULT_BUDGET) -> np.ndarray:
    """Concatenates fn(weights) -> (b, ...) over all resamples -> (n_resamples, ...)."""
    out = None
    pos = 0
    for w in iter_bootstrap_weights(entropy, n_items, n_resamples, budget):
        v = np.asarray(fn(w), dtype=np.float64)
        if out is None:
            out = np.empty((n_resamples,) + v.shape[1:], dtype=np.float64)
        out[pos: pos + v.shape[0]] = v
        pos += v.shape[0]
    return out


def percentile_ci(draws, level: float = 0.95):
    """(lo, hi) percentile interval over axis 0, ignoring NaN draws."""
    draws = np.asarray(draws, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(draws, [alpha, 1.0 - alpha], axis=0)
    return lo, hi


def bootstrap_median_difference(x, y, entropy_x, entropy_y, n_resamples: int, budget: int = DEFAULT_BUDGET):
    """median(x) - median(y) and its bootstrap draws, resampling x and y
    independently (each within its own group)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.nan, np.full(n_resamples, np.nan)

    def medians(values, entropy):
        rng = rng_from_entropy(entropy)
        chunk = max(1, int(budget) // values.size)
        out = np.empty(n_resamples, dtype=np.float64)
        done = 0
        while done < n_resamples:
            b = min(chunk, n_resamples - done)
            idx = rng.integers(0, values.size, size=(b, values.size))
            out[done: done + b] = np.median(values[idx], axis=1)
            done += b
        return out

    return float(np.median(x) - np.median(y)), medians(x, entropy_x) - medians(y, entropy_y)


# --- paired randomisation ------------------------------------------------------


def paired_randomisation(a: CountSet, b: CountSet, entropy, n_resamples: int, budget: int = DEFAULT_BUDGET) -> dict:
    """metric -> {"observed": (C,), "null": (R, C), "p_value": (C,)} for the
    pooled difference a - b; one-sided p (a > b)."""
    if a.tp_p.shape != b.tp_p.shape or not np.array_equal(a.n_targets, b.n_targets):
        raise ValueError("paired methods must share files, columns and targets")
    n, C = a.tp_p.shape
    if n == 0:
        raise ValueError("no files")
    obs_a, obs_b = a.pooled(), b.pooled()
    null = {m: np.empty((n_resamples, C), dtype=np.float64) for m in METRICS}
    nt = a.n_targets.sum(0)[None, :]
    rng = rng_from_entropy(entropy)
    chunk = max(1, int(budget) // n)
    done = 0
    while done < n_resamples:
        r = min(chunk, n_resamples - done)
        swap = (rng.random((r, n)) < 0.5).astype(np.float64)
        keep = 1.0 - swap
        ma = metric_dict(keep @ a.tp_p + swap @ b.tp_p, (keep @ a.n_pred + swap @ b.n_pred)[:, None],
                         keep @ a.tp_r + swap @ b.tp_r, nt)
        mb = metric_dict(swap @ a.tp_p + keep @ b.tp_p, (swap @ a.n_pred + keep @ b.n_pred)[:, None],
                         swap @ a.tp_r + keep @ b.tp_r, nt)
        for m in METRICS:
            null[m][done: done + r] = ma[m] - mb[m]
        done += r
    out = {}
    for m in METRICS:
        observed = obs_a[m] - obs_b[m]
        out[m] = {
            "observed": observed,
            "null": null[m],
            "p_value": permutation_summary(observed, null[m])["p_value"],
        }
    return out
