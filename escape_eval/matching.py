"""Vectorised any-match tolerance counts for all strata at once, and span IoU.

For predicted boundary offsets B, target offsets T and an integer tolerance
k >= 0, a boundary b matches a target t iff |b - t| <= k, and

  TPp = #{b in B : exists t in T, |b - t| <= k}   (precision numerator, denominator |B|)
  TPr = #{t in T : exists b in B, |b - t| <= k}   (recall numerator, denominator |T|)

A stratum s restricts only the target side: TPp_s counts boundaries within k
of some target in T^s (the denominator stays |B|), and TPr_s counts the targets
of T^s that some boundary matches (denominator |T^s|).

Predictions come as a 2-D integer array with one boundary set per row (one row
per H0 resample, or a single row for BLT or whitespace). Each call returns
(rows x n_strata) int64 counts.

  precision: per k, every byte position gets a signature id naming the set of
    strata that have a target within k of it, i.e. a dilated target mask split
    by strata. Scoring a chunk is one gather, one bincount, and a sparse
    signature -> strata product.
  recall: rows are concatenated with per-row offsets (row * W, W > n_bytes + 2k)
    into one sorted array. A single searchsorted then finds, for every
    (row, target), the first boundary >= t - k. The target matches iff that
    boundary is <= t + k, which also keeps it in the same row. Matches are summed
    per stratum with np.add.reduceat over targets grouped by stratum.

The test-suite checks these counts exactly against the naive pure-Python
reference in tests/test_eval_reference.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from escape_eval.targets import AXES, Targets


@dataclass(frozen=True)
class StrataLayout:
    axes: tuple
    keys: tuple              # ((axis, stratum), ...) in column order; keys[0] == ("all", "all")
    target_cols: np.ndarray  # (n_targets, len(axes)) int64: column index of each target per axis

    @property
    def n_strata(self) -> int:
        return len(self.keys)

    def n_targets_per_stratum(self) -> np.ndarray:
        if self.target_cols.size == 0:
            return np.zeros(self.n_strata, dtype=np.int64)
        return np.bincount(self.target_cols.ravel(), minlength=self.n_strata).astype(np.int64)


def stratum_sort_key(axis: str, value: str):
    if axis == "depth":
        return (0, int(value), "")
    return (1, 0, str(value))


def build_layout(targets: Targets, axes=AXES) -> StrataLayout:
    """Column layout for one target set. ("all", "all") is always column 0,
    even with zero targets; other strata exist only if a target has them."""
    axes = tuple(axes)
    if not axes or axes[0] != "all":
        raise ValueError("axes must start with 'all'")
    n = len(targets)
    keys = [("all", "all")]
    cols = [np.zeros(n, dtype=np.int64)]
    for axis in axes[1:]:
        values = [str(v) for v in targets.stratum_values(axis)]
        uniq = sorted(set(values), key=lambda v, a=axis: stratum_sort_key(a, v))
        base = len(keys)
        lut = {v: base + i for i, v in enumerate(uniq)}
        keys.extend((axis, v) for v in uniq)
        cols.append(np.fromiter((lut[v] for v in values), dtype=np.int64, count=n))
    target_cols = np.stack(cols, axis=1) if n else np.empty((0, len(axes)), dtype=np.int64)
    return StrataLayout(axes, tuple(keys), target_cols)


class ToleranceScorer:
    """Precomputed any-match counter for one file, one target set and one k."""

    def __init__(self, target_offsets, target_cols, n_strata: int, n_bytes: int, k: int):
        if int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        self.k = int(k)
        self.n_bytes = int(n_bytes)
        self.n_strata = int(n_strata)
        self.targets = np.asarray(target_offsets, dtype=np.int64)
        if self.targets.ndim != 1:
            raise ValueError("target offsets must be 1-D")
        if self.targets.size > 1 and np.any(np.diff(self.targets) <= 0):
            raise ValueError("target offsets must be strictly increasing")
        if self.targets.size and (self.targets[0] < 0 or self.targets[-1] > self.n_bytes):
            raise ValueError("target offsets must lie in [0, n_bytes]")
        tc = np.asarray(target_cols, dtype=np.int64).reshape(self.targets.size, -1)
        if tc.size and (tc.min() < 0 or tc.max() >= self.n_strata):
            raise ValueError("target column index out of range")
        self._build_precision(tc)
        self._build_recall(tc)

    @property
    def n_targets(self) -> int:
        return int(self.targets.size)

    @property
    def n_signatures(self) -> int:
        return self._G

    # -- precision ---------------------------------------------------------
    def _build_precision(self, tc: np.ndarray) -> None:
        self._G = 0
        self._sig = None
        self._MT = None
        n, k, T = self.n_bytes, self.k, self.targets
        if T.size == 0 or n == 0:
            return
        lo = np.clip(T - k, 0, n)
        hi = np.clip(T + k + 1, 0, n)
        diff = np.bincount(lo, minlength=n + 1)[: n + 1] - np.bincount(hi, minlength=n + 1)[: n + 1]
        hit = np.flatnonzero(np.cumsum(diff[:n]) > 0)
        if hit.size == 0:
            return
        rlo = np.searchsorted(T, hit - k, side="left")
        rhi = np.searchsorted(T, hit + k, side="right")
        base = T.size + 1
        ukey, inv = np.unique(rlo * base + rhi, return_inverse=True)
        index: dict = {}
        members_list: list = []
        range_sig = np.empty(ukey.size, dtype=np.int64)
        for i, key in enumerate(ukey.tolist()):
            lo_i, hi_i = divmod(key, base)
            if hi_i - lo_i == 1:
                members = tuple(sorted(tc[lo_i].tolist()))
            else:
                members = tuple(np.unique(tc[lo_i:hi_i]).tolist())
            sid = index.get(members)
            if sid is None:
                sid = len(members_list)
                index[members] = sid
                members_list.append(members)
            range_sig[i] = sid
        G = len(members_list)
        sig = np.full(n, G, dtype=np.int32 if G < np.iinfo(np.int32).max else np.int64)
        sig[hit] = range_sig[inv.reshape(-1)]
        lens = np.fromiter((len(m) for m in members_list), dtype=np.int64, count=G)
        rows = np.repeat(np.arange(G, dtype=np.int64), lens)
        cols = np.fromiter((c for m in members_list for c in m), dtype=np.int64, count=int(lens.sum()))
        M = sparse.csr_matrix(
            (np.ones(cols.size, dtype=np.int64), (rows, cols)), shape=(G, self.n_strata)
        )
        self._G = G
        self._sig = sig
        self._MT = M.T.tocsr()

    def precision_counts(self, pred, *, check: bool = True) -> np.ndarray:
        pred = np.asarray(pred)
        if pred.ndim != 2:
            raise ValueError("pred must be 2-D (rows x boundaries)")
        rows, m = pred.shape
        out = np.zeros((rows, self.n_strata), dtype=np.int64)
        if rows == 0 or m == 0 or self._G == 0:
            return out
        if check and (pred.min() < 0 or pred.max() >= self.n_bytes):
            raise ValueError("predicted offsets must lie in [0, n_bytes)")
        G1 = self._G + 1
        s = self._sig[pred].astype(np.int64)
        s += (np.arange(rows, dtype=np.int64) * G1)[:, None]
        cnt = np.bincount(s.ravel(), minlength=rows * G1).reshape(rows, G1)[:, : self._G]
        res = self._MT @ np.ascontiguousarray(cnt.T)
        return np.ascontiguousarray(np.asarray(res, dtype=np.int64).T)

    # -- recall ------------------------------------------------------------
    def _build_recall(self, tc: np.ndarray) -> None:
        self._groups = []
        if self.targets.size == 0:
            return
        for a in range(tc.shape[1]):
            colvals = tc[:, a]
            perm = np.argsort(colvals, kind="stable")
            sorted_cols = colvals[perm]
            starts = np.flatnonzero(np.r_[True, sorted_cols[1:] != sorted_cols[:-1]])
            identity = bool(np.array_equal(perm, np.arange(perm.size)))
            self._groups.append((None if identity else perm, starts, sorted_cols[starts]))

    def recall_counts(self, pred, *, check: bool = True) -> np.ndarray:
        pred = np.asarray(pred)
        if pred.ndim != 2:
            raise ValueError("pred must be 2-D (rows x boundaries)")
        rows, m = pred.shape
        out = np.zeros((rows, self.n_strata), dtype=np.int64)
        if rows == 0 or m == 0 or self.targets.size == 0:
            return out
        if check:
            if pred.min() < 0 or pred.max() >= self.n_bytes:
                raise ValueError("predicted offsets must lie in [0, n_bytes)")
            if m > 1 and np.any(np.diff(pred, axis=1) < 0):
                raise ValueError("each prediction row must be sorted ascending")
        W = self.n_bytes + 2 * self.k + 2
        offs = np.arange(rows, dtype=np.int64)[:, None] * W
        flat = (pred.astype(np.int64) + offs).ravel()
        q = ((self.targets - self.k)[None, :] + offs).ravel()
        idx = np.searchsorted(flat, q, side="left")
        in_range = idx < flat.size
        np.minimum(idx, flat.size - 1, out=idx)
        upper = ((self.targets + self.k)[None, :] + offs).ravel()
        matched = (in_range & (flat[idx] <= upper)).reshape(rows, self.targets.size)
        mu8 = matched.view(np.uint8)
        for perm, starts, cols in self._groups:
            src = mu8 if perm is None else mu8[:, perm]
            out[:, cols] = np.add.reduceat(src, starts, axis=1, dtype=np.int64)
        return out

    def counts(self, pred, *, check: bool = True) -> tuple:
        return self.precision_counts(pred, check=check), self.recall_counts(pred, check=check)


def match_counts(pred, targets, n_bytes: int, k: int) -> tuple:
    """(TPp, TPr) for one boundary set against one target set, no strata."""
    T = np.unique(np.asarray(targets, dtype=np.int64))
    P = np.unique(np.asarray(pred, dtype=np.int64))[None, :]
    scorer = ToleranceScorer(T, np.zeros((T.size, 1), dtype=np.int64), 1, n_bytes, k)
    return int(scorer.precision_counts(P)[0, 0]), int(scorer.recall_counts(P)[0, 0])


# --- span IoU (secondary metric) ---------------------------------------------


def _range_max(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """max(values[lo..hi]) inclusive, vectorised with a sparse table."""
    if lo.size == 0:
        return np.empty(0, dtype=values.dtype)
    length = hi - lo + 1
    level = np.frexp(length.astype(np.float64))[1].astype(np.int64) - 1  # floor(log2(length))
    max_level = int(level.max())
    tables = [values]
    for j in range(1, max_level + 1):
        prev = tables[-1]
        half = 1 << (j - 1)
        tables.append(np.maximum(prev[:-half], prev[half:]))
    out = np.empty(lo.size, dtype=values.dtype)
    for j in np.unique(level).tolist():
        sel = level == j
        t = tables[j]
        out[sel] = np.maximum(t[lo[sel]], t[hi[sel] - (1 << j) + 1])
    return out


def best_patch_iou(span_starts, span_ends, patch_starts, n_bytes: int) -> np.ndarray:
    """For each span [s, e) with e > s: max over single patches of
    |span ∩ patch| / |span ∪ patch|. patch_starts must start at 0, be strictly
    increasing and < n_bytes (the last patch runs to n_bytes).

    Only the patches containing s and e-1 can straddle the span. Every patch in
    between lies inside it, with IoU = len / (e - s), so the longest one wins
    (range-maximum query)."""
    s = np.asarray(span_starts, dtype=np.int64)
    e = np.asarray(span_ends, dtype=np.int64)
    if s.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(e <= s) or s.min() < 0 or e.max() > n_bytes:
        raise ValueError("spans must be non-empty and inside [0, n_bytes]")
    ps = np.asarray(patch_starts, dtype=np.int64)
    if ps.size == 0 or ps[0] != 0 or np.any(np.diff(ps) <= 0) or ps[-1] >= n_bytes:
        raise ValueError("patch_starts must start at 0, increase strictly and stay below n_bytes")
    pe = np.append(ps[1:], n_bytes)
    plen = pe - ps
    j0 = np.searchsorted(ps, s, side="right") - 1
    j1 = np.searchsorted(ps, e - 1, side="right") - 1

    def iou(j):
        inter = np.minimum(e, pe[j]) - np.maximum(s, ps[j])
        union = np.maximum(e, pe[j]) - np.minimum(s, ps[j])
        return inter / union

    best = np.maximum(iou(j0), iou(j1))
    mid = (j1 - j0) >= 2
    if mid.any():
        longest = _range_max(plen, j0[mid] + 1, j1[mid] - 1)
        best[mid] = np.maximum(best[mid], longest / (e[mid] - s[mid]))
    return best
