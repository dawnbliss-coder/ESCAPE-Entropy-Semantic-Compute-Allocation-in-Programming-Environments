"""Array implementations of `metrics.match`/`counts`/`shuffled_boundaries(fixed_eof)`.

Results are identical to the reference functions in metrics.py (tests/scoring_v1/
test_fast.py checks this on random inputs and on whole scoring runs):

- matching makes the same greedy ordered decisions, but skips runs of unmatched
  endpoints by binary search instead of one step at a time;
- the null consumes the generator through the same single `rng.permutation(count)`
  call, so every permutation is the same draw as the reference.

numba compiles the matching loop when installed; otherwise a pure-Python loop with
the same logic runs (correct, but slow on real corpora).
"""
from __future__ import annotations

import numpy as np

from .metrics import InvalidNull

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
    HAVE_NUMBA = False
else:
    HAVE_NUMBA = True


@njit(cache=True)
def _lower_bound(a, lo, value):
    hi = a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def match_count(p, t, k):
    """len(metrics.match(p, t, k)) for strictly increasing int64 arrays p, t."""
    i = 0
    j = 0
    n = 0
    while i < p.shape[0] and j < t.shape[0]:
        if p[i] < t[j] - k:
            i = _lower_bound(p, i + 1, t[j] - k)
        elif t[j] < p[i] - k:
            j = _lower_bound(t, j + 1, p[i] - k)
        else:
            n += 1
            i += 1
            j += 1
    return n


def as_sorted(values):
    return np.unique(np.asarray(list(values), dtype=np.int64))


def counts_fast(pred, targets, k):
    """metrics.counts for arrays already passed through `as_sorted`."""
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")
    tp = match_count(pred, targets, k) if pred.size and targets.size else 0
    return np.array([tp, pred.size - tp, targets.size - tp], dtype=float)


def f1_of(c):
    """metrics(c)["f1"] for count rows c[..., (tp, fp, fn)]."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den), where=den != 0)


class FixedEofNull:
    """shuffled_boundaries(patches, n_bytes, rng, "fixed_eof") as a reusable sampler."""

    def __init__(self, patches, n_bytes):
        lengths = np.array([p["end_byte"] - p["start_byte"] for p in patches], dtype=np.int64)
        reasons = [p["termination_reason"] for p in patches]
        # Same validity checks, in the same order, as the reference.
        if any(length <= 0 or reason not in {"entropy", "max_length", "eof"}
               for length, reason in zip(lengths.tolist(), reasons)) or int(lengths.sum()) != n_bytes:
            raise InvalidNull("invalid patch lengths/labels/span")
        if reasons and (reasons[-1] != "eof" or sum(r == "eof" for r in reasons) != 1):
            raise InvalidNull("observed patches need exactly one final EOF pair")
        self.count = len(reasons) - (1 if reasons else 0)
        self.lengths = lengths[:self.count]
        self.entropy = np.array([r == "entropy" for r in reasons[:self.count]], dtype=bool)
        self.n_bytes = n_bytes

    def draw(self, rng):
        order = rng.permutation(self.count)
        # The final EOF patch is appended after these, so it never yields a boundary
        # and every endpoint here is < n_bytes; lengths are positive, so ends > 0.
        ends = np.cumsum(self.lengths[order])
        return ends[self.entropy[order]]


if HAVE_NUMBA:
    # Compile once at import, so forked workers inherit the machine code.
    match_count(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), 0)
