"""Baselines: the density-matched null H0 and the whitespace baseline.

H0 (per file), built from the BLT boundaries file alone:
  - X_f = the init offsets (read from the file, never hard-coded); a = max(X_f).
  - Patches that start before a keep their BLT layout (the forced prefix).
  - The lengths of the BLT patches starting at >= a (these sum to n_bytes - a)
    are uniformly permuted and laid out consecutively from a. The candidate
    boundaries are a + cumsum(perm)[:-1], all strictly inside (a, n_bytes).
  - The scored set has |B_f| boundaries: B_f counts the 'entropy' rows. With no
    length caps, |B_f| == #candidates and every candidate is used. With length
    caps, |B_f| < #candidates and |B_f| candidates are chosen uniformly without
    replacement.
  - Preserved: boundary count, multiset of patch lengths, exact span, forced
    prefix. Randomised: the order of the patch lengths, hence positions.
  - Scored 'entropy' offsets below a are kept in place as part of the forced
    prefix. That only happens if init rows are not a prefix of the rows, and a
    BLT file with init rows at bytes 0 (and 1) never has one.

RNG: every file has its own streams,
  Generator(PCG64(SeedSequence([master_seed, file_key, purpose_code]))),
file_key = int.from_bytes(sha256(file_id)[:8], "little"). Results are therefore
independent of file processing order. Resample r of a file is identical however
the R resamples are chunked: Generator.permuted and Generator.random consume
their streams row by row (checked in tests/test_eval_baselines.py).

Whitespace baseline: W_f = unique byte_offset of kind == 'newline' rows
(stream_b/whitespace_extractor.py) minus X_f. It is not density matched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

PURPOSE_H0_PERMUTATION = 101  # order of patch lengths for scored H0 sets
PURPOSE_H0_SUBSAMPLE = 102    # which candidates are kept when |B_f| < #candidates
PURPOSE_H0_IOU = 103          # independent full H0 layouts for the IoU metric


def file_key(file_id: str) -> int:
    return int.from_bytes(hashlib.sha256(file_id.encode("utf-8")).digest()[:8], "little")


def file_rng(master_seed: int, file_id: str, purpose: int) -> Generator:
    if int(master_seed) < 0:
        raise ValueError("master_seed must be a non-negative integer")
    return Generator(PCG64(SeedSequence([int(master_seed), file_key(file_id), int(purpose)])))


@dataclass(frozen=True)
class H0Spec:
    n_bytes: int
    forced_offsets: np.ndarray  # X_f
    a: int                      # max(X_f)
    prefix_starts: np.ndarray   # BLT patch starts < a, kept as-is
    post_lengths: np.ndarray    # int64 lengths of BLT patches starting at >= a, in file order
    fixed_scored: np.ndarray    # scored offsets < a, kept in place (normally empty)
    n_select: int               # scored offsets > a, re-drawn among the candidates

    @property
    def n_candidates(self) -> int:
        return max(0, int(self.post_lengths.size) - 1)

    @property
    def n_scored(self) -> int:
        return int(self.fixed_scored.size) + int(self.n_select)


def h0_spec(byte_offsets, triggers, patch_lengths, n_bytes: int, scored_trigger: str = "entropy") -> H0Spec:
    off = np.asarray(byte_offsets, dtype=np.int64)
    trig = np.asarray([str(t) for t in triggers], dtype=object)
    plen = np.asarray(patch_lengths, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    if n_bytes == 0:
        if off.size:
            raise ValueError("n_bytes == 0 but boundaries are present")
        return H0Spec(0, empty, 0, empty, empty, empty, 0)
    forced = off[trig == "init"]
    if forced.size == 0:
        raise ValueError("no 'init' rows: the forced prefix is undefined")
    a = int(forced.max())
    post = off >= a
    post_lengths = plen[post]
    if int(post_lengths.sum()) != n_bytes - a:
        raise ValueError(f"patch lengths from offset {a} sum to {int(post_lengths.sum())}, expected {n_bytes - a}")
    scored = off[trig == scored_trigger]
    fixed = scored[scored < a]
    n_select = int(np.count_nonzero(scored > a))
    n_candidates = max(0, post_lengths.size - 1)
    if n_select > n_candidates:
        raise ValueError(f"{n_select} scored boundaries but only {n_candidates} H0 candidate positions")
    return H0Spec(int(n_bytes), forced, a, off[off < a], post_lengths, fixed, n_select)


class H0Sampler:
    """Draws H0 boundary sets for one file; see the module docstring."""

    def __init__(self, spec: H0Spec, master_seed: int, file_id: str):
        self.spec = spec
        self._rng_perm = file_rng(master_seed, file_id, PURPOSE_H0_PERMUTATION)
        self._rng_sub = file_rng(master_seed, file_id, PURPOSE_H0_SUBSAMPLE)
        self._rng_iou = file_rng(master_seed, file_id, PURPOSE_H0_IOU)

    def _permuted_lengths(self, rng: Generator, rows: int) -> np.ndarray:
        lengths = self.spec.post_lengths
        arr = np.empty((rows, lengths.size), dtype=np.int64)
        arr[:] = lengths
        rng.permuted(arr, axis=1, out=arr)
        return arr

    def draw(self, rows: int, *, return_candidates: bool = False):
        """(rows x n_scored) int64 H0 sets, each row strictly increasing.
        With return_candidates, also the (rows x n_candidates) full candidate
        layouts the rows were chosen from (None if nothing was drawn)."""
        spec = self.spec
        if rows < 0:
            raise ValueError("rows must be >= 0")
        candidates = None
        if spec.n_select == 0:
            chosen = np.empty((rows, 0), dtype=np.int64)
        else:
            perm = self._permuted_lengths(self._rng_perm, rows)
            candidates = np.cumsum(perm[:, :-1], axis=1)
            candidates += spec.a
            if spec.n_select < spec.n_candidates:
                keys = self._rng_sub.random((rows, spec.n_candidates))
                idx = np.argpartition(keys, spec.n_select - 1, axis=1)[:, : spec.n_select]
                idx.sort(axis=1)
                chosen = np.take_along_axis(candidates, idx, axis=1)
            else:
                chosen = candidates
        if spec.fixed_scored.size:
            fixed = np.broadcast_to(spec.fixed_scored, (rows, spec.fixed_scored.size))
            chosen = np.concatenate([fixed, chosen], axis=1)
        if return_candidates:
            return chosen, candidates
        return chosen

    def draw_layouts(self, rows: int) -> np.ndarray:
        """(rows x n_patches) patch starts of full H0 segmentations (forced
        prefix, then the permuted lengths from a), from an independent stream.
        Used for IoU, where the whole patch layout matters, not just the scored subset."""
        spec = self.spec
        n_prefix = spec.prefix_starts.size
        m = spec.post_lengths.size
        starts = np.empty((rows, n_prefix + m), dtype=np.int64)
        starts[:, :n_prefix] = spec.prefix_starts
        if m:
            starts[:, n_prefix] = spec.a
            if m > 1:
                perm = self._permuted_lengths(self._rng_iou, rows)
                starts[:, n_prefix + 1:] = spec.a + np.cumsum(perm[:, :-1], axis=1)
        return starts


def whitespace_boundaries(newline_offsets, exclude_offsets=()) -> np.ndarray:
    """W_f: unique newline offsets minus X_f."""
    w = np.unique(np.asarray(newline_offsets, dtype=np.int64))
    excl = np.asarray(list(exclude_offsets), dtype=np.int64)
    if excl.size:
        w = w[~np.isin(w, excl)]
    return w


def whitespace_segment_starts(newline_offsets, n_bytes: int) -> np.ndarray:
    """Line segmentation for IoU: segments start at 0 and at every newline offset."""
    if n_bytes == 0:
        return np.empty(0, dtype=np.int64)
    w = np.unique(np.asarray(newline_offsets, dtype=np.int64))
    w = w[(w > 0) & (w < n_bytes)]
    return np.concatenate([np.zeros(1, dtype=np.int64), w])


# Word-boundary baseline (the P3 prose confound control). Constituent starts in prose
# largely coincide with word starts (after a space), and the whitespace baseline is
# newline-only - empty for single-line paragraphs. This baseline splits at every word
# start instead, so BLT can be compared against word segmentation directly.
_WORD_TABLE = np.zeros(256, dtype=bool)
for _c in b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_":
    _WORD_TABLE[_c] = True


def word_start_offsets(content: bytes) -> np.ndarray:
    """Byte offsets of word starts: byte i is a word byte ([A-Za-z0-9_]) and byte i-1
    is not (or i == 0). Multi-byte UTF-8 continuation bytes (>= 0x80) are never word
    bytes, so this is a byte-level segmentation control, not a tokenizer."""
    b = np.frombuffer(content, dtype=np.uint8)
    if b.size == 0:
        return np.empty(0, dtype=np.int64)
    is_word = _WORD_TABLE[b]
    starts = np.zeros(b.size, dtype=bool)
    starts[0] = is_word[0]
    starts[1:] = is_word[1:] & ~is_word[:-1]
    return np.flatnonzero(starts).astype(np.int64)


def word_boundaries(content: bytes, exclude_offsets=()) -> np.ndarray:
    """S_f: unique word-start offsets minus X_f (same exclusion convention as the
    whitespace baseline). Not density matched."""
    w = word_start_offsets(content)
    excl = np.asarray(list(exclude_offsets), dtype=np.int64)
    if excl.size:
        w = w[~np.isin(w, excl)]
    return w


def word_segment_starts(content: bytes, n_bytes: int) -> np.ndarray:
    """Word segmentation for IoU: segments start at 0 and at every word start."""
    if n_bytes == 0:
        return np.empty(0, dtype=np.int64)
    w = word_start_offsets(content)
    w = w[(w > 0) & (w < n_bytes)]
    return np.concatenate([np.zeros(1, dtype=np.int64), w])
