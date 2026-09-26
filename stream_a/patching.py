"""Pure boundary logic (numpy only, no torch): per-byte entropy -> patch starts ->
trigger classification -> patch lengths -> BPP statistics.

Index convention (derived and checked in docs/STREAM_A_FIDELITY.md): the patcher reads
tokens [BOS, b_0+4, ..., b_{n-1}+4]. The prediction made at token position j is the
distribution of token j+1, so byte i (token i+1) is predicted at position i and

    entropies[i] = H(p(b_i | BOS, b_0 .. b_{i-1}))        (nats)

HF `BltPatcher.patch_lengths_from_entropies` (modeling_blt.py) and the reference
`find_entropy_patch_start_ids` (facebookresearch/blt bytelatent/data/patcher.py) both
force tokens 0 (BOS) and 1 (byte 0) to start patches, and start a patch at token
t >= 2 iff entropy[t-1] > threshold (strict). In byte coordinates:

    byte 0 starts a patch                              -> trigger "init"
    byte i >= 1 starts a patch iff entropies[i] > tau  -> trigger "entropy"

With max_patch_length = m, every patch longer than m is cut into consecutive pieces of
m bytes (HF `process_patch_lengths` == reference `split_large_numbers`); each cut is a
"length_cap" boundary. The BOS token's own one-token patch is not a byte and is dropped.

The threshold comparison is done in float64 on the stored float32 entropies, so the
stored values alone reproduce every trigger decision.
"""

from dataclasses import dataclass

import numpy as np

TRIGGER_INIT = "init"
TRIGGER_ENTROPY = "entropy"
TRIGGER_LENGTH_CAP = "length_cap"
_TRIGGER_DTYPE = "<U10"

BOS_ID = 1  # reference tokenizers/constants.py BOS_ID; checkpoint tokenizer.json "<s>" = 1
BYTE_TOKEN_OFFSET = 4  # reference constants.py OFFSET: byte b -> token id b + 4


def bytes_to_token_ids(content: bytes) -> np.ndarray:
    """[BOS] + [b + 4 for b in content]. No EOS: it would only add a prediction slot after
    the last byte and cannot change any earlier prediction under causal attention."""
    ids = np.empty(len(content) + 1, dtype=np.int64)
    ids[0] = BOS_ID
    ids[1:] = np.frombuffer(content, dtype=np.uint8).astype(np.int64) + BYTE_TOKEN_OFFSET
    return ids


def byte_entropies_from_token_entropies(token_entropies, n_bytes: int) -> np.ndarray:
    """Token position j predicts token j+1, and byte i is token i+1, so byte i's entropy is
    the prediction made at token position i. The last position (predicting past the end)
    is dropped."""
    te = np.asarray(token_entropies)
    if te.shape != (n_bytes + 1,):
        raise ValueError(f"expected {n_bytes + 1} token entropies for {n_bytes} bytes, got {te.shape}")
    return te[:n_bytes].astype(np.float32)


@dataclass(frozen=True)
class Patches:
    starts: np.ndarray    # int64[n_patches], first byte of each patch, strictly increasing
    triggers: np.ndarray  # <U10[n_patches], init | entropy | length_cap
    lengths: np.ndarray   # int32[n_patches], sums to n_bytes

    @property
    def n_patches(self) -> int:
        return int(self.starts.shape[0])


def compute_patches(entropies, tau: float, max_patch_length=None) -> Patches:
    ent = np.asarray(entropies)
    if ent.ndim != 1:
        raise ValueError(f"entropies must be 1-D, got shape {ent.shape}")
    n = ent.shape[0]
    if n == 0:
        return Patches(np.zeros(0, np.int64), np.zeros(0, _TRIGGER_DTYPE), np.zeros(0, np.int32))

    entropy_starts = np.flatnonzero(ent[1:].astype(np.float64) > float(tau)) + 1
    starts = np.concatenate(([0], entropy_starts)).astype(np.int64)
    triggers = np.full(starts.shape[0], TRIGGER_ENTROPY, dtype=_TRIGGER_DTYPE)
    triggers[0] = TRIGGER_INIT

    if max_patch_length is not None:
        m = int(max_patch_length)
        if m < 1:
            raise ValueError(f"max_patch_length must be >= 1, got {max_patch_length}")
        lengths = np.diff(np.append(starts, n))
        n_cuts = (lengths - 1) // m  # a patch of length L gets ceil(L/m) - 1 cuts
        total_cuts = int(n_cuts.sum())
        if total_cuts:
            owner = np.repeat(np.arange(starts.shape[0]), n_cuts)
            first_cut_index = np.repeat(np.cumsum(n_cuts) - n_cuts, n_cuts)
            within = np.arange(total_cuts) - first_cut_index
            cut_starts = starts[owner] + m * (within + 1)
            starts = np.concatenate((starts, cut_starts))
            triggers = np.concatenate((triggers, np.full(total_cuts, TRIGGER_LENGTH_CAP, dtype=_TRIGGER_DTYPE)))
            order = np.argsort(starts, kind="stable")
            starts, triggers = starts[order], triggers[order]

    lengths = np.diff(np.append(starts, n)).astype(np.int32)
    return Patches(starts, triggers, lengths)


def byte_starts_from_token_patch_lengths(token_patch_lengths, n_bytes: int) -> np.ndarray:
    """Convert HF/reference token-level patch lengths (over [BOS, bytes...], possibly with
    trailing zero-length patches) into byte-level patch starts. Used only to cross-check
    compute_patches against the model's own patching code."""
    pl = np.asarray(token_patch_lengths).reshape(-1)
    pl = pl[pl > 0]
    token_starts = np.concatenate(([0], np.cumsum(pl)[:-1]))
    byte_starts = token_starts[token_starts >= 1] - 1  # token t is byte t-1; drop BOS
    return byte_starts[byte_starts < n_bytes].astype(np.int64)


def reference_start_ids(token_entropies, threshold: float, include_next_token: bool = True) -> np.ndarray:
    """Line-by-line numpy port of facebookresearch/blt bytelatent/data/patcher.py
    find_entropy_patch_start_ids (threshold branch, monotonicity=False, threshold_add=None)
    for one sequence. Returns token-level start ids. Cross-check only."""
    ent = np.asarray(token_entropies, dtype=np.float64)
    first_ids = np.array([0, 1], dtype=np.int64)
    mask = ent[1:] > threshold
    if not include_next_token:
        mask = mask[:-1]
    return np.concatenate((first_ids, np.flatnonzero(mask) + first_ids.shape[0]))


def bpp_stats(lengths) -> dict:
    """Bytes-per-patch statistics over ALL patches (init, entropy and length-capped)."""
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0:
        nan = float("nan")
        return {"mean_bpp": nan, "var_bpp": nan, "median_bpp": nan, "min_bpp": nan,
                "max_bpp": nan, "p10_bpp": nan, "p90_bpp": nan}
    return {
        "mean_bpp": float(lengths.mean()),
        "var_bpp": float(lengths.var()),  # population variance (ddof=0)
        "median_bpp": float(np.median(lengths)),
        "min_bpp": float(lengths.min()),
        "max_bpp": float(lengths.max()),
        "p10_bpp": float(np.percentile(lengths, 10)),
        "p90_bpp": float(np.percentile(lengths, 90)),
    }


def entropy_stats(entropies, tau: float) -> dict:
    ent = np.asarray(entropies, dtype=np.float64)
    if ent.size == 0:
        nan = float("nan")
        return {"ent_mean": nan, "ent_p10": nan, "ent_p50": nan, "ent_p90": nan, "ent_max": nan,
                "frac_bytes_above_tau": nan}
    return {
        "ent_mean": float(ent.mean()),
        "ent_p10": float(np.percentile(ent, 10)),
        "ent_p50": float(np.percentile(ent, 50)),
        "ent_p90": float(np.percentile(ent, 90)),
        "ent_max": float(ent.max()),
        "frac_bytes_above_tau": float((ent > float(tau)).mean()),
    }


def chunk_context_meta(n_bytes: int, chunk_len) -> dict:
    """Additive BPP-summary columns describing the swa512 8192-token chunk resets.

    The token stream is [BOS] + n_bytes byte tokens, cut into independent chunks of
    chunk_len tokens. A chunk boundary falls strictly inside the stream at every
    multiple of chunk_len < n_bytes + 1, i.e. n_chunk_edges = n_bytes // chunk_len.
    Byte i is predicted by token i+1; bytes whose predicting token lies in chunk >= 2
    (token index >= chunk_len, i.e. byte index >= chunk_len - 1) have truncated
    context. Only files longer than chunk_len - 1 bytes are affected; downstream
    consumers can flag or stratify by these columns without re-deriving the rule."""
    if chunk_len is None:
        return {"chunk_len": None, "n_chunk_edges": 0, "n_bytes_truncated_context": 0}
    c = int(chunk_len)
    if c < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    n = int(n_bytes)
    return {
        "chunk_len": c,
        "n_chunk_edges": n // c,
        "n_bytes_truncated_context": max(0, n - (c - 1)),
    }
