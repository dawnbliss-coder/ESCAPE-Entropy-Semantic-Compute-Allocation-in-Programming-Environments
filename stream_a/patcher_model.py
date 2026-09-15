"""Stream A model layer: load the BLT patcher (entropy model) from the fetched checkpoint
and compute next-byte entropies under an explicit, recorded context regime.

The only input is bytes. Nothing here reads structure/, whitespace/ or any parse data.

Context regimes (evidence in docs/STREAM_A_FIDELITY.md):
  swa512   Reference-faithful, and the PRIMARY regime.
           - Every layer attends to itself plus the previous 511 tokens. Reference
             entropy_model.py hard-codes sliding_window=512 with xformers
             local_block_causal, and xformers _materialize_causal_mask keeps keys
             i-511..i.
           - The token stream is cut into independent 8192-token chunks, as the
             reference calculate_entropies does.
  hf_full  What transformers' BltPatcher.forward does natively: create_causal_mask, i.e.
           full causal attention over the whole input with no chunking. Sensitivity
           comparison only.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import BltConfig
from transformers.models.blt.modeling_blt import BltPatcher

from stream_a.patching import bytes_to_token_ids

CONTEXT_MODES = ("swa512", "hf_full")
SLIDING_WINDOW = 512
CHUNK_LEN = 8192
PAD_ID = 0  # right padding only; causal/banded masks keep real positions from seeing it
TENSOR_PREFIX = "model.patcher."
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_patcher(ckpt_dir, device="cpu", dtype: str = "float32", attn_implementation: str = "sdpa"):
    """Build transformers' own BltPatcher from the checkpoint's config.json and load the
    fetched `model.patcher.*` tensors with strict key matching."""
    ckpt_dir = Path(ckpt_dir)
    blt_config = BltConfig.from_pretrained(ckpt_dir)
    patcher_config = blt_config.patcher_config
    patcher_config._attn_implementation = attn_implementation
    device = torch.device(device)
    # Build the module and load the weights directly on the target device. This avoids a
    # transient ~600 MB host copy (random-init fp32 parameters plus the CPU state dict), which
    # matters on a host with full swap where earlyoom kills the largest process.
    with device:
        model = BltPatcher(patcher_config)
    raw = load_file(str(ckpt_dir / "patcher.safetensors"), device=str(device))
    if not all(k.startswith(TENSOR_PREFIX) for k in raw):
        raise ValueError("patcher.safetensors contains tensors outside model.patcher.*")
    state = {k[len(TENSOR_PREFIX):]: v for k, v in raw.items()}
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=DTYPES[dtype]).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, blt_config


def attention_mask_4d(length: int, context_mode: str, device) -> torch.Tensor | None:
    """Boolean [1, 1, L, L] mask, True = may attend. None selects the native HF causal
    path (create_causal_mask); an explicit 4D mask is passed through as-is
    (transformers masking_utils: "an already prepared 4D mask ... is returned as-is")."""
    if context_mode == "hf_full":
        return None
    if context_mode != "swa512":
        raise ValueError(f"unknown context_mode {context_mode!r}")
    idx = torch.arange(length, device=device)
    diff = idx[:, None] - idx[None, :]
    return ((diff >= 0) & (diff < SLIDING_WINDOW))[None, None]


def entropy_nats(logits: torch.Tensor) -> torch.Tensor:
    """H = -sum_v p log p over all 260 logits, natural log (reference patcher.py entropy())."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def chunk_spans(n_tokens: int, context_mode: str) -> list:
    if context_mode == "hf_full":
        return [(0, n_tokens)]
    return [(s, min(s + CHUNK_LEN, n_tokens)) for s in range(0, n_tokens, CHUNK_LEN)]


@torch.inference_mode()
def run_chunks(model, chunks: list, context_mode: str, device, return_logits: bool = False) -> list:
    """One forward pass over a batch of token-id chunks (right-padded to a common length).
    Returns [(entropies float32[len], logits float32[len, 260] or None), ...]."""
    length = max(len(c) for c in chunks)
    ids = torch.full((len(chunks), length), PAD_ID, dtype=torch.long)
    for row, chunk in enumerate(chunks):
        ids[row, : len(chunk)] = torch.as_tensor(chunk)
    ids = ids.to(device)
    mask = attention_mask_4d(length, context_mode, device)
    _, _, logits = model(input_ids=ids, attention_mask=mask)
    ent = entropy_nats(logits)
    out = []
    for row, chunk in enumerate(chunks):
        e = ent[row, : len(chunk)].cpu().numpy().astype(np.float32)
        lg = logits[row, : len(chunk)].float().cpu().numpy() if return_logits else None
        out.append((e, lg))
    return out


def token_entropies(model, content: bytes, context_mode: str, device, return_logits: bool = False):
    """Per-token entropies for [BOS] + bytes, chunk by chunk (one sequence per forward)."""
    ids = bytes_to_token_ids(content)
    ents, logits = [], []
    for s, e in chunk_spans(len(ids), context_mode):
        ((ent, lg),) = run_chunks(model, [ids[s:e]], context_mode, device, return_logits)
        ents.append(ent)
        if return_logits:
            logits.append(lg)
    return np.concatenate(ents), (np.concatenate(logits) if return_logits else None)


@torch.inference_mode()
def greedy_continuation(model, prompt: bytes, n_new: int, context_mode: str, device) -> bytes:
    """Greedy next-byte decoding with the patcher itself (a byte LM) - sanity check only."""
    ids = list(bytes_to_token_ids(prompt))
    new = []
    for _ in range(n_new):
        window = ids[-CHUNK_LEN:]
        ((_, lg),) = run_chunks(model, [np.array(window)], context_mode, device, return_logits=True)
        nxt = int(lg[-1].argmax())
        if nxt < 4:  # special token (BOE/BOS/EOS/PAD) - stop
            break
        ids.append(nxt)
        new.append(nxt - 4)
    return bytes(new)
