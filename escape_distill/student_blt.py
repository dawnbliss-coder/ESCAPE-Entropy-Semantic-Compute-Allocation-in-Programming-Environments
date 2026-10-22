"""A small BLT-style byte-level entropy model, written from scratch.

This is the *student* of the distillation experiment (escape_distill). It is deliberately
a faithful-but-small reimplementation of the BLT patcher (the entropy model), so that the
student reproduces the teacher's job -- predict the next byte and, through that, where
patch boundaries fall -- while being trained only on the teacher's soft outputs.

Architecture choices mirror the reference patcher documented in docs/STREAM_A_FIDELITY.md:

  * vocabulary 260: ids 0..3 are special (the pipeline uses BOS=1), byte b -> id b+4;
  * pre-norm residual transformer blocks;
  * RMSNorm(eps=1e-5);
  * rotary position embeddings (RoPE), theta 10000, interleaved pairs;
  * SwiGLU feed-forward  down(silu(gate x) * up x);
  * attention scale head_dim ** -0.5.

We train and evaluate on windows of at most 512 bytes. On such windows a 512-token
sliding window is identical to full causal attention (docs/STREAM_A_FIDELITY.md G3:
"swa512 vs causal on <= 512-token inputs: 0.0"), so the student uses plain causal
attention and is directly comparable to the swa512 teacher on the same windows.

The config here is a small default (a few million parameters). It is NOT the 100M-param
checkpoint; the point of the experiment is to see how well a small from-scratch student can
absorb the teacher's next-byte distribution, not to re-train BLT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 260          # reference tokenizer: 256 bytes + 4 special ids
BOS_ID = 1
BYTE_TOKEN_OFFSET = 4
PAD_ID = 0


@dataclass
class StudentConfig:
    vocab_size: int = VOCAB_SIZE
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn_mult: float = 8 / 3      # SwiGLU keeps ~parameter-parity with a 4x dense FFN
    ffn_multiple_of: int = 64
    max_seq_len: int = 513       # BOS + up to 512 bytes
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def ffn_hidden(self) -> int:
        h = int(self.ffn_mult * self.dim)
        return self.ffn_multiple_of * ((h + self.ffn_multiple_of - 1) // self.ffn_multiple_of)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """cos/sin tables for interleaved-pair RoPE, shape [seq_len, head_dim]."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)                      # [seq_len, half]
    ang = torch.repeat_interleave(ang, 2, dim=-1)    # [seq_len, head_dim], pairs share angle
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """(x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...) for interleaved RoPE pairs."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, Dh]; cos/sin: [L, Dh]
    cos = cos[None, None]
    sin = sin[None, None]
    return x * cos + rotate_half_interleaved(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Plain causal attention: valid because L <= 512 == sliding window.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        h = cfg.ffn_hidden()
        self.gate = nn.Linear(cfg.dim, h, bias=False)
        self.up = nn.Linear(cfg.dim, h, bias=False)
        self.down = nn.Linear(h, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentBLT(nn.Module):
    """From-scratch BLT entropy model. Input token ids [B, L] -> next-token logits [B, L, V]."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._rope = None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, L, device, dtype):
        head_dim = self.cfg.dim // self.cfg.n_heads
        if self._rope is None or self._rope[0].shape[0] < L or self._rope[0].device != device:
            self._rope = build_rope_cache(max(L, self.cfg.max_seq_len), head_dim,
                                          self.cfg.rope_theta, device, dtype)
        cos, sin = self._rope
        return cos[:L].to(dtype), sin[:L].to(dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self._rope_cache(L, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def entropy_nats(logits: torch.Tensor) -> torch.Tensor:
    """H = -sum_v p log p, natural log -- identical definition to the teacher
    (stream_a.patcher_model.entropy_nats)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


if __name__ == "__main__":
    cfg = StudentConfig()
    m = StudentBLT(cfg)
    print("config:", asdict(cfg))
    print("ffn_hidden:", cfg.ffn_hidden())
    print("parameters:", f"{m.num_parameters():,}")
    ids = torch.randint(4, 260, (2, 128))
    ids[:, 0] = BOS_ID
    out = m(ids)
    print("logits:", tuple(out.shape))
    print("entropy sample:", entropy_nats(out)[0, :5].tolist())
