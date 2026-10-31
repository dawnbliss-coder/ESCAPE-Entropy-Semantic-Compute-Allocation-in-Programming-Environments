"""Independent re-implementation of the reference entropy model's forward pass, written
from facebookresearch/blt @ 9774ed4f (bytelatent/transformer.py LMTransformer and
bytelatent/base_transformer.py), NOT from transformers' modeling_blt.py.

Used only by the fidelity battery, to check that the HF port computes the same function
as the original architecture on the same weights.

What agreement proves:
  - HF's RoPE pairing, RMSNorm, attention scaling, residual layout and SiLU-gated FFN
    equal the reference's.
  - The explicit masks (causal, sliding window) behave as intended.
What it cannot prove: the conversion assigned each tensor to the right role (e.g.
gate vs up). Both implementations consume the same role-assigned tensors. Role errors
would instead show up as a broken language model, which the behavioural checks in the
battery test (repetition, continuation, bits-per-byte).

Reference code -> this file:
  precompute_freqs_cis   base_transformer.py:93-123
  apply_rotary_emb       base_transformer.py:155-168  (pairs (x0,x1),(x2,x3),...)
  Attention.forward      base_transformer.py:361-425  (sdpa branch)
  FeedForward.forward    base_transformer.py:484-489  (w2(silu(w1 x) * w3 x))
  TransformerBlock       base_transformer.py:548-566  (pre-norm residual)
  LMTransformer.forward  transformer.py:105-137       (embed -> layers -> norm -> output)
"""

import torch
import torch.nn.functional as F

# HF parameter name -> reference parameter name (per layer where applicable).
HF_TO_REFERENCE = {
    "embed_tokens.weight": "tok_embeddings.weight",
    "norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}


def to_reference_names(hf_state: dict) -> dict:
    out = {}
    for name, tensor in hf_state.items():
        if name.startswith("layers."):
            _, idx, rest = name.split(".", 2)
            out[f"layers.{idx}.{HF_TO_REFERENCE[rest]}"] = tensor
        else:
            out[HF_TO_REFERENCE[name]] = tensor
    return out


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    fc = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[3], 2, 2).float()
    xq_out = (xq_ * fc).sum(5).flatten(3)
    xk_out = (xk_ * fc).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class ReferenceEntropyModel:
    def __init__(self, reference_state: dict, n_layers=14, dim=768, n_heads=12, norm_eps=1e-5,
                 rope_theta=10000.0, max_seqlen=8192, device="cpu", dtype=torch.float32):
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in reference_state.items()}
        self.n_layers, self.dim, self.n_heads = n_layers, dim, n_heads
        self.head_dim = dim // n_heads
        self.norm_eps = norm_eps
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seqlen, rope_theta).to(device)

    def _rms_norm(self, x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight, self.norm_eps)

    @torch.inference_mode()
    def logits(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens [B, S]; mask: boolean [1|B, 1, S, S] (True = attend) or None for causal."""
        w = self.w
        bsz, seqlen = tokens.shape
        h = F.embedding(tokens, w["tok_embeddings.weight"])
        freq_cis = self.freqs_cis[0:seqlen]
        for i in range(self.n_layers):
            p = f"layers.{i}."
            x = self._rms_norm(h, w[p + "attention_norm.weight"])
            xq = F.linear(x, w[p + "attention.wq.weight"]).view(bsz, seqlen, self.n_heads, self.head_dim)
            xk = F.linear(x, w[p + "attention.wk.weight"]).view(bsz, seqlen, self.n_heads, self.head_dim)
            xv = F.linear(x, w[p + "attention.wv.weight"]).view(bsz, seqlen, self.n_heads, self.head_dim)
            xq, xk = apply_rotary_emb(xq, xk, freq_cis)
            xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
            if mask is None:
                out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
            out = out.transpose(1, 2).reshape(bsz, seqlen, self.dim)
            h = h + F.linear(out, w[p + "attention.wo.weight"])
            hn = self._rms_norm(h, w[p + "ffn_norm.weight"])
            ffn = F.linear(F.silu(F.linear(hn, w[p + "feed_forward.w1.weight"])) * F.linear(hn, w[p + "feed_forward.w3.weight"]),
                           w[p + "feed_forward.w2.weight"])
            h = h + ffn
        return F.linear(self._rms_norm(h, w["norm.weight"]), w["output.weight"])
