"""The BLT patcher (entropy model) as a distillation teacher, with two interchangeable
backends:

  * "transformers"  -- transformers' own BltPatcher (stream_a.patcher_model.load_patcher),
                       the exact object used for every number in the report. Preferred on
                       Ada, where transformers 5.17 is installed.
  * "reference"     -- stream_a.reference_lm.ReferenceEntropyModel, a pure-torch
                       re-implementation of the same architecture on the same weights
                       (fidelity gate G3: max |delta logits| 1.05e-4 vs the HF port). Needs
                       no transformers/xformers, so it runs anywhere torch runs.

Either way we only need, for a batch of <=512-byte windows, the teacher's next-token
logits [B, L, 260] with no gradient. On <=512-token windows the swa512 sliding window
equals full causal attention (docs/STREAM_A_FIDELITY.md G3), so a single causal forward is
exact and needs no 8192-token chunking. Right padding is harmless under causal attention.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

TENSOR_PREFIX = "model.patcher."
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class Teacher:
    def __init__(self, backend, model, device):
        self.backend = backend
        self.model = model
        self.device = device

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, L] on device, L <= 513. Returns float32 logits [B, L, 260]."""
        if self.backend == "reference":
            return self.model.logits(input_ids, mask=None).float()   # causal == swa512 for L<=512
        from stream_a.patcher_model import attention_mask_4d
        mask = attention_mask_4d(input_ids.shape[1], "swa512", input_ids.device)
        _, _, logits = self.model(input_ids=input_ids, attention_mask=mask)
        return logits.float()


def _load_hf_state(ckpt_dir: Path, device: str, dtype: str):
    raw = load_file(str(ckpt_dir / "patcher.safetensors"), device=device)
    if not all(k.startswith(TENSOR_PREFIX) for k in raw):
        raise ValueError("patcher.safetensors contains tensors outside model.patcher.*")
    state = {k[len(TENSOR_PREFIX):]: v.to(DTYPES[dtype]) for k, v in raw.items()}
    return state


def load_teacher(ckpt_dir: str, device: str = "cuda", dtype: str = "float32",
                 backend: str = "auto") -> Teacher:
    ckpt_dir = Path(ckpt_dir)
    if backend == "auto":
        try:
            import transformers  # noqa: F401
            backend = "transformers"
        except Exception:
            backend = "reference"

    if backend == "transformers":
        from stream_a.patcher_model import load_patcher
        model, _ = load_patcher(ckpt_dir, device=device, dtype=dtype, attn_implementation="sdpa")
        return Teacher("transformers", model, device)

    if backend == "reference":
        from stream_a.reference_lm import ReferenceEntropyModel, to_reference_names
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        pc = cfg.get("patcher_config", cfg)
        hf_state = _load_hf_state(ckpt_dir, device, dtype)
        ref_state = to_reference_names(hf_state)
        model = ReferenceEntropyModel(
            ref_state,
            n_layers=pc["num_hidden_layers"], dim=pc["hidden_size"],
            n_heads=pc["num_attention_heads"], norm_eps=pc.get("norm_eps", 1e-5),
            rope_theta=pc.get("rope_theta", 10000.0),
            max_seqlen=pc.get("max_position_embeddings", 8192),
            device=device, dtype=DTYPES[dtype])
        return Teacher("reference", model, device)

    raise ValueError(f"unknown teacher backend {backend!r}")


# Back-compat helper used by train.py / evaluate.py.
def teacher_logits(teacher: Teacher, input_ids: torch.Tensor) -> torch.Tensor:
    return teacher.logits(input_ids)
