"""CPU regression tests for the real patcher (the fidelity battery's core claims on short
inputs). Skipped when the checkpoint has not been fetched (python -m stream_a.fetch_patcher)."""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stream_a.fetch_patcher import checkpoint_dir, verify  # noqa: E402

pytestmark = pytest.mark.model

PY = b"def area(r):\n    if r < 0:\n        raise ValueError(r)\n    return 3.14159 * r * r\n\nprint(area(2))\n"
CPP = b"int sum(const int* a, int n) {\n    int s = 0;\n    for (int i = 0; i < n; ++i) {\n        s += a[i];\n    }\n    return s;\n}\n"
PROSE = "Naïve travellers often underestimate how cold the mountain pass becomes after sunset.".encode("utf-8")


@pytest.fixture(scope="module")
def patcher(repo_root):
    if verify(repo_root):
        pytest.skip("patcher checkpoint not fetched/verified")
    from stream_a.patcher_model import load_patcher

    torch.backends.cuda.matmul.allow_tf32 = False
    model, config = load_patcher(checkpoint_dir(repo_root), device="cpu", dtype="float32")
    return model, config


def hf_forward(model, content, **kwargs):
    from stream_a.patching import bytes_to_token_ids

    ids = torch.as_tensor(bytes_to_token_ids(content))[None]
    with torch.inference_mode():
        ent, pl, logits = model(input_ids=ids, **kwargs)
    return ent[0].numpy(), pl[0].to(torch.int64).numpy(), logits[0].numpy()


def test_weights_match_provenance(patcher, repo_root):
    model, _ = patcher
    prov = json.loads((checkpoint_dir(repo_root) / "provenance.json").read_text())
    assert sum(p.numel() for p in model.parameters()) == prov["n_params"]


@pytest.mark.parametrize("content", [PY, CPP, PROSE])
def test_entropy_is_nats_and_matches_float64(patcher, content):
    model, _ = patcher
    ent, _, logits = hf_forward(model, content)
    lg = logits.astype(np.float64)
    logp = lg - lg.max(-1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(-1, keepdims=True))
    h_ln = -(np.exp(logp) * logp).sum(-1)
    assert np.max(np.abs(ent - h_ln)) <= 1e-4
    assert np.max(np.abs(ent - h_ln / math.log(2))) > 1e-2
    assert ent.max() <= math.log(260) + 1e-4


@pytest.mark.parametrize("content", [PY, CPP, PROSE])
@pytest.mark.parametrize("cap", [None, 5])
def test_boundaries_equal_hf_patch_lengths(patcher, content, cap):
    from stream_a.patching import byte_entropies_from_token_entropies, byte_starts_from_token_patch_lengths, compute_patches

    model, config = patcher
    tau = float(config.patching_threshold)
    ent, pl, _ = hf_forward(model, content, patch_size=4, threshold=tau, max_patch_length=cap)
    hf_starts = byte_starts_from_token_patch_lengths(pl, len(content))
    ours = compute_patches(byte_entropies_from_token_entropies(ent, len(content)), tau, cap)
    assert hf_starts.tolist() == ours.starts.tolist()


def test_swa512_mask_equals_causal_for_short_input_and_reference_matches(patcher, repo_root):
    from safetensors.torch import load_file

    from stream_a.patcher_model import TENSOR_PREFIX, attention_mask_4d
    from stream_a.patching import bytes_to_token_ids
    from stream_a.reference_lm import ReferenceEntropyModel, to_reference_names

    model, _ = patcher
    ids = torch.as_tensor(bytes_to_token_ids(PY + CPP))[None]
    L = ids.shape[1]
    assert L <= 512
    with torch.inference_mode():
        _, _, native = model(input_ids=ids)
        _, _, swa = model(input_ids=ids, attention_mask=attention_mask_4d(L, "swa512", "cpu"))
    assert torch.max(torch.abs(native - swa)).item() <= 1e-4

    raw = load_file(str(checkpoint_dir(repo_root) / "patcher.safetensors"))
    ref = ReferenceEntropyModel(to_reference_names({k[len(TENSOR_PREFIX):]: v for k, v in raw.items()}))
    assert torch.max(torch.abs(ref.logits(ids) - native)).item() <= 1e-3
