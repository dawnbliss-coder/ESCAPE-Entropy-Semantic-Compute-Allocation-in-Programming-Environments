"""Stream-A-format extraction for the distilled student (and the teacher-512 control).

Writes the same artifact layout as `stream_a/extract.py` under a new, honest run_id:
  entropies/{run_id}/{file_id}.npy        float32[n_bytes], global next-byte entropy (nats)
  boundaries/{run_id}/{file_id}.parquet   file_id, byte_offset, trigger, entropy,
                                          patch_index, patch_length (one row per patch;
                                          byte_offset = first byte of the patch)

Context policies (pre-registered in docs/ADA_DISTILL_RUNBOOK.md; validated on the golden
fixtures in results/student_cv1_golden_v1/ and results/tau_sweep_golden_v1/):

  rolling    (default, primary)  stride-1 fresh windows: byte i is predicted from
             [BOS] + (last min(i,512) bytes) with absolute RoPE positions 0..min(i,512).
             Exactly the regime the student was trained on; no artificial chunk resets.
  continuous (sensitivity only)  KV-cache incremental decoding over the whole file with
             absolute positions 0..i and a rolling 513-token cache. For files longer than
             512 bytes the model uses RoPE positions it never saw in training
             (extrapolation); this mode is out-of-distribution and must be labelled as a
             sensitivity variant, never the primary number.

--model student : distilled StudentBLT (results/distill/full/student/student.pt)
--model teacher : frozen BLT patcher under the SAME 512-byte fresh-window budget
                  (the teacher-512 control; use --file-ids for the pre-registered subset)

Resumable: files whose artifacts already exist and match n_bytes are skipped
(--skip-existing, on by default). Shard with --shard i --n-shards k (deterministic:
sorted file ids, round-robin by index).
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from escape_common.io import atomic_save_npy, atomic_to_parquet, update_json  # noqa: E402
from escape_common.schema import (RUNS_JSON_PATH, boundary_path,  # noqa: E402
                                  corpus_bin_path, entropy_path, golden_bin_path)  # noqa: E402
from escape_distill.student_blt import (BOS_ID, BYTE_TOKEN_OFFSET, PAD_ID,  # noqa: E402
                                        StudentBLT, StudentConfig, rotate_half_interleaved)  # noqa: E402
from stream_a.patching import compute_patches  # noqa: E402

TAU = 1.335442066192627
MAX_CTX = 512
BATCH_WINDOWS = 128
BOUNDARY_COLUMNS = ["file_id", "byte_offset", "trigger", "entropy", "patch_index", "patch_length"]

ENTROPY_DEFINITION = (
    "entropies[i] = -sum_v p(v) ln p(v) over all 260 patcher logits of the prediction made at token "
    "position i of [BOS, b_0+4, ..., b_{n-1}+4], i.e. the distribution of byte i given BOS and bytes < i (nats)"
)
BOUNDARY_CONVENTION = (
    "one row per patch; byte_offset = first byte of the patch (start-of-patch convention); "
    "trigger init = byte 0 (architecturally forced), entropy = entropies[byte_offset] > tau (strict, float64 "
    "comparison of the stored float32), length_cap = cut of a patch longer than max_patch_length"
)


def load_student(pt: Path, device: str):
    ckpt = torch.load(pt, map_location=device, weights_only=False)
    model = StudentBLT(StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()  # load_state_dict copies into the CPU-built params; move them
    return model, ckpt


def load_teacher(ckpt_dir: Path, device: str):
    from safetensors.torch import load_file
    from stream_a.reference_lm import ReferenceEntropyModel, to_reference_names
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    pc = cfg.get("patcher_config", cfg)
    raw = load_file(str(ckpt_dir / "patcher.safetensors"), device=device)
    state = {k[len("model.patcher."):]: v.to(torch.float32) for k, v in raw.items()}
    model = ReferenceEntropyModel(
        to_reference_names(state), n_layers=pc["num_hidden_layers"], dim=pc["hidden_size"],
        n_heads=pc["num_attention_heads"], norm_eps=pc.get("norm_eps", 1e-5),
        rope_theta=pc.get("rope_theta", 10000.0),
        max_seqlen=pc.get("max_position_embeddings", 8192), device=device, dtype=torch.float32)
    return model, pc


@torch.inference_mode()
def rolling_entropies(logits_fn, content: bytes, device, batch_windows: int = 128) -> np.ndarray:
    """Stride-1 fresh windows; returns float32 entropy[n_bytes]."""
    n = len(content)
    ent = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch_windows):
        chunk = list(range(s, min(s + batch_windows, n)))
        L = max(min(i, MAX_CTX) for i in chunk) + 1
        batch = torch.full((len(chunk), L), PAD_ID, dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            ctx = content[max(0, i - MAX_CTX): i]
            batch[r, 0] = BOS_ID
            batch[r, 1: len(ctx) + 1] = torch.from_numpy(
                np.frombuffer(ctx, dtype=np.uint8).astype(np.int64) + BYTE_TOKEN_OFFSET).to(device)
        logits = logits_fn(batch).float()
        for r, i in enumerate(chunk):
            pos = min(i, MAX_CTX)
            logp = torch.log_softmax(logits[r, pos], -1)
            ent[i] = -float((logp.exp() * logp).sum())
    return ent


@torch.inference_mode()
def continuous_entropies(student, content: bytes, device) -> np.ndarray:
    """KV-cache incremental decoding (absolute positions, rolling 513-token cache).
    OOD beyond position 512 for files longer than 512 bytes; sensitivity variant only."""
    n = len(content)
    ids = np.concatenate(([BOS_ID], np.frombuffer(content, dtype=np.uint8).astype(np.int64) + BYTE_TOKEN_OFFSET))
    ent = np.zeros(n, dtype=np.float32)
    head_dim = student.cfg.dim // student.cfg.n_heads
    cos, sin = student._rope_cache(len(ids), device, torch.float32)
    cache_k = [None] * student.cfg.n_layers
    cache_v = [None] * student.cfg.n_layers
    for t in range(len(ids)):
        tok = torch.tensor([int(ids[t])], dtype=torch.long, device=device)
        x = student.tok_emb(tok)
        c, s = cos[t: t + 1], sin[t: t + 1]
        for li, blk in enumerate(student.blocks):
            xn = blk.attn_norm(x)
            q = blk.attn.wq(xn).view(1, 1, student.cfg.n_heads, head_dim).transpose(1, 2)
            k = blk.attn.wk(xn).view(1, 1, student.cfg.n_heads, head_dim).transpose(1, 2)
            v = blk.attn.wv(xn).view(1, 1, student.cfg.n_heads, head_dim).transpose(1, 2)
            q = q * c[None, None] + rotate_half_interleaved(q) * s[None, None]
            k = k * c[None, None] + rotate_half_interleaved(k) * s[None, None]
            if cache_k[li] is None:
                cache_k[li], cache_v[li] = k, v
            else:
                cache_k[li] = torch.cat([cache_k[li], k], dim=2)
                cache_v[li] = torch.cat([cache_v[li], v], dim=2)
                if cache_k[li].shape[2] > MAX_CTX + 1:  # BOS + 512 bytes
                    cache_k[li] = cache_k[li][:, :, 1:]
                    cache_v[li] = cache_v[li][:, :, 1:]
            scores = (q @ cache_k[li].transpose(-1, -2)) * (head_dim ** -0.5)
            att = torch.softmax(scores, dim=-1)
            out = att @ cache_v[li]
            x = x + blk.attn.wo(out.transpose(1, 2).reshape(1, 1, student.cfg.dim))
            x = x + blk.ffn(blk.ffn_norm(x))
        logits = student.lm_head(student.final_norm(x))[0, 0]
        if t < n:
            logp = torch.log_softmax(logits, -1)
            ent[t] = -float((logp.exp() * logp).sum())
    return ent


def boundaries_frame(file_id: str, entropies: np.ndarray) -> pd.DataFrame:
    patches = compute_patches(entropies, TAU)
    return pd.DataFrame({
        "file_id": pd.Series([file_id] * patches.n_patches, dtype="string"),
        "byte_offset": patches.starts.astype(np.int64),
        "trigger": pd.Series(patches.triggers.tolist(), dtype="string"),
        "entropy": entropies[patches.starts].astype(np.float32),
        "patch_index": np.arange(patches.n_patches, dtype=np.int32),
        "patch_length": patches.lengths.astype(np.int32),
    }, columns=BOUNDARY_COLUMNS)


def read_verified_bytes(path: Path, n_bytes: int, sha256: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"missing input file {path}")
    content = path.read_bytes()
    if len(content) != n_bytes:
        raise ValueError(f"{path}: {len(content)} bytes on disk, manifest says {n_bytes}")
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"{path}: sha256 differs from manifest")
    return content


def register_run(root: Path, run_id: str, entry: dict) -> dict:
    def update(runs):
        if run_id in runs:
            old = runs[run_id]
            mismatched = {k: (old.get(k), entry.get(k)) for k in
                          ("backend", "tau", "max_patch_length", "checkpoint", "checkpoint_revision",
                           "entropy_definition", "boundary_convention", "context_mode", "dtype",
                           "student_pt_sha256", "patcher_weights_sha256")
                          if old.get(k) != entry.get(k)}
            if mismatched:
                raise SystemExit(f"run_id {run_id} already exists with different settings: {mismatched}")
            return runs
        entry["invocations"] = [{k: entry[k] for k in ("date", "code_sha256") if k in entry}]
        runs[run_id] = entry
        return runs
    return update_json(root / RUNS_JSON_PATH, update)[run_id]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--input", choices=("golden", "corpus"), required=True)
    ap.add_argument("--domains", nargs="+", default=["py", "cpp", "prose"])
    ap.add_argument("--file-ids", nargs="*", help="explicit file list (use scripts/ada_final/shard_ids.py for shards)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", choices=("student", "teacher"), default="student")
    ap.add_argument("--student-pt", default=str(REPO_ROOT / "results/distill/full/student/student.pt"))
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"))
    ap.add_argument("--mode", choices=("rolling", "continuous"), default="rolling")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-windows", type=int, default=BATCH_WINDOWS)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--shard", type=int, default=None, help="0-based shard index")
    ap.add_argument("--n-shards", type=int, default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_parquet(root / "corpus/manifest.parquet")
    if args.input == "golden":
        from escape_common.schema import golden_file_ids
        gids = sorted(fid for d in args.domains for fid in golden_file_ids(root, d))
        sel = manifest[manifest["file_id"].isin(gids)]
    else:
        sel = manifest[manifest["domain"].isin(args.domains)]
    if args.file_ids:
        sel = sel[sel["file_id"].isin(set(args.file_ids))]
    sel = sel.sort_values(["domain", "n_bytes", "file_id"]).reset_index(drop=True)
    if args.shard is not None and args.n_shards:
        sel = sel.iloc[args.shard:: args.n_shards]
    if args.limit:
        sel = sel.head(args.limit)
    print(f"[extract] {len(sel)} files selected (shard {args.shard}/{args.n_shards})", flush=True)

    if args.model == "student":
        model, ckpt = load_student(Path(args.student_pt), device)
        params = model.num_parameters()
        c = ckpt["config"]
        model_identity = {"checkpoint": f"escape_distill/student-bl-t-{c['dim']}-{c['n_layers']}-{c['n_heads']}",
                          "checkpoint_revision": "c4384eb81f7d801aab2746dc453b6b1679136f9a",
                          "student_pt_sha256": hashlib.sha256(Path(args.student_pt).read_bytes()).hexdigest(),
                          "n_params": params}
        logits_fn = lambda b: model(b)  # noqa: E731
    else:
        model, pc = load_teacher(Path(args.ckpt_dir), device)
        model_identity = {"checkpoint": "itazap/blt-1b-hf",
                          "checkpoint_revision": "91aa6b8e168046ad91e517d2191e1e974f50cc01",
                          "patcher_weights_sha256": hashlib.sha256(
                              (Path(args.ckpt_dir) / "patcher.safetensors").read_bytes()).hexdigest()}
        logits_fn = lambda b: model.logits(b, mask=None).float()  # noqa: E731
    print(f"[extract] model={args.model} mode={args.mode} device={device} "
          f"{model_identity.get('n_params', '')}", flush=True)

    entry = {
        "backend": "patcher",
        "created_by": "escape_distill/extract.py",
        "entropy_definition": ENTROPY_DEFINITION,
        "boundary_convention": BOUNDARY_CONVENTION,
        "tau": TAU, "max_patch_length": None,
        "context_mode": {"rolling": "fresh-512-stride1", "continuous": "kv-cache-absolute-positions-OOD-beyond-512"}[args.mode],
        "dtype": "float32", "sliding_window": MAX_CTX if args.model == "teacher" else None,
        "chunk_len": None,
        "add_bos": True, "add_eos": False, "attn_implementation": "sdpa",
        "seed": None, "seed_note": "deterministic inference (eval mode, float32)",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "code_sha256": "escape_distill/extract.py",
        **model_identity,
    }
    register_run(root, args.run_id, entry)

    done = skipped = failed = 0
    for _, row in sel.iterrows():
        fid, domain = row["file_id"], row["domain"]
        n_bytes, sha = int(row["n_bytes"]), row["sha256"]
        ep, bp = entropy_path(root, args.run_id, fid), boundary_path(root, args.run_id, fid)
        if not args.no_skip_existing and ep.exists() and bp.exists():
            try:
                if np.load(ep, mmap_mode="r").shape == (n_bytes,) and \
                        int(pd.read_parquet(bp, columns=["patch_length"])["patch_length"].sum()) == n_bytes:
                    skipped += 1
                    continue
            except Exception:
                pass
        try:
            if args.input == "golden":
                path = golden_bin_path(root, domain, fid)
            else:
                path = corpus_bin_path(root, domain, fid)
            content = read_verified_bytes(path, n_bytes, sha)
            t0 = time.time()
            if args.mode == "rolling":
                ent = rolling_entropies(logits_fn, content, device, args.batch_windows)
            else:
                if args.model != "student":
                    raise SystemExit("--mode continuous supports --model student only")
                ent = continuous_entropies(model, content, device)
            if not np.isfinite(ent).all():
                raise FloatingPointError(f"non-finite entropy in {fid}")
            atomic_save_npy(ep, ent.astype(np.float32))
            atomic_to_parquet(boundaries_frame(fid, ent), bp)
            done += 1
            if done % args.progress_every == 0:
                print(f"[extract] {done} done / {skipped} skipped / {failed} failed; "
                      f"last {fid} {n_bytes}B in {time.time()-t0:.1f}s", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[extract] FAIL {fid}: {exc}", flush=True)
    print(f"[extract] DONE run_id={args.run_id} shard={args.shard}: "
          f"{done} done, {skipped} skipped, {failed} failed", flush=True)
    if failed:
        raise SystemExit(f"{failed} files failed")


if __name__ == "__main__":
    main()
