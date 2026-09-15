"""Resumable, failure-aware Stream A extraction (Phases 2-5, 7 and 10).

Pipeline: bytes -> BLT patcher -> per-byte entropy (nats) -> patch boundaries
(init / entropy / length_cap).

Outputs per file:
  entropies/{run_id}/{file_id}.npy        float32[n_bytes]; entropies[i] belongs to byte i
  boundaries/{run_id}/{file_id}.parquet   file_id, byte_offset, trigger, entropy, patch_index, patch_length
Outputs per run:
  runs.json entry                         every knob that affects outputs
  runs/{run_id}/status.jsonl              append-only event log (running / success / failed)
  runs/{run_id}/status.parquet            latest status per file (rebuilt after each invocation)
  runs/{run_id}/failures/{file_id}.json   traceback and attempts for failed files
  runs/{run_id}/logs/                     console logs (gitignored)

Inputs:
  - Only raw bytes are read: corpus/{domain}/{file_id}.bin or golden_fixtures/{domain}/{file_id}.bin.
  - Every file's n_bytes and sha256 are checked against corpus/manifest.parquet before inference.
  - No parse information (structure/, whitespace/, taxonomy) is read at any point.

Resumability:
  - A file is done when both artifacts exist and are consistent with its n_bytes.
  - Artifacts are written atomically (temp + rename), entropies first and boundaries last, so an
    interrupted run leaves complete outputs or none.
  - Re-running the same command resumes. --only-failed re-attempts only files whose latest status
    is 'failed'.

Examples (repo root, .venv_a):
  python -m stream_a.extract --input golden
  python -m stream_a.extract --input corpus --domains py cpp --split calib
  python -m stream_a.extract --input corpus --domains prose --split main
  python -m stream_a.extract --input corpus --split main --only-failed
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_common.io import atomic_save_npy, atomic_to_parquet, atomic_write_json, sha256_file, update_json  # noqa: E402
from escape_common.schema import (  # noqa: E402
    BOUNDARY_COLUMNS,
    MANIFEST_PATH,
    RUNS_JSON_PATH,
    boundary_path,
    corpus_bin_path,
    entropy_path,
    golden_bin_path,
    golden_file_ids,
)
from stream_a.patching import byte_entropies_from_token_entropies, compute_patches  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "itazap/blt-1b-hf"

# Fields that change outputs: a run_id may only ever be reused with identical values.
OUTPUT_AFFECTING_KEYS = (
    "backend", "tau", "sliding_window", "chunk_len", "max_patch_length", "context_mode", "dtype",
    "checkpoint", "checkpoint_revision", "patcher_weights_sha256", "add_bos", "add_eos",
    "attn_implementation", "entropy_definition", "boundary_convention",
)
ENTROPY_DEFINITION = (
    "entropies[i] = -sum_v p(v) ln p(v) over all 260 patcher logits of the prediction made at token "
    "position i of [BOS, b_0+4, ..., b_{n-1}+4], i.e. the distribution of byte i given BOS and bytes < i (nats)"
)
BOUNDARY_CONVENTION = (
    "one row per patch; byte_offset = first byte of the patch (start-of-patch convention); "
    "trigger init = byte 0 (architecturally forced), entropy = entropies[byte_offset] > tau (strict, float64 "
    "comparison of the stored float32), length_cap = cut of a patch longer than max_patch_length"
)


class InputIntegrityError(RuntimeError):
    """Bytes on disk do not match the manifest (deterministic - never retried)."""


# ---------------------------------------------------------------------------------------
# Backends: the only thing that differs between the real patcher and the GPU-free mock.
# ---------------------------------------------------------------------------------------
class PatcherBackend:
    name = "patcher"

    def __init__(self, root: Path, context_mode: str, dtype: str, device: str, attn_implementation: str = "sdpa"):
        import torch

        from stream_a.fetch_patcher import REVISION, checkpoint_dir, verify
        from stream_a.patcher_model import CHUNK_LEN, SLIDING_WINDOW, load_patcher

        problems = verify(root)
        if problems:
            raise SystemExit(f"patcher checkpoint missing or unverified ({problems}); run python -m stream_a.fetch_patcher")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.device = torch.device(device)
        self.context_mode = context_mode
        self.dtype = dtype
        ckpt = checkpoint_dir(root)
        self.model, blt_config = load_patcher(ckpt, device=self.device, dtype=dtype, attn_implementation=attn_implementation)
        self.tau = float(blt_config.patching_threshold)
        self.config_max_patch_length = blt_config.max_patch_length
        self.config = {
            "backend": self.name,
            "context_mode": context_mode,
            "sliding_window": SLIDING_WINDOW if context_mode == "swa512" else None,
            "chunk_len": CHUNK_LEN if context_mode == "swa512" else None,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "checkpoint": CHECKPOINT,
            "checkpoint_revision": REVISION,
            "patcher_weights_sha256": sha256_file(ckpt / "patcher.safetensors"),
            "add_bos": True,
            "add_eos": False,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": __import__("transformers").__version__,
            "device": str(self.device),
            "hardware": _hardware(torch, self.device),
            "tf32_matmul": False,
        }

    def token_entropies(self, content: bytes) -> np.ndarray:
        from stream_a.patcher_model import token_entropies

        try:
            ent, _ = token_entropies(self.model, content, self.context_mode, self.device)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            raise
        return ent


class MockBackend:
    """Deterministic, byte-only stand-in for tests and GPU-free end-to-end runs. Its outputs
    are NOT model measurements. Runs created with it are marked backend="mock" / checkpoint="MOCK"."""

    name = "mock"
    SEPARATORS = np.frombuffer(b" \n\t()[]{};,.:=", dtype=np.uint8)

    def __init__(self, tau: float = 1.335442066192627):
        self.tau = tau
        self.config_max_patch_length = None
        self.config = {"backend": self.name, "context_mode": "mock", "sliding_window": None, "chunk_len": None,
                       "dtype": "float32", "attn_implementation": None, "checkpoint": "MOCK",
                       "checkpoint_revision": None, "patcher_weights_sha256": None, "add_bos": True,
                       "add_eos": False, "device": "cpu", "hardware": platform.processor() or platform.machine()}

    def token_entropies(self, content: bytes) -> np.ndarray:
        b = np.frombuffer(content, dtype=np.uint8)
        prev = np.concatenate(([ord("\n")], b))  # token position j "sees" the byte before byte j
        jitter = (np.arange(len(content) + 1) * 2654435761 % 97) / 1000.0
        high = np.isin(prev, self.SEPARATORS)
        return np.where(high, 2.0, 0.4).astype(np.float32) + jitter.astype(np.float32)


def _hardware(torch, device) -> str:
    cpu = platform.processor() or platform.machine()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} ({props.total_memory / 2**30:.1f} GiB) / {cpu}"
    return f"CPU {cpu}"


# ---------------------------------------------------------------------------------------
# Run registry and provenance
# ---------------------------------------------------------------------------------------
def git_state(root: Path) -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD") or None,
            "commit_dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def code_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(list((root / "stream_a").glob("*.py")) + list((root / "escape_common").glob("*.py"))):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def make_run_id(config: dict, tau: float, max_patch_length) -> str:
    mpl = "none" if max_patch_length is None else str(max_patch_length)
    if config["backend"] == "mock":
        return f"mock-t{tau:.4f}-mpl{mpl}"
    return f"{config['context_mode']}-{config['dtype']}-t{tau:.4f}-mpl{mpl}-{config['checkpoint_revision'][:8]}"


def register_run(root: Path, run_id: str, entry: dict) -> dict:
    def update(runs):
        if run_id in runs:
            old = runs[run_id]
            mismatched = {k: (old.get(k), entry.get(k)) for k in OUTPUT_AFFECTING_KEYS if old.get(k) != entry.get(k)}
            if mismatched:
                raise SystemExit(f"run_id {run_id} already exists with different output-affecting settings: {mismatched}")
            history = old.setdefault("invocations", [])
            history.append({k: entry[k] for k in ("date", "commit", "commit_dirty", "code_sha256", "torch_version",
                                                   "transformers_version", "device", "hardware") if k in entry})
            for k in ("torch_version", "transformers_version", "cuda_version", "device", "hardware"):
                if entry.get(k) != old.get(k):
                    print(f"WARNING: {k} changed for run {run_id}: {old.get(k)!r} -> {entry.get(k)!r}")
            return runs
        entry["invocations"] = [{k: entry[k] for k in ("date", "commit", "commit_dirty", "code_sha256") if k in entry}]
        runs[run_id] = entry
        return runs

    return update_json(root / RUNS_JSON_PATH, update)[run_id]


# ---------------------------------------------------------------------------------------
# Per-file work
# ---------------------------------------------------------------------------------------
def read_verified_bytes(path: Path, n_bytes: int, sha256: str) -> bytes:
    if not path.exists():
        raise InputIntegrityError(f"missing input file {path}")
    content = path.read_bytes()
    if len(content) != n_bytes:
        raise InputIntegrityError(f"{path}: {len(content)} bytes on disk, manifest says {n_bytes}")
    if hashlib.sha256(content).hexdigest() != sha256:
        raise InputIntegrityError(f"{path}: sha256 differs from manifest")
    return content


def boundaries_frame(file_id: str, entropies: np.ndarray, patches) -> pd.DataFrame:
    return pd.DataFrame({
        "file_id": pd.Series([file_id] * patches.n_patches, dtype="string"),
        "byte_offset": patches.starts.astype(np.int64),
        "trigger": pd.Series(patches.triggers.tolist(), dtype="string"),
        "entropy": entropies[patches.starts].astype(np.float32),
        "patch_index": np.arange(patches.n_patches, dtype=np.int32),
        "patch_length": patches.lengths.astype(np.int32),
    }, columns=BOUNDARY_COLUMNS)


def artifacts_done(root: Path, run_id: str, file_id: str, n_bytes: int) -> bool:
    ep, bp = entropy_path(root, run_id, file_id), boundary_path(root, run_id, file_id)
    if not (ep.exists() and bp.exists()):
        return False
    try:
        ent = np.load(ep, mmap_mode="r")
        if ent.dtype != np.float32 or ent.shape != (n_bytes,):
            return False
        lengths = pd.read_parquet(bp, columns=["patch_length"])["patch_length"]
        return int(lengths.sum()) == n_bytes
    except Exception:
        return False


def process_file(root: Path, run_id: str, backend, file_id: str, content: bytes, tau: float, max_patch_length) -> dict:
    n = len(content)
    t0 = time.time()
    token_ent = backend.token_entropies(content)
    ent = byte_entropies_from_token_entropies(token_ent, n)
    if not np.all(np.isfinite(ent)):
        raise FloatingPointError(f"non-finite entropy in {file_id}")
    patches = compute_patches(ent, tau, max_patch_length)
    atomic_save_npy(entropy_path(root, run_id, file_id), ent)
    atomic_to_parquet(boundaries_frame(file_id, ent, patches), boundary_path(root, run_id, file_id))
    return {
        "n_bytes": n,
        "n_patches": patches.n_patches,
        "n_entropy": int((patches.triggers == "entropy").sum()),
        "n_length_cap": int((patches.triggers == "length_cap").sum()),
        "duration_s": round(time.time() - t0, 4),
    }


# ---------------------------------------------------------------------------------------
# Status log
# ---------------------------------------------------------------------------------------
class StatusLog:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "status.jsonl"
        self.f = open(self.path, "a", encoding="utf-8")
        self.n_since_sync = 0

    def event(self, **fields):
        fields["t"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.f.write(json.dumps(fields, sort_keys=True) + "\n")
        self.f.flush()
        self.n_since_sync += 1
        if self.n_since_sync >= 50:
            os.fsync(self.f.fileno())
            self.n_since_sync = 0

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


def latest_status(run_dir: Path) -> dict:
    path = run_dir / "status.jsonl"
    latest = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line from a crash
                latest[ev["file_id"]] = ev
    return latest


def write_status_snapshot(root: Path, run_id: str, selected: pd.DataFrame) -> pd.DataFrame:
    run_dir = root / "runs" / run_id
    latest = latest_status(run_dir)
    rows = []
    for fid, ev in latest.items():
        status = ev["status"]
        if status == "running":
            status = "interrupted"  # started, never finished: treated as pending on resume
        rows.append({"file_id": fid, "status": status, "attempt": ev.get("attempt"), "t": ev.get("t"),
                     "n_bytes": ev.get("n_bytes"), "n_patches": ev.get("n_patches"), "error": ev.get("error")})
    for fid in set(selected["file_id"]) - set(latest):
        rows.append({"file_id": fid, "status": "pending", "attempt": None, "t": None, "n_bytes": None,
                     "n_patches": None, "error": None})
    snap = pd.DataFrame(rows, columns=["file_id", "status", "attempt", "t", "n_bytes", "n_patches", "error"])
    snap = snap.sort_values("file_id").reset_index(drop=True)
    atomic_to_parquet(snap, run_dir / "status.parquet")
    return snap


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------
def select_files(root: Path, input_kind: str, domains, split: str, file_ids, limit) -> pd.DataFrame:
    manifest = pd.read_parquet(root / MANIFEST_PATH)
    sel = manifest[manifest["domain"].isin(domains)]
    if split != "all":
        sel = sel[sel["split"] == split]
    if input_kind == "golden":
        golden = {fid for d in domains for fid in golden_file_ids(root, d)}
        sel = sel[sel["file_id"].isin(golden)]
        sel = sel.assign(path=[golden_bin_path(root, d, f) for d, f in zip(sel["domain"], sel["file_id"])])
    else:
        sel = sel.assign(path=[corpus_bin_path(root, d, f) for d, f in zip(sel["domain"], sel["file_id"])])
    if file_ids:
        sel = sel[sel["file_id"].isin(set(file_ids))]
    sel = sel.sort_values(["domain", "n_bytes", "file_id"]).reset_index(drop=True)
    if limit:
        sel = sel.head(limit)
    return sel


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream A extraction: bytes -> entropy -> boundaries")
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--input", choices=("golden", "corpus"), required=True)
    ap.add_argument("--domains", nargs="+", default=["py", "cpp", "prose"])
    ap.add_argument("--split", choices=("calib", "main", "all"), default="all")
    ap.add_argument("--file-ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", choices=("patcher", "mock"), default="patcher")
    ap.add_argument("--context-mode", choices=("swa512", "hf_full"), default="swa512")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-patch-length", default="config",
                    help="'config' (checkpoint value, null for itazap/blt-1b-hf), 'none', or an integer")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--only-failed", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.backend == "patcher":
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("WARNING: CUDA not available - falling back to CPU (slow; long files may not fit in RAM)")
        backend = PatcherBackend(root, args.context_mode, args.dtype, device)
    else:
        backend = MockBackend()

    if args.max_patch_length == "config":
        max_patch_length = backend.config_max_patch_length
    elif args.max_patch_length == "none":
        max_patch_length = None
    else:
        max_patch_length = int(args.max_patch_length)
    tau = backend.tau

    run_id = make_run_id(backend.config, tau, max_patch_length)
    entry = {**backend.config, **git_state(root), "tau": tau, "max_patch_length": max_patch_length,
             "code_sha256": code_digest(root), "seed": None,
             "seed_note": "no randomness in Stream A: inference and patching are deterministic",
             "python_version": platform.python_version(),
             "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "entropy_definition": ENTROPY_DEFINITION, "boundary_convention": BOUNDARY_CONVENTION,
             "created_by": "stream_a/extract.py"}
    register_run(root, run_id, entry)

    selected = select_files(root, args.input, args.domains, args.split, args.file_ids, args.limit)
    run_dir = root / "runs" / run_id
    latest = latest_status(run_dir)
    if args.only_failed:
        selected = selected[[latest.get(f, {}).get("status") == "failed" for f in selected["file_id"]]]

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"extract_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    log_f = open(log_path, "a", encoding="utf-8")

    def say(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    say(f"run_id={run_id} input={args.input} domains={args.domains} split={args.split} selected={len(selected)} "
        f"device={backend.config.get('device')} dtype={backend.config.get('dtype')}")
    status = StatusLog(run_dir)
    n_done = n_skipped = n_failed = 0
    bytes_done = 0
    t_start = time.time()
    try:
        for i, row in enumerate(selected.itertuples(index=False), start=1):
            fid, n_bytes = row.file_id, int(row.n_bytes)
            if artifacts_done(root, run_id, fid, n_bytes):
                n_skipped += 1
                continue
            attempt = 0
            while True:
                attempt += 1
                status.event(file_id=fid, status="running", attempt=attempt)
                try:
                    content = read_verified_bytes(Path(row.path), n_bytes, row.sha256)
                    info = process_file(root, run_id, backend, fid, content, tau, max_patch_length)
                    status.event(file_id=fid, status="success", attempt=attempt, **info)
                    n_done += 1
                    bytes_done += n_bytes
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    retryable = not isinstance(e, InputIntegrityError) and attempt <= args.retries
                    status.event(file_id=fid, status="failed" if not retryable else "retrying", attempt=attempt,
                                 error=f"{type(e).__name__}: {e}"[:500])
                    if not retryable:
                        atomic_write_json(run_dir / "failures" / f"{fid}.json", {
                            "file_id": fid, "attempts": attempt, "error_type": type(e).__name__, "error": str(e),
                            "traceback": traceback.format_exc(),
                            "t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
                        n_failed += 1
                        say(f"FAILED {fid} after {attempt} attempt(s): {type(e).__name__}: {e}")
                        break
                    time.sleep(min(2 ** (attempt - 1), 30))
            if i % args.progress_every == 0 or i == len(selected):
                elapsed = time.time() - t_start
                rate = bytes_done / max(elapsed, 1e-9)
                say(f"[{i}/{len(selected)}] done={n_done} skipped={n_skipped} failed={n_failed} "
                    f"{rate / 1e3:.1f} kB/s elapsed={elapsed:.0f}s")
    finally:
        status.close()
        snap = write_status_snapshot(root, run_id, select_files(root, args.input, args.domains, args.split,
                                                                 args.file_ids, args.limit))
        counts = snap["status"].value_counts().to_dict()
        say(f"finished: processed={n_done} skipped_already_done={n_skipped} failed={n_failed} "
            f"bytes={bytes_done} elapsed={time.time() - t_start:.0f}s status_snapshot={counts}")
        log_f.close()
    return run_id


if __name__ == "__main__":
    main()
