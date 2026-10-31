"""Fetch ONLY the patcher (entropy model) tensors of the pinned itazap/blt-1b-hf
checkpoint, plus its small config/tokenizer files, into
checkpoints/itazap__blt-1b-hf/{REVISION}/.

model.safetensors is 15.4 GB (the hash n-gram embeddings alone are 3.07B
float32 parameters), while H1 needs nothing but the patcher. The patcher's 129
BF16 tensors (99.5M parameters) occupy one contiguous 199 MB byte range of that
file, so they are fetched with HTTP Range requests and re-packed VERBATIM (no
dtype conversion, no re-serialisation of values) into a standalone safetensors
file with the original tensor names.

Integrity: the revision is pinned by commit sha; every range response must be
HTTP 206 with exactly the Content-Range requested; contiguity and per-tensor byte
lengths (dtype x shape) are checked against the source header, whose sha256 is
recorded; per-tensor sha256 and the output file's sha256 go to provenance.json.
The whole-file LFS sha256 cannot be checked without the full 15.4 GB download; it
is recorded in provenance.json so a later full download can confirm it.

Usage (from the repo root):
    python -m stream_a.fetch_patcher           # idempotent: verifies, then skips
    python -m stream_a.fetch_patcher --force   # re-fetch
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escape_common.io import atomic_open, atomic_write_bytes, atomic_write_json, sha256_file  # noqa: E402

REPO_ID = "itazap/blt-1b-hf"
REVISION = "91aa6b8e168046ad91e517d2191e1e974f50cc01"
WEIGHTS_FILE = "model.safetensors"
SMALL_FILES = (
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "README.md",
)
TENSOR_PREFIX = "model.patcher."
PATCHER_FILE = "patcher.safetensors"
PROVENANCE_FILE = "provenance.json"
CHUNK_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
USER_AGENT = "escape-stream-a-fetch-patcher/1.0"


def checkpoint_dir(root=".") -> Path:
    return Path(root) / "checkpoints" / REPO_ID.replace("/", "__") / REVISION


def _resolve_url(filename: str) -> str:
    return f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{filename}"


def _http_get(url: str, byte_range=None, retries: int = 6, timeout: int = 120):
    """GET with retry + exponential backoff. byte_range is inclusive (lo, hi)."""
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if byte_range is not None:
                    lo, hi = byte_range
                    content_range = resp.headers.get("Content-Range", "")
                    if resp.status != 206:
                        raise OSError(f"expected HTTP 206 for a range request, got {resp.status}")
                    if not content_range.startswith(f"bytes {lo}-{hi}/"):
                        raise OSError(f"unexpected Content-Range {content_range!r} for bytes {lo}-{hi}")
                    if len(body) != hi - lo + 1:
                        raise OSError(f"short range read: {len(body)} != {hi - lo + 1}")
                return body
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries - 1} in {wait}s after: {e}", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def verify(root=".") -> list:
    """Re-check a fetched checkpoint dir against its provenance.json.
    Returns a list of problems (empty = verified)."""
    d = checkpoint_dir(root)
    problems = []
    prov_path = d / PROVENANCE_FILE
    if not prov_path.exists():
        return [f"missing {prov_path}"]
    prov = json.loads(prov_path.read_text())
    out = d / PATCHER_FILE
    if not out.exists():
        return [f"missing {out}"]
    if sha256_file(out) != prov["output"]["sha256"]:
        problems.append(f"{PATCHER_FILE} sha256 differs from provenance")
    with open(out, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    if sorted(header) != sorted(prov["tensors"]):
        problems.append("tensor key set differs from provenance")
    for name, meta in prov["small_files"].items():
        p = d / name
        if not p.exists() or sha256_file(p) != meta["sha256"]:
            problems.append(f"{name} missing or sha256 differs")
    return problems


def fetch(root=".", force: bool = False) -> Path:
    d = checkpoint_dir(root)
    if not force and (d / PROVENANCE_FILE).exists():
        problems = verify(root)
        if not problems:
            print(f"already fetched and verified: {d}")
            return d
        print(f"existing checkpoint dir failed verification ({problems}); re-fetching")

    api_url = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"
    api = json.loads(_http_get(api_url))
    if api.get("sha") != REVISION:
        raise SystemExit(f"API returned sha {api.get('sha')!r}, expected pinned {REVISION}")
    siblings = {s["rfilename"]: s for s in api.get("siblings", [])}

    small_files = {}
    for name in SMALL_FILES:
        body = _http_get(_resolve_url(name))
        atomic_write_bytes(d / name, body)
        small_files[name] = {"sha256": hashlib.sha256(body).hexdigest(), "n_bytes": len(body)}
        print(f"fetched {name} ({len(body)} bytes)")

    url = _resolve_url(WEIGHTS_FILE)
    (n_header,) = struct.unpack("<Q", _http_get(url, (0, 7)))
    header_raw = _http_get(url, (8, 8 + n_header - 1))
    header = json.loads(header_raw)
    header.pop("__metadata__", None)
    data_start = 8 + n_header

    names = sorted((k for k in header if k.startswith(TENSOR_PREFIX)), key=lambda k: header[k]["data_offsets"][0])
    if not names:
        raise SystemExit(f"no tensors with prefix {TENSOR_PREFIX!r} in {WEIGHTS_FILE}")
    span_lo = header[names[0]]["data_offsets"][0]
    cursor = span_lo
    for k in names:
        a, b = header[k]["data_offsets"]
        expected = math.prod(header[k]["shape"]) * DTYPE_BYTES[header[k]["dtype"]]
        if a != cursor:
            raise SystemExit(f"patcher tensors are not contiguous at {k} ({a} != {cursor})")
        if b - a != expected:
            raise SystemExit(f"{k}: byte length {b - a} != dtype x shape = {expected}")
        cursor = b
    span_hi = cursor
    for k, v in header.items():
        a, b = v["data_offsets"]
        if not k.startswith(TENSOR_PREFIX) and a < span_hi and b > span_lo:
            raise SystemExit(f"non-patcher tensor {k} overlaps the patcher byte span")

    new_header = {
        k: {
            "dtype": header[k]["dtype"],
            "shape": header[k]["shape"],
            "data_offsets": [header[k]["data_offsets"][0] - span_lo, header[k]["data_offsets"][1] - span_lo],
        }
        for k in names
    }
    new_header["__metadata__"] = {
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_file": WEIGHTS_FILE,
        "source_header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "subset": TENSOR_PREFIX + "*",
        "note": "tensor bytes copied verbatim from the source file (no dtype conversion)",
    }
    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    # Stream chunks straight to disk, hashing each tensor as its bytes pass by.
    tensor_hashers = {k: hashlib.sha256() for k in names}
    total = span_hi - span_lo
    t0 = time.time()
    out_path = d / PATCHER_FILE
    with atomic_open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk_lo in range(span_lo, span_hi, CHUNK_BYTES):
            chunk_hi = min(chunk_lo + CHUNK_BYTES, span_hi)
            body = _http_get(url, (data_start + chunk_lo, data_start + chunk_hi - 1))
            f.write(body)
            for k in names:
                a, b = header[k]["data_offsets"]
                lo, hi = max(a, chunk_lo), min(b, chunk_hi)
                if lo < hi:
                    tensor_hashers[k].update(body[lo - chunk_lo : hi - chunk_lo])
            done = chunk_hi - span_lo
            rate = done / max(time.time() - t0, 1e-9) / 1e6
            print(f"  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({rate:.2f} MB/s)")

    provenance = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "weights_file": WEIGHTS_FILE,
        "weights_file_lfs": siblings.get(WEIGHTS_FILE, {}).get("lfs"),
        "weights_file_size": siblings.get(WEIGHTS_FILE, {}).get("size"),
        "whole_file_sha256_verified_locally": False,
        "source_header_n_bytes": n_header,
        "source_header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "source_data_section_start": data_start,
        "patcher_span_in_data_section": [span_lo, span_hi],
        "n_tensors": len(names),
        "n_params": sum(math.prod(header[k]["shape"]) for k in names),
        "tensors": {
            k: {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "sha256": tensor_hashers[k].hexdigest()}
            for k in names
        },
        "small_files": small_files,
        "output": {"file": PATCHER_FILE, "sha256": sha256_file(out_path), "n_bytes": out_path.stat().st_size},
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "tool": "stream_a/fetch_patcher.py",
    }
    atomic_write_json(d / PROVENANCE_FILE, provenance)
    problems = verify(root)
    if problems:
        raise SystemExit(f"post-fetch verification failed: {problems}")
    print(f"OK: {len(names)} tensors, {provenance['n_params'] / 1e6:.2f}M params -> {out_path}")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if present and verified")
    args = ap.parse_args()
    fetch(args.root, force=args.force)


if __name__ == "__main__":
    main()
