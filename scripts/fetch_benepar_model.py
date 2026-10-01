#!/usr/bin/env python3
"""Fetch benepar's `benepar_en3` parsing model with parallel, resumable HTTP range requests,
verify it against benepar's own package index (size + md5), and install it exactly where
`benepar.download('benepar_en3')` would put it: the NLTK download directory, under
`models/`, with the verified zip kept and its contents extracted beside it.

`benepar.download` (nltk.downloader) fetches the 66.2 MB zip over a single
connection. GitHub release assets were served at ~15-30 KB/s here, and the single stream was
cut at exactly 60 MiB:
  "Integrity check failed for 'benepar_en3': size mismatch (got 62914560, expected 66207553)".
The model itself is unchanged: same URL, same index checksum.

Usage (inside the Stream B venv):
  .venv/bin/python scripts/fetch_benepar_model.py [--workers 8]
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INDEX_URL = "https://kitaev.com/benepar/index.xml"  # benepar/integrations/downloader.py BENEPAR_SERVER_INDEX
PACKAGE = "benepar_en3"
CHUNK_BYTES = 4 * 1024 * 1024
USER_AGENT = "escape-fetch-benepar/1.0"


def http_get(url: str, byte_range=None, timeout: int = 180) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if byte_range is not None:
            if resp.status != 206:
                raise OSError(f"expected HTTP 206, got {resp.status}")
            if len(data) != byte_range[1] - byte_range[0] + 1:
                raise OSError(f"short range read: {len(data)} bytes")
        return data


def package_entry() -> dict:
    xml = http_get(INDEX_URL).decode("utf-8")
    m = re.search(r'<package ([^>]*id="%s"[^>]*)/>' % re.escape(PACKAGE), xml)
    if not m:
        raise SystemExit(f"{PACKAGE} not found in {INDEX_URL}")
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    return {"url": attrs["url"], "size": int(attrs["size"]), "md5": attrs["checksum"],
            "subdir": attrs["subdir"], "unzipped_size": int(attrs.get("unzipped_size", 0))}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_chunk(url: str, idx: int, lo: int, hi: int, part_dir: Path, retries: int = 10):
    part = part_dir / f"{idx:05d}.part"
    want = hi - lo + 1
    if part.exists() and part.stat().st_size == want:
        return idx, "cached"
    last = None
    for attempt in range(retries):
        try:
            data = http_get(url, (lo, hi))
            tmp = part_dir / f".{idx:05d}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, part)
            return idx, f"fetched (attempt {attempt + 1})"
        except Exception as exc:  # noqa: BLE001 - network errors of every kind are retried
            last = exc
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"chunk {idx} [{lo}, {hi}] failed after {retries} attempts: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    import nltk.data
    import nltk.downloader

    download_dir = Path(nltk.downloader.Downloader(server_index_url=INDEX_URL).default_download_dir())
    entry = package_entry()
    models = download_dir / entry["subdir"]
    models.mkdir(parents=True, exist_ok=True)
    final_zip = models / f"{PACKAGE}.zip"
    print(f"index: {INDEX_URL}\npackage: {PACKAGE} size={entry['size']} md5={entry['md5']}\nurl: {entry['url']}\n"
          f"install dir: {models}", flush=True)

    if final_zip.exists() and final_zip.stat().st_size == entry["size"] and md5_file(final_zip) == entry["md5"] \
            and (models / PACKAGE).is_dir():
        print("already installed and verified")
    else:
        part_dir = download_dir / f".partial_{PACKAGE}"
        part_dir.mkdir(parents=True, exist_ok=True)
        size = entry["size"]
        ranges = [(i, lo, min(lo + CHUNK_BYTES, size) - 1) for i, lo in enumerate(range(0, size, CHUNK_BYTES))]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_chunk, entry["url"], i, lo, hi, part_dir) for i, lo, hi in ranges]
            for n, fut in enumerate(as_completed(futures), start=1):
                idx, how = fut.result()
                print(f"  chunk {idx:3d} {how} ({n}/{len(ranges)}, {time.time() - t0:.0f}s)", flush=True)

        tmp_zip = models / f".{PACKAGE}.zip.tmp"
        h = hashlib.md5()
        with open(tmp_zip, "wb") as out:
            for i, _, _ in ranges:
                data = (part_dir / f"{i:05d}.part").read_bytes()
                out.write(data)
                h.update(data)
        got_size, got_md5 = tmp_zip.stat().st_size, h.hexdigest()
        if got_size != entry["size"] or got_md5 != entry["md5"]:
            tmp_zip.unlink()
            raise SystemExit(f"verification FAILED: size {got_size} vs {entry['size']}, md5 {got_md5} vs {entry['md5']}")
        os.replace(tmp_zip, final_zip)
        print(f"verified: size {got_size} == index, md5 {got_md5} == index ({time.time() - t0:.0f}s)", flush=True)

        # same layout as nltk.downloader for unzip="1" packages: extract into the package's subdir
        with zipfile.ZipFile(final_zip) as z:
            unzipped = sum(info.file_size for info in z.infolist())
            z.extractall(models)
        print(f"extracted {unzipped} bytes (index unzipped_size {entry['unzipped_size']})", flush=True)
        shutil.rmtree(part_dir, ignore_errors=True)

    found = nltk.data.find(f"models/{PACKAGE}")  # the lookup benepar.integrations.downloader.locate_model uses
    print(f"OK: nltk.data.find('models/{PACKAGE}') -> {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
