"""File ids for the post-submission arms (optional extensions).

  subset   the pre-registered teacher-512 control subset (300 Python + 300 C++ test files,
           600-2,000 bytes; results/final/teacher512_subset_{python,cpp}.txt) plus every
           English test-split file. The byte-balanced student and the adapted teacher are
           extracted and scored here, paired with the runs already scored on these files.
  hf_full  test-split files for the hf_full sensitivity run: all Python and English files and
           the C++ files under 8,192 bytes (one swa512 chunk, so no chunk reset; full attention
           over the longer C++ files, up to 999 KB, is not feasible).
Splits come from the frozen calibration record, never from outcomes.
    python scripts/ada_distill/newarm_ids.py subset|hf_full [--domain py|cpp|prose]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_ids() -> set[str]:
    rec = json.loads((ROOT / "results/final/c-v1-2721954/calibration.json").read_text())
    return {m["sample_id"] for m in rec["split_manifest"] if m["split"] == "test"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("subset", "hf_full"))
    ap.add_argument("--domain", choices=("py", "cpp", "prose"))
    args = ap.parse_args()
    m = pd.read_parquet(ROOT / "corpus/manifest.parquet")
    m = m[m.file_id.isin(test_ids())]
    if args.which == "subset":
        code = set()
        for lang in ("python", "cpp"):
            code |= set((ROOT / f"results/final/teacher512_subset_{lang}.txt").read_text().split())
        sel = m[m.file_id.isin(code) | (m.domain == "prose")]
    else:
        sel = m[(m.domain != "cpp") | (m.n_bytes < 8192)]
    if args.domain:
        sel = sel[sel.domain == args.domain]
    print(" ".join(sorted(sel.file_id)))


if __name__ == "__main__":
    main()
