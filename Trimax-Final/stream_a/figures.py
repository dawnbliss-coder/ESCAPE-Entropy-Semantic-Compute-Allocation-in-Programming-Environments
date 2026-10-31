"""Stream A figures, drawn only from saved artifacts: entropies/, boundaries/,
bpp_summary.parquet and runs.json. Outputs go to results/figures/stream_a/{run_id}/.

  entropy_heatmap_{file_id}.png   source bytes coloured by per-byte entropy (nats), patch starts
                                  marked (entropy-triggered: solid bar; init: dotted; length-capped:
                                  dashed), plus an entropy trace with tau
  patch_length_distribution.png   patch-length histogram per domain (all patches)
  bpp_distributions.png           per-file mean_bpp and var_bpp by domain (code vs prose, py vs cpp)
  figures.json                    inputs, run config and generation time

Usage:
  python -m stream_a.figures --run-id RUN [--heatmap-files ID ...] [--max-bytes 700]
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from escape_common.io import atomic_write_json  # noqa: E402
from escape_common.schema import (  # noqa: E402
    BPP_SUMMARY_PATH,
    RUNS_JSON_PATH,
    boundary_path,
    corpus_bin_path,
    domain_of,
    entropy_path,
    golden_bin_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_ORDER = ("py", "cpp", "prose")
DOMAIN_COLORS = {"py": "#3572A5", "cpp": "#f34b7d", "prose": "#6a8759"}


def read_source(root: Path, file_id: str) -> bytes:
    domain = domain_of(file_id)
    for path in (corpus_bin_path(root, domain, file_id), golden_bin_path(root, domain, file_id)):
        if path.exists():
            return path.read_bytes()
    raise FileNotFoundError(f"no bytes for {file_id}")


def entropy_heatmap(root: Path, run_id: str, file_id: str, tau: float, out: Path, max_bytes: int) -> None:
    content = read_source(root, file_id)
    ent = np.load(entropy_path(root, run_id, file_id))
    b = pd.read_parquet(boundary_path(root, run_id, file_id))
    n = min(len(content), max_bytes)
    starts = {int(o): t for o, t in zip(b["byte_offset"], b["trigger"]) if o < n}

    # grid layout: one row per source line (wrapped at `wrap` bytes), one column per byte
    wrap = 100
    cells, row, col = [], 0, 0
    for i in range(n):
        cells.append((i, row, col))
        if content[i] == 0x0A:
            row, col = row + 1, 0
        else:
            col += 1
            if col >= wrap:
                row, col = row + 1, 0
    n_rows = row + 1
    n_cols = max(c for _, _, c in cells) + 1 if cells else 1

    fig_w = min(22, 2 + n_cols * 0.16)
    fig_h = 1.8 + n_rows * 0.26 + 2.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_rows * 0.26 + 0.6, 2.0], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("magma_r")
    vmax = max(3.0, float(np.percentile(ent[:n], 99))) if n else 3.0
    for i, r, c in cells:
        colour = cmap(min(float(ent[i]) / vmax, 1.0))
        ax.add_patch(plt.Rectangle((c, n_rows - 1 - r), 1, 1, color=colour, linewidth=0))
        ch = content[i]
        if 0x21 <= ch < 0x7F:
            glyph = chr(ch)
        elif ch == 0x0A:
            glyph = "↵"
        elif ch == 0x20:
            glyph = ""
        elif ch >= 0xC0:  # lead byte of a multi-byte UTF-8 character: draw the whole character here
            width = 2 if ch < 0xE0 else (3 if ch < 0xF0 else 4)
            glyph = content[i:i + width].decode("utf-8", errors="replace")
        else:  # UTF-8 continuation byte or control byte
            glyph = "˙"
        if glyph:
            ax.text(c + 0.5, n_rows - 1 - r + 0.5, glyph, ha="center", va="center", fontsize=6.5,
                    family="monospace", color="white" if float(ent[i]) / vmax > 0.55 else "black")
        trig = starts.get(i)
        if trig is not None:
            style = {"entropy": "-", "init": ":", "length_cap": "--"}.get(trig, "-")
            ax.plot([c, c], [n_rows - 1 - r, n_rows - r], color="#00b4d8", linestyle=style, linewidth=1.6)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_axis_off()
    shown = f"first {n} of {len(content)} bytes" if n < len(content) else f"{len(content)} bytes"
    ax.set_title(f"{file_id} ({shown}) - per-byte entropy (nats), patch starts in cyan "
                 f"(solid: entropy-triggered, dotted: init, dashed: length-capped)", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="entropy (nats)")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(np.arange(n), ent[:n], color="#333333", linewidth=0.8)
    ax2.axhline(tau, color="#d62828", linestyle="--", linewidth=1, label=f"tau = {tau:.4f} nats")
    ent_starts = [o for o, t in starts.items() if t == "entropy"]
    ax2.scatter(ent_starts, ent[ent_starts], s=8, color="#00b4d8", zorder=3, label="entropy-triggered start")
    ax2.set_xlim(0, max(n, 1))
    ax2.set_xlabel("byte offset")
    ax2.set_ylabel("entropy (nats)")
    ax2.legend(fontsize=8, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def patch_length_distribution(root: Path, run_id: str, bpp: pd.DataFrame, out: Path, max_len: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(1, max_len + 2)
    for domain in DOMAIN_ORDER:
        ids = bpp.loc[bpp["domain"] == domain, "file_id"]
        if ids.empty:
            continue
        lengths = np.concatenate([pd.read_parquet(boundary_path(root, run_id, f), columns=["patch_length"])["patch_length"].to_numpy()
                                  for f in ids])
        clipped = np.minimum(lengths, max_len)
        weights = np.full(clipped.shape, 1.0 / clipped.size)
        ax.hist(clipped, bins=bins, weights=weights, histtype="step", linewidth=1.6, color=DOMAIN_COLORS[domain],
                label=f"{domain}: {len(ids)} files, {lengths.size} patches, mean {lengths.mean():.2f}")
    ax.set_yscale("log")
    ax.set_xlabel(f"patch length in bytes (last bin = >= {max_len})")
    ax.set_ylabel("fraction of patches")
    ax.set_title(f"Patch-length distribution - run {run_id}")
    ax.legend(fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bpp_distributions(bpp: pd.DataFrame, run_id: str, out: Path) -> None:
    domains = [d for d in DOMAIN_ORDER if (bpp["domain"] == d).any()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rng = np.random.default_rng(0)  # jitter only
    for ax, stat, label in ((axes[0], "mean_bpp", "mean bytes per patch (per file)"),
                            (axes[1], "var_bpp", "variance of patch length (per file, log)")):
        data = [bpp.loc[bpp["domain"] == d, stat].dropna().to_numpy() for d in domains]
        ax.boxplot(data, tick_labels=[f"{d}\n(n={len(x)})" for d, x in zip(domains, data)], showfliers=False)
        for i, (d, x) in enumerate(zip(domains, data), start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, size=x.size), x, s=6, alpha=0.5, color=DOMAIN_COLORS[d])
        ax.set_ylabel(label)
        if stat == "var_bpp":
            ax.set_yscale("log")
    fig.suptitle(f"BPP per file by domain - run {run_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream A figures from saved artifacts")
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--heatmap-files", nargs="*",
                    help="file_ids for entropy heatmaps (default: first golden fixture of each domain in the run)")
    ap.add_argument("--max-bytes", type=int, default=700)
    ap.add_argument("--out", default="results/figures/stream_a")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    runs = json.loads((root / RUNS_JSON_PATH).read_text())
    tau = float(runs[args.run_id]["tau"])
    bpp = pd.read_parquet(root / BPP_SUMMARY_PATH)
    bpp = bpp[bpp["run_id"] == args.run_id].reset_index(drop=True)
    out_dir = root / args.out / args.run_id
    written = []
    heat = args.heatmap_files
    if not heat:
        heat = []
        for d in DOMAIN_ORDER:
            golden = sorted(p.stem for p in (root / "golden_fixtures" / d).glob("*.bin"))
            present = [f for f in golden if boundary_path(root, args.run_id, f).exists()]
            if present:
                heat.append(present[0])
    for fid in heat:
        path = out_dir / f"entropy_heatmap_{fid}.png"
        entropy_heatmap(root, args.run_id, fid, tau, path, args.max_bytes)
        written.append(path)
    if not bpp.empty:
        for name, fn in (("patch_length_distribution.png", lambda p: patch_length_distribution(root, args.run_id, bpp, p)),
                         ("bpp_distributions.png", lambda p: bpp_distributions(bpp, args.run_id, p))):
            fn(out_dir / name)
            written.append(out_dir / name)
    atomic_write_json(out_dir / "figures.json", {
        "run_id": args.run_id, "run_entry": runs[args.run_id], "heatmap_files": heat,
        "n_files_by_domain": bpp["domain"].value_counts().to_dict() if not bpp.empty else {},
        "outputs": [str(p.relative_to(root)) for p in written],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })
    for p in written:
        print(f"wrote {p.relative_to(root)}")


if __name__ == "__main__":
    main()
