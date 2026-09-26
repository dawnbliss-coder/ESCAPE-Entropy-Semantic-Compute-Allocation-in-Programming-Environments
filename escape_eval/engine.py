"""Streaming per-file scoring engine (PRELIMINARY).

For every file of a file table, and every requested (target kind, tolerance k), the engine
produces:
  - observed any-match counts for BLT entropy-triggered boundaries, for the whitespace
    (newline) baseline, for the word-boundary baseline (the P3 prose confound control),
    and separately for length-capped boundaries;
  - R density-matched H0 resamples, scored with the same scorer. They are added directly
    into pooled per-resample sums, so memory is O(R x strata), not O(files x R);
  - per-file mean H0 counts, which the file-level bootstrap needs;
  - optionally span IoU (a secondary metric).

One H0 draw per resample per file is shared by every kind, k and stratum, so all those
comparisons are paired. NullSums pool over the files given to one run_engine call:
call it once per domain so that domains never mix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from escape_common.validate import validate_boundaries
from escape_eval import data as D
from escape_eval.baselines import (H0Sampler, h0_spec, whitespace_boundaries, whitespace_segment_starts,
                                   word_boundaries, word_segment_starts)
from escape_eval.matching import ToleranceScorer, best_patch_iou, build_layout
from escape_eval.stats import metric_dict
from escape_eval.targets import AXES, build_targets, unique_spans

COUNT_COLUMNS = ["file_id", "domain", "kind", "k", "axis", "stratum", "method", "tp_p", "tp_r", "n_pred", "n_targets"]
IOU_COLUMNS = ["file_id", "domain", "n_spans", "iou_sum_blt", "iou_sum_whitespace", "iou_sum_word", "iou_sum_h0"]
METHOD_BLT, METHOD_WS, METHOD_WORD, METHOD_CAP, METHOD_H0 = "blt", "whitespace", "word", "length_cap", "h0_mean"


@dataclass(frozen=True)
class EngineConfig:
    ks: tuple
    kinds: tuple = ("start", "end")
    axes: tuple = AXES
    n_resamples: int = 10_000
    master_seed: int = 20260915
    budget: int = 4_000_000  # max H0 boundary offsets held per chunk
    iou: bool = False
    iou_resamples: int = 100
    validate: bool = True


class NullSums:
    """Pooled per-resample H0 numerators keyed by (kind, k, axis, stratum)."""

    def __init__(self, n_resamples: int):
        if int(n_resamples) < 1:
            raise ValueError("n_resamples must be >= 1")
        self.R = int(n_resamples)
        self.tp_p: dict = {}
        self.tp_r: dict = {}

    def add(self, kind: str, k: int, keys, start: int, tp_p: np.ndarray, tp_r: np.ndarray) -> None:
        stop = start + tp_p.shape[0]
        for j, (axis, stratum) in enumerate(keys):
            key = (kind, int(k), axis, str(stratum))
            if key not in self.tp_p:
                self.tp_p[key] = np.zeros(self.R, dtype=np.int64)
                self.tp_r[key] = np.zeros(self.R, dtype=np.int64)
            self.tp_p[key][start:stop] += tp_p[:, j]
            self.tp_r[key][start:stop] += tp_r[:, j]

    def merged(self, other: "NullSums") -> "NullSums":
        """Sum of two independent groups (e.g. py + cpp -> code)."""
        if other.R != self.R:
            raise ValueError("cannot merge NullSums with different R")
        out = NullSums(self.R)
        for src in (self, other):
            for key in src.tp_p:
                if key not in out.tp_p:
                    out.tp_p[key] = np.zeros(self.R, dtype=np.int64)
                    out.tp_r[key] = np.zeros(self.R, dtype=np.int64)
                out.tp_p[key] += src.tp_p[key]
                out.tp_r[key] += src.tp_r[key]
        return out

    def null_metrics(self, kind: str, k: int, keys, n_pred_total: float, n_targets_total) -> dict:
        """metric -> (R, C) pooled H0 metrics for the given stratum keys."""
        zeros = np.zeros(self.R, dtype=np.int64)
        tp_p = np.stack([self.tp_p.get((kind, int(k), a, str(s)), zeros) for a, s in keys], axis=1)
        tp_r = np.stack([self.tp_r.get((kind, int(k), a, str(s)), zeros) for a, s in keys], axis=1)
        return metric_dict(tp_p, n_pred_total, tp_r, np.asarray(n_targets_total, dtype=np.float64)[None, :])


@dataclass
class EngineResult:
    counts: pd.DataFrame
    null: NullSums
    iou: pd.DataFrame
    problems: list
    n_files_scored: int
    seconds: float


def load_file_boundaries(stream_a_root, run_id: str, run_entry: dict, file_id: str, n_bytes: int, validate: bool):
    b = D.load_boundaries(stream_a_root, run_id, file_id)
    if validate:
        problems = validate_boundaries(b, n_bytes=n_bytes, tau=run_entry["tau"],
                                       max_patch_length=run_entry["max_patch_length"], file_id=file_id)
        if problems:
            raise ValueError("; ".join(problems[:5]))
    return (b["byte_offset"].to_numpy(dtype=np.int64),
            b["trigger"].astype(str).to_numpy(dtype=object),
            b["patch_length"].to_numpy(dtype=np.int64))


def run_engine(files: pd.DataFrame, stream_a_root, run_id: str, run_entry: dict, taxonomies: dict,
               cfg: EngineConfig, log=print) -> EngineResult:
    """files: a file table (escape_eval.data). taxonomies: domain -> taxonomy dict."""
    t0 = time.time()
    null = NullSums(cfg.n_resamples)
    count_rows, iou_rows, problems = [], [], []
    n_scored = 0
    for i, row in enumerate(files.itertuples(index=False), start=1):
        fid, domain, n = str(row.file_id), str(row.domain), int(row.n_bytes)
        try:
            off, trig, plen = load_file_boundaries(stream_a_root, run_id, run_entry, fid, n, cfg.validate)
            structure = D.load_structure(row.structure_path, fid)
            D.check_structure_bounds(structure, n, fid)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{fid}: {exc}")
            continue
        init = off[trig == "init"]
        blt = off[trig == "entropy"]
        cap = off[trig == "length_cap"]
        newline_raw = None
        ws = None
        if row.whitespace_path is not None and not pd.isna(row.whitespace_path):
            newline_raw = D.load_newline_offsets(row.whitespace_path, fid)
            ws = whitespace_boundaries(newline_raw, init)
        content = D.load_source_bytes(row.bin_path, n)
        word_raw = word_boundaries(content, init)

        scorers = []
        for kind in cfg.kinds:
            targets = build_targets(structure, kind, taxonomy=taxonomies[domain], exclude_offsets=init, n_bytes=n)
            layout = build_layout(targets, cfg.axes)
            n_t = layout.n_targets_per_stratum()
            for k in cfg.ks:
                scorer = ToleranceScorer(targets.offsets, layout.target_cols, layout.n_strata, n, int(k))
                scorers.append((kind, int(k), layout, n_t, scorer))
                for method, pred in ((METHOD_BLT, blt), (METHOD_WS, ws), (METHOD_WORD, word_raw), (METHOD_CAP, cap)):
                    if pred is None:
                        continue
                    tp_p, tp_r = scorer.counts(pred[None, :])
                    for j, (axis, stratum) in enumerate(layout.keys):
                        count_rows.append((fid, domain, kind, int(k), axis, str(stratum), method,
                                           float(tp_p[0, j]), float(tp_r[0, j]), int(pred.size), int(n_t[j])))

        spec = h0_spec(off, trig, plen, n)
        sampler = H0Sampler(spec, cfg.master_seed, fid)
        sums = [[np.zeros(s[2].n_strata, dtype=np.int64), np.zeros(s[2].n_strata, dtype=np.int64)] for s in scorers]
        chunk = max(1, cfg.budget // max(1, spec.n_scored))
        done = 0
        while done < cfg.n_resamples:
            r = min(chunk, cfg.n_resamples - done)
            draws = sampler.draw(r)
            for s_idx, (kind, k, layout, n_t, scorer) in enumerate(scorers):
                tp_p, tp_r = scorer.counts(draws, check=False)
                null.add(kind, k, layout.keys, done, tp_p, tp_r)
                sums[s_idx][0] += tp_p.sum(axis=0)
                sums[s_idx][1] += tp_r.sum(axis=0)
            done += r
        for s_idx, (kind, k, layout, n_t, scorer) in enumerate(scorers):
            for j, (axis, stratum) in enumerate(layout.keys):
                count_rows.append((fid, domain, kind, k, axis, str(stratum), METHOD_H0,
                                   sums[s_idx][0][j] / cfg.n_resamples, sums[s_idx][1][j] / cfg.n_resamples,
                                   int(spec.n_scored), int(n_t[j])))

        if cfg.iou:
            s, e = unique_spans(structure)
            if s.size:
                iou_blt = float(best_patch_iou(s, e, off, n).sum())
                iou_ws = (float(best_patch_iou(s, e, whitespace_segment_starts(newline_raw, n), n).sum())
                          if newline_raw is not None else np.nan)
                iou_word = float(best_patch_iou(s, e, word_segment_starts(content, n), n).sum())
                layouts = sampler.draw_layouts(cfg.iou_resamples)
                iou_h0 = float(np.mean([best_patch_iou(s, e, layouts[r], n).sum() for r in range(layouts.shape[0])]))
                iou_rows.append((fid, domain, int(s.size), iou_blt, iou_ws, iou_word, iou_h0))
        n_scored += 1
        if log is not None and (i % 50 == 0 or i == len(files)):
            log(f"  engine: {i}/{len(files)} files ({time.time() - t0:.1f}s)")

    return EngineResult(
        counts=pd.DataFrame(count_rows, columns=COUNT_COLUMNS),
        null=null,
        iou=pd.DataFrame(iou_rows, columns=IOU_COLUMNS),
        problems=problems,
        n_files_scored=n_scored,
        seconds=time.time() - t0,
    )


def count_sets(counts: pd.DataFrame, file_ids, kind: str, k: int, keys, methods) -> dict:
    """method -> stats.CountSet over (files x keys), files in the given order. Strata a file
    does not have count as zero targets and zero hits. n_pred always comes from the file's
    'all' row, so every stratum's precision denominator is the full boundary count."""
    from escape_eval.stats import CountSet

    file_ids = list(file_ids)
    fidx = pd.Series(np.arange(len(file_ids)), index=pd.Index(file_ids, dtype=object))
    kidx = {(a, str(s)): j for j, (a, s) in enumerate(keys)}
    sub = counts[(counts["kind"] == kind) & (counts["k"] == int(k))]
    out = {}
    for method in methods:
        m = sub[sub["method"] == method]
        if m.empty:
            continue
        n_files, C = len(file_ids), len(keys)
        tp_p = np.zeros((n_files, C))
        tp_r = np.zeros((n_files, C))
        n_targets = np.zeros((n_files, C))
        n_pred = np.zeros(n_files)
        all_rows = m[m["axis"] == "all"]
        present = all_rows["file_id"].isin(fidx.index)
        n_pred[fidx[all_rows.loc[present, "file_id"]].to_numpy()] = all_rows.loc[present, "n_pred"].to_numpy()
        col = np.fromiter((kidx.get((a, str(s)), -1) for a, s in zip(m["axis"], m["stratum"])), dtype=np.int64, count=len(m))
        keep = (col >= 0) & m["file_id"].isin(fidx.index).to_numpy()
        rows = fidx[m.loc[keep, "file_id"]].to_numpy()
        np.add.at(tp_p, (rows, col[keep]), m.loc[keep, "tp_p"].to_numpy(dtype=np.float64))
        np.add.at(tp_r, (rows, col[keep]), m.loc[keep, "tp_r"].to_numpy(dtype=np.float64))
        np.add.at(n_targets, (rows, col[keep]), m.loc[keep, "n_targets"].to_numpy(dtype=np.float64))
        out[method] = CountSet(tp_p, tp_r, n_pred, n_targets)
    return out


def stratum_keys(counts: pd.DataFrame, kind: str, k: int, axis: str) -> list:
    """Sorted (axis, stratum) keys present for one kind/k/axis."""
    from escape_eval.matching import stratum_sort_key

    sub = counts[(counts["kind"] == kind) & (counts["k"] == int(k)) & (counts["axis"] == axis)]
    values = sorted(set(sub["stratum"].astype(str)), key=lambda v: stratum_sort_key(axis, v))
    return [(axis, v) for v in values]
