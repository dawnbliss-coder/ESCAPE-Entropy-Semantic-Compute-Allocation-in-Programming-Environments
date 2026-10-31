"""Order-preserving per-file map over a Dataset, optionally in forked workers.

Results come back in job order, so callers aggregate exactly as a serial loop
would and outputs do not depend on the number of workers. Workers inherit the
parent's already-built Dataset index through fork instead of re-reading the JSONL.
"""
from __future__ import annotations

import multiprocessing

_DATASET = None


def _call(task):
    fn, job = task
    return fn(_DATASET, *job)


def ordered_map(fn, jobs, dataset, workers=1):
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield fn(dataset, *job)
        return
    global _DATASET
    _DATASET = dataset
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            yield from pool.imap(_call, [(fn, job) for job in jobs], chunksize=1)
    finally:
        _DATASET = None
