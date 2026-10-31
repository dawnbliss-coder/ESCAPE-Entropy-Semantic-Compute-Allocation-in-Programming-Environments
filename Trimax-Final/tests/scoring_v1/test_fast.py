"""The array fast path must reproduce the reference scorer exactly."""
import json
from pathlib import Path

import numpy as np
import pytest

from escape_scoring.calibration import calibrate
from escape_scoring.contract import Dataset
from escape_scoring.engine import evaluate
from escape_scoring.fast import FixedEofNull, as_sorted, counts_fast, match_count
from escape_scoring.fixtures import create_fixture
from escape_scoring.metrics import counts, match, rng_for, shuffled_boundaries

ROOT = Path(__file__).resolve().parents[2]
ADA_RUN = ROOT / "results/ada/synthetic-2719654"


def test_match_count_equals_reference_on_random_sets():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 400))
        p = sorted(set(rng.integers(0, n, rng.integers(0, 60)).tolist()))
        t = sorted(set(rng.integers(0, n, rng.integers(0, 60)).tolist()))
        k = int(rng.choice([0, 1, 2, 4, 8, 50]))
        assert match_count(as_sorted(p), as_sorted(t), k) == len(match(p, t, k))
        assert counts_fast(as_sorted(p), as_sorted(t), k).tolist() == counts(p, t, k).tolist()


def test_fixed_eof_null_draws_equal_reference():
    rng = np.random.default_rng(1)
    for case in range(200):
        lengths = rng.integers(1, 12, rng.integers(1, 80)).tolist()
        reasons = rng.choice(["entropy", "max_length"], len(lengths) - 1).tolist() + ["eof"]
        patches, start = [], 0
        for length, reason in zip(lengths, reasons):
            patches.append({"start_byte": start, "end_byte": start + length, "termination_reason": reason})
            start += length
        a, b = rng_for(7, f"s{case}", "x"), rng_for(7, f"s{case}", "x")
        sampler = FixedEofNull(patches, start)
        for _ in range(20):
            assert sampler.draw(b).tolist() == shuffled_boundaries(patches, start, a, "fixed_eof")


def _strip_environment(scores):
    scores = json.loads(json.dumps(scores))
    for key in ("python", "numpy"):
        scores["configuration"].pop(key)
    return scores


@pytest.mark.skipif(not (ADA_RUN / "data").exists(), reason="retrieved Ada run not present")
@pytest.mark.parametrize("workers", [1, 3])
def test_reproduces_reference_ada_run_exactly(tmp_path, workers):
    """Ada job 2719654 was scored by the reference pure-Python implementation."""
    dataset = Dataset(ADA_RUN / "data")
    record = calibrate(dataset, tmp_path / "calibration.json", workers=workers)
    assert record == json.loads((ADA_RUN / "calibration.json").read_text())
    result = evaluate(dataset, record, tmp_path / "evaluation", workers=workers)
    reference = json.loads((ADA_RUN / "evaluation/scores.json").read_text())
    assert _strip_environment(result) == _strip_environment(reference)


def test_worker_count_does_not_change_random_target_scores(tmp_path):
    create_fixture(tmp_path / "data", aligned=False)
    dataset = Dataset(tmp_path / "data")
    outputs = []
    for workers in (1, 4):
        record = calibrate(dataset, tmp_path / f"k{workers}.json", permutations=300, workers=workers)
        outputs.append((record, evaluate(dataset, record, tmp_path / f"s{workers}", permutations=300,
                                         bootstrap=50, workers=workers)))
    assert outputs[0][0] == outputs[1][0]
    assert _strip_environment(outputs[0][1]) == _strip_environment(outputs[1][1])


def test_baselines_match_stream_b_and_legacy_definitions(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "stream_b"))
    from whitespace_extractor import extract_indent_changes, extract_newlines
    from escape_eval.baselines import word_start_offsets
    from escape_scoring.baselines import indent_change, newline, score_baselines, word_start
    rng = np.random.default_rng(3)
    alphabet = np.frombuffer(b"ab_ 9\n\t\r(){};x\xc3\xa9", dtype=np.uint8)
    for _ in range(300):
        content = bytes(rng.choice(alphabet, rng.integers(0, 120)).tolist())
        assert newline(content) == extract_newlines(content)
        assert indent_change(content) == extract_indent_changes(content)
        legacy = [int(o) for o in word_start_offsets(content) if o > 0]
        assert word_start(content) == legacy
    create_fixture(tmp_path / "data")
    dataset = Dataset(tmp_path / "data")
    record = calibrate(dataset, tmp_path / "k.json", permutations=50)
    result = score_baselines(dataset, record, tmp_path / "baselines.json", bootstrap=20, workers=2)
    assert {r["proposer"] for r in result["rows"]} == {"blt", "newline", "indent_change", "word_start"}
