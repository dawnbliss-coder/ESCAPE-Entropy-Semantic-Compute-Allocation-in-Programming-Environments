import itertools
import json
import numpy as np
import pytest

from escape_scoring.calibration import calibrate, split_manifest, verify_frozen
from escape_scoring.contract import Dataset, derived_boundaries, digest
from escape_scoring.engine import evaluate
from escape_scoring.fixtures import create_fixture
from escape_scoring.metrics import (InvalidNull, counts, match, metrics, predictions,
                                    rng_for, shuffled_boundaries, target_nodes)


@pytest.fixture
def dataset(tmp_path):
    create_fixture(tmp_path / "data")
    return Dataset(tmp_path / "data")


def test_exact_jitter_duplicates_and_one_to_one():
    assert float(metrics(counts([10, 20], [10, 20, 20], 0))["f1"]) == 1
    assert float(metrics(counts([10], [11], 0))["f1"]) == 0
    assert float(metrics(counts([10], [11], 1))["f1"]) == 1
    assert counts([10], [9, 11], 1).tolist() == [1, 0, 1]
    assert counts([9, 11], [10], 1).tolist() == [1, 1, 0]


def test_greedy_matches_bruteforce_cardinality():
    def brute(p, t, k):
        if not p or not t: return 0
        return max([brute(p[1:], t, k)] + [1 + brute(p[1:], t[:j] + t[j+1:], k)
                    for j in range(len(t)) if abs(p[0] - t[j]) <= k])
    subsets = [list(c) for n in range(4) for c in itertools.combinations(range(5), n)]
    for p in subsets:
        for t in subsets:
            for k in [0, 1, 2]:
                assert len(match(p, t, k)) == brute(p, t, k)


def test_exclusions_and_capped_boundary(dataset):
    s = dataset.load(dataset.manifest[0]["sample_id"])
    s.patcher["patches"][0]["termination_reason"] = "max_length"
    s.patcher["boundaries"] = derived_boundaries(s.patcher["patches"])
    assert s.patcher["patches"][0]["end_byte"] not in predictions(s)
    t = target_nodes(s)
    duplicate = {**s.nodes[0], "node_id": "duplicate"}
    s.nodes.append(duplicate)
    assert set(target_nodes(s)) == set(t)
    for node_type, depth, start, end in [("program", 1, 1, 4), ("x", 0, 2, 5), ("x", 1, 3, 3), ("x", 1, 0, 3)]:
        s.nodes.append({**duplicate, "node_id": str(len(s.nodes)), "node_type": node_type,
                        "depth": depth, "start_byte": start, "end_byte": end})
    assert set(target_nodes(s)) == set(t)


def test_null_preserves_tuple_labels_length_and_count():
    patches = [{"start_byte": 0, "end_byte": 2, "termination_reason": "entropy"},
               {"start_byte": 2, "end_byte": 7, "termination_reason": "max_length"},
               {"start_byte": 7, "end_byte": 10, "termination_reason": "eof"}]
    rng = rng_for(1, "a", "test")
    seen = {tuple(shuffled_boundaries(patches, 10, rng)) for _ in range(40)}
    assert seen == {(2,), (7,)}  # cap tuple moves WITH its length, never assigned randomly.
    with pytest.raises(InvalidNull):
        shuffled_boundaries(patches, 11, rng)
    class Reverse:
        def permutation(self, n): return np.arange(n)[::-1]
    with pytest.raises(InvalidNull, match="EOF"):
        shuffled_boundaries(patches, 10, Reverse(), "strict")


def test_random_independent_boundaries_near_null_and_aligned_beats_null():
    rng = np.random.default_rng(13)
    random_observed, aligned_observed, null_scores = [], [], []
    for _ in range(80):
        starts = [0, *sorted(rng.choice(np.arange(1, 201), 12, replace=False).tolist())]
        patches = [{"start_byte": a, "end_byte": b, "termination_reason": "eof" if b == 240 else "entropy"}
                   for a, b in zip(starts, starts[1:] + [240])]
        target = sorted(rng.choice(np.arange(1, 201), 12, replace=False).tolist())
        random_observed.append(float(metrics(counts(starts[1:], target, 0))["f1"]))
        aligned_observed.append(float(metrics(counts(starts[1:], starts[1:], 0))["f1"]))
        draws = [shuffled_boundaries(patches, 240, rng) for _ in range(50)]
        null_scores.append((np.mean([metrics(counts(p, target, 0))["f1"] for p in draws]),
                            np.mean([metrics(counts(p, starts[1:], 0))["f1"] for p in draws])))
    assert abs(np.mean(random_observed) - np.mean(np.array(null_scores)[:, 0])) < .035
    assert np.mean(aligned_observed) - np.mean(np.array(null_scores)[:, 1]) > .7


def test_calibration_never_loads_test_and_persists(dataset, tmp_path):
    split = split_manifest(dataset.manifest)
    assert all(sum(m["split"] == "calibration" and m["language"] == lang for m in split) == 1
               for lang in ["cpp", "python", "english"])
    assert split_manifest(list(reversed(dataset.manifest))) == split
    real_load = dataset.load
    test_ids = {m["sample_id"] for m in split if m["split"] == "test"}
    def guarded(sid):
        assert sid not in test_ids
        return real_load(sid)
    dataset.load = guarded
    output = tmp_path / "k.json"
    record = calibrate(dataset, output, permutations=30)
    assert json.loads(output.read_text()) == record
    assert record["candidates"] == [0, 1, 2, 4, 8]
    assert record["selected_k"] == {"cpp": 0, "python": 0, "english": 0}
    with pytest.raises(FileExistsError):
        calibrate(dataset, output, permutations=2)
    record["selected_k"]["cpp"] = 8
    with pytest.raises(ValueError, match="changed"):
        verify_frozen(dataset, record)


def test_evaluation_determinism_and_frozen_k(dataset, tmp_path):
    record = calibrate(dataset, tmp_path / "k.json", permutations=30)
    a = evaluate(dataset, record, tmp_path / "a", permutations=50, bootstrap=30)
    b = evaluate(dataset, record, tmp_path / "b", permutations=50, bootstrap=30)
    assert a == b
    global_start = next(r for r in a["scores"] if r["axis"] == "global" and r["kind"] == "start")
    assert global_start["f1"] == 1
    assert global_start["f1_margin"] > .7
    assert global_start["f1_permutation_p"] == 1/51
    assert a["synthetic"]
    assert (tmp_path / "a" / "scores.csv").exists()
    with pytest.raises(TypeError):
        evaluate(dataset, record, tmp_path / "c", k=8)


def test_duplicate_source_split_rejected(dataset):
    dataset.manifest[1]["text_sha256"] = dataset.manifest[0]["text_sha256"]
    with pytest.raises(ValueError, match="duplicate source"):
        split_manifest(dataset.manifest)


def test_calibration_criterion_is_median_per_file(tmp_path):
    create_fixture(tmp_path / "many", files_per_language=15, aligned=False)
    data = Dataset(tmp_path / "many")
    record = calibrate(data, tmp_path / "many-k.json", permutations=40)
    for language, chosen in record["selected_k"].items():
        margins = {k: [r["margin"] for r in record["curves"] if r["language"] == language and r["k"] == k]
                   for k in record["candidates"]}
        assert all(len(v) == 3 for v in margins.values())
        medians = {k: np.median(v) for k, v in margins.items()}
        assert chosen == min(k for k, v in medians.items() if v >= max(medians.values()) - 1e-12)


def test_all_tolerance_ties_choose_zero(dataset, tmp_path):
    real_load = dataset.load
    def without_targets(sid):
        s = real_load(sid)
        s.nodes = []
        return s
    dataset.load = without_targets
    record = calibrate(dataset, tmp_path / "tie.json", permutations=3)
    assert set(record["selected_k"].values()) == {0}


def test_stratum_denominator_includes_zero_target_files(dataset, tmp_path):
    # All files carry predictions; one file's distinct node type must not
    # remove predictions from other files of the same language.
    record = calibrate(dataset, tmp_path / "k.json", permutations=3)
    test_rows = [m for m in record["split_manifest"] if m["split"] == "test" and m["language"] == "cpp"]
    sid = test_rows[0]["sample_id"]
    real_rows = dataset.rows
    def rows(kind, sample_id):
        values = real_rows(kind, sample_id)
        if kind == "ground_truth_nodes" and sample_id == sid:
            for v in values: v["node_type"] = "rare"
        return values
    dataset.rows = rows
    result = evaluate(dataset, record, tmp_path / "result", permutations=3, bootstrap=3)
    rare = next(r for r in result["scores"] if r["axis"] == "node_type" and r["stratum"] == "rare" and r["kind"] == "start")
    assert rare["n_files"] == 4
    assert rare["tp"] == 7 and rare["fp"] == 21
