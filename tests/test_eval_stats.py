"""Pooled metrics, permutation p-values, cluster bootstrap and paired randomisation."""

import math

import numpy as np
import pytest

from escape_eval import stats as S


def test_prf_values_and_zero_handling():
    p, r, f = S.prf(3, 4, 2, 5)
    assert p == pytest.approx(0.75) and r == pytest.approx(0.4)
    assert f == pytest.approx(2 * 0.75 * 0.4 / (0.75 + 0.4))
    p, r, f = S.prf(0, 0, 0, 5)
    assert math.isnan(p) and r == 0 and math.isnan(f)
    p, r, f = S.prf(0, 3, 0, 5)
    assert p == 0 and r == 0 and f == 0


def test_permutation_p_value_formula():
    null = np.array([0.1, 0.2, 0.3])
    assert S.permutation_p_value(0.2, null) == pytest.approx((1 + 2) / (3 + 1))
    assert S.permutation_p_value(0.35, null) == pytest.approx(1 / 4)
    summary = S.permutation_summary([0.35], null[:, None])
    assert summary["null_mean"][0] == pytest.approx(0.2)
    assert summary["z"][0] == pytest.approx((0.35 - 0.2) / np.std(null, ddof=1))


def test_bootstrap_is_deterministic_and_chunk_invariant():
    ent = S.seed_entropy(5, "py|k=1", S.PURPOSE_BOOTSTRAP)
    fn = lambda w: w[:, :3]  # noqa: E731
    a = S.bootstrap_draws(ent, 20, 300, fn, budget=40)   # 2 resamples per chunk
    b = S.bootstrap_draws(ent, 20, 300, fn, budget=10_000)
    np.testing.assert_array_equal(a, b)
    weights = next(S.iter_bootstrap_weights(ent, 20, 5))
    assert np.all(weights.sum(axis=1) == 20)


def test_bootstrap_ci_covers_truth_on_easy_case():
    rng = np.random.default_rng(0)
    n_files = 300
    n_pred = rng.integers(20, 60, size=n_files).astype(float)
    tp = rng.binomial(n_pred.astype(int), 0.3).astype(float)
    cs = S.CountSet(tp[:, None], tp[:, None], n_pred, n_pred[:, None])
    draws = S.bootstrap_draws(S.seed_entropy(1, "x", S.PURPOSE_BOOTSTRAP), n_files, 2000,
                              lambda w: cs.pooled(w)["precision"])
    lo, hi = S.percentile_ci(draws)
    assert lo[0] < 0.3 < hi[0]
    assert hi[0] - lo[0] < 0.05


def test_paired_randomisation():
    rng = np.random.default_rng(1)
    n_files = 40
    n_pred = rng.integers(10, 30, size=n_files).astype(float)
    n_t = rng.integers(10, 30, size=(n_files, 1)).astype(float)
    same = S.CountSet(np.minimum(n_pred, 5)[:, None], np.minimum(n_t, 5), n_pred, n_t)
    out = S.paired_randomisation(same, same, S.seed_entropy(0, "t", S.PURPOSE_RANDOMISATION_WS), 200)
    assert out["f1"]["observed"][0] == 0 and out["f1"]["p_value"][0] == 1.0
    better = S.CountSet(np.minimum(n_pred, 9)[:, None], np.minimum(n_t, 9), n_pred, n_t)
    worse = S.CountSet(np.minimum(n_pred, 2)[:, None], np.minimum(n_t, 2), n_pred, n_t)
    out = S.paired_randomisation(better, worse, S.seed_entropy(0, "t", S.PURPOSE_RANDOMISATION_WS), 500)
    assert out["f1"]["observed"][0] > 0 and out["f1"]["p_value"][0] < 0.01
    again = S.paired_randomisation(better, worse, S.seed_entropy(0, "t", S.PURPOSE_RANDOMISATION_WS), 500)
    np.testing.assert_array_equal(out["f1"]["null"], again["f1"]["null"])


def test_median_difference_bootstrap():
    x = np.arange(100, dtype=float)
    y = np.arange(100, dtype=float) + 10
    diff, draws = S.bootstrap_median_difference(x, y, [1, 2, 3], [4, 5, 6], 500)
    assert diff == pytest.approx(-10)
    lo, hi = S.percentile_ci(draws)
    assert lo < -10 < hi
