import numpy as np

from pbench.metrics import (
    METRICS,
    discrimination_rank,
    per_pert_metrics,
    rowwise_pearson,
)


def _obs(seed=0, m=200, g=50):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=g) * 3
    obs = shared + rng.normal(size=(m, g))
    de_idx = np.argsort(-np.abs(obs), axis=1)[:, :10]
    return obs, de_idx, shared


def test_perfect_prediction():
    obs, de_idx, shared = _obs()
    df = per_pert_metrics(obs, obs, de_idx, train_mean=shared)
    assert list(df.columns) == list(METRICS)
    np.testing.assert_allclose(df["pearson_all"], 1.0)
    np.testing.assert_allclose(df["pearson_de"], 1.0)
    np.testing.assert_allclose(df["pearson_centered"], 1.0)
    np.testing.assert_allclose(df["disc_rank"], 0.0)


def test_train_mean_scores_zero_centered_but_high_raw():
    obs, de_idx, shared = _obs()
    pred = np.tile(shared, (len(obs), 1))
    df = per_pert_metrics(pred, obs, de_idx, train_mean=shared)
    assert df["pearson_all"].mean() > 0.8
    np.testing.assert_allclose(df["pearson_centered"], 0.0)


def test_constant_prediction_scores_zero():
    obs, de_idx, shared = _obs()
    df = per_pert_metrics(np.zeros_like(obs), obs, de_idx, train_mean=shared)
    assert not df.isna().any().any()
    np.testing.assert_allclose(df["pearson_all"], 0.0)


def test_shuffled_prediction_is_chance_rank():
    obs, de_idx, shared = _obs()
    pred = obs[np.random.default_rng(1).permutation(len(obs))]
    rank = discrimination_rank(pred, obs)
    assert 0.4 < rank.mean() < 0.6


def test_rowwise_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
    expected = [np.corrcoef(a[i], b[i])[0, 1] for i in range(5)]
    np.testing.assert_allclose(rowwise_pearson(a, b), expected)


def test_rank_single_row_is_nan():
    assert np.isnan(discrimination_rank(np.ones((1, 3)), np.ones((1, 3)))).all()
