import numpy as np

from pbench.metrics import rowwise_pearson
from pbench.models import KNNModel, NoChange, RidgeModel, TrainMean


def _linear(seed=0, n=120, d=6, g=40):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d))
    W = rng.normal(size=(d, g))
    shared = rng.normal(size=g) * 2
    D = shared + E @ W + 0.1 * rng.normal(size=(n, g))
    return E, D


def test_reference_models():
    E, D = _linear()
    assert np.all(NoChange().fit(E[:100], D[:100]).predict(E[100:]) == 0)
    tm = TrainMean().fit(None, D[:100]).predict(E[100:])
    np.testing.assert_allclose(tm, np.tile(D[:100].mean(0), (20, 1)), rtol=1e-6)


def test_ridge_recovers_linear_signal():
    E, D = _linear()
    model = RidgeModel(seed=0).fit(E[:100], D[:100])
    m = D[:100].mean(0)
    score = rowwise_pearson(model.predict(E[100:]) - m, D[100:] - m).mean()
    assert score > 0.95
    assert model.best_ in model.alphas


def test_ridge_records_inner_scores_for_every_alpha():
    E, D = _linear()
    model = RidgeModel(alphas=(1.0, 10.0), seed=0).fit(E, D)
    assert set(model.inner_scores_) == {1.0, 10.0}
    assert model.inner_scores_[model.best_] == max(model.inner_scores_.values())


def test_knn_returns_neighbour_mean():
    E = np.array([[1.0, 0], [0, 1.0], [0.9, 0.1]])
    D = np.array([[1.0, 1.0], [5.0, 5.0], [3.0, 3.0]])
    model = KNNModel(ks=(2,)).fit(E, D)
    np.testing.assert_allclose(model.predict(np.array([[1.0, 0.05]])), [[2.0, 2.0]])


def test_knn_caps_k_to_train_size():
    E, D = _linear(n=8)
    pred = KNNModel(ks=(1, 3, 50), inner_folds=2).fit(E, D).predict(E[:2])
    assert pred.shape == (2, D.shape[1])
    assert np.isfinite(pred).all()
