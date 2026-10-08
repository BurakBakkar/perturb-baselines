"""Models map an embedding of the perturbed gene to a predicted Δ over all measured genes.

Models only ever see (E_train, D_train) and E_test: no gene names, no held-out data.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pbench.metrics import rowwise_pearson


class NoChange:
    def __init__(self, seed: int = 0):
        pass

    def fit(self, E, D):
        self.n_genes_ = D.shape[1]
        return self

    def predict(self, E):
        return np.zeros((len(E), self.n_genes_), np.float32)


class TrainMean:
    def __init__(self, seed: int = 0):
        pass

    def fit(self, E, D):
        self.mean_ = np.asarray(D, np.float64).mean(axis=0)
        return self

    def predict(self, E):
        return np.tile(self.mean_, (len(E), 1)).astype(np.float32)


class _FixedRidge:
    def __init__(self, alpha):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, E, D):
        self.pipe.fit(E, D)
        return self

    def predict(self, E):
        return self.pipe.predict(E)


def _unit(E):
    E = np.asarray(E, np.float64)
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(norm > 0, norm, 1.0)


class _FixedKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, E, D):
        self.E_ = _unit(E)
        self.D_ = np.asarray(D, np.float64)
        return self

    def predict(self, E):
        sims = _unit(E) @ self.E_.T
        k = min(self.k, len(self.E_))
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        return self.D_[idx].mean(axis=1)


def _inner_score(make, E, D, n_folds, seed) -> float:
    """Mean centered Pearson on inner CV: the same headline metric used for reporting."""
    kf = KFold(n_splits=min(n_folds, len(E)), shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(E):
        pred = make().fit(E[tr], D[tr]).predict(E[va])
        m = D[tr].mean(axis=0)
        scores.append(rowwise_pearson(pred - m, D[va] - m).mean())
    return float(np.mean(scores))


class _Selected:
    """Pick one hyperparameter by inner CV, then refit on all training perturbations."""

    grid: tuple
    fixed: type

    def __init__(self, inner_folds: int, seed: int):
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, E, D):
        E = np.asarray(E, np.float64)
        D = np.asarray(D, np.float64)
        scores = {v: _inner_score(lambda v=v: self.fixed(v), E, D, self.inner_folds, self.seed)
                  for v in self.grid}
        self.best_ = max(scores, key=scores.get)
        self.inner_scores_ = scores
        self.model_ = self.fixed(self.best_).fit(E, D)
        return self

    def predict(self, E):
        return self.model_.predict(np.asarray(E, np.float64)).astype(np.float32)


class RidgeModel(_Selected):
    fixed = _FixedRidge

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6), inner_folds=5, seed=0):
        super().__init__(inner_folds, seed)
        self.alphas = self.grid = tuple(alphas)


class KNNModel(_Selected):
    fixed = _FixedKNN

    def __init__(self, ks=(1, 3, 5, 10, 20), inner_folds=5, seed=0):
        super().__init__(inner_folds, seed)
        self.ks = self.grid = tuple(ks)


MODELS = {"ridge": RidgeModel, "knn": KNNModel}
