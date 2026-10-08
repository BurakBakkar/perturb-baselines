"""Perturbation-level splits."""
from __future__ import annotations

from collections.abc import Sequence

from sklearn.model_selection import KFold


def kfold_splits(perts: Sequence[str], n_folds: int = 5, seed: int = 0):
    perts = sorted(perts)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [([perts[i] for i in tr], [perts[i] for i in te]) for tr, te in kf.split(perts)]


def fair_set(perts: Sequence[str], coverages: dict[str, set[str]]):
    keep = [p for p in perts if all(p in cov for cov in coverages.values())]
    dropped = {name: sorted(p for p in perts if p not in cov) for name, cov in coverages.items()}
    return keep, dropped
