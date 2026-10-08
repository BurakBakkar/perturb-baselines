"""Per-perturbation metrics on Δ. The only metric code used for reported numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # noqa: F401  (for discrimination_rank)

METRICS = ("pearson_all", "pearson_de", "pearson_centered", "disc_rank")


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.zeros(len(a))
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def discrimination_rank(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """For each held-out perturbation i: how many *other* perturbations' observed Δ lie closer
    to pred[i] than its own observed Δ does, normalized to [0, 1].

    0 = pred[i] is closest to its own truth (perfect); ~0.5 = chance; 1 = farthest.
    Distance is L1 (cityblock) over all genes. Returns NaN for every row when m < 2
    (no other perturbations to compare against).
    """
    # TODO(user): implement — see the request in the conversation.
    raise NotImplementedError


def per_pert_metrics(pred, obs, de_idx, train_mean) -> pd.DataFrame:
    pred = np.asarray(pred, np.float64)
    obs = np.asarray(obs, np.float64)
    m = np.asarray(train_mean, np.float64)[None, :]
    return pd.DataFrame({
        "pearson_all": rowwise_pearson(pred, obs),
        "pearson_de": rowwise_pearson(np.take_along_axis(pred, de_idx, axis=1),
                                      np.take_along_axis(obs, de_idx, axis=1)),
        "pearson_centered": rowwise_pearson(pred - m, obs - m),
        "disc_rank": discrimination_rank(pred, obs),
    })
