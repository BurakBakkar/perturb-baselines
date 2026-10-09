"""Per-perturbation metrics on Δ. The only metric code used for reported numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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
    m = len(pred)
    if m < 2:
        return np.full(m, np.nan)
    dist = cdist(np.asarray(pred, np.float64), np.asarray(obs, np.float64), "cityblock")
    own = np.diag(dist)[:, None]
    less = (dist < own).sum(axis=1)
    ties = (dist == own).sum(axis=1) - 1  # exclude the diagonal itself
    # Ties count half, so a constant predictor (all distances tie) lands at chance, 0.5.
    return (less + 0.5 * ties) / (m - 1)


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


def target_gene_metrics(pred, obs, de_idx, perts, genes) -> pd.DataFrame:
    """What a method predicts for the knocked-down gene itself, and pearson_de without it.

    scGPT's perturbation model sees which input gene is perturbed, so it can learn to lower that
    gene; gene-embedding baselines cannot. Separating the target from the other DE genes shows how
    much of a pearson_de score is just the knockdown itself. NaN target values when the target is
    not a measured gene.
    """
    pred = np.asarray(pred, np.float64)
    obs = np.asarray(obs, np.float64)
    pos = {g: i for i, g in enumerate(np.asarray(genes).tolist())}
    rows = []
    for i, p in enumerate(perts):
        t = pos.get(p)
        idx = np.array([j for j in de_idx[i] if j != t])
        rows.append({
            "pred_target": pred[i, t] if t is not None else np.nan,
            "obs_target": obs[i, t] if t is not None else np.nan,
            "pearson_de_offtarget": rowwise_pearson(pred[i, idx][None], obs[i, idx][None])[0],
        })
    return pd.DataFrame(rows)
