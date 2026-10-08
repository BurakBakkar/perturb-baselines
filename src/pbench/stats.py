"""Uncertainty and significance for per-perturbation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pbench.metrics import METRICS


def bootstrap_ci(x, n_boot=1000, seed=0, alpha=0.05):
    x = np.asarray(x, np.float64)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method", sort=False):
        for metric in [m for m in METRICS if m in df.columns]:
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy())
            rows.append({"method": method, "metric": metric, "mean": mean,
                         "ci_lo": lo, "ci_hi": hi, "n": int(g[metric].notna().sum())})
    return pd.DataFrame(rows)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def paired_tests(df: pd.DataFrame, metric: str, reference: str) -> pd.DataFrame:
    wide = df.pivot_table(index=["fold", "pert"], columns="method", values=metric)
    rows = []
    for method in wide.columns:
        if method == reference:
            continue
        pair = wide[[reference, method]].dropna()
        diff = pair[method] - pair[reference]
        p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(pair[method], pair[reference]).pvalue)
        rows.append({"method": method, "reference": reference, "metric": metric,
                     "mean_diff": float(diff.mean()), "n": len(pair), "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out
