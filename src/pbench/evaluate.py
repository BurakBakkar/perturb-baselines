"""The single evaluator: score every predictions file in a fold directory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pbench.data import PerturbData
from pbench.metrics import per_pert_metrics
from pbench.preds import load_preds


def _frame(method, fold, perts, metrics: pd.DataFrame) -> pd.DataFrame:
    metrics.insert(0, "pert", list(perts))
    metrics.insert(0, "fold", fold)
    metrics.insert(0, "method", method)
    return metrics


def evaluate_fold(data: PerturbData, train_perts, test_perts, fold_dir, fold: int) -> pd.DataFrame:
    # Round to float32 like the saved predictions, so train_mean − m is exactly 0.
    train_mean = data.subset(train_perts).delta.astype(np.float64).mean(axis=0).astype(np.float32)
    expected = set(test_perts)
    out = []
    for path in sorted(Path(fold_dir).glob("*.npz")):
        preds = load_preds(path)
        if preds.genes.tolist() != data.genes.tolist():
            raise ValueError(f"{path.name}: gene order differs from the data cache")
        if set(preds.perts.tolist()) != expected:
            raise ValueError(f"{path.name}: perturbations differ from fold {fold} test set")
        obs = data.subset(preds.perts.tolist())
        m = per_pert_metrics(preds.delta_pred, obs.delta, obs.de_idx, train_mean)
        out.append(_frame(path.stem, fold, preds.perts, m))
    test = data.subset(test_perts)
    ceiling = per_pert_metrics(test.delta_a, test.delta_b, test.de_idx, train_mean)
    out.append(_frame("noise_ceiling", fold, test.perts, ceiling))
    return pd.concat(out, ignore_index=True)
