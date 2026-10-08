"""The one interface every method (Phase 1 and Phase 2) shares: a predictions .npz."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Preds:
    perts: np.ndarray
    genes: np.ndarray
    delta_pred: np.ndarray


def save_preds(path, perts: Sequence[str], genes: Sequence[str], delta_pred) -> None:
    delta_pred = np.asarray(delta_pred, np.float32)
    if delta_pred.shape != (len(perts), len(genes)):
        raise ValueError(f"delta_pred shape {delta_pred.shape} != {(len(perts), len(genes))}")
    if not np.isfinite(delta_pred).all():
        raise ValueError("delta_pred must be finite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, perts=np.asarray(perts, str), genes=np.asarray(genes, str),
                        delta_pred=delta_pred)


def load_preds(path) -> Preds:
    with np.load(path, allow_pickle=False) as z:
        return Preds(perts=z["perts"], genes=z["genes"], delta_pred=z["delta_pred"])
