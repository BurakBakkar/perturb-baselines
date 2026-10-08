"""Predict Δ for held-out perturbations: run a pool of control cells through the model with
the perturbation flag set, average the predicted cells, subtract the mean control cell."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch

from scgpt_ft.data import Cells, pert_flags


class _Batch:
    """The two attributes of a GEARS batch that TransformerGenerator.pred_perturb reads."""

    def __init__(self, x: torch.Tensor, n: int):
        self.x = x
        self.pert = ["_"] * n

    def to(self, device):
        self.x = self.x.to(device)
        return self


@torch.no_grad()
def predict_delta(model, cells: Cells, perts, gene_ids, pool_size: int, batch_size: int,
                  seed: int, amp: bool) -> np.ndarray:
    n_genes = cells.X.shape[1]
    ctrl_mean = cells.ctrl_mean()
    out = np.empty((len(perts), n_genes), np.float32)
    for i, p in enumerate(perts):
        # Seed by name, so a perturbation's pool doesn't depend on which others are predicted.
        rng = np.random.default_rng([seed, zlib.crc32(p.encode())])
        n_ctrl = len(cells.ctrl_rows)
        rows = rng.choice(cells.ctrl_rows, pool_size, replace=pool_size > n_ctrl)
        flags = torch.as_tensor(pert_flags(p, cells.gene_index, n_genes), dtype=torch.float32)
        total = np.zeros(n_genes, np.float64)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            vals = torch.as_tensor(cells.dense(chunk))
            x = torch.stack([vals.reshape(-1), flags.repeat(len(chunk))], dim=1)
            pred = model.pred_perturb(_Batch(x, len(chunk)), include_zero_gene="all",
                                      gene_ids=gene_ids, amp=amp)
            total += pred.float().cpu().numpy().astype(np.float64).sum(axis=0)
        out[i] = total / len(rows) - ctrl_mean
    return out


def save_preds(path, perts, genes, delta_pred) -> None:
    """Same file contract as pbench.preds.save_preds (re-implemented: no cross-env imports)."""
    delta_pred = np.asarray(delta_pred, np.float32)
    if delta_pred.shape != (len(perts), len(genes)):
        raise ValueError(f"delta_pred shape {delta_pred.shape} != {(len(perts), len(genes))}")
    if not np.isfinite(delta_pred).all():
        raise ValueError("delta_pred must be finite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, perts=np.asarray(perts, str), genes=np.asarray(genes, str),
                        delta_pred=delta_pred)
