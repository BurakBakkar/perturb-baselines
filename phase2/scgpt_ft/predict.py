"""Predict Δ for held-out perturbations: run a pool of control cells through the model with
the perturbation flag set, average the predicted cells, subtract the mean control cell."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
from scgpt.utils import map_raw_id_to_vocab_id

from scgpt_ft.data import Cells, pert_flags


def forward_all_genes(model, values: torch.Tensor, flags: torch.Tensor, gene_ids,
                      amp: bool) -> torch.Tensor:
    """TransformerGenerator.pred_perturb(include_zero_gene="all") without its all-False padding
    mask: same outputs, but lets PyTorch use memory-efficient attention over 5,000 genes."""
    model.eval()
    device = next(model.parameters()).device
    values, flags = values.to(device), flags.to(device)
    ids = torch.arange(values.shape[1], device=device)
    mapped = map_raw_id_to_vocab_id(ids, gene_ids).repeat(len(values), 1)
    with torch.cuda.amp.autocast(enabled=amp):
        out = model(mapped, values, flags, src_key_padding_mask=None, CLS=False, CCE=False,
                    MVC=False, ECS=False, do_sample=True)
    return out["mlm_output"].float()


def _release_cached_gpu_memory() -> None:
    # Training (~9 GB) and inference (~8 GB) caches together exceed 16 GB; on WSL the driver then
    # spills to host RAM and runs ~3x slower instead of raising OOM.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_delta(model, cells: Cells, perts, gene_ids, pool_size: int, batch_size: int,
                  seed: int, amp: bool) -> np.ndarray:
    _release_cached_gpu_memory()
    try:
        return _predict_delta(model, cells, perts, gene_ids, pool_size, batch_size, seed, amp)
    finally:
        _release_cached_gpu_memory()


@torch.no_grad()
def _predict_delta(model, cells: Cells, perts, gene_ids, pool_size: int, batch_size: int,
                   seed: int, amp: bool) -> np.ndarray:
    n_genes = cells.X.shape[1]
    ctrl_mean = cells.ctrl_mean()
    out = np.empty((len(perts), n_genes), np.float32)
    for i, p in enumerate(perts):
        # Seed by name, so a perturbation's pool doesn't depend on which others are predicted.
        rng = np.random.default_rng([seed, zlib.crc32(p.encode())])
        n_ctrl = len(cells.ctrl_rows)
        rows = rng.choice(cells.ctrl_rows, pool_size, replace=pool_size > n_ctrl)
        flags = torch.as_tensor(pert_flags(p, cells.gene_index, n_genes))
        total = np.zeros(n_genes, np.float64)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            vals = torch.as_tensor(cells.dense(chunk))
            pred = forward_all_genes(model, vals, flags.repeat(len(chunk), 1), gene_ids, amp)
            total += pred.cpu().numpy().astype(np.float64).sum(axis=0)
        out[i] = total / len(rows) - ctrl_mean
    return out


def subset_rows(perts, delta: np.ndarray, subset) -> np.ndarray:
    """Rows of `delta` (one per name in `perts`) for the names in `subset`, in that order."""
    index = {p: i for i, p in enumerate(perts)}
    return delta[[index[p] for p in subset]]


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
