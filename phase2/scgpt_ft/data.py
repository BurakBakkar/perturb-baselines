"""Single cells from the GEARS h5ad → (control cell, perturbation flags, perturbed cell) pairs.

This reproduces GEARS's training pairs (each perturbed cell is paired with a random control
cell, which is the model input) without materialising one graph object per cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def target_gene(condition: str) -> str | None:
    """Same rule as pbench.preprocess: exactly one non-'ctrl' part, upper-cased."""
    parts = [p for p in condition.split("+") if p != "ctrl"]
    return parts[0].upper() if len(parts) == 1 else None


@dataclass
class Cells:
    X: sp.csr_matrix  # float32 [n_cells, G], log-normalized expression
    genes: np.ndarray  # gene_name as stored in the h5ad (used for vocab lookup)
    perts: dict[str, np.ndarray]  # upper-case target gene → row indices
    ctrl_rows: np.ndarray

    @cached_property
    def gene_index(self) -> dict[str, int]:
        return {g.upper(): i for i, g in enumerate(self.genes.tolist())}

    @property
    def out_genes(self) -> np.ndarray:
        return np.array([g.upper() for g in self.genes.tolist()])

    def ctrl_mean(self) -> np.ndarray:
        return np.asarray(self.X[self.ctrl_rows].mean(axis=0), np.float64).ravel()

    def dense(self, rows) -> np.ndarray:
        return np.asarray(self.X[rows].toarray(), np.float32)


def cells_from_arrays(X, genes, conditions) -> Cells:
    conditions = np.asarray(conditions).astype(str)
    groups = pd.Series(np.arange(len(conditions))).groupby(conditions).indices
    perts = {}
    for cond, rows in groups.items():
        t = target_gene(cond) if cond != "ctrl" else None
        if t is not None:
            perts[t] = np.asarray(rows)
    return Cells(X=sp.csr_matrix(X, dtype=np.float32), genes=np.asarray(genes).astype(str),
                 perts=perts, ctrl_rows=np.asarray(groups["ctrl"]))


def load_cells(h5ad_path) -> Cells:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    return cells_from_arrays(adata.X, adata.var["gene_name"].astype(str).to_numpy(),
                             adata.obs["condition"].astype(str).to_numpy())


def pert_flags(target: str, gene_index: dict[str, int], n_genes: int) -> np.ndarray:
    """scGPT's perturbation tokens: 1 at the target gene's input position, 0 elsewhere.

    A target that is not a measured gene has no position, so it gets no flag at all.
    """
    flags = np.zeros(n_genes, np.int64)
    if target in gene_index:
        flags[gene_index[target]] = 1
    return flags


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, cells: Cells, perts, max_cells: int, seed: int):
        unknown = [p for p in perts if p not in cells.perts]
        if unknown:
            raise KeyError(f"perturbations not in the data: {unknown[:5]}")
        self.cells, self.perts, self.max_cells, self.seed = cells, list(perts), max_cells, seed
        n_genes = cells.X.shape[1]
        self._flags = {p: pert_flags(p, cells.gene_index, n_genes) for p in self.perts}
        self.resample(0)

    def resample(self, epoch: int) -> None:
        """Draw ≤ max_cells cells per perturbation and a random control partner for each."""
        rng = np.random.default_rng([self.seed, epoch])
        items = []
        for p in self.perts:
            rows = self.cells.perts[p]
            if len(rows) > self.max_cells:
                rows = rng.choice(rows, self.max_cells, replace=False)
            ctrl = rng.choice(self.cells.ctrl_rows, len(rows), replace=True)
            items += [(p, int(r), int(c)) for r, c in zip(rows, ctrl, strict=True)]
        self.items = items

    def rows_used(self) -> set[int]:
        return {r for _, r, _ in self.items} | {c for _, _, c in self.items}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        p, r, c = self.items[i]
        return self.cells.dense([c])[0], self._flags[p], self.cells.dense([r])[0]
