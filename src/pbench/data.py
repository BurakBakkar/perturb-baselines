"""Cached pseudobulk perturbation data: everything Phase 1 needs, nothing single-cell."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PerturbData:
    perts: np.ndarray  # str[P]  target gene symbol of each perturbation (upper-case)
    genes: np.ndarray  # str[G]  measured genes, column order of every matrix below
    delta: np.ndarray  # float32[P, G]  perturbation mean − control mean
    delta_a: np.ndarray  # float32[P, G]  same, from a random half of the perturbation's cells
    delta_b: np.ndarray  # float32[P, G]  same, from the other half
    de_idx: np.ndarray  # int64[P, n_de]  column indices of the top DE genes
    n_cells: np.ndarray  # int64[P]

    def __post_init__(self) -> None:
        n_p, n_g = len(self.perts), len(self.genes)
        for name in ("delta", "delta_a", "delta_b"):
            shape = getattr(self, name).shape
            if shape != (n_p, n_g):
                raise ValueError(f"{name} has shape {shape}, expected {(n_p, n_g)}")
        if self.de_idx.ndim != 2 or self.de_idx.shape[0] != n_p:
            raise ValueError(f"de_idx must be [P, n_de], got {self.de_idx.shape}")
        if self.n_cells.shape != (n_p,):
            raise ValueError(f"n_cells must be [P], got {self.n_cells.shape}")
        if len(set(self.perts.tolist())) != n_p:
            raise ValueError("perturbation names must be unique")

    def subset(self, perts: Sequence[str]) -> PerturbData:
        index = {p: i for i, p in enumerate(self.perts.tolist())}
        missing = [p for p in perts if p not in index]
        if missing:
            raise KeyError(f"unknown perturbations: {missing[:5]}")
        rows = np.array([index[p] for p in perts], dtype=np.int64)
        return PerturbData(
            perts=self.perts[rows],
            genes=self.genes,
            delta=self.delta[rows],
            delta_a=self.delta_a[rows],
            delta_b=self.delta_b[rows],
            de_idx=self.de_idx[rows],
            n_cells=self.n_cells[rows],
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> PerturbData:
        with np.load(path, allow_pickle=False) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls)})
