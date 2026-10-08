"""Embeddings that need no external resources."""
from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
from sklearn.decomposition import PCA

from pbench.data import PerturbData


class RandomEmbedding:
    """Control: an arbitrary but fixed Gaussian vector per gene name."""

    name = "random"

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def coverage(self, data: PerturbData) -> set[str]:
        return set(data.perts.tolist())

    def embed(self, train: PerturbData, genes: Sequence[str]) -> np.ndarray:
        rows = [np.random.default_rng([self.seed, zlib.crc32(g.encode())]).normal(size=self.dim)
                for g in genes]
        return np.asarray(rows, np.float32)


class PCAEmbedding:
    """Gene g is described by how g's expression moves across the *training* knockdowns."""

    name = "pca"

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def coverage(self, data: PerturbData) -> set[str]:
        return set(data.genes.tolist())

    def embed(self, train: PerturbData, genes: Sequence[str]) -> np.ndarray:
        M = np.asarray(train.delta, np.float64).T  # genes × train perts
        n_comp = min(self.dim, M.shape[0], M.shape[1])
        Z = PCA(n_components=n_comp, random_state=self.seed).fit_transform(M)
        col = {g: i for i, g in enumerate(train.genes.tolist())}
        return Z[[col[g] for g in genes]].astype(np.float32)
