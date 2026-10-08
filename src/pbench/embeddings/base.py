"""Embedding protocol: a vector for the perturbed gene, built from training data only."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from pbench.data import PerturbData


class Embedding(Protocol):
    name: str

    def coverage(self, data: PerturbData) -> set[str]: ...

    def embed(self, train: PerturbData, genes: Sequence[str]) -> np.ndarray: ...


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.astype(str).str.upper()
    df = df[~df.index.duplicated(keep="first")]
    return df.astype(np.float32)


class StaticEmbedding:
    """A fixed gene → vector table (prior knowledge or a pretrained model)."""

    def __init__(self, name: str, loader: Callable[[], pd.DataFrame]):
        self.name = name
        self._loader = loader
        self._table: pd.DataFrame | None = None

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = normalize_table(self._loader())
        return self._table

    def coverage(self, data) -> set[str]:
        return set(self.table.index)

    def embed(self, train, genes: Sequence[str]) -> np.ndarray:
        return self.table.loc[list(genes)].to_numpy()
