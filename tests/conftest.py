import numpy as np
import pandas as pd
import pytest

from pbench.data import PerturbData


def make_synthetic(n_perts=60, n_genes=80, dim=8, n_de=10, noise=0.1, seed=0):
    """Δ = shared response + Z·W + noise, where Z is a known per-perturbation embedding."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i}" for i in range(n_genes)])
    perts = genes[:n_perts].copy()
    Z = rng.normal(size=(n_perts, dim))
    W = rng.normal(size=(dim, n_genes))
    shared = rng.normal(size=n_genes) * 2.0
    signal = shared + Z @ W
    delta = signal + noise * rng.normal(size=signal.shape)
    delta_a = signal + noise * rng.normal(size=signal.shape)
    delta_b = signal + noise * rng.normal(size=signal.shape)
    de_idx = np.argsort(-np.abs(delta), axis=1)[:, :n_de]
    data = PerturbData(
        perts=perts,
        genes=genes,
        delta=delta.astype(np.float32),
        delta_a=delta_a.astype(np.float32),
        delta_b=delta_b.astype(np.float32),
        de_idx=de_idx.astype(np.int64),
        n_cells=np.full(n_perts, 50, dtype=np.int64),
    )
    return data, pd.DataFrame(Z, index=perts)


@pytest.fixture
def synthetic():
    return make_synthetic()
