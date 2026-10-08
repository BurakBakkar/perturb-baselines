import numpy as np
import pandas as pd

from pbench.embeddings.base import StaticEmbedding
from pbench.embeddings.simple import PCAEmbedding, RandomEmbedding


def test_static_embedding_uppercases_and_dedupes():
    table = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=["actb", "ACTB", "Gapdh"])
    emb = StaticEmbedding("t", lambda: table)
    assert emb.coverage(None) == {"ACTB", "GAPDH"}
    np.testing.assert_allclose(emb.embed(None, ["GAPDH", "ACTB"]), [[5, 6], [1, 2]])


def test_random_is_deterministic_per_name(synthetic):
    data, _ = synthetic
    emb = RandomEmbedding(dim=4, seed=0)
    a = emb.embed(data.subset(["G0", "G1"]), ["G5", "G0"])
    b = emb.embed(data.subset(["G2"]), ["G0"])
    np.testing.assert_allclose(a[1], b[0])
    assert a.shape == (2, 4)
    assert emb.coverage(data) == set(data.perts.tolist())


def test_pca_coverage_is_measured_genes(synthetic):
    data, _ = synthetic
    assert PCAEmbedding().coverage(data) == set(data.genes.tolist())


def test_pca_uses_only_train_rows(synthetic):
    data, _ = synthetic
    train = data.subset([f"G{i}" for i in range(40)])
    E = PCAEmbedding(dim=5).embed(train, ["G45", "G3"])
    assert E.shape == (2, 5)
    # Same train set → same embedding, regardless of what else exists in `data`.
    E2 = PCAEmbedding(dim=5).embed(train, ["G45", "G3"])
    np.testing.assert_allclose(E, E2)
