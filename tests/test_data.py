import numpy as np
import pytest

from pbench.data import PerturbData


def test_subset_reorders_rows(synthetic):
    data, _ = synthetic
    sub = data.subset(["G3", "G1"])
    assert sub.perts.tolist() == ["G3", "G1"]
    np.testing.assert_array_equal(sub.delta[0], data.delta[3])
    np.testing.assert_array_equal(sub.de_idx[1], data.de_idx[1])
    assert sub.genes is data.genes


def test_subset_unknown_raises(synthetic):
    data, _ = synthetic
    with pytest.raises(KeyError):
        data.subset(["NOPE"])


def test_roundtrip(tmp_path, synthetic):
    data, _ = synthetic
    path = tmp_path / "d.npz"
    data.save(path)
    back = PerturbData.load(path)
    assert back.perts.tolist() == data.perts.tolist()
    np.testing.assert_array_equal(back.delta, data.delta)
    np.testing.assert_array_equal(back.de_idx, data.de_idx)


def test_shape_validation(synthetic):
    data, _ = synthetic
    with pytest.raises(ValueError):
        PerturbData(
            perts=data.perts, genes=data.genes, delta=data.delta[:, :5],
            delta_a=data.delta_a, delta_b=data.delta_b,
            de_idx=data.de_idx, n_cells=data.n_cells,
        )


def test_duplicate_perts_rejected(synthetic):
    data, _ = synthetic
    perts = data.perts.copy()
    perts[1] = perts[0]
    with pytest.raises(ValueError, match="unique"):
        PerturbData(perts=perts, genes=data.genes, delta=data.delta, delta_a=data.delta_a,
                    delta_b=data.delta_b, de_idx=data.de_idx, n_cells=data.n_cells)
