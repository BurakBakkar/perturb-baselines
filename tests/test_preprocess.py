import numpy as np
import scipy.sparse as sp

from pbench.preprocess import compute_perturb_data, target_gene


def test_target_gene():
    assert target_gene("kif11+ctrl") == "KIF11"
    assert target_gene("ctrl+KIF11") == "KIF11"
    assert target_gene("ctrl") is None
    assert target_gene("A+B") is None


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n_g = 6
    ctrl = rng.normal(0, 0.1, size=(40, n_g))
    a = rng.normal(0, 0.1, size=(20, n_g))
    a[:, 2] += 3.0  # perturbation A strongly shifts gene 2
    b = rng.normal(0, 0.1, size=(12, n_g))
    b[:, 4] -= 2.0
    few = rng.normal(0, 0.1, size=(3, n_g))
    combo = rng.normal(0, 0.1, size=(15, n_g))
    X = np.vstack([ctrl, a, b, few, combo])
    cond = ["ctrl"] * 40 + ["A+ctrl"] * 20 + ["B+ctrl"] * 12 + ["C+ctrl"] * 3 + ["A+B"] * 15
    genes = np.array([f"g{i}" for i in range(n_g)])
    return X, np.array(cond), genes


def test_compute_delta_and_de():
    X, cond, genes = _toy()
    data, log = compute_perturb_data(sp.csr_matrix(X), cond, genes, n_de=2, min_cells=10)
    assert data.perts.tolist() == ["A", "B"]
    ctrl_mean = X[:40].mean(0)
    np.testing.assert_allclose(data.delta[0], X[40:60].mean(0) - ctrl_mean, atol=1e-5)
    assert data.de_idx[0, 0] == 2
    assert data.de_idx[1, 0] == 4
    assert data.n_cells.tolist() == [20, 12]
    assert log["n_control_cells"] == 40


def test_halves_partition_cells():
    X, cond, genes = _toy()
    data, _ = compute_perturb_data(X, cond, genes, n_de=2, min_cells=10)
    # mean of the two half-deltas, weighted by half sizes, equals the full delta
    n = 20
    w_a, w_b = (n // 2) / n, (n - n // 2) / n
    np.testing.assert_allclose(w_a * data.delta_a[0] + w_b * data.delta_b[0], data.delta[0],
                               atol=1e-5)
    assert not np.allclose(data.delta_a[0], data.delta_b[0])


def test_compute_drops_small_and_combo_perts():
    X, cond, genes = _toy()
    data, log = compute_perturb_data(X, cond, genes, n_de=2, min_cells=10)
    assert "C" not in data.perts.tolist()
    assert log["dropped_few_cells"] == ["C"]
    assert log["dropped_combo"] == ["A+B"]
    assert np.isfinite(data.delta).all()


def test_deterministic_given_seed():
    X, cond, genes = _toy()
    d1, _ = compute_perturb_data(X, cond, genes, n_de=2, seed=3)
    d2, _ = compute_perturb_data(X, cond, genes, n_de=2, seed=3)
    np.testing.assert_array_equal(d1.delta_a, d2.delta_a)
