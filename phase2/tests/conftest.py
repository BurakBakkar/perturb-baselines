import numpy as np
import pytest
import scipy.sparse as sp

from scgpt_ft.data import cells_from_arrays


def make_cells(n_genes=12, n_ctrl=40, cells_per_pert=(30, 5, 20, 8), seed=0):
    """Targets: G0, G1, G2 are measured genes; NOTMEASURED is not among the genes."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])  # lower-case, as some h5ads store them
    targets = ["G0", "G1", "G2", "NOTMEASURED"]
    conds = ["ctrl"] * n_ctrl
    for t, n in zip(targets, cells_per_pert, strict=True):
        conds += [f"{t}+ctrl"] * n
    X = rng.poisson(1.0, size=(len(conds), n_genes)).astype(np.float32)
    return cells_from_arrays(sp.csr_matrix(X), genes, np.array(conds))


@pytest.fixture
def cells():
    return make_cells()
