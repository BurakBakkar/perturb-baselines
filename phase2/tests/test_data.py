import numpy as np

from scgpt_ft.data import PairDataset, pert_flags, target_gene


def test_target_gene():
    assert target_gene("aars+ctrl") == "AARS"
    assert target_gene("ctrl") is None
    assert target_gene("A+B") is None


def test_cells_groups_and_gene_index(cells):
    assert set(cells.perts) == {"G0", "G1", "G2", "NOTMEASURED"}
    assert len(cells.ctrl_rows) == 40
    assert cells.gene_index["G3"] == 3
    assert cells.out_genes[0] == "G0"


def test_flags_mark_target_index(cells):
    f = pert_flags("G2", cells.gene_index, 12)
    assert f.dtype == np.int64 and f.sum() == 1 and f[2] == 1


def test_flags_unmeasured_target_all_zero(cells):
    assert pert_flags("NOTMEASURED", cells.gene_index, 12).sum() == 0


def test_pair_dataset_caps_and_handles_small_perts(cells):
    ds = PairDataset(cells, ["G0", "G1"], max_cells=10, seed=0)
    assert len(ds) == 10 + 5  # G0 capped at 10, G1 has only 5
    ctrl, flags, target = ds[0]
    assert ctrl.shape == target.shape == flags.shape == (12,)
    assert ctrl.dtype == np.float32 and flags.dtype == np.int64


def test_pair_dataset_only_uses_listed_perts_and_controls(cells):
    ds = PairDataset(cells, ["G0", "G2"], max_cells=100, seed=0)
    allowed = set(cells.perts["G0"]) | set(cells.perts["G2"]) | set(cells.ctrl_rows)
    for epoch in range(3):
        ds.resample(epoch)
        assert ds.rows_used() <= allowed
    held_out = set(cells.perts["G1"]) | set(cells.perts["NOTMEASURED"])
    assert not (ds.rows_used() & held_out)


def test_pair_dataset_resample_is_seeded(cells):
    a = PairDataset(cells, ["G0"], max_cells=10, seed=3)
    b = PairDataset(cells, ["G0"], max_cells=10, seed=3)
    assert a.items == b.items
    a.resample(1)
    assert a.items != b.items


def test_pair_dataset_rejects_unknown_pert(cells):
    import pytest

    with pytest.raises(KeyError):
        PairDataset(cells, ["NOPE"], max_cells=10, seed=0)
