from scgpt_ft.split import gears_simulation_split, make_gears_split
from tests.conftest import make_cells


def _conds(n=40):
    return ["ctrl"] * 5 + [f"G{i}+ctrl" for i in range(n) for _ in range(2)]


def test_gears_split_partitions_conditions_deterministically():
    s = gears_simulation_split(_conds(), seed=1)
    allc = {f"G{i}+ctrl" for i in range(40)}
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == allc
    assert not set(s["train"]) & set(s["test"]) and not set(s["val"]) & set(s["test"])
    assert gears_simulation_split(_conds(), seed=1) == s


def test_make_gears_split_schema(cells):
    conds = ["ctrl"] * 40 + ["G0+ctrl"] * 30 + ["G1+ctrl"] * 5 + ["G2+ctrl"] * 20 + \
        ["NOTMEASURED+ctrl"] * 8
    out = make_gears_split(cells, conds, seed=1, n_top=5)
    assert set(out) == {"train", "val", "test", "test_measured", "top_expressed_genes", "seed"}
    assert set(out["test_measured"]) <= set(out["test"])
    assert "NOTMEASURED" not in out["test_measured"]
    assert len(out["top_expressed_genes"]) == 5
    assert all(g == g.upper() for g in out["top_expressed_genes"])


def test_conftest_import_path():
    assert make_cells().X.shape[1] == 12
