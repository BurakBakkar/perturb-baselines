from pbench.splits import fair_set, kfold_splits

PERTS = [f"P{i}" for i in range(23)]


def test_folds_disjoint_and_cover():
    folds = kfold_splits(PERTS, n_folds=5, seed=0)
    assert len(folds) == 5
    tests = [set(te) for _, te in folds]
    assert set().union(*tests) == set(PERTS)
    assert sum(len(t) for t in tests) == len(PERTS)
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert set(tr) | set(te) == set(PERTS)


def test_folds_deterministic_and_order_independent():
    a = kfold_splits(PERTS, seed=0)
    b = kfold_splits(list(reversed(PERTS)), seed=0)
    assert a == b
    assert kfold_splits(PERTS, seed=1) != a


def test_fair_set_drops_uncovered_and_reports():
    keep, dropped = fair_set(["A", "B", "C"], {"x": {"A", "B", "C"}, "pca": {"A", "C"}})
    assert keep == ["A", "C"]
    assert dropped == {"x": [], "pca": ["B"]}
