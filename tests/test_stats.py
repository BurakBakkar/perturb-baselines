import numpy as np
import pandas as pd

from pbench.stats import bootstrap_ci, holm, paired_tests, summarize


def _long(seed=0, n=100):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.3, 0.1, size=n)
    values = {
        "ref": base,
        "better": base + 0.05 + rng.normal(0, 0.001, size=n),
        "same": base.copy(),  # identical → p must be 1
    }
    rows = [{"method": m, "fold": i % 5, "pert": f"P{i}", "pearson_centered": v[i]}
            for m, v in values.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, size=500)
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi
    assert abs(mean - 1.0) < 0.15


def test_bootstrap_ignores_nan():
    mean, lo, hi = bootstrap_ci(np.array([1.0, np.nan, 1.0]))
    assert mean == lo == hi == 1.0


def test_holm_monotone_and_bounded():
    adj = holm(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_paired_detects_shift():
    df = _long()
    res = paired_tests(df, "pearson_centered", reference="ref").set_index("method")
    assert res.loc["better", "p_holm"] < 1e-6
    assert res.loc["better", "mean_diff"] > 0.04
    assert res.loc["same", "p_holm"] == 1.0


def test_summarize_shape():
    df = _long()
    s = summarize(df)
    assert set(s.columns) == {"method", "metric", "mean", "ci_lo", "ci_hi", "n"}
    assert len(s) == 3  # 3 methods × 1 metric column present
