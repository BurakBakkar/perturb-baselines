import numpy as np
import pandas as pd

from pbench.report import best_baseline, breakdown, covariates, split_method
from pbench.splits import kfold_splits


def _df():
    rows = []
    for i in range(30):
        for method, val in [("train_mean", 0.0), ("ridge__go", 0.2), ("ridge__scgpt", 0.3),
                            ("noise_ceiling", 0.9)]:
            bump = 0.2 if (method == "ridge__scgpt" and i < 10) else 0.0
            rows.append({"method": method, "fold": i % 5, "pert": f"P{i}",
                         "pearson_centered": val + bump})
    return pd.DataFrame(rows)


def test_split_method():
    assert split_method("ridge__go") == ("ridge", "go")
    assert split_method("train_mean") == ("train_mean", None)


def test_best_baseline_excludes_fm_and_ceiling():
    assert best_baseline(_df()) == "ridge__go"


def test_breakdown_localizes_gain():
    df = _df()
    cov = pd.DataFrame({"fold": [i % 5 for i in range(30)], "pert": [f"P{i}" for i in range(30)],
                        "effect_size": np.arange(30, dtype=float)})
    out = breakdown(df, cov, ["ridge__scgpt"], "ridge__go", "pearson_centered", n_bins=3)
    low = out[(out.covariate == "effect_size") & (out["bin"] == 0)]["mean_diff"].item()
    high = out[(out.covariate == "effect_size") & (out["bin"] == 2)]["mean_diff"].item()
    assert low > high
    np.testing.assert_allclose(high, 0.1)


def test_covariates_use_train_only_for_similarity(synthetic):
    data, _ = synthetic
    folds = kfold_splits(data.perts.tolist(), 5, 0)
    cov = covariates(data, folds, go_counts=pd.Series({"G0": 3}))
    assert len(cov) == len(data.perts)
    assert cov["nn_similarity"].between(-1, 1).all()
    assert cov.set_index("pert").loc["G0", "go_terms"] == 3
    assert cov.set_index("pert").loc["G1", "go_terms"] == 0
