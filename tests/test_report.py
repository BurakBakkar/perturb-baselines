import json

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


def test_best_baseline_excludes_corrected_ceiling():
    df = _df()
    extra = df[df.method == "noise_ceiling"].assign(method="noise_ceiling_sb")
    assert best_baseline(pd.concat([df, extra])) == "ridge__go"


def test_fm_vs_reference_tests_only_fm_methods():
    from pbench.report import fm_vs
    out = fm_vs(_df().assign(method=lambda d: d.method.replace({"ridge__go": "ridge__pca"})),
                "ridge__pca")
    assert out["method"].tolist() == ["ridge__scgpt"]
    assert out["mean_diff"].item() > 0


def test_finetune_extras(tmp_path):
    from pbench.report import finetune_extras

    rng = np.random.default_rng(0)
    rows = []
    for m, shift in (("finetune__scgpt", 0.1), ("ridge__scgpt", 0.05), ("knn__scgpt", 0.0)):
        for fold in (0, 1):
            for i in range(20):
                base = rng.normal(0.3, 0.05)
                rows.append({"method": m, "fold": fold, "pert": f"P{fold}_{i}",
                             "pearson_all": base + shift, "pearson_de": base + shift,
                             "pearson_centered": base + shift, "disc_rank": 0.3 - shift})
    df = pd.DataFrame(rows)
    run, final = tmp_path / "run", tmp_path / "final"
    run.mkdir()
    final.mkdir()
    g = df[df.method.isin(["finetune__scgpt", "ridge__scgpt"])]
    pd.concat([g.assign(view="all"), g.assign(view="measured")]).to_parquet(
        run / "metrics_gears_sim.parquet")
    (run / "gate.json").write_text(json.dumps({"passed": True}))

    finetune_extras(df, run, final)

    ft = pd.read_csv(final / "paired_tests_finetune_vs_static.csv")
    assert set(ft.reference) == {"ridge__scgpt", "knn__scgpt"} and len(ft) == 8
    assert (ft[ft.metric == "pearson_centered"].mean_diff > 0).all()
    assert "p_holm" in ft.columns
    assert "view: measured" in (final / "gears_sim.md").read_text()
    assert json.loads((final / "gate.json").read_text()) == {"passed": True}


def test_finetune_extras_noop_without_finetune(tmp_path):
    from pbench.report import finetune_extras

    finetune_extras(_df(), tmp_path, tmp_path)
    assert not (tmp_path / "paired_tests_finetune_vs_static.csv").exists()
