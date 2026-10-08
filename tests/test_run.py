import json

import numpy as np

from pbench.embeddings.base import StaticEmbedding
from pbench.embeddings.simple import PCAEmbedding, RandomEmbedding
from pbench.preds import load_preds
from pbench.run import Config, run_experiment
from pbench.splits import kfold_splits
from tests.conftest import make_synthetic


def _cfg(tmp_path, data, name, folds=None):
    path = tmp_path / f"{name}.npz"
    data.save(path)
    return Config(data=path, raw_dir=tmp_path, out_dir=tmp_path / name,
                  embeddings=[], models=["ridge", "knn"], n_folds=5, seed=0, folds=folds)


def _embs(truth):
    return {"random": RandomEmbedding(dim=8), "pca": PCAEmbedding(dim=8),
            "truth": StaticEmbedding("truth", lambda: truth)}


def test_smoke_ridge_on_true_embedding_beats_train_mean(tmp_path):
    data, truth = make_synthetic()
    df = run_experiment(_cfg(tmp_path, data, "smoke"), embeddings=_embs(truth))
    means = df.groupby("method")["pearson_centered"].mean()
    expected = {"no_change", "train_mean", "noise_ceiling", "ridge__truth", "knn__pca"}
    assert expected <= set(means.index)
    assert means["ridge__truth"] > 0.8
    assert abs(means["train_mean"]) < 1e-9
    assert means["noise_ceiling"] > means["ridge__random"]
    assert (tmp_path / "smoke" / "metrics.parquet").exists()


def test_no_leakage_from_held_out_rows(tmp_path):
    data_a, truth = make_synthetic()
    _, test = kfold_splits(data_a.perts.tolist(), 5, 0)[0]
    rows = np.isin(data_a.perts, test)
    rng = np.random.default_rng(99)
    poisoned = {}
    for field in ("delta", "delta_a", "delta_b"):
        arr = getattr(data_a, field).copy()
        arr[rows] = rng.normal(size=(rows.sum(), arr.shape[1]))
        poisoned[field] = arr
    data_b = type(data_a)(perts=data_a.perts, genes=data_a.genes, de_idx=data_a.de_idx,
                          n_cells=data_a.n_cells, **poisoned)
    run_experiment(_cfg(tmp_path, data_a, "a", folds=[0]), embeddings=_embs(truth))
    run_experiment(_cfg(tmp_path, data_b, "b", folds=[0]), embeddings=_embs(truth))
    files = sorted((tmp_path / "a" / "preds" / "cv" / "fold0").glob("*.npz"))
    assert len(files) == 2 + 3 * 2
    for f in files:
        pa = load_preds(f).delta_pred
        pb = load_preds(tmp_path / "b" / "preds" / "cv" / "fold0" / f.name).delta_pred
        np.testing.assert_array_equal(pa, pb, err_msg=f.name)


def test_fair_set_written_and_applied(tmp_path):
    data, truth = make_synthetic()
    partial = truth.drop(index=["G0", "G1"])
    embs = {"truth": StaticEmbedding("truth", lambda: partial)}
    df = run_experiment(_cfg(tmp_path, data, "fair"), embeddings=embs)
    assert not {"G0", "G1"} & set(df["pert"])
    fair = json.loads((tmp_path / "fair" / "fair_set.json").read_text())
    assert fair["dropped"]["truth"] == ["G0", "G1"]
    assert fair["n_kept"] == len(data.perts) - 2
