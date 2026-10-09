import json

import numpy as np
import pandas as pd
import pytest

from pbench.evaluate import evaluate_fold
from pbench.external import (
    FINETUNE_METHOD,
    ExternalConfig,
    export_splits,
    flag_diagnostic,
    load_config,
    score_cv,
    score_gears,
)
from pbench.models import TrainMean
from pbench.preds import save_preds
from pbench.splits import kfold_splits
from tests.conftest import make_synthetic


def _baseline_run(tmp_path, data, n_folds=5, seed=0, folds=(0, 1)):
    """A fake Phase 1 run dir: fair_set.json + metrics.parquet with train_mean rows."""
    run = tmp_path / "base"
    keep = data.perts.tolist()
    (run / "preds").mkdir(parents=True)
    (run / "fair_set.json").write_text(json.dumps({"n_kept": len(keep), "kept": keep}))
    frames = []
    for k, (train, test) in enumerate(kfold_splits(keep, n_folds, seed)):
        if k not in folds:
            continue
        d = run / "preds" / "cv" / f"fold{k}"
        tm = TrainMean().fit(None, data.subset(train).delta).predict(np.zeros((len(test), 0)))
        save_preds(d / "train_mean.npz", test, data.genes, tm)
        frames.append(evaluate_fold(data, train, test, d, k))
    pd.concat(frames, ignore_index=True).to_parquet(run / "metrics.parquet")
    return run


def _cfg(tmp_path, data_path, run, gate=None):
    return ExternalConfig(
        data=data_path, raw_dir=tmp_path / "raw", out_dir=tmp_path / "out", baseline_run=run,
        splits_dir=tmp_path / "splits", preds_dir=tmp_path / "preds", cv_folds=[0, 1],
        gate=gate or {"mean_ref": 0.404, "scgpt_lo": 0.290, "scgpt_hi": 0.344, "tol": 0.05},
    )


@pytest.fixture
def setup(tmp_path):
    data, _ = make_synthetic()
    data_path = tmp_path / "data.npz"
    data.save(data_path)
    run = _baseline_run(tmp_path, data)
    return data, data_path, run


def test_export_splits_matches_kfold(tmp_path, setup):
    data, _, run = setup
    paths = export_splits(run, 5, 0, [0, 1], tmp_path / "splits")
    expected = kfold_splits(data.perts.tolist(), 5, 0)
    for k, p in zip([0, 1], paths, strict=True):
        got = json.loads(p.read_text())
        assert got["fold"] == k
        assert got["train"] == expected[k][0] and got["test"] == expected[k][1]


def test_score_cv_merges_external_with_baselines(tmp_path, setup):
    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    for k in (0, 1):
        test = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())["test"]
        perfect = data.subset(test).delta
        save_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz",
                   test, data.genes, perfect)
    df = score_cv(cfg, data)
    ft = df[df.method == FINETUNE_METHOD]
    assert set(ft.fold) == {0, 1}
    assert np.allclose(ft.pearson_all, 1.0)
    assert {"train_mean", "noise_ceiling", "noise_ceiling_sb"} <= set(df.method)
    # noise ceilings come only from the baseline run, never duplicated
    assert len(df[(df.method == "noise_ceiling") & (df.fold == 0)]) == len(ft[ft.fold == 0])


def test_score_cv_rejects_stale_split(tmp_path, setup):
    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 1, [0, 1], cfg.splits_dir)  # wrong seed
    with pytest.raises(ValueError, match="does not match"):
        score_cv(cfg, data)


def test_score_cv_rejects_wrong_perts(tmp_path, setup):
    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    for k in (0, 1):
        test = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())["test"][:-1]
        save_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz",
                   test, data.genes, data.subset(test).delta)
    with pytest.raises(ValueError, match="perturbations differ"):
        score_cv(cfg, data)


def test_score_cv_requires_external_preds(tmp_path, setup):
    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    with pytest.raises(FileNotFoundError):
        score_cv(cfg, data)


def _gears_split(cfg, data):
    perts = data.perts.tolist()
    split = {"train": perts[:40], "val": perts[40:45], "test": perts[45:],
             "test_measured": perts[45:55], "top_expressed_genes": data.genes[:30].tolist(),
             "seed": 1}
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    (cfg.splits_dir / "gears_sim.json").write_text(json.dumps(split))
    return split


def test_score_gears_writes_reference_preds_and_gate(tmp_path, setup):
    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    split = _gears_split(cfg, data)
    for view, key in (("all", "test"), ("measured", "test_measured")):
        save_preds(cfg.preds_dir / "gears_sim" / view / f"{FINETUNE_METHOD}.npz",
                   split[key], data.genes, data.subset(split[key]).delta)
    df, gate = score_gears(cfg, data)
    assert set(df.view) == {"all", "measured"}
    assert {FINETUNE_METHOD, "train_mean", "no_change"} <= set(df.method)
    assert gate["finetune__scgpt"] == pytest.approx(1.0)
    assert gate["scgpt_ok"] is False  # perfect is far above the published scGPT range
    assert set(gate) >= {"train_mean", "calibration_ok", "passed", "flag_diagnostic"}


def test_flag_diagnostic():
    rng = np.random.default_rng(0)
    same = np.tile(rng.normal(size=50), (5, 1))
    assert flag_diagnostic(same) == pytest.approx(1.0)
    assert flag_diagnostic(rng.normal(size=(5, 50))) < 0.5


def test_load_config_dispatches(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data: d.npz\nraw_dir: r\nout_dir: o\nbaseline_run: b\nsplits_dir: s\n"
                 "preds_dir: p\ncv_folds: [0, 1]\n")
    assert isinstance(load_config(p), ExternalConfig)
    p.write_text("data: d.npz\nraw_dir: r\nout_dir: o\nembeddings: [random]\nmodels: [ridge]\n")
    assert type(load_config(p)).__name__ == "Config"


def test_score_external_writes_target_gene_diagnostic(tmp_path, setup):
    from pbench.external import score_external

    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    for k in (0, 1):
        test = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())["test"]
        save_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz",
                   test, data.genes, data.subset(test).delta)
    score_external(cfg)
    tg = pd.read_parquet(cfg.out_dir / "target_gene.parquet")
    assert {FINETUNE_METHOD, "train_mean"} == set(tg.method)
    assert set(tg.fold) == {0, 1}
    ft = tg[tg.method == FINETUNE_METHOD]
    assert np.allclose(ft.pred_target, ft.obs_target)


def test_score_external_skips_gears_without_predictions(tmp_path, setup, capsys):
    from pbench.external import score_external

    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    _gears_split(cfg, data)  # split exists, but no GEARS predictions yet
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    for k in (0, 1):
        test = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())["test"]
        save_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz",
                   test, data.genes, data.subset(test).delta)
    df = score_external(cfg)
    assert FINETUNE_METHOD in set(df.method)
    assert not (cfg.out_dir / "gate.json").exists()
    assert "GEARS predictions missing" in capsys.readouterr().out


def test_score_external_warns_when_gate_fails(tmp_path, setup, capsys):
    from pbench.external import score_external

    data, data_path, run = setup
    cfg = _cfg(tmp_path, data_path, run)
    split = _gears_split(cfg, data)
    for view, key in (("all", "test"), ("measured", "test_measured")):
        save_preds(cfg.preds_dir / "gears_sim" / view / f"{FINETUNE_METHOD}.npz",
                   split[key], data.genes, data.subset(split[key]).delta)  # perfect: fails gate
    score_external(cfg)
    assert json.loads((cfg.out_dir / "gate.json").read_text())["passed"] is False
    assert "SANITY GATE FAILED" in capsys.readouterr().out


def test_score_cv_requires_baseline_rows_for_every_fold(tmp_path):
    data, _ = make_synthetic()
    data_path = tmp_path / "data.npz"
    data.save(data_path)
    run = _baseline_run(tmp_path, data, folds=(0,))  # baseline has fold 0 only
    cfg = _cfg(tmp_path, data_path, run)
    export_splits(run, 5, 0, [0, 1], cfg.splits_dir)
    for k in (0, 1):
        test = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())["test"]
        save_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz",
                   test, data.genes, data.subset(test).delta)
    with pytest.raises(ValueError, match="no baseline rows for fold"):
        score_cv(cfg, data)
