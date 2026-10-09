"""Bridge to methods trained outside this package (Phase 2): export splits, score their preds.

Phase 2 runs in its own environment and shares only files with pbench: the fold lists exported
here, and predictions .npz files that are scored by the same evaluator as every baseline.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pbench.data import PerturbData
from pbench.evaluate import evaluate_fold
from pbench.metrics import rowwise_pearson, target_gene_metrics
from pbench.models import NoChange, TrainMean
from pbench.preds import load_preds, save_preds
from pbench.splits import kfold_splits

FINETUNE_METHOD = "finetune__scgpt"
_PATH_KEYS = ("data", "raw_dir", "out_dir", "baseline_run", "splits_dir", "preds_dir")


def export_splits(run_dir, n_folds: int, seed: int, folds, out_dir) -> list[Path]:
    """Write the exact train/test perturbation lists a finished pbench run used for `folds`."""
    keep = json.loads((Path(run_dir) / "fair_set.json").read_text())["kept"]
    splits = kfold_splits(keep, n_folds, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in folds:
        train, test = splits[k]
        path = out / f"fold{k}.json"
        path.write_text(json.dumps({"fold": k, "train": train, "test": test}, indent=2))
        paths.append(path)
    return paths


@dataclass
class ExternalConfig:
    data: Path
    raw_dir: Path
    out_dir: Path
    baseline_run: Path
    splits_dir: Path
    preds_dir: Path
    cv_folds: list[int]
    n_folds: int = 5
    seed: int = 0
    gate: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> ExternalConfig:
        raw = dict(raw)
        for key in _PATH_KEYS:
            raw[key] = Path(raw[key])
        return cls(**raw)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if "baseline_run" in raw:
        return ExternalConfig.from_dict(raw)
    from pbench.run import Config

    return Config.from_yaml(path)


def flag_diagnostic(delta_pred: np.ndarray) -> float:
    """Mean pairwise Pearson between different perturbations' predictions (1.0 = identical)."""
    c = np.corrcoef(np.asarray(delta_pred, np.float64))
    off = c[~np.eye(len(c), dtype=bool)]
    return float(np.nanmean(off))


def score_cv(cfg: ExternalConfig, data: PerturbData) -> pd.DataFrame:
    keep = json.loads((cfg.baseline_run / "fair_set.json").read_text())["kept"]
    data = data.subset(keep)
    base = pd.read_parquet(cfg.baseline_run / "metrics.parquet")
    base = base[base.fold.isin(cfg.cv_folds)]
    missing = sorted(set(cfg.cv_folds) - set(base.fold))
    if missing:
        raise ValueError(f"no baseline rows for fold(s) {missing} in {cfg.baseline_run}")
    expected = kfold_splits(keep, cfg.n_folds, cfg.seed)
    frames = [base]
    for k in cfg.cv_folds:
        split = json.loads((cfg.splits_dir / f"fold{k}.json").read_text())
        if (split["train"], split["test"]) != (expected[k][0], expected[k][1]):
            raise ValueError(f"fold{k}.json does not match the baseline run's fold {k}")
        fold_dir = cfg.preds_dir / "cv" / f"fold{k}"
        if not list(fold_dir.glob("*.npz")):
            raise FileNotFoundError(f"no predictions in {fold_dir}")
        rows = evaluate_fold(data, split["train"], split["test"], fold_dir, k)
        rows = rows[~rows.method.str.startswith("noise_ceiling")]
        clash = set(rows.method) & set(base.method)
        if clash:
            raise ValueError(f"external methods clash with baseline names: {sorted(clash)}")
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def score_gears(cfg: ExternalConfig, data: PerturbData) -> tuple[pd.DataFrame, dict]:
    split = json.loads((cfg.splits_dir / "gears_sim.json").read_text())
    train = data.subset(split["train"])
    frames = []
    for view, key in (("all", "test"), ("measured", "test_measured")):
        test = split[key]
        fold_dir = cfg.preds_dir / "gears_sim" / view
        empty = np.zeros((len(test), 0))
        save_preds(fold_dir / "train_mean.npz", test, data.genes,
                   TrainMean().fit(None, train.delta).predict(empty))
        save_preds(fold_dir / "no_change.npz", test, data.genes,
                   NoChange().fit(None, train.delta).predict(empty))
        rows = evaluate_fold(data, split["train"], test, fold_dir, 0)
        rows.insert(1, "view", view)
        frames.append(rows)

    gene_pos = {g: i for i, g in enumerate(data.genes.tolist())}
    top = np.array([gene_pos[g] for g in split["top_expressed_genes"]])
    measured = cfg.preds_dir / "gears_sim" / "measured"
    gate: dict = {"metric": "pearson_delta_top1000_expressed", "reference": cfg.gate,
                  "n_test_measured": len(split["test_measured"])}
    for method in (FINETUNE_METHOD, "train_mean"):
        p = load_preds(measured / f"{method}.npz")
        obs = data.subset(p.perts.tolist()).delta
        gate[method] = float(rowwise_pearson(p.delta_pred[:, top], obs[:, top]).mean())
    ft = load_preds(measured / f"{FINETUNE_METHOD}.npz").delta_pred
    gate["flag_diagnostic"] = flag_diagnostic(ft)
    g = cfg.gate
    gate["calibration_ok"] = bool(abs(gate["train_mean"] - g["mean_ref"]) <= g["tol"])
    gate["scgpt_ok"] = bool(g["scgpt_lo"] - g["tol"] <= gate[FINETUNE_METHOD]
                            <= g["scgpt_hi"] + g["tol"])
    gate["flag_ok"] = bool(gate["flag_diagnostic"] < 0.999)
    gate["passed"] = gate["calibration_ok"] and gate["scgpt_ok"] and gate["flag_ok"]
    return pd.concat(frames, ignore_index=True), gate


def target_gene_table(cfg: ExternalConfig, data: PerturbData) -> pd.DataFrame:
    """target_gene_metrics for every method (baselines and external) on the CV folds."""
    frames = []
    for k in cfg.cv_folds:
        dirs = [cfg.baseline_run / "preds" / "cv" / f"fold{k}", cfg.preds_dir / "cv" / f"fold{k}"]
        for path in sorted(p for d in dirs for p in d.glob("*.npz")):
            preds = load_preds(path)
            obs = data.subset(preds.perts.tolist())
            m = target_gene_metrics(preds.delta_pred, obs.delta, obs.de_idx,
                                    preds.perts.tolist(), data.genes)
            m.insert(0, "pert", preds.perts.tolist())
            m.insert(0, "fold", k)
            m.insert(0, "method", path.stem)
            frames.append(m)
    return pd.concat(frames, ignore_index=True)


def score_external(cfg: ExternalConfig) -> pd.DataFrame:
    data = PerturbData.load(cfg.data)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gears_preds = [cfg.preds_dir / "gears_sim" / v / f"{FINETUNE_METHOD}.npz"
                   for v in ("all", "measured")]
    if not (cfg.splits_dir / "gears_sim.json").exists():
        print("[score-external] no GEARS split; skipping the sanity gate", flush=True)
    elif not all(p.exists() for p in gears_preds):
        print("[score-external] GEARS predictions missing; skipping the sanity gate", flush=True)
    else:
        gdf, gate = score_gears(cfg, data)
        gdf.to_parquet(out / "metrics_gears_sim.parquet")
        (out / "gate.json").write_text(json.dumps(gate, indent=2))
        print(f"[score-external] GEARS gate: {json.dumps(gate)}", flush=True)
        if not gate["passed"]:
            print("[score-external] WARNING: SANITY GATE FAILED (see gate.json). Do not report CV "
                  "results until this is explained (spec §7).", flush=True)
    if not all((cfg.splits_dir / f"fold{k}.json").exists() for k in cfg.cv_folds):
        print("[score-external] CV splits not exported yet; skipping CV scoring", flush=True)
        return pd.DataFrame()
    if not all((cfg.preds_dir / "cv" / f"fold{k}").exists() for k in cfg.cv_folds):
        print("[score-external] CV predictions missing; skipping CV scoring", flush=True)
        return pd.DataFrame()
    df = score_cv(cfg, data)
    df.to_parquet(out / "metrics.parquet")
    shutil.copy(cfg.baseline_run / "fair_set.json", out / "fair_set.json")
    target_gene_table(cfg, data).to_parquet(out / "target_gene.parquet")
    diag = {}
    for k in cfg.cv_folds:
        p = load_preds(cfg.preds_dir / "cv" / f"fold{k}" / f"{FINETUNE_METHOD}.npz")
        diag[f"fold{k}"] = flag_diagnostic(p.delta_pred)
    (out / "diagnostics.json").write_text(json.dumps({"flag_diagnostic": diag}, indent=2))
    print(f"[score-external] wrote {out / 'metrics.parquet'} ({len(df)} rows)", flush=True)
    return df
