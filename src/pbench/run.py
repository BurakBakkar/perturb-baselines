"""Run a benchmark config: fair set → CV folds → predictions files → metrics."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pbench.data import PerturbData
from pbench.embeddings.base import Embedding
from pbench.evaluate import evaluate_fold
from pbench.models import MODELS, NoChange, TrainMean
from pbench.preds import save_preds
from pbench.splits import fair_set, kfold_splits


@dataclass
class Config:
    data: Path
    raw_dir: Path
    out_dir: Path
    embeddings: list[str]
    models: list[str]
    n_folds: int = 5
    seed: int = 0
    folds: list[int] | None = None

    @classmethod
    def from_yaml(cls, path) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        for key in ("data", "raw_dir", "out_dir"):
            raw[key] = Path(raw[key])
        return cls(**raw)


def predict_fold(train: PerturbData, test_perts, embeddings, model_names, seed):
    """Everything here sees only `train` and the *names* of the test perturbations."""
    n_test = len(test_perts)
    placeholder = np.zeros((n_test, 0))
    out = {
        "no_change": NoChange().fit(None, train.delta).predict(placeholder),
        "train_mean": TrainMean().fit(None, train.delta).predict(placeholder),
    }
    genes = train.perts.tolist() + list(test_perts)
    n_train = len(train.perts)
    for ename, emb in embeddings.items():
        E = emb.embed(train, genes)
        for mname in model_names:
            model = MODELS[mname](seed=seed).fit(E[:n_train], train.delta)
            out[f"{mname}__{ename}"] = model.predict(E[n_train:])
            print(f"  {mname}__{ename}: best={model.best_}", flush=True)
    return out


def run_experiment(cfg: Config, embeddings: dict[str, Embedding] | None = None) -> pd.DataFrame:
    if embeddings is None:
        from pbench.embeddings import build_embeddings

        embeddings = build_embeddings(cfg.embeddings, cfg.raw_dir)
    data = PerturbData.load(cfg.data)
    coverages = {name: emb.coverage(data) for name, emb in embeddings.items()}
    keep, dropped = fair_set(data.perts.tolist(), coverages)
    data = data.subset(keep)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fair_set.json").write_text(json.dumps({"n_kept": len(keep), "kept": keep,
                                                   "dropped": dropped}, indent=2))
    print(f"[run] fair set: {len(keep)} perturbations", flush=True)

    frames = []
    for k, (train_perts, test_perts) in enumerate(kfold_splits(keep, cfg.n_folds, cfg.seed)):
        if cfg.folds is not None and k not in cfg.folds:
            continue
        print(f"[run] fold {k}: {len(train_perts)} train / {len(test_perts)} test", flush=True)
        train = data.subset(train_perts)
        preds = predict_fold(train, test_perts, embeddings, cfg.models, cfg.seed)
        fold_dir = out / "preds" / "cv" / f"fold{k}"
        if fold_dir.exists():  # never score predictions left over from an earlier config
            shutil.rmtree(fold_dir)
        for method, delta_pred in preds.items():
            save_preds(fold_dir / f"{method}.npz", test_perts, data.genes, delta_pred)
        frames.append(evaluate_fold(data, train_perts, test_perts, fold_dir, k))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "metrics.parquet")
    return df
