"""Turn a GEARS-style single-cell AnnData into a PerturbData cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pbench.data import PerturbData


def target_gene(condition: str) -> str | None:
    parts = [p for p in condition.split("+") if p != "ctrl"]
    if len(parts) != 1:
        return None
    return parts[0].upper()


def _rows_dense(X, rows: np.ndarray) -> np.ndarray:
    sub = X[rows]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return sub.astype(np.float64)


def _mean_var(X, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = _rows_dense(X, rows)
    var = sub.var(axis=0, ddof=1) if len(rows) > 1 else np.zeros(sub.shape[1])
    return sub.mean(axis=0), var


def compute_perturb_data(X, conditions, genes, *, n_de=20, min_cells=10, seed=0):
    conditions = np.asarray(conditions).astype(str)
    genes = np.asarray(genes).astype(str)
    groups = pd.Series(np.arange(len(conditions))).groupby(conditions).indices
    if "ctrl" not in groups or len(groups["ctrl"]) < 2:
        raise ValueError("need at least 2 control cells labelled 'ctrl'")
    ctrl_rows = groups["ctrl"]
    ctrl_mean, ctrl_var = _mean_var(X, ctrl_rows)
    n_ctrl = len(ctrl_rows)
    n_de = min(n_de, len(genes))

    log = {"n_control_cells": int(n_ctrl), "dropped_combo": [], "dropped_few_cells": []}
    by_gene: dict[str, np.ndarray] = {}
    for cond, rows in sorted(groups.items()):
        if cond == "ctrl":
            continue
        gene = target_gene(cond)
        if gene is None:
            log["dropped_combo"].append(cond)
            continue
        if gene in by_gene:
            raise ValueError(f"two conditions map to target gene {gene}")
        if len(rows) < min_cells:
            log["dropped_few_cells"].append(gene)
            continue
        by_gene[gene] = rows

    rng = np.random.default_rng(seed)
    perts = sorted(by_gene)
    n_p, n_g = len(perts), len(genes)
    delta = np.empty((n_p, n_g), np.float32)
    delta_a = np.empty_like(delta)
    delta_b = np.empty_like(delta)
    de_idx = np.empty((n_p, n_de), np.int64)
    n_cells = np.empty(n_p, np.int64)
    for i, gene in enumerate(perts):
        rows = by_gene[gene]
        mean, var = _mean_var(X, rows)
        d = mean - ctrl_mean
        t = d / np.sqrt(var / len(rows) + ctrl_var / n_ctrl + 1e-12)
        delta[i] = d
        de_idx[i] = np.argsort(-np.abs(t), kind="stable")[:n_de]
        perm = rng.permutation(rows)
        half = len(rows) // 2
        delta_a[i] = _rows_dense(X, perm[:half]).mean(0) - ctrl_mean
        delta_b[i] = _rows_dense(X, perm[half:]).mean(0) - ctrl_mean
        n_cells[i] = len(rows)

    log["n_perts"] = n_p
    data = PerturbData(
        perts=np.array(perts), genes=genes, delta=delta, delta_a=delta_a,
        delta_b=delta_b, de_idx=de_idx, n_cells=n_cells,
    )
    return data, log


def preprocess_gears(raw_dir: str, out: str) -> None:
    import anndata as ad

    hits = sorted(Path(raw_dir).rglob("perturb_processed.h5ad"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one perturb_processed.h5ad under {raw_dir}, got {hits}")
    adata = ad.read_h5ad(hits[0])
    genes = adata.var["gene_name"].astype(str).str.upper().to_numpy()
    data, log = compute_perturb_data(adata.X, adata.obs["condition"].astype(str), genes)
    log["source"] = str(hits[0])
    log["n_cells_total"] = int(adata.n_obs)
    log["n_genes"] = int(len(genes))
    data.save(out)
    Path(out).with_suffix(".log.json").write_text(json.dumps(log, indent=2))
