"""GEARS 'simulation' split (GEARS's own DataSplitter), for comparability with published runs."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from gears.data_utils import DataSplitter

from scgpt_ft.data import Cells, target_gene


def gears_simulation_split(conditions, seed: int = 1) -> dict[str, list[str]]:
    """Same call as GEARS PertData.prepare_split('simulation', seed) on an obs-only AnnData."""
    obs = pd.DataFrame({"condition": pd.Categorical(np.asarray(conditions).astype(str))})
    adata = ad.AnnData(obs=obs)
    adata, _ = DataSplitter(adata, split_type="simulation").split_data(
        train_gene_set_size=0.75, combo_seen2_train_frac=0.75, seed=seed)
    out = {}
    for name in ("train", "val", "test"):
        conds = adata.obs.loc[adata.obs["split"] == name, "condition"].astype(str).unique()
        out[name] = sorted(c for c in conds if c != "ctrl")
    return out


def make_gears_split(cells: Cells, conditions, seed: int = 1, n_top: int = 1000) -> dict:
    s = gears_simulation_split(conditions, seed)
    to_t = {name: sorted({target_gene(c) for c in s[name]} - {None}) for name in s}
    top = np.argsort(-cells.ctrl_mean(), kind="stable")[:n_top]
    return {**to_t,
            "test_measured": [p for p in to_t["test"] if p in cells.gene_index],
            "top_expressed_genes": cells.out_genes[top].tolist(), "seed": seed}
