"""Gene embeddings from GenePT (text) and single-cell foundation models (scGPT, Geneformer)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _find_one(root, name: str) -> Path:
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name} under {root}, found {hits}")
    return hits[0]


def load_genept(genept_dir) -> pd.DataFrame:
    with open(_find_one(genept_dir, "GenePT_gene_embedding_ada_text.pickle"), "rb") as f:
        table = pickle.load(f)
    # Stack straight into one float32 matrix; from_dict(orient="index") peaks at several GB.
    genes = list(table)
    matrix = np.asarray([table.pop(g) for g in genes], dtype=np.float32)
    return pd.DataFrame(matrix, index=genes)


def scgpt_table(weight: np.ndarray, vocab: dict[str, int]) -> pd.DataFrame:
    genes = [g for g in vocab if not g.startswith("<")]
    return pd.DataFrame(weight[[vocab[g] for g in genes]], index=genes)


def load_scgpt(ckpt_dir) -> pd.DataFrame:
    import torch

    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=True)
    key = "encoder.embedding.weight"
    if key not in state:
        raise KeyError(f"{key} missing; embedding-like keys: {[k for k in state if 'emb' in k]}")
    vocab = json.loads((ckpt_dir / "vocab.json").read_text())
    return scgpt_table(state[key].float().numpy(), vocab)


def geneformer_table(weight, token_dict: dict[str, int], name_to_id: dict[str, str]):
    rows = {sym.upper(): weight[token_dict[ens]]
            for sym, ens in name_to_id.items() if ens in token_dict}
    return pd.DataFrame.from_dict(rows, orient="index")


def load_geneformer(gf_dir) -> pd.DataFrame:
    from safetensors import safe_open

    gf_dir = Path(gf_dir)
    path = gf_dir / "Geneformer-V2-104M" / "model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        weight = f.get_tensor("bert.embeddings.word_embeddings.weight").float().numpy()
    with open(gf_dir / "geneformer" / "token_dictionary_gc104M.pkl", "rb") as f:
        token_dict = pickle.load(f)
    with open(gf_dir / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb") as f:
        name_to_id = pickle.load(f)
    return geneformer_table(weight, token_dict, name_to_id)
