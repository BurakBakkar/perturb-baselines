"""Prior-knowledge gene embeddings: GO annotations and the STRING interaction network."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD


def read_gaf_pairs(path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if "NOT" in cols[3]:
                continue
            rows.append((cols[2].upper(), cols[4]))
    return pd.DataFrame(rows, columns=["gene", "go"]).drop_duplicates().reset_index(drop=True)


def go_annotation_counts(path) -> pd.Series:
    return read_gaf_pairs(path).groupby("gene")["go"].nunique()


def load_go(path, dim: int = 64, seed: int = 0) -> pd.DataFrame:
    pairs = read_gaf_pairs(path)
    gene = pairs["gene"].astype("category")
    term = pairs["go"].astype("category")
    M = sp.csr_matrix((np.ones(len(pairs)), (gene.cat.codes, term.cat.codes)),
                      shape=(len(gene.cat.categories), len(term.cat.categories)))
    n_comp = min(dim, M.shape[1] - 1)
    Z = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(M)
    return pd.DataFrame(Z, index=gene.cat.categories)


def load_string(links_path, info_path, dim: int = 64, min_score: int = 700) -> pd.DataFrame:
    info = pd.read_csv(info_path, sep="\t", usecols=[0, 1])
    id_to_name = dict(zip(info.iloc[:, 0], info.iloc[:, 1].str.upper(), strict=True))
    links = pd.read_csv(links_path, sep=" ")
    links = links[links["combined_score"] >= min_score]
    names = pd.Categorical(pd.concat([links["protein1"], links["protein2"]]).map(id_to_name))
    n_edges = len(links)
    codes = names.codes
    n = len(names.categories)
    A = sp.csr_matrix((np.ones(n_edges), (codes[:n_edges], codes[n_edges:])), shape=(n, n))
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    A_norm = inv_sqrt @ A @ inv_sqrt
    k = dim + 1
    # Largest eigenvectors of D^-1/2 A D^-1/2 = smallest of the normalized Laplacian.
    if n <= 4 * k:
        vals, vecs = np.linalg.eigh(A_norm.toarray())
    else:
        vals, vecs = eigsh(A_norm, k=k, which="LA")
    order = np.argsort(-vals)
    Z = vecs[:, order[1:k]]  # drop the trivial top eigenvector
    return pd.DataFrame(Z, index=names.categories)
