import gzip
import pickle

import numpy as np
import pandas as pd

from pbench.embeddings.foundation import geneformer_table, load_genept, scgpt_table
from pbench.embeddings.knowledge import go_annotation_counts, load_go, load_string, read_gaf_pairs

GAF = """!gaf-version: 2.2
UniProtKB\tP1\tactb\t\tGO:1\tPMID\tIDA\t\tC\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP1\tACTB\t\tGO:2\tPMID\tIDA\t\tC\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP1\tACTB\t\tGO:2\tPMID\tIEA\t\tC\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP2\tGAPDH\tNOT\tGO:1\tPMID\tIDA\t\tF\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP2\tGAPDH\t\tGO:3\tPMID\tIDA\t\tF\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP3\tTP53\t\tGO:3\tPMID\tIDA\t\tF\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
UniProtKB\tP3\tTP53\t\tGO:4\tPMID\tIDA\t\tF\t\t\tprotein\ttaxon:9606\t20200101\tUniProt
"""


def _gaf(tmp_path):
    p = tmp_path / "goa.gaf.gz"
    with gzip.open(p, "wt") as f:
        f.write(GAF)
    return p


def test_gaf_pairs_drop_not_and_dupes(tmp_path):
    pairs = read_gaf_pairs(_gaf(tmp_path))
    assert set(map(tuple, pairs[["gene", "go"]].to_numpy())) == {
        ("ACTB", "GO:1"), ("ACTB", "GO:2"), ("GAPDH", "GO:3"), ("TP53", "GO:3"), ("TP53", "GO:4")}


def test_go_embedding_and_counts(tmp_path):
    path = _gaf(tmp_path)
    emb = load_go(path, dim=2)
    assert set(emb.index) == {"ACTB", "GAPDH", "TP53"}
    assert emb.shape[1] == 2
    assert go_annotation_counts(path).to_dict() == {"ACTB": 2, "GAPDH": 1, "TP53": 2}


def test_string_separates_cliques(tmp_path):
    names = [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)]
    info = pd.DataFrame({"#string_protein_id": [f"9606.ENSP{i}" for i in range(10)],
                         "preferred_name": names, "protein_size": 100, "annotation": "x"})
    edges = []
    for block in (range(5), range(5, 10)):
        edges += [(i, j) for i in block for j in block if i != j]
    edges += [(4, 5), (5, 4)]
    links = pd.DataFrame({"protein1": [f"9606.ENSP{i}" for i, _ in edges],
                          "protein2": [f"9606.ENSP{j}" for _, j in edges],
                          "combined_score": 900})
    links.loc[len(links)] = ["9606.ENSP0", "9606.ENSP9", 100]  # below threshold
    ip, lp = tmp_path / "info.txt.gz", tmp_path / "links.txt.gz"
    info.to_csv(ip, sep="\t", index=False, compression="gzip")
    links.to_csv(lp, sep=" ", index=False, compression="gzip")
    emb = load_string(lp, ip, dim=1)
    v = emb.loc[names].to_numpy()[:, 0]
    assert np.sign(v[:4]).tolist().count(np.sign(v[0])) == 4
    assert np.sign(v[0]) != np.sign(v[9])


def test_genept_loader(tmp_path):
    d = tmp_path / "GenePT" / "inner"
    d.mkdir(parents=True)
    with open(d / "GenePT_gene_embedding_ada_text.pickle", "wb") as f:
        pickle.dump({"ACTB": [1.0, 2.0], "gapdh": [3.0, 4.0]}, f)
    emb = load_genept(tmp_path)
    assert emb.loc["ACTB"].tolist() == [1.0, 2.0]
    assert emb.shape == (2, 2)


def test_scgpt_table_skips_special_tokens():
    W = np.arange(8, dtype=np.float32).reshape(4, 2)
    t = scgpt_table(W, {"<pad>": 0, "ACTB": 1, "<cls>": 2, "TP53": 3})
    assert t.index.tolist() == ["ACTB", "TP53"]
    np.testing.assert_allclose(t.loc["TP53"], [6, 7])


def test_geneformer_table_maps_symbols():
    W = np.arange(6, dtype=np.float32).reshape(3, 2)
    t = geneformer_table(W, {"<pad>": 0, "ENSG1": 1, "ENSG2": 2},
                         {"actb": "ENSG1", "TP53": "ENSG2", "MISSING": "ENSG9"})
    assert sorted(t.index) == ["ACTB", "TP53"]
    np.testing.assert_allclose(t.loc["ACTB"], [2, 3])
