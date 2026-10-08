"""Embedding registry: name → Embedding instance."""
from __future__ import annotations

from pbench.download import RESOURCE_PATHS
from pbench.embeddings.base import Embedding, StaticEmbedding
from pbench.embeddings.simple import PCAEmbedding, RandomEmbedding

FM_EMBEDDINGS = frozenset({"scgpt", "geneformer"})


def build_embeddings(names: list[str], raw_dir) -> dict[str, Embedding]:
    from pbench.embeddings import foundation, knowledge

    paths = RESOURCE_PATHS(raw_dir)
    factories = {
        "random": lambda: RandomEmbedding(),
        "pca": lambda: PCAEmbedding(),
        "go": lambda: StaticEmbedding("go", lambda: knowledge.load_go(paths["go_gaf"])),
        "string": lambda: StaticEmbedding(
            "string", lambda: knowledge.load_string(paths["string_links"], paths["string_info"])),
        "genept": lambda: StaticEmbedding(
            "genept", lambda: foundation.load_genept(paths["genept"])),
        "scgpt": lambda: StaticEmbedding("scgpt", lambda: foundation.load_scgpt(paths["scgpt"])),
        "geneformer": lambda: StaticEmbedding(
            "geneformer", lambda: foundation.load_geneformer(paths["geneformer"])),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(f"unknown embeddings: {sorted(unknown)}")
    return {n: factories[n]() for n in names}
