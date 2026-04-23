"""Common embedding extraction helpers used across scripts.

Heavy model dependencies are imported lazily to keep lightweight callers usable.
"""

from .alphagenome import (
    AG_DIM,
    fuse_embeddings_with_ag,
    infer_genome_id_from_fasta,
    load_ag_embeddings,
    load_all_ag_embeddings,
)
from .fasta import load_sequences, parse_fasta_dict, parse_fasta_rows

__all__ = [
    "AG_DIM",
    "ESM2Embedder",
    "Evo2Embedder",
    "ProtBERTEmbedder",
    "fuse_embeddings_with_ag",
    "infer_genome_id_from_fasta",
    "load_ag_embeddings",
    "load_all_ag_embeddings",
    "load_sequences",
    "parse_fasta_dict",
    "parse_fasta_rows",
]


def __getattr__(name):
    if name == "Evo2Embedder":
        from .evo2 import Evo2Embedder

        return Evo2Embedder
    if name == "ESM2Embedder":
        from .esm2 import ESM2Embedder

        return ESM2Embedder
    if name == "ProtBERTEmbedder":
        from .protbert import ProtBERTEmbedder

        return ProtBERTEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
