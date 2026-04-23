import glob
from pathlib import Path
from typing import Dict

import numpy as np

AG_DIM = 1536


def load_ag_embeddings(ag_emb_dir: str | Path, genome_id: str) -> Dict[str, np.ndarray]:
    npz_path = Path(ag_emb_dir) / f"{genome_id}.npz"
    if not npz_path.exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    return dict(zip(data["seq_ids"].tolist(), data["embeddings"]))


def load_all_ag_embeddings(ag_emb_dir: str | Path) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for npz_path in glob.glob(str(Path(ag_emb_dir) / "*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        for sid, emb in zip(data["seq_ids"].tolist(), data["embeddings"]):
            merged[sid] = emb
    return merged


def infer_ag_level(embedding: np.ndarray) -> str:
    arr = np.asarray(embedding)
    if arr.ndim <= 1:
        return "pooled"
    return "token"


def infer_ag_dim(embedding: np.ndarray) -> int:
    arr = np.asarray(embedding)
    if arr.ndim == 0:
        raise ValueError("AlphaGenome embedding must have at least 1 dimension")
    return int(arr.shape[-1])


def summarize_ag_embeddings(ag_dict: Dict[str, np.ndarray]) -> dict[str, object]:
    level_counts: dict[str, int] = {}
    dim_counts: dict[str, int] = {}
    for emb in ag_dict.values():
        level = infer_ag_level(emb)
        dim = infer_ag_dim(emb)
        level_counts[level] = level_counts.get(level, 0) + 1
        dim_counts[str(dim)] = dim_counts.get(str(dim), 0) + 1
    return {
        "count": len(ag_dict),
        "level_counts": level_counts,
        "dim_counts": dim_counts,
    }


def infer_genome_id_from_fasta(fasta_path: str | Path, keep_segments: int = 2) -> str:
    stem = Path(fasta_path).stem
    parts = stem.split("_")
    if len(parts) < keep_segments:
        return stem
    return "_".join(parts[:keep_segments])


def fuse_embeddings_with_ag(
    seq_embeddings: Dict[str, np.ndarray],
    ag_dict: Dict[str, np.ndarray],
    ag_dim: int = AG_DIM,
) -> Dict[str, np.ndarray]:
    fused: Dict[str, np.ndarray] = {}
    for seq_id, seq_emb in seq_embeddings.items():
        ag_emb = ag_dict.get(seq_id, np.zeros(ag_dim, dtype=np.float32))
        seq_len = seq_emb.shape[0]
        ag_broadcast = np.tile(ag_emb, (seq_len, 1))
        fused[seq_id] = np.concatenate([seq_emb, ag_broadcast], axis=1)
    return fused
