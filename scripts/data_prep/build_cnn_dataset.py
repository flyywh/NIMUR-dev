#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder, parse_fasta_rows
from utils.mibig_multilabel import build_multihot_labels, load_split_index_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train/val/test fused features from protein FASTA using ESM2 + ProtBERT."
    )
    p.add_argument("--train-pos-faa", required=True)
    p.add_argument("--train-neg-faa", required=True)
    p.add_argument("--val-pos-faa", required=True)
    p.add_argument("--val-neg-faa", required=True)
    p.add_argument("--test-pos-faa", required=True)
    p.add_argument("--test-neg-faa", required=True)
    p.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split-index", default="", help="Optional split_index.tsv for multi-label targets")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--esm-max-len", type=int, default=1022)
    p.add_argument("--chunk-size", type=int, default=30, help="Chunk size for saving")
    p.add_argument("--max-seq-len", type=int, default=2400, help="Max sequence length for ProtBERT")
    p.add_argument("--limit-per-file", type=int, default=-1)
    return p.parse_args()


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_split_rows(pos_path: Path, neg_path: Path, limit_per_file: int) -> tuple[list[str], np.ndarray, dict[str, str]]:
    seq_map: dict[str, str] = {}
    ids: list[str] = []
    labels: list[int] = []

    for fasta_path, label in ((pos_path, 1), (neg_path, 0)):
        rows = parse_fasta_rows(fasta_path)
        if limit_per_file > 0:
            rows = rows[:limit_per_file]
        for seq_id, seq in rows:
            if seq_id in seq_map:
                raise ValueError(f"Duplicate sequence id across split inputs: {seq_id}")
            seq_map[seq_id] = seq
            ids.append(seq_id)
            labels.append(label)

    return ids, np.asarray(labels, dtype=np.int64), seq_map


def cache_path(cache_dir: Path, split_name: str, model_name: str) -> Path:
    return cache_dir / f"{split_name}_{model_name}.pt"


def load_cached_features(path: Path, expected_ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    ids = obj.get("ids")
    feats = obj.get("features")
    if ids != expected_ids or feats is None:
        return None
    if torch.is_tensor(feats):
        return feats.cpu().numpy()
    return np.asarray(feats, dtype=np.float32)


def save_cached_features(path: Path, ids: list[str], features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "features": torch.tensor(features, dtype=torch.float32)}, path)


def encode_split_esm(
    embedder: ESM2Embedder,
    split_name: str,
    ids: list[str],
    seq_map: dict[str, str],
    chunk_len: int,
) -> np.ndarray:
    feats = []
    for seq_id in tqdm(ids, desc=f"ESM2 {split_name}", leave=False):
        vec = embedder.encode_mean_chunked(seq_id, seq_map[seq_id], chunk_len=chunk_len).astype(np.float32)
        feats.append(vec)
    return np.stack(feats, axis=0)


def encode_split_protbert(
    embedder: ProtBERTEmbedder,
    split_name: str,
    ids: list[str],
    seq_map: dict[str, str],
    max_len: int,
) -> np.ndarray:
    feats = []
    for seq_id in tqdm(ids, desc=f"ProtBERT {split_name}", leave=False):
        vec = embedder.encode_mean(seq_id, seq_map[seq_id], max_len=max_len).astype(np.float32)
        feats.append(vec)
    return np.stack(feats, axis=0)


def maybe_empty_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.out_dir).with_suffix("").parent / "cache_esm_protbert"
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": (Path(args.train_pos_faa), Path(args.train_neg_faa)),
        "val": (Path(args.val_pos_faa), Path(args.val_neg_faa)),
        "test": (Path(args.test_pos_faa), Path(args.test_neg_faa)),
    }

    split_data: dict[str, dict[str, object]] = {}
    for split_name, (pos_path, neg_path) in split_files.items():
        ids, labels, seq_map = load_split_rows(pos_path, neg_path, args.limit_per_file)
        split_data[split_name] = {"ids": ids, "labels": labels, "seq_map": seq_map}

    classes: list[str] | None = None
    if args.split_index:
        labels_by_id, classes = load_split_index_labels(args.split_index)
        for split_name, data in split_data.items():
            ids = data["ids"]  # type: ignore[index]
            data["labels"] = build_multihot_labels(ids, labels_by_id, classes)

    esm_features: dict[str, np.ndarray] = {}
    esm = ESM2Embedder(device=device, layer=33)
    for split_name, data in split_data.items():
        ids = data["ids"]  # type: ignore[index]
        seq_map = data["seq_map"]  # type: ignore[index]
        cp = cache_path(cache_dir, split_name, "esm2")
        cached = load_cached_features(cp, ids)
        if cached is not None:
            esm_features[split_name] = cached
            print(f"[CACHE] reuse {cp}")
            continue
        feats = encode_split_esm(esm, split_name, ids, seq_map, args.esm_max_len)
        esm_features[split_name] = feats
        save_cached_features(cp, ids, feats)
    del esm
    gc.collect()
    maybe_empty_cuda_cache(device)

    prot_features: dict[str, np.ndarray] = {}
    prot = ProtBERTEmbedder(args.protbert, device=device)
    for split_name, data in split_data.items():
        ids = data["ids"]  # type: ignore[index]
        seq_map = data["seq_map"]  # type: ignore[index]
        cp = cache_path(cache_dir, split_name, "protbert")
        cached = load_cached_features(cp, ids)
        if cached is not None:
            prot_features[split_name] = cached
            print(f"[CACHE] reuse {cp}")
            continue
        feats = encode_split_protbert(prot, split_name, ids, seq_map, args.max_seq_len)
        prot_features[split_name] = feats
        save_cached_features(cp, ids, feats)
    del prot
    gc.collect()
    maybe_empty_cuda_cache(device)

    meta = {
        "device": device,
        "esm_dim": None,
        "protbert_dim": None,
        "feature_dim": None,
        "classes": classes,
        "num_classes": (len(classes) if classes is not None else 1),
        "cache_dir": str(cache_dir),
        "chunk_size": args.chunk_size,
        "source_files": {k: [str(v[0]), str(v[1])] for k, v in split_files.items()},
        "split_index": args.split_index,
    }

    # Save each split in chunks
    for split_name, data in split_data.items():
        ids = data["ids"]
        labels = data["labels"]
        esm_arr = esm_features[split_name]
        prot_arr = prot_features[split_name]
        fused = np.concatenate([esm_arr, prot_arr], axis=1).astype(np.float32)

        # Create split directory
        split_dir = Path(args.out_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save in chunks
        for chunk_start in range(0, len(ids), args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, len(ids))
            chunk_ids = ids[chunk_start:chunk_end]
            chunk_features = fused[chunk_start:chunk_end]
            chunk_labels = labels[chunk_start:chunk_end]

            chunk_path = split_dir / f"chunk_{chunk_start:06d}.pt"
            torch.save({
                "ids": chunk_ids,
                "X": torch.tensor(chunk_features, dtype=torch.float32),
                "y": torch.tensor(chunk_labels, dtype=torch.float32),
            }, chunk_path)

        meta[f"{split_name}_size"] = int(len(ids))
        if meta["esm_dim"] is None:
            meta["esm_dim"] = int(esm_arr.shape[1])
            meta["protbert_dim"] = int(prot_arr.shape[1])
            meta["feature_dim"] = int(fused.shape[1])

    # Save metadata
    meta_path = Path(args.out_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[DONE] saved to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
