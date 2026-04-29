#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import gc
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ORACLE-KG train/val/test directories from MiBiG split FASTAs and split_index.tsv"
    )
    parser.add_argument("--split-index", required=True)
    parser.add_argument("--train-pos-fasta", required=True)
    parser.add_argument("--train-neg-fasta", required=True)
    parser.add_argument("--val-pos-fasta", required=True)
    parser.add_argument("--val-neg-fasta", required=True)
    parser.add_argument("--test-pos-fasta", required=True)
    parser.add_argument("--test-neg-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fused-cache-dir", default="", help="Optional output dir for per-sequence fused ESM2+ProtBERT cache")
    parser.add_argument("--split-cache-dir", default="", help="Optional split cache dir from build_cnn_dataset/build_esm_protbert_dataset")
    parser.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--esm-max-len", type=int, default=1022)
    parser.add_argument("--protbert-max-len", type=int, default=2400)
    return parser.parse_args()


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def header_to_stem(header: str, max_base_len: int = 120) -> str:
    base = re.sub(r"[^A-Za-z0-9._=-]+", "_", header).strip("._")
    if not base:
        base = "seq"
    if len(base) > max_base_len:
        base = base[:max_base_len]
    digest = hashlib.sha1(header.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{digest}"


def cache_file_for_id(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{header_to_stem(sample_id)}.pt"


def load_sequences(paths: list[Path]) -> dict[str, str]:
    seq_by_id: dict[str, str] = {}
    for path in paths:
        for record in SeqIO.parse(str(path), "fasta"):
            seq_id = record.id.strip()
            sequence = str(record.seq).upper().strip()
            if not seq_id or not sequence:
                continue
            seq_by_id[seq_id] = sequence
    if not seq_by_id:
        raise ValueError("No sequences loaded from provided FASTA files")
    return seq_by_id


def read_index(path: Path):
    records_by_split: dict[str, list[tuple[str, list[str], str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    type_names: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"seq_id", "split", "binary_label", "classes"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            seq_id = (row.get("seq_id") or "").strip()
            split = (row.get("split") or "").strip()
            classes = (row.get("classes") or "").strip()
            if not seq_id or split not in records_by_split or not classes:
                continue
            labels = [label.strip() for label in classes.split(",") if label.strip()]
            if not labels:
                continue
            type_names.update(labels)
            records_by_split[split].append((seq_id, labels, ""))
    if not any(records_by_split.values()):
        raise ValueError(f"No valid records found in {path}")
    return records_by_split, {name: idx for idx, name in enumerate(sorted(type_names))}


def attach_sequences(records_by_split, seq_by_id: dict[str, str]):
    enriched: dict[str, list[tuple[str, list[str], str]]] = {}
    for split, records in records_by_split.items():
        rows = []
        missing = []
        for seq_id, labels, _ in records:
            sequence = seq_by_id.get(seq_id)
            if sequence is None:
                missing.append(seq_id)
                continue
            rows.append((seq_id, labels, sequence))
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing sequences for split={split}: {preview} (total missing={len(missing)})")
        enriched[split] = rows
    return enriched


def save_split(records: list[tuple[str, list[str], str]], type2id: dict[str, int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    relation2id = {"no": 0, "yes": 1}
    protein2id = {query: idx for idx, (query, _labels, _seq) in enumerate(records)}

    with (out_dir / "protein2id.txt").open("w", encoding="utf-8") as f:
        for key, value in protein2id.items():
            f.write(f"{key}\t{value}\n")

    with (out_dir / "type2id.txt").open("w", encoding="utf-8") as f:
        for key, value in type2id.items():
            f.write(f"{key}\t{value}\n")

    with (out_dir / "relation2id.txt").open("w", encoding="utf-8") as f:
        for key, value in relation2id.items():
            f.write(f"{key}\t{value}\n")

    with (out_dir / "sequence.txt").open("w", encoding="utf-8") as f:
        for _query, _labels, sequence in records:
            f.write(sequence + "\n")

    with (out_dir / "protein_relation_type.txt").open("w", encoding="utf-8") as f:
        for query, labels, _seq in records:
            pid = protein2id[query]
            label_set = set(labels)
            for type_name, tid in type2id.items():
                rid = relation2id["yes"] if type_name in label_set else relation2id["no"]
                f.write(f"{pid}\t{rid}\t{tid}\n")


def write_kg_tsv(all_records: list[tuple[str, str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write("query\ttype\tgos\tsequence\n")
        for seq_id, classes, sequence in all_records:
            f.write(f"{seq_id}\t{classes}\t\t{sequence}\n")


def load_split_cached_map(cache_dir: Path, split_name: str, model_name: str) -> dict[str, np.ndarray] | None:
    import torch

    path = cache_dir / f"{split_name}_{model_name}.pt"
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    ids = obj.get("ids")
    features = obj.get("features")
    if ids is None or features is None:
        return None
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    return {seq_id: features[idx] for idx, seq_id in enumerate(ids)}


def write_fused_cache(
    *,
    records_by_split: dict[str, list[tuple[str, list[str], str]]],
    fused_cache_dir: Path,
    split_cache_dir: Path | None,
    protbert_path: str,
    device: str,
    esm_max_len: int,
    protbert_max_len: int,
) -> dict[str, object]:
    import torch

    fused_cache_dir.mkdir(parents=True, exist_ok=True)
    per_split_maps: dict[str, dict[str, np.ndarray]] = {}
    if split_cache_dir is not None:
        for split_name in ("train", "val", "test"):
            esm_map = load_split_cached_map(split_cache_dir, split_name, "esm2")
            prot_map = load_split_cached_map(split_cache_dir, split_name, "protbert")
            if esm_map is None or prot_map is None:
                per_split_maps = {}
                break
            merged: dict[str, np.ndarray] = {}
            for seq_id, esm_vec in esm_map.items():
                prot_vec = prot_map.get(seq_id)
                if prot_vec is None:
                    continue
                merged[seq_id] = np.concatenate([esm_vec, prot_vec], axis=0).astype(np.float32)
            per_split_maps[split_name] = merged

    esm_embedder = None
    prot_embedder = None
    total = 0
    for split_name in ("train", "val", "test"):
        cached_map = per_split_maps.get(split_name)
        for seq_id, _labels, sequence in records_by_split[split_name]:
            out_path = cache_file_for_id(fused_cache_dir, seq_id)
            if out_path.exists():
                total += 1
                continue
            fused = cached_map.get(seq_id) if cached_map else None
            if fused is None:
                if esm_embedder is None:
                    esm_embedder = ESM2Embedder(device=device, layer=33)
                if prot_embedder is None:
                    prot_embedder = ProtBERTEmbedder(protbert_path, device=device)
                esm_vec = esm_embedder.encode_mean_chunked(seq_id, sequence, chunk_len=esm_max_len).astype(np.float32)
                prot_vec = prot_embedder.encode_mean(seq_id, sequence, max_len=protbert_max_len).astype(np.float32)
                fused = np.concatenate([esm_vec, prot_vec], axis=0).astype(np.float32)
            torch.save(torch.tensor(fused, dtype=torch.float32), out_path)
            total += 1

    meta = {
        "name": "esm2_protbert_fused",
        "dir": str(fused_cache_dir),
        "level": "pooled",
        "dim": 2304,
        "device": device,
        "esm_dim": 1280,
        "protbert_dim": 1024,
        "protbert_path": protbert_path,
        "esm_max_len": esm_max_len,
        "protbert_max_len": protbert_max_len,
        "count": total,
    }
    (fused_cache_dir / "_cache_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    del esm_embedder
    del prot_embedder
    gc.collect()
    return meta


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    seq_by_id = load_sequences(
        [
            Path(args.train_pos_fasta),
            Path(args.train_neg_fasta),
            Path(args.val_pos_fasta),
            Path(args.val_neg_fasta),
            Path(args.test_pos_fasta),
            Path(args.test_neg_fasta),
        ]
    )
    records_by_split, type2id = read_index(Path(args.split_index))
    records_by_split = attach_sequences(records_by_split, seq_by_id)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_records: list[tuple[str, str, str]] = []
    for split in ("train", "val", "test"):
        split_records = records_by_split[split]
        save_split(split_records, type2id, out_root / split)
        for seq_id, labels, sequence in split_records:
            all_records.append((seq_id, "-".join(labels), sequence))

    write_kg_tsv(all_records, out_root / "mibig_kg.tsv")
    summary = {
        "split_index": str(Path(args.split_index)),
        "num_types": len(type2id),
        "types": sorted(type2id.keys()),
        "split_sizes": {split: len(records_by_split[split]) for split in ("train", "val", "test")},
        "output_dir": str(out_root),
    }

    if args.fused_cache_dir:
        cache_meta = write_fused_cache(
            records_by_split=records_by_split,
            fused_cache_dir=Path(args.fused_cache_dir),
            split_cache_dir=(Path(args.split_cache_dir) if args.split_cache_dir else None),
            protbert_path=args.protbert,
            device=device,
            esm_max_len=int(args.esm_max_len),
            protbert_max_len=int(args.protbert_max_len),
        )
        summary["fused_cache"] = cache_meta

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
