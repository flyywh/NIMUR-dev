#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifest import build_manifest, run_cmd, write_manifest


class ProteinDataset:
    def __init__(self, data_dict: dict[str, tuple[Any, Any]]):
        self.items = list(data_dict.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _name, value = self.items[idx]
        return value

    def __add__(self, other: "ProteinDataset") -> "ProteinDataset":
        return ProteinDataset(dict(self.items + other.items))


class DnaDataset:
    def __init__(self, data_dict: dict[str, tuple[Any, Any]]):
        self.items = list(data_dict.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _name, value = self.items[idx]
        return value


class ProteinDnaDataset:
    def __init__(self, data_dict: dict[str, tuple[Any, Any]]):
        self.items = list(data_dict.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _name, value = self.items[idx]
        return value


class ClassifierDataset:
    def __init__(self, data_dict: dict[str, tuple[Any, Any]]):
        self.sequences = list(data_dict.values())
        self.indices = list(data_dict.keys())

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]


for _cls in (ProteinDataset, DnaDataset, ProteinDnaDataset, ClassifierDataset):
    _cls.__module__ = "__main__"
    setattr(sys.modules["__main__"], _cls.__name__, _cls)


RESOURCE_MAP: dict[str, tuple[str, str]] = {
    "fasta": ("data/fasta", "fasta/*"),
    "ProtBERT": ("data/ProtBERT", "ProtBERT/*"),
    "ng": ("data/ng", "ng/*"),
    "NIMROD-EE": ("default_models", "NIMROD-EE"),
    "NIMROD-ESM": ("default_models", "NIMROD-ESM"),
    "NIMROD-EVO": ("default_models", "NIMROD-EVO"),
    "ORACLE": ("default_models", "ORACLE"),
    "ORACLE-CNN": ("default_models", "ORACLE-CNN"),
    "classes.pt": ("data", "classes.pt"),
    "label_dict.pt": ("data", "label_dict.pt"),
}

MIBIG_NAME_MAP = {
    "nrps": "NRPS",
    "other": "Other",
    "pks": "PKS",
    "ribosomal": "Ribosomal",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
}

BASE_CLASS_ORDER = ["NRPS", "Other", "PKS", "Ribosomal", "Saccharide", "Terpene"]


def _paths(values: List[str]) -> List[Path]:
    return [Path(v) for v in values if v]


def _maybe(value: str) -> List[str]:
    return [value] if value else []


def _public_params(params: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in params.items() if not k.startswith("_")}


def _legacy_sh_cmd(script_name: str, args: List[str]) -> List[str]:
    return ["bash", script_name, *args]


def _legacy_py_cmd(script_name: str, args: List[str]) -> List[str]:
    return ["python", script_name, *args]


def _set_cuda_visible_devices(gpus: str) -> None:
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"Using GPUs: {gpus}")


def _require_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def _require_dir(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Missing directory: {p}")
    return p


def _parse_fasta_rows(path: str | Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    cur_id = ""
    cur_seq: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    rows.append((cur_id, "".join(cur_seq).upper()))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id:
        rows.append((cur_id, "".join(cur_seq).upper()))
    return [(seq_id, seq) for seq_id, seq in rows if seq]


def _parse_fasta_dict(path: str | Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for seq_id, seq in _parse_fasta_rows(path):
        data.setdefault(seq_id, seq)
    return data


def _load_sequences(paths: Iterable[str | Path]) -> Dict[str, str]:
    all_seq: Dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        all_seq.update(_parse_fasta_dict(p))
    return all_seq


def _extract_mibig_id_and_labels(desc: str) -> tuple[str | None, set[str]]:
    m_id = re.search(r"BGC_ID=(BGC\d+)", desc)
    m_cls = re.search(r"bgc_class=([^|]+)", desc)
    if not m_id or not m_cls:
        return None, set()

    bgc_id = m_id.group(1)
    labels: set[str] = set()
    for part in m_cls.group(1).split(";"):
        match = re.search(r"class:([^,;]+)", part)
        if not match:
            continue
        cls = match.group(1).strip().lower()
        mapped = MIBIG_NAME_MAP.get(cls)
        if mapped:
            labels.add(mapped)
    return bgc_id, labels


def _load_mibig_real_samples(in_fasta: Path) -> tuple[list[str], dict[str, str], dict[str, set[str]]]:
    seq_by_id: dict[str, str] = {}
    labels_by_id: dict[str, set[str]] = {}
    cur_desc = ""
    cur_seq: list[str] = []

    def _flush() -> None:
        nonlocal cur_desc, cur_seq
        if not cur_desc:
            return
        bgc_id, labels = _extract_mibig_id_and_labels(cur_desc)
        if bgc_id and labels:
            seq = "".join(cur_seq).upper()
            if seq and bgc_id not in seq_by_id:
                seq_by_id[bgc_id] = seq
            if bgc_id:
                labels_by_id.setdefault(bgc_id, set()).update(labels)
        cur_desc = ""
        cur_seq = []

    with in_fasta.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                cur_desc = line[1:]
            else:
                cur_seq.append(line)
    _flush()

    ids = [sample_id for sample_id in seq_by_id if sample_id in labels_by_id and labels_by_id[sample_id]]
    if not ids:
        raise ValueError(f"No usable MiBiG records parsed from {in_fasta}")
    return ids, seq_by_id, labels_by_id


def _make_mibig_chimera(
    seq_a: str,
    seq_b: str,
    frac_min: float,
    frac_max: float,
    rng: random.Random,
) -> str:
    max_len = min(len(seq_a), len(seq_b))
    seg_len = max(1, int(max_len * rng.uniform(frac_min, frac_max)))
    seg_len = min(seg_len, len(seq_a), len(seq_b))
    start_a = rng.randint(0, len(seq_a) - seg_len)
    start_b = rng.randint(0, len(seq_b) - seg_len)
    seg_b = seq_b[start_b : start_b + seg_len]
    return seq_a[:start_a] + seg_b + seq_a[start_a + seg_len :]


def _add_mibig_negative_samples(
    ids: list[str],
    seq_by_id: dict[str, str],
    labels_by_id: dict[str, set[str]],
    *,
    neg_ratio: float,
    min_seq_len: int,
    swap_frac_min: float,
    swap_frac_max: float,
    neg_prefix: str,
    rng: random.Random,
) -> list[str]:
    candidates = [sample_id for sample_id in ids if len(seq_by_id[sample_id]) >= min_seq_len]
    if len(candidates) < 2:
        return []

    n_neg = int(len(ids) * neg_ratio)
    new_ids: list[str] = []
    for idx in range(1, n_neg + 1):
        for _ in range(20):
            a, b = rng.sample(candidates, 2)
            if labels_by_id[a].isdisjoint(labels_by_id[b]):
                break
        else:
            a, b = rng.sample(candidates, 2)

        neg_id = f"{neg_prefix}{idx:09d}"[:12]
        if neg_id in seq_by_id:
            continue

        seq_by_id[neg_id] = _make_mibig_chimera(
            seq_by_id[a],
            seq_by_id[b],
            swap_frac_min,
            swap_frac_max,
            rng,
        )
        labels_by_id[neg_id] = {"neg"}
        new_ids.append(neg_id)
    return new_ids


def _split_train_valid_ids(ids: list[str], train_ratio: float, rng: random.Random) -> tuple[list[str], list[str]]:
    ordered = list(ids)
    rng.shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, []
    split_idx = int(len(ordered) * train_ratio)
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered[:split_idx], ordered[split_idx:]


def _write_selected_fasta(path: Path, ordered_ids: list[str], seq_by_id: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample_id in ordered_ids:
            f.write(f">{sample_id}\n{seq_by_id[sample_id]}\n")


def _write_oracle_label_file(
    path: Path,
    ordered_ids: list[str],
    labels_by_id: dict[str, set[str]],
    class_order: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample_id in ordered_ids:
            f.write(sample_id + "\n")
            for cls in class_order:
                if cls in labels_by_id[sample_id]:
                    f.write(cls + "\n")
            f.write("\n")


def _write_mibig_kg_tsv(
    path: Path,
    ordered_ids: list[str],
    labels_by_id: dict[str, set[str]],
    seq_by_id: dict[str, str],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("query\ttype\tgos\tsequence\n")
        for sample_id in ordered_ids:
            labels = sorted(label for label in labels_by_id[sample_id] if label != "neg")
            if not labels:
                continue
            f.write(f"{sample_id}\t{'-'.join(labels)}\t\t{seq_by_id[sample_id]}\n")


def _prepare_cnn_inputs_from_samples(
    *,
    ids: list[str],
    seq_by_id: dict[str, str],
    labels_by_id: dict[str, set[str]],
    output_dir: Path,
    train_ratio: float,
    seed: int,
    add_neg: bool,
    neg_ratio: float,
    min_seq_len: int,
    swap_frac_min: float,
    swap_frac_max: float,
    neg_prefix: str,
) -> dict[str, object]:
    seq_map = {k: v for k, v in seq_by_id.items()}
    label_map = {k: set(v) for k, v in labels_by_id.items()}
    rng = random.Random(seed)

    neg_ids: list[str] = []
    if add_neg:
        neg_ids = _add_mibig_negative_samples(
            ids,
            seq_map,
            label_map,
            neg_ratio=neg_ratio,
            min_seq_len=min_seq_len,
            swap_frac_min=swap_frac_min,
            swap_frac_max=swap_frac_max,
            neg_prefix=neg_prefix,
            rng=rng,
        )

    all_ids = ids + neg_ids
    train_ids, valid_ids = _split_train_valid_ids(all_ids, train_ratio, rng)
    class_order = BASE_CLASS_ORDER + (["neg"] if add_neg else [])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_fasta = output_dir / "train.fasta"
    valid_fasta = output_dir / "valid.fasta"
    label_file = output_dir / "label.txt"

    _write_selected_fasta(train_fasta, train_ids, seq_map)
    _write_selected_fasta(valid_fasta, valid_ids, seq_map)
    _write_oracle_label_file(label_file, train_ids + valid_ids, label_map, class_order)

    summary = {
        "real_samples": len(ids),
        "neg_samples": len(neg_ids),
        "total_samples": len(all_ids),
        "train_samples": len(train_ids),
        "valid_samples": len(valid_ids),
        "train_fasta": str(train_fasta),
        "valid_fasta": str(valid_fasta),
        "label_file": str(label_file),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _prepare_mibig_cnn_inputs(
    *,
    input_fasta: Path,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    add_neg: bool,
    neg_ratio: float,
    min_seq_len: int,
    swap_frac_min: float,
    swap_frac_max: float,
    neg_prefix: str,
) -> dict[str, object]:
    ids, seq_by_id, labels_by_id = _load_mibig_real_samples(input_fasta)
    return _prepare_cnn_inputs_from_samples(
        ids=ids,
        seq_by_id=seq_by_id,
        labels_by_id=labels_by_id,
        output_dir=output_dir,
        train_ratio=train_ratio,
        seed=seed,
        add_neg=add_neg,
        neg_ratio=neg_ratio,
        min_seq_len=min_seq_len,
        swap_frac_min=swap_frac_min,
        swap_frac_max=swap_frac_max,
        neg_prefix=neg_prefix,
    )


def _write_stage1_label_tsv(path: Path, ordered_ids: list[str], labels_by_id: dict[str, set[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("sample_id\tlabels\n")
        for sample_id in ordered_ids:
            f.write(f"{sample_id}\t{','.join(sorted(labels_by_id[sample_id]))}\n")


def _load_stage1_labels(path: Path) -> dict[str, set[str]]:
    labels_by_id: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            labels = (row.get("labels") or "").strip()
            if not sample_id:
                continue
            labels_by_id[sample_id] = {label for label in labels.split(",") if label}
    return labels_by_id


def _parse_csv_option(raw: str | None, *, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if raw.strip().lower() in {"", "none"}:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _cache_file_for_id(cache_dir: Path, sample_id: str) -> Path:
    return cache_dir / f"{header_to_stem(sample_id)}.pt"


def _cache_index_path(cache_dir: Path) -> Path:
    return cache_dir / "_cache_meta.json"


def _write_cache_index(cache_dir: Path, meta: dict[str, object]) -> None:
    _cache_index_path(cache_dir).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_cache_index(cache_dir: Path) -> dict[str, object]:
    path = _cache_index_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_cache_entry(
    *,
    name: str,
    cache_dir: Path,
    level: str,
    sequence_field: str,
    dim: int,
    dtype: str = "float32",
    source: str = "local",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": name,
        "dir": str(cache_dir),
        "level": level,
        "sequence_field": sequence_field,
        "dim": int(dim),
        "dtype": dtype,
        "source": source,
    }
    if extra:
        entry.update(extra)
    _write_cache_index(cache_dir, entry)
    return entry


def _load_stage1_samples(stage1_dir: Path) -> tuple[list[str], dict[str, str], dict[str, set[str]], dict[str, object]]:
    sample_fasta = stage1_dir / "samples.fasta"
    label_tsv = stage1_dir / "labels.tsv"
    manifest_path = stage1_dir / "stage1_manifest.json"
    _require_file(sample_fasta)
    _require_file(label_tsv)
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seq_by_id = _parse_fasta_dict(sample_fasta)
    labels_by_id = _load_stage1_labels(label_tsv)
    ordered_ids = list(seq_by_id.keys())
    return ordered_ids, seq_by_id, labels_by_id, manifest


def _extract_stage1_caches(
    *,
    output_dir: Path,
    ordered_ids: list[str],
    seq_by_id: dict[str, str],
    extract_targets: list[str],
    esm_max_seq_len: int,
    protbert_max_len: int,
    protbert_path: str,
    evo2_max_len: int,
    evo2_layer_name: str,
    alphagenome_dir: str,
) -> dict[str, dict[str, object]]:
    torch = _load_torch()

    cache_root = output_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_entries: dict[str, dict[str, object]] = {}

    targets = set(extract_targets)
    if "esm2" in targets:
        from utils.embed_extract import ESM2Embedder

        esm_cache = cache_root / "esm2"
        esm_cache.mkdir(parents=True, exist_ok=True)
        embedder = ESM2Embedder(device="cuda" if torch.cuda.is_available() else "cpu", layer=33)
        sample_dim = 0
        for sample_id in ordered_ids:
            cache_file = _cache_file_for_id(esm_cache, sample_id)
            if cache_file.exists():
                if sample_dim == 0:
                    sample_dim = int(torch.load(cache_file, map_location="cpu", weights_only=False).shape[-1])
                continue
            seq_use = seq_by_id[sample_id][:esm_max_seq_len] if esm_max_seq_len > 0 else seq_by_id[sample_id]
            emb = embedder.encode_matrix_with_oom_fallback(sample_id, seq_use)
            tensor = torch.tensor(emb, dtype=torch.float32)
            sample_dim = int(tensor.shape[-1])
            torch.save(tensor, cache_file)
        cache_entries["esm2"] = _make_cache_entry(
            name="esm2",
            cache_dir=esm_cache,
            level="token",
            sequence_field="dna_seq",
            dim=sample_dim,
            extra={"max_seq_len": int(esm_max_seq_len), "count": len(ordered_ids)},
        )

    if "protbert" in targets:
        from utils.embed_extract import ProtBERTEmbedder

        protbert_cache = cache_root / "protbert"
        protbert_cache.mkdir(parents=True, exist_ok=True)
        embedder = ProtBERTEmbedder(protbert_path, device="cuda" if torch.cuda.is_available() else "cpu")
        sample_dim = 0
        for sample_id in ordered_ids:
            cache_file = _cache_file_for_id(protbert_cache, sample_id)
            if cache_file.exists():
                if sample_dim == 0:
                    sample_dim = int(torch.load(cache_file, map_location="cpu", weights_only=False).shape[-1])
                continue
            emb = embedder.encode_mean(sample_id, seq_by_id[sample_id], max_len=protbert_max_len)
            tensor = torch.tensor(emb, dtype=torch.float32)
            sample_dim = int(tensor.shape[-1])
            torch.save(tensor, cache_file)
        cache_entries["protbert"] = _make_cache_entry(
            name="protbert",
            cache_dir=protbert_cache,
            level="pooled",
            sequence_field="dna_seq",
            dim=sample_dim,
            extra={
                "max_seq_len": int(protbert_max_len),
                "protbert_path": protbert_path,
                "count": len(ordered_ids),
            },
        )

    if "evo2" in targets:
        from utils.embed_extract import Evo2Embedder

        evo2_cache = cache_root / "evo2"
        evo2_cache.mkdir(parents=True, exist_ok=True)
        embedder = Evo2Embedder(
            layer_name=evo2_layer_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_len=evo2_max_len,
        )
        sample_dim = 0
        for sample_id in ordered_ids:
            cache_file = _cache_file_for_id(evo2_cache, sample_id)
            if cache_file.exists():
                if sample_dim == 0:
                    sample_dim = int(torch.load(cache_file, map_location="cpu", weights_only=False).shape[-1])
                continue
            emb = embedder.encode_matrix(sample_id, seq_by_id[sample_id])
            tensor = torch.tensor(emb, dtype=torch.float32)
            sample_dim = int(tensor.shape[-1])
            torch.save(tensor, cache_file)
        cache_entries["evo2"] = _make_cache_entry(
            name="evo2",
            cache_dir=evo2_cache,
            level="token",
            sequence_field="dna_seq",
            dim=sample_dim,
            extra={
                "max_seq_len": int(evo2_max_len),
                "layer_name": evo2_layer_name,
                "count": len(ordered_ids),
            },
        )

    if "alphagenome" in targets:
        from utils.embed_extract.alphagenome import (
            infer_ag_dim,
            infer_ag_level,
            load_all_ag_embeddings,
            summarize_ag_embeddings,
        )

        if not alphagenome_dir:
            raise ValueError("stage1-mibig extract includes alphagenome but --alphagenome-dir is empty")
        ag_dir = _require_dir(alphagenome_dir)
        ag_cache = cache_root / "alphagenome"
        ag_cache.mkdir(parents=True, exist_ok=True)
        ag_map = load_all_ag_embeddings(ag_dir)
        sample_dim = 0
        sample_level = ""
        matched = 0
        matched_ids: list[str] = []
        missing: list[str] = []
        level_mismatch: list[str] = []
        dim_mismatch: list[str] = []
        for sample_id in ordered_ids:
            emb = ag_map.get(sample_id)
            if emb is None:
                missing.append(sample_id)
                continue
            tensor = torch.tensor(emb, dtype=torch.float32)
            current_level = infer_ag_level(emb)
            current_dim = infer_ag_dim(emb)
            if not sample_level:
                sample_level = current_level
            elif sample_level != current_level:
                level_mismatch.append(sample_id)
                continue
            if sample_dim == 0:
                sample_dim = current_dim
            elif sample_dim != current_dim:
                dim_mismatch.append(sample_id)
                continue
            torch.save(tensor, _cache_file_for_id(ag_cache, sample_id))
            matched += 1
            matched_ids.append(sample_id)
        if matched == 0:
            raise ValueError(f"No AlphaGenome embeddings matched stage1 sample ids under {ag_dir}")
        if level_mismatch:
            raise ValueError(
                "AlphaGenome embeddings have mixed level types for matched samples; "
                f"examples={level_mismatch[:5]}"
            )
        if dim_mismatch:
            raise ValueError(
                "AlphaGenome embeddings have inconsistent last-dim for matched samples; "
                f"examples={dim_mismatch[:5]}"
            )
        if missing:
            (ag_cache / "_missing_ids.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
        ag_summary = summarize_ag_embeddings({sid: ag_map[sid] for sid in matched_ids})
        cache_entries["alphagenome"] = _make_cache_entry(
            name="alphagenome",
            cache_dir=ag_cache,
            level=sample_level,
            sequence_field="dna_seq",
            dim=sample_dim,
            source="external_npz",
            extra={
                "alphagenome_dir": str(ag_dir),
                "count": matched,
                "missing_count": len(missing),
                "summary": ag_summary,
            },
        )

    return cache_entries


def _package_kg_from_samples(
    *,
    ordered_ids: list[str],
    seq_by_id: dict[str, str],
    labels_by_id: dict[str, set[str]],
    output_dir: Path,
    train_ratio: float,
    seed: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "mibig_kg.tsv"
    _write_mibig_kg_tsv(tsv_path, ordered_ids, labels_by_id, seq_by_id)
    (output_dir / "mibig_kg_summary.json").write_text(
        json.dumps(
            {
                "num_samples": len(ordered_ids),
                "kg_tsv": str(tsv_path),
                "classes": sorted({label for sample_id in ordered_ids for label in labels_by_id[sample_id]}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return _native_run_kg(
        {
            "input_tsv": str(tsv_path),
            "output_dir": str(output_dir),
            "train_ratio": train_ratio,
            "seed": seed,
        }
    )


def _get_stage1_cache_entry(stage1_manifest: dict[str, object], name: str) -> dict[str, object]:
    cache_info = stage1_manifest.get("cache", {})
    if not isinstance(cache_info, dict):
        raise ValueError("stage1 manifest cache section is invalid")
    entry = cache_info.get(name)
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str) and entry:
        torch = _load_torch()
        cache_dir = Path(entry)
        meta_path = _cache_index_path(cache_dir)
        if meta_path.exists():
            return _read_cache_index(cache_dir)
        for p in sorted(cache_dir.glob("*.pt")):
            tensor = torch.load(p, map_location="cpu", weights_only=False)
            if not torch.is_tensor(tensor):
                tensor = torch.tensor(tensor)
            tensor = tensor.float()
            level = "pooled" if tensor.ndim == 1 else "token"
            dim = int(tensor.shape[-1]) if tensor.ndim >= 1 else 0
            return {
                "name": name,
                "dir": str(cache_dir),
                "level": level,
                "sequence_field": "dna_seq",
                "dim": dim,
            }
    raise ValueError(f"stage1 cache entry not found for {name}")


def _load_cached_tensor(cache_dir: Path, sample_id: str):
    torch = _load_torch()
    path = _cache_file_for_id(cache_dir, sample_id)
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    return tensor.float()


def _compute_and_cache_tensor(
    *,
    embedding_name: str,
    cache_dir: Path,
    sample_id: str,
    sequence: str,
    cache_entry: dict[str, object],
):
    torch = _load_torch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = _cache_file_for_id(cache_dir, sample_id)

    if embedding_name == "esm2":
        from utils.embed_extract import ESM2Embedder

        embedder = ESM2Embedder(device="cuda" if torch.cuda.is_available() else "cpu", layer=33)
        max_seq_len = int(cache_entry.get("max_seq_len", 1022))
        seq_use = sequence[:max_seq_len] if max_seq_len > 0 else sequence
        tensor = torch.tensor(embedder.encode_matrix_with_oom_fallback(sample_id, seq_use), dtype=torch.float32)
    elif embedding_name == "protbert":
        from utils.embed_extract import ProtBERTEmbedder

        model_path = str(cache_entry.get("protbert_path", "local_assets/data/ProtBERT/ProtBERT"))
        max_len = int(cache_entry.get("max_seq_len", 1024))
        embedder = ProtBERTEmbedder(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(embedder.encode_mean(sample_id, sequence, max_len=max_len), dtype=torch.float32)
    elif embedding_name == "evo2":
        from utils.embed_extract import Evo2Embedder

        embedder = Evo2Embedder(
            layer_name=str(cache_entry.get("layer_name", "blocks.21.mlp.l3")),
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_len=int(cache_entry.get("max_seq_len", 16384)),
        )
        tensor = torch.tensor(embedder.encode_matrix(sample_id, sequence), dtype=torch.float32)
    else:
        raise FileNotFoundError(f"Missing cache for {sample_id} in {cache_dir} and embedding {embedding_name} is not computable on the fly")

    torch.save(tensor, out_path)
    return tensor


def _load_or_compute_tensor(
    *,
    embedding_name: str,
    cache_entry: dict[str, object],
    sample_id: str,
    sequence: str,
):
    cache_dir = Path(str(cache_entry["dir"]))
    try:
        return _load_cached_tensor(cache_dir, sample_id)
    except FileNotFoundError:
        return _compute_and_cache_tensor(
            embedding_name=embedding_name,
            cache_dir=cache_dir,
            sample_id=sample_id,
            sequence=sequence,
            cache_entry=cache_entry,
        )


def _vector_from_tensor_for_kd(tensor):
    if tensor.ndim == 1:
        return tensor
    return tensor.mean(dim=0)


def _tensor_for_cnn_from_tensor(tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _build_oracle_cnn_dataset_from_cache(
    *,
    ids: list[str],
    seq_by_id: dict[str, str],
    labels_by_id: dict[str, set[str]],
    cache_entry: dict[str, object],
    embedding_name: str,
    output_dir: Path,
    train_ratio: float,
    seed: int,
    add_neg: bool,
    neg_ratio: float,
    min_seq_len: int,
    swap_frac_min: float,
    swap_frac_max: float,
    neg_prefix: str,
) -> dict[str, object]:
    torch = _load_torch()

    prep = _prepare_cnn_inputs_from_samples(
        ids=ids,
        seq_by_id=seq_by_id,
        labels_by_id=labels_by_id,
        output_dir=output_dir / "oracle_cnn_inputs",
        train_ratio=train_ratio,
        seed=seed,
        add_neg=add_neg,
        neg_ratio=neg_ratio,
        min_seq_len=min_seq_len,
        swap_frac_min=swap_frac_min,
        swap_frac_max=swap_frac_max,
        neg_prefix=neg_prefix,
    )

    with Path(prep["label_file"]).open("r", encoding="utf-8") as f:
        label_data = f.readlines()
    label_dict, classes = _oracle_generate_dict(label_data, use_v2=True)

    def _make_dataset(sample_ids: list[str]) -> ClassifierDataset:
        data_dict: dict[str, tuple[Any, Any]] = {}
        for sample_id in sample_ids:
            tensor = _load_or_compute_tensor(
                embedding_name=embedding_name,
                cache_entry=cache_entry,
                sample_id=sample_id,
                sequence=seq_by_id[sample_id],
            )
            data_dict[sample_id] = (_tensor_for_cnn_from_tensor(tensor), label_dict[sample_id])
        return ClassifierDataset(data_dict)

    dataset_dir = output_dir / "oracle_cnn"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(label_dict, dataset_dir / "label_dict.pt")
    torch.save(classes, dataset_dir / "classes.pt")
    torch.save(_make_dataset(prep["train_ids"]), dataset_dir / "oracle_train_dataset.pt")
    torch.save(_make_dataset(prep["valid_ids"]), dataset_dir / "oracle_valid_dataset.pt")

    return {
        "root": str(output_dir),
        "inputs_dir": str(output_dir / "oracle_cnn_inputs"),
        "dataset_dir": str(dataset_dir),
        "oracle_train_dataset": str(dataset_dir / "oracle_train_dataset.pt"),
        "oracle_valid_dataset": str(dataset_dir / "oracle_valid_dataset.pt"),
        "embedding_name": embedding_name,
        "cache_dir": str(cache_entry["dir"]),
        "cache_level": str(cache_entry.get("level", "")),
    }


def _split_train_val_test_ids(ids: list[str], train_ratio: float, val_ratio: float, rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    ordered = list(ids)
    rng.shuffle(ordered)
    n = len(ordered)
    if n == 0:
        return [], [], []
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, min(n_train, n))
    n_val = max(1, min(n_val, n - n_train - 1)) if n >= 3 else max(0, min(n_val, n - n_train))
    train_ids = ordered[:n_train]
    val_ids = ordered[n_train : n_train + n_val]
    test_ids = ordered[n_train + n_val :]
    if not test_ids and val_ids:
        test_ids = [val_ids.pop()]
    if not val_ids and len(train_ids) > 1:
        val_ids = [train_ids.pop()]
    return train_ids, val_ids, test_ids


def _build_kd_dataset_from_cache(
    *,
    ids: list[str],
    seq_by_id: dict[str, str],
    labels_by_id: dict[str, set[str]],
    cache_entries: list[dict[str, object]],
    embedding_names: list[str],
    output_file: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    add_neg: bool,
    neg_ratio: float,
    min_seq_len: int,
    swap_frac_min: float,
    swap_frac_max: float,
    neg_prefix: str,
) -> dict[str, object]:
    np = _load_numpy()
    torch = _load_torch()

    seq_map = {k: v for k, v in seq_by_id.items()}
    label_map = {k: set(v) for k, v in labels_by_id.items()}
    rng = random.Random(seed)
    neg_ids: list[str] = []
    if add_neg:
        neg_ids = _add_mibig_negative_samples(
            ids,
            seq_map,
            label_map,
            neg_ratio=neg_ratio,
            min_seq_len=min_seq_len,
            swap_frac_min=swap_frac_min,
            swap_frac_max=swap_frac_max,
            neg_prefix=neg_prefix,
            rng=rng,
        )

    all_ids = ids + neg_ids
    train_ids, val_ids, test_ids = _split_train_val_test_ids(all_ids, train_ratio, val_ratio, rng)

    def _label_for(sample_id: str) -> int:
        return 0 if "neg" in label_map[sample_id] else 1

    def _make_block(sample_ids: list[str]):
        feats = []
        labels = []
        out_ids = []
        for sample_id in sample_ids:
            parts = []
            for embedding_name, cache_entry in zip(embedding_names, cache_entries):
                tensor = _load_or_compute_tensor(
                    embedding_name=embedding_name,
                    cache_entry=cache_entry,
                    sample_id=sample_id,
                    sequence=seq_map[sample_id],
                )
                parts.append(_vector_from_tensor_for_kd(tensor))
            feats.append(torch.cat(parts, dim=0))
            labels.append(float(_label_for(sample_id)))
            out_ids.append(sample_id)
        return {
            "X": torch.stack(feats, dim=0) if feats else torch.empty(0, 0),
            "y": torch.tensor(labels, dtype=torch.float32),
            "ids": out_ids,
        }

    payload = {
        "train": _make_block(train_ids),
        "val": _make_block(val_ids),
        "test": _make_block(test_ids),
        "meta": {
            "embedding_names": embedding_names,
            "feature_dim": int(sum(int(entry["dim"]) for entry in cache_entries)),
            "train_size": len(train_ids),
            "val_size": len(val_ids),
            "test_size": len(test_ids),
            "neg_samples": len(neg_ids),
            "seq_by_id": {sample_id: seq_map[sample_id] for sample_id in all_ids},
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_file)
    return {
        "dataset": str(output_file),
        "embedding_names": embedding_names,
        "feature_dim": payload["meta"]["feature_dim"],
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
    }


def _load_torch():
    import torch

    return torch


def _load_numpy():
    import numpy as np

    return np


def _load_pandas():
    import pandas as pd

    return pd


def _load_tqdm():
    from tqdm import tqdm

    return tqdm


def _make_mock_esm_dataset(pos_path: str, neg_path: str, dim: int):
    torch = _load_torch()
    data: Dict[str, tuple[Any, Any]] = {}
    for seq_id, seq in _parse_fasta_rows(pos_path):
        length = max(8, min(len(seq), 256))
        emb = torch.randn(length, dim, dtype=torch.float32)
        data[seq_id] = (emb, torch.tensor([1], dtype=torch.long))
    for seq_id, seq in _parse_fasta_rows(neg_path):
        length = max(8, min(len(seq), 256))
        emb = torch.randn(length, dim, dtype=torch.float32)
        data[seq_id] = (emb, torch.tensor([0], dtype=torch.long))
    return ProteinDataset(data)


def _make_mock_evo_dataset(pos_path: str, neg_path: str, dim: int):
    torch = _load_torch()
    data: Dict[str, tuple[Any, Any]] = {}
    for seq_id, seq in _parse_fasta_rows(pos_path):
        length = max(8, min(len(seq), 256))
        emb = torch.randn(length, dim, dtype=torch.float32)
        data[seq_id] = (emb, 1)
    for seq_id, seq in _parse_fasta_rows(neg_path):
        length = max(8, min(len(seq), 256))
        emb = torch.randn(length, dim, dtype=torch.float32)
        data[seq_id] = (emb, 0)
    return list(data.values())


def _build_esm_dataset(pos_path: str, neg_path: str, *, max_seq_len: int | None = None):
    torch = _load_torch()
    tqdm = _load_tqdm()
    from utils.embed_extract import ESM2Embedder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ESM2Embedder(device=device, layer=33)
    data: Dict[str, tuple[Any, Any]] = {}

    for fasta_path, label in ((pos_path, 1), (neg_path, 0)):
        rows = _parse_fasta_rows(fasta_path)
        for seq_id, seq in tqdm(rows, desc=f"Embedding {Path(fasta_path).stem}", leave=False):
            seq_use = seq[:max_seq_len] if max_seq_len is not None else seq
            emb = embedder.encode_matrix_with_oom_fallback(seq_id, seq_use)
            data[seq_id] = (
                torch.tensor(emb, dtype=torch.float32),
                torch.tensor([label], dtype=torch.long),
            )
    return ProteinDataset(data)


class _EvoEmbeddingGenerator:
    def __init__(
        self,
        *,
        layer_name: str = "blocks.21.mlp.l3",
        max_len: int = 16384,
        compression: str = "gzip",
    ):
        from utils.embed_extract import Evo2Embedder

        self.embedder = Evo2Embedder(layer_name=layer_name, max_len=max_len)
        self.max_len = max_len
        self.compression = compression

    def clean_sequence(self, seq: str) -> str:
        return self.embedder.clean_dna(seq)

    def generate_embedding(self, header: str, sequence: str, out_root: str | Path) -> bool:
        pd = _load_pandas()

        clean_seq = self.clean_sequence(sequence)
        if not clean_seq:
            print(f"Warning: sequence {header} is empty, skipped")
            return False

        try:
            seq_embedding = self.embedder.encode_matrix(header, clean_seq)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if ("out of memory" in msg) or ("canuse32bitindexmath" in msg):
                print(f"OOM error: sequence {header} failed: {exc}")
                return False
            raise

        columns = ["token_idx"] + [f"dim{i}" for i in range(seq_embedding.shape[1])]
        rows = [[token_idx, *vec.tolist()] for token_idx, vec in enumerate(seq_embedding)]
        df = pd.DataFrame(rows, columns=columns)

        out_root_path = Path(out_root)
        out_root_path.mkdir(parents=True, exist_ok=True)
        safe_name = header_to_stem(header)
        if self.compression == "gzip":
            csv_path = out_root_path / f"{safe_name}.csv.gz"
            df.to_csv(csv_path, index=False, compression="gzip")
        else:
            csv_path = out_root_path / f"{safe_name}.csv"
            df.to_csv(csv_path, index=False)
        return True


def header_to_stem(header: str, max_base_len: int = 120) -> str:
    base = re.sub(r"[^A-Za-z0-9._=-]+", "_", header).strip("._")
    if not base:
        base = "seq"
    if len(base) > max_base_len:
        base = base[:max_base_len]
    digest = hashlib.sha1(header.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{digest}"


def _existing_embedding_path(out_root: str | Path, header: str) -> Path | None:
    out_dir = Path(out_root)
    stem = header_to_stem(header)
    candidates = [
        out_dir / f"{stem}.csv.gz",
        out_dir / f"{stem}.csv",
        out_dir / f"{header}.csv.gz",
        out_dir / f"{header}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_mapping_tsv(out_root: str | Path, rows: List[List[str]]) -> Path:
    map_path = Path(out_root) / "header_to_file_map.tsv"
    with map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["header", "safe_stem", "file_path", "status"])
        writer.writerows(rows)
    return map_path


def _validate_and_regenerate_evo_embeddings(
    fasta_file: str | Path,
    out_root: str | Path,
    *,
    layer_name: str,
    max_len: int,
    report_every: int,
    compression: str = "gzip",
    skip_existing: bool = True,
) -> tuple[int, int, int]:
    pd = _load_pandas()
    tqdm = _load_tqdm()

    Path(out_root).mkdir(parents=True, exist_ok=True)
    fasta_dict = _parse_fasta_dict(fasta_file)
    print(f"Loaded {len(fasta_dict)} sequences from {fasta_file}")
    print(f"Output directory: {out_root}")

    generator = _EvoEmbeddingGenerator(
        layer_name=layer_name,
        max_len=max_len,
        compression=compression,
    )

    valid_count = 0
    regenerated_count = 0
    error_count = 0
    map_rows: List[List[str]] = []

    pbar = tqdm(fasta_dict.items(), total=len(fasta_dict), desc="Validating sequences")
    for idx, (header, sequence) in enumerate(pbar, start=1):
        csv_path = _existing_embedding_path(out_root, header)
        safe_stem = header_to_stem(header)
        seq_len = len(generator.clean_sequence(sequence))
        if generator.max_len is not None:
            seq_len = min(seq_len, generator.max_len)

        if csv_path is None:
            success = generator.generate_embedding(header, sequence, out_root)
            if success:
                regenerated_count += 1
                csv_path = _existing_embedding_path(out_root, header)
                map_rows.append([header, safe_stem, str(csv_path or ""), "generated"])
            else:
                error_count += 1
                map_rows.append([header, safe_stem, "", "error"])
            continue

        if skip_existing:
            valid_count += 1
            map_rows.append([header, safe_stem, str(csv_path), "exists_skip"])
            continue

        try:
            df = pd.read_csv(csv_path)
            csv_rows = len(df)
            if csv_rows == seq_len:
                valid_count += 1
                map_rows.append([header, safe_stem, str(csv_path), "valid"])
            else:
                success = generator.generate_embedding(header, sequence, out_root)
                if success:
                    regenerated_count += 1
                    csv_path = _existing_embedding_path(out_root, header)
                    map_rows.append([header, safe_stem, str(csv_path or ""), "regenerated"])
                else:
                    error_count += 1
                    map_rows.append([header, safe_stem, "", "error"])
        except Exception:
            success = generator.generate_embedding(header, sequence, out_root)
            if success:
                regenerated_count += 1
                csv_path = _existing_embedding_path(out_root, header)
                map_rows.append([header, safe_stem, str(csv_path or ""), "regenerated"])
            else:
                error_count += 1
                map_rows.append([header, safe_stem, "", "error"])

        if report_every > 0 and idx % report_every == 0:
            pbar.write(
                f"[Progress] {idx}/{len(fasta_dict)} "
                f"valid={valid_count} regenerated={regenerated_count} errors={error_count}"
            )

    map_path = _write_mapping_tsv(out_root, map_rows)
    print(f"Header mapping TSV: {map_path}")
    return valid_count, regenerated_count, error_count


def _evo_extract_name_from_file(path: str | Path) -> str:
    base = Path(path).name
    if base.endswith(".csv.gz"):
        return base[: -len(".csv.gz")]
    if base.endswith(".csv"):
        return base[: -len(".csv")]
    return Path(base).stem


def _save_evo_shard(shard_records: list[tuple[str, Any, int]], shard_path: Path) -> None:
    torch = _load_torch()
    names = [x[0] for x in shard_records]
    embeddings = [x[1] for x in shard_records]
    labels = [x[2] for x in shard_records]
    torch.save({"names": names, "embeddings": embeddings, "labels": labels}, shard_path)


def _evo_process_directory(
    dna_dir: str | Path,
    *,
    label: int,
    out_dir: str | Path,
    prefix: str,
    samples_per_shard: int = 64,
    dtype: str = "float16",
) -> dict[str, object]:
    pd = _load_pandas()
    torch = _load_torch()
    tqdm = _load_tqdm()

    dna_files = sorted(
        glob.glob(str(Path(dna_dir) / "*.csv")) + glob.glob(str(Path(dna_dir) / "*.csv.gz"))
    )
    shard_dir = Path(out_dir) / f"{prefix}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_records: list[tuple[str, Any, int]] = []
    shard_meta: list[dict[str, object]] = []
    total = 0
    shard_idx = 0
    target_dtype = torch.float16 if dtype == "float16" else torch.float32

    for dna_file in tqdm(dna_files, desc=f"Processing {prefix}", leave=False):
        try:
            name = _evo_extract_name_from_file(dna_file)
            dna_data = pd.read_csv(dna_file).iloc[:, 1:].values
            dna_tensor = torch.tensor(dna_data, dtype=target_dtype)
            shard_records.append((name, dna_tensor, int(label)))
            total += 1

            if len(shard_records) >= samples_per_shard:
                shard_name = f"{prefix}_part_{shard_idx:05d}.pt"
                shard_path = shard_dir / shard_name
                _save_evo_shard(shard_records, shard_path)
                shard_meta.append(
                    {"path": str(shard_path.relative_to(Path(out_dir))), "count": len(shard_records)}
                )
                shard_records = []
                shard_idx += 1
        except Exception as exc:
            print(f"Error processing file {dna_file}: {exc}")

    if shard_records:
        shard_name = f"{prefix}_part_{shard_idx:05d}.pt"
        shard_path = shard_dir / shard_name
        _save_evo_shard(shard_records, shard_path)
        shard_meta.append(
            {"path": str(shard_path.relative_to(Path(out_dir))), "count": len(shard_records)}
        )

    return {
        "format": "sharded_v1",
        "version": 1,
        "dtype": dtype,
        "num_samples": total,
        "shards": shard_meta,
    }


def _kg_parse_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    query, bgc_type, _gos, sequence = parts[0], parts[1], parts[2], parts[3]
    labels = [x for x in bgc_type.split("-") if x]
    if not query or not labels or not sequence:
        return None
    return query, labels, sequence


def _kg_read_tsv(path: Path):
    rows = []
    type2id: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("query"):
                continue
            parsed = _kg_parse_line(line)
            if parsed is None:
                continue
            query, labels, sequence = parsed
            rows.append((query, labels, sequence))
            for label in labels:
                if label not in type2id:
                    type2id[label] = len(type2id)
    if not rows:
        raise ValueError(f"No valid records found in {path}")
    return rows, type2id


def _kg_save_split(records, type2id: dict[str, int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    relation2id = {"no": 0, "yes": 1}

    protein2id: dict[str, int] = {}
    query2seq: dict[str, str] = {}
    for query, _labels, seq in records:
        if query not in protein2id:
            protein2id[query] = len(protein2id)
            query2seq[query] = seq

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
        for query, _pid in sorted(protein2id.items(), key=lambda x: x[1]):
            f.write(query2seq[query] + "\n")

    with (out_dir / "protein_relation_type.txt").open("w", encoding="utf-8") as f:
        for query, labels, _seq in records:
            pid = protein2id[query]
            label_set = set(labels)
            for type_name, tid in type2id.items():
                rid = relation2id["yes"] if type_name in label_set else relation2id["no"]
                f.write(f"{pid}\t{rid}\t{tid}\n")


def _load_map_rows(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            header = (row.get("header") or "").strip()
            stem = (row.get("safe_stem") or "").strip()
            if header and stem:
                rows.append((header, stem))
    return rows


def _iter_esm_items(obj) -> Iterator[Tuple[str, Any]]:
    torch = _load_torch()

    if hasattr(obj, "items") and isinstance(obj.items, list):
        for key, value in obj.items:
            if isinstance(value, tuple) and len(value) >= 1:
                yield key, value[0]
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, tuple) and len(value) >= 1:
                yield key, value[0]
            elif torch.is_tensor(value):
                yield key, value
        return

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            if isinstance(value, tuple) and len(value) >= 1:
                yield f"item_{idx}", value[0]
        return

    raise TypeError(f"Unsupported dataset object type: {type(obj)}")


def _load_needed_embeddings(dataset_paths: List[Path], needed_headers: List[str]) -> dict[str, Any]:
    torch = _load_torch()

    found: dict[str, Any] = {}
    needed = set(needed_headers)
    for path in dataset_paths:
        if not path.exists():
            continue
        print(f"[EE-PROTEIN] loading {path}")
        obj = torch.load(path, weights_only=False)
        matched = 0
        for header, emb in _iter_esm_items(obj):
            if header in needed and header not in found:
                found[header] = emb
                matched += 1
        print(f"[EE-PROTEIN] matched {matched} headers from {path.name}")
        if len(found) == len(needed):
            break
    return found


def _save_ee_split(rows: List[Tuple[str, str]], found: dict[str, Any], out_dir: Path, split_name: str):
    torch = _load_torch()

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    missing: List[Tuple[str, str]] = []
    for header, stem in rows:
        emb = found.get(header)
        if emb is None:
            missing.append((header, stem))
            continue
        torch.save(emb, out_dir / f"{stem}.pt")
        ok += 1
    print(f"[EE-PROTEIN] {split_name}: saved={ok}, missing={len(missing)}")
    return missing


def _fallback_rows_from_files(split_dir: Path) -> List[Tuple[str, str]]:
    rows = []
    files = sorted(glob.glob(str(split_dir / "*.csv")) + glob.glob(str(split_dir / "*.csv.gz")))
    for file_path in files:
        base = Path(file_path).name
        stem = base[:-7] if base.endswith(".csv.gz") else base[:-4]
        rows.append((stem, stem))
    return rows


def _read_split_rows(split_dir: Path, split_name: str) -> List[Tuple[str, str]]:
    map_tsv = split_dir / "header_to_file_map.tsv"
    if map_tsv.exists():
        rows = _load_map_rows(map_tsv)
        if rows:
            return rows
    rows = _fallback_rows_from_files(split_dir)
    if not rows:
        raise ValueError(f"No csv/csv.gz files in {split_dir} ({split_name})")
    print(f"[KD-DATA] map missing for {split_name}, fallback to filename-based ids: {split_dir}")
    return rows


def _find_evo_csv(split_dir: Path, stem: str) -> Path:
    p1 = split_dir / f"{stem}.csv.gz"
    if p1.exists():
        return p1
    p2 = split_dir / f"{stem}.csv"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing Evo embedding csv for stem={stem} in {split_dir}")


def _evo_mean(csv_path: Path):
    pd = _load_pandas()
    return pd.read_csv(csv_path).iloc[:, 1:].values.astype("float32").mean(axis=0)


def _merge_embeddings(pro_emb, dna_emb, label: int):
    torch = _load_torch()

    protein = pro_emb
    dna = dna_emb
    protein_len = protein.size(0)
    dna_len = dna.size(0)

    if label == 1:
        if dna_len == protein_len * 3:
            pass
        elif dna_len == protein_len * 3 + 3:
            dna = dna[:-3, :]
        else:
            raise ValueError(
                f"Positive sample length mismatch: protein={protein_len}, dna={dna_len}"
            )
    else:
        if dna_len == protein_len * 3:
            pass
        elif dna_len == protein_len * 3 - 3:
            protein = protein[:-1, :]
        else:
            raise ValueError(
                f"Negative sample length mismatch: protein={protein_len}, dna={dna_len}"
            )

    if dna.size(0) != protein.size(0) * 3:
        raise RuntimeError(
            f"Length mismatch after adjustment: protein={protein.size(0)}, dna={dna.size(0)}"
        )

    dna_reshaped = dna.view(protein.size(0), 3, -1)
    dna_concatenated = dna_reshaped.reshape(protein.size(0), -1)
    return torch.cat([protein, dna_concatenated], dim=1)


def _merge_dna_and_pt(dna_file: Path, pt_file: Path, label: int):
    pd = _load_pandas()
    torch = _load_torch()

    dna_data = pd.read_csv(dna_file)
    dna_embeddings = torch.tensor(dna_data.iloc[:, 1:].values, dtype=torch.float32)
    protein_data = torch.load(pt_file, weights_only=False)

    if isinstance(protein_data, tuple):
        protein_embeddings = protein_data[0]
    elif isinstance(protein_data, dict) and "representations" in protein_data:
        protein_embeddings = protein_data["representations"][33]
    else:
        protein_embeddings = protein_data
    return _merge_embeddings(protein_embeddings, dna_embeddings, label), label


def _process_ee_directory(dna_dir: Path, protein_dir: Path, *, label: int) -> dict[str, tuple[Any, Any]]:
    tqdm = _load_tqdm()
    data_dict: dict[str, tuple[Any, Any]] = {}
    dna_files = sorted(
        glob.glob(str(dna_dir / "*.csv")) + glob.glob(str(dna_dir / "*.csv.gz"))
    )

    for dna_file in tqdm(dna_files, desc=f"Processing {'positive' if label == 1 else 'negative'}"):
        try:
            name = _evo_extract_name_from_file(dna_file)
            protein_file = protein_dir / f"{name}.pt"
            if not protein_file.exists():
                print(f"Protein file missing: {protein_file}")
                continue
            data_dict[name] = _merge_dna_and_pt(Path(dna_file), protein_file, label)
        except Exception as exc:
            print(f"Error processing {dna_file}: {exc}")
    return data_dict


def _load_esm_feature_map(dataset_paths: List[Path]) -> dict[str, Any]:
    torch = _load_torch()

    out: dict[str, Any] = {}
    for path in dataset_paths:
        if not path.exists():
            continue
        print(f"[KD-DATA] loading ESM dataset: {path}")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        matched = 0
        for header, emb in _iter_esm_items(obj):
            if header in out:
                continue
            if not torch.is_tensor(emb):
                emb = torch.tensor(emb)
            out[header] = emb.float().mean(dim=0).numpy()
            matched += 1
        print(f"[KD-DATA] ESM samples loaded from {path.name}: {matched}")
    return out


def _oracle_generate_dict(label_data: List[str], *, use_v2: bool):
    torch = _load_torch()
    from collections import defaultdict

    label_dict = defaultdict(list)
    current_id = None
    for line in label_data:
        line = line.strip()
        if not line:
            continue
        if use_v2:
            if line.startswith("BGC") or line.startswith("NEG"):
                current_id = line[:12]
            elif current_id:
                label_dict[current_id].append(line)
        else:
            if line.startswith("BGC"):
                current_id = line
            elif current_id:
                label_dict[current_id].append(line)

    all_labels = set()
    for labels in label_dict.values():
        all_labels.update(labels)

    classes = sorted(all_labels)
    tensor_dict = {}
    for bgc_id, labels in label_dict.items():
        vector = [1 if cls in labels else 0 for cls in classes]
        tensor_dict[bgc_id] = torch.tensor(vector, dtype=torch.float32)
    return tensor_dict, classes


def _oracle_get_label(seq_id: str, label_dict: dict[str, Any], *, use_v2: bool):
    if use_v2:
        key12 = seq_id[:12]
        if key12 in label_dict:
            return label_dict[key12]
        if seq_id in label_dict:
            return label_dict[seq_id]
        raise KeyError(f"Label not found for sequence id={seq_id}")
    return label_dict[seq_id[:12]]


def _oracle_fasta_to_esm(
    fasta_file: str,
    label_dict: dict[str, Any],
    *,
    max_seq_len: int | None,
    use_v2: bool,
    cache_dir: Path | None = None,
):
    torch = _load_torch()
    tqdm = _load_tqdm()
    from utils.embed_extract import ESM2Embedder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    records = _parse_fasta_rows(fasta_file)
    print(f"Processing {len(records)} sequences from {fasta_file}")
    embeddings: dict[str, tuple[Any, Any]] = {}
    truncated_count = 0
    for seq_id, seq in tqdm(records, desc="Processing sequences"):
        seq_use = seq
        if max_seq_len is not None and len(seq_use) > max_seq_len:
            seq_use = seq_use[:max_seq_len]
            truncated_count += 1
        cache_file = _cache_file_for_id(cache_dir, seq_id) if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            token_tensor = torch.load(cache_file, map_location="cpu", weights_only=False)
            token_representations = token_tensor.float()
        else:
            if embedder is None:
                embedder = ESM2Embedder(device=device, layer=33)
            token_representations = torch.tensor(
                embedder.encode_matrix_with_oom_fallback(seq_id, seq_use),
                dtype=torch.float32,
            )
            if cache_file is not None:
                torch.save(token_representations, cache_file)
        embeddings[seq_id] = (
            token_representations,
            _oracle_get_label(seq_id, label_dict, use_v2=use_v2),
        )
    if truncated_count:
        print(f"Warning: Truncated {truncated_count} sequences to max length {max_seq_len}")
    return embeddings


def _native_run_assets(params: Dict[str, object]) -> int:
    required_files = [
        REPO_ROOT / "config" / "esm2_config.json",
        REPO_ROOT / "config" / "evo2_config.json",
        REPO_ROOT / "scripts" / "trainers" / "esm2_train.py",
        REPO_ROOT / "scripts" / "trainers" / "evo2_train.py",
        REPO_ROOT / "scripts" / "trainers" / "ORACLE_train.py",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    download_models = params.get("download_models") or []
    if download_models:
        from huggingface_hub import snapshot_download

        for model_name in download_models:
            if model_name not in RESOURCE_MAP:
                raise ValueError(f"Unknown resource name: {model_name}")
            local_dir, pattern = RESOURCE_MAP[model_name]
            target_dir = REPO_ROOT / "local_assets" / local_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {model_name} -> {target_dir}")
            snapshot_download(
                repo_id="x1324x4/igem2025njuchina",
                allow_patterns=[pattern],
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

    print("Asset check complete.")
    return 0


def _native_run_esm2(params: Dict[str, object]) -> int:
    _set_cuda_visible_devices(str(params.get("gpus", "")))

    out_dir = Path(str(params["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in (
        "train_pos_fasta",
        "train_neg_fasta",
        "valid_pos_fasta",
        "valid_neg_fasta",
        "test_pos_fasta",
        "test_neg_fasta",
    ):
        _require_file(str(params[key]))

    torch = _load_torch()

    if params.get("mock"):
        train_dataset = _make_mock_esm_dataset(
            str(params["train_pos_fasta"]),
            str(params["train_neg_fasta"]),
            int(params.get("mock_dim", 1280)),
        )
        valid_dataset = _make_mock_esm_dataset(
            str(params["valid_pos_fasta"]),
            str(params["valid_neg_fasta"]),
            int(params.get("mock_dim", 1280)),
        )
        test_dataset = _make_mock_esm_dataset(
            str(params["test_pos_fasta"]),
            str(params["test_neg_fasta"]),
            int(params.get("mock_dim", 1280)),
        )
    else:
        train_dataset = _build_esm_dataset(
            str(params["train_pos_fasta"]),
            str(params["train_neg_fasta"]),
        )
        valid_dataset = _build_esm_dataset(
            str(params["valid_pos_fasta"]),
            str(params["valid_neg_fasta"]),
        )
        test_dataset = _build_esm_dataset(
            str(params["test_pos_fasta"]),
            str(params["test_neg_fasta"]),
        )

    torch.save(train_dataset, out_dir / "train_dataset_esm2.pt")
    torch.save(valid_dataset, out_dir / "valid_dataset_esm2.pt")
    torch.save(test_dataset, out_dir / "test_dataset_esm2.pt")
    print(f"Prepared: {out_dir / 'train_dataset_esm2.pt'}")
    print(f"Prepared: {out_dir / 'valid_dataset_esm2.pt'}")
    print(f"Prepared: {out_dir / 'test_dataset_esm2.pt'}")
    return 0


def _native_run_evo2(params: Dict[str, object]) -> int:
    _set_cuda_visible_devices(str(params.get("gpus", "")))

    work_dir = Path(str(params["output_dir"]))
    work_dir.mkdir(parents=True, exist_ok=True)

    for key in ("train_pos_fasta", "train_neg_fasta", "test_pos_fasta", "test_neg_fasta"):
        _require_file(str(params[key]))

    torch = _load_torch()

    if params.get("mock"):
        train_dataset = _make_mock_evo_dataset(
            str(params["train_pos_fasta"]),
            str(params["train_neg_fasta"]),
            int(params.get("mock_dim", 1920)),
        )
        test_dataset = _make_mock_evo_dataset(
            str(params["test_pos_fasta"]),
            str(params["test_neg_fasta"]),
            int(params.get("mock_dim", 1920)),
        )
        torch.save(train_dataset, work_dir / "train.pt")
        torch.save(test_dataset, work_dir / "test.pt")
        print(f"Prepared Evo2 datasets: {work_dir / 'train.pt'} {work_dir / 'test.pt'}")
        return 0

    emb_dir = work_dir / "emb"
    split_cfg = {
        "train_pos": str(params["train_pos_fasta"]),
        "train_neg": str(params["train_neg_fasta"]),
        "test_pos": str(params["test_pos_fasta"]),
        "test_neg": str(params["test_neg_fasta"]),
    }

    for split_name, fasta_path in split_cfg.items():
        out_root = emb_dir / split_name
        out_root.mkdir(parents=True, exist_ok=True)
        _validate_and_regenerate_evo_embeddings(
            fasta_path,
            out_root,
            layer_name=str(params.get("layer", "blocks.21.mlp.l3")),
            max_len=int(params.get("max_len", 16384)),
            report_every=int(params.get("report_every", 50)),
            compression="gzip" if str(params.get("dtype", "float16")) in {"float16", "float32"} else "gzip",
            skip_existing=True,
        )

    train_build = work_dir / "train_build"
    test_build = work_dir / "test_build"
    train_build.mkdir(parents=True, exist_ok=True)
    test_build.mkdir(parents=True, exist_ok=True)

    train_manifest = _evo_process_directory(
        emb_dir / "train_pos",
        label=1,
        out_dir=train_build,
        prefix="positive",
        samples_per_shard=int(params.get("samples_per_shard", 64)),
        dtype=str(params.get("dtype", "float16")),
    )
    neg_train_manifest = _evo_process_directory(
        emb_dir / "train_neg",
        label=0,
        out_dir=train_build,
        prefix="negative",
        samples_per_shard=int(params.get("samples_per_shard", 64)),
        dtype=str(params.get("dtype", "float16")),
    )
    train_combined = {
        "format": "sharded_v1",
        "version": 1,
        "dtype": str(params.get("dtype", "float16")),
        "num_samples": int(train_manifest["num_samples"]) + int(neg_train_manifest["num_samples"]),
        "shards": list(train_manifest["shards"]) + list(neg_train_manifest["shards"]),
    }
    torch.save(train_manifest, train_build / "positive_dataset.pt")
    torch.save(neg_train_manifest, train_build / "negative_dataset.pt")
    torch.save(train_combined, train_build / "combined_dataset.pt")

    test_manifest = _evo_process_directory(
        emb_dir / "test_pos",
        label=1,
        out_dir=test_build,
        prefix="positive",
        samples_per_shard=int(params.get("samples_per_shard", 64)),
        dtype=str(params.get("dtype", "float16")),
    )
    neg_test_manifest = _evo_process_directory(
        emb_dir / "test_neg",
        label=0,
        out_dir=test_build,
        prefix="negative",
        samples_per_shard=int(params.get("samples_per_shard", 64)),
        dtype=str(params.get("dtype", "float16")),
    )
    test_combined = {
        "format": "sharded_v1",
        "version": 1,
        "dtype": str(params.get("dtype", "float16")),
        "num_samples": int(test_manifest["num_samples"]) + int(neg_test_manifest["num_samples"]),
        "shards": list(test_manifest["shards"]) + list(neg_test_manifest["shards"]),
    }
    torch.save(test_manifest, test_build / "positive_dataset.pt")
    torch.save(neg_test_manifest, test_build / "negative_dataset.pt")
    torch.save(test_combined, test_build / "combined_dataset.pt")

    shutil.copy2(train_build / "combined_dataset.pt", work_dir / "train.pt")
    shutil.copy2(test_build / "combined_dataset.pt", work_dir / "test.pt")
    print(f"Prepared Evo2 datasets: {work_dir / 'train.pt'} {work_dir / 'test.pt'}")
    return 0


def _native_run_kg(params: Dict[str, object]) -> int:
    input_tsv = _require_file(str(params["input_tsv"]))
    train_ratio = float(params.get("train_ratio", 0.8))
    seed = int(params.get("seed", 42))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    rows, type2id = _kg_read_tsv(input_tsv)
    random.seed(seed)
    random.shuffle(rows)

    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    out_root = Path(str(params["output_dir"]))
    _kg_save_split(train_rows, type2id, out_root / "train")
    _kg_save_split(val_rows, type2id, out_root / "val")
    print(f"total={len(rows)} train={len(train_rows)} val={len(val_rows)}")
    print(f"types={len(type2id)}")
    print(f"out={out_root}")
    return 0


def _native_run_ee(params: Dict[str, object]) -> int:
    _set_cuda_visible_devices(str(params.get("gpus", "")))

    evo_emb_root = _require_dir(str(params["evo_emb_root"]))
    esm_train_dataset = _require_file(str(params["esm_train_dataset"]))
    esm_test_dataset = _require_file(str(params["esm_test_dataset"]))
    esm_valid_raw = str(params.get("esm_valid_dataset", ""))
    esm_valid_dataset = Path(esm_valid_raw) if esm_valid_raw else None
    if esm_valid_dataset is not None and not esm_valid_dataset.exists():
        print(f"Warning: missing valid dataset: {esm_valid_dataset} (continue)")
        esm_valid_dataset = None

    for split_dir in ("train_pos", "train_neg", "test_pos", "test_neg"):
        _require_dir(evo_emb_root / split_dir)

    out_dir = Path(str(params["output_dir"])).resolve()
    dna_pos_dir = out_dir / "dna_pos"
    dna_neg_dir = out_dir / "dna_neg"
    pro_pos_dir = out_dir / "protein_pos"
    pro_neg_dir = out_dir / "protein_neg"
    map_dir = out_dir / "maps"

    for path in (dna_pos_dir, dna_neg_dir, pro_pos_dir, pro_neg_dir, map_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    train_pos_rows = _read_split_rows(evo_emb_root / "train_pos", "train_pos")
    train_neg_rows = _read_split_rows(evo_emb_root / "train_neg", "train_neg")
    test_pos_rows = _read_split_rows(evo_emb_root / "test_pos", "test_pos")
    test_neg_rows = _read_split_rows(evo_emb_root / "test_neg", "test_neg")

    for src_dir, dst_dir in (
        (evo_emb_root / "train_pos", dna_pos_dir),
        (evo_emb_root / "test_pos", dna_pos_dir),
        (evo_emb_root / "train_neg", dna_neg_dir),
        (evo_emb_root / "test_neg", dna_neg_dir),
    ):
        for csv_file in list(src_dir.glob("*.csv")) + list(src_dir.glob("*.csv.gz")):
            target = dst_dir / csv_file.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(csv_file.resolve())

    pos_rows = train_pos_rows + test_pos_rows
    neg_rows = train_neg_rows + test_neg_rows
    needed_headers = [header for header, _stem in pos_rows + neg_rows]
    dataset_paths = [esm_train_dataset]
    if esm_valid_dataset is not None:
        dataset_paths.append(esm_valid_dataset)
    dataset_paths.append(esm_test_dataset)
    found = _load_needed_embeddings(dataset_paths, needed_headers)

    miss_pos = _save_ee_split(pos_rows, found, pro_pos_dir, "positive")
    miss_neg = _save_ee_split(neg_rows, found, pro_neg_dir, "negative")
    total_missing = len(miss_pos) + len(miss_neg)
    if total_missing:
        miss_file = out_dir / "missing_esm2_embeddings.tsv"
        with miss_file.open("w", encoding="utf-8") as f:
            f.write("split\theader\tsafe_stem\n")
            for header, stem in miss_pos:
                f.write(f"pos\t{header}\t{stem}\n")
            for header, stem in miss_neg:
                f.write(f"neg\t{header}\t{stem}\n")
        print(f"[EE-PROTEIN] missing list: {miss_file}")
        if not params.get("allow_missing"):
            raise RuntimeError(f"Missing {total_missing} ESM2 embeddings. See {miss_file}")

    pos_data_dict = _process_ee_directory(dna_pos_dir, pro_pos_dir, label=1)
    neg_data_dict = _process_ee_directory(dna_neg_dir, pro_neg_dir, label=0)

    torch = _load_torch()
    pos_dataset = ProteinDnaDataset(pos_data_dict)
    neg_dataset = ProteinDnaDataset(neg_data_dict)
    combined_dataset = ProteinDnaDataset({**pos_data_dict, **neg_data_dict})

    torch.save(pos_dataset, out_dir / "positive_protein_dna_dataset.pt")
    torch.save(neg_dataset, out_dir / "negative_protein_dna_dataset.pt")
    torch.save(combined_dataset, out_dir / "combined_protein_dna_dataset.pt")

    sample_count = len(combined_dataset)
    print(f"[EE-DATA] combined samples={sample_count}")
    if sample_count <= 0:
        raise RuntimeError("EE combined dataset is empty; check DNA/protein alignment and input quality")
    print(f"EE dataset prepared: {out_dir / 'combined_protein_dna_dataset.pt'}")
    return 0


def _native_run_kd_fusion(params: Dict[str, object]) -> int:
    np = _load_numpy()
    torch = _load_torch()

    evo_root = _require_dir(str(params["evo_emb_root"]))
    split_cfg = {
        "train_pos": (evo_root / "train_pos", 1),
        "train_neg": (evo_root / "train_neg", 0),
        "test_pos": (evo_root / "test_pos", 1),
        "test_neg": (evo_root / "test_neg", 0),
    }
    for split_dir, _label in split_cfg.values():
        _require_dir(split_dir)

    seqs = _load_sequences(
        [
            str(params["train_pos_fasta"]),
            str(params["train_neg_fasta"]),
            str(params["test_pos_fasta"]),
            str(params["test_neg_fasta"]),
        ]
    )
    print(f"[KD-DATA] loaded sequences: {len(seqs)}")

    esm_paths = [Path(str(params["esm_train"]))]
    if params.get("esm_valid"):
        esm_paths.append(Path(str(params["esm_valid"])))
    esm_paths.append(Path(str(params["esm_test"])))
    esm_map = _load_esm_feature_map(esm_paths)
    print(f"[KD-DATA] ESM feature map size: {len(esm_map)}")

    from utils.embed_extract import ProtBERTEmbedder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    protbert = ProtBERTEmbedder(str(params["protbert"]), device=device)
    rng = np.random.default_rng(int(params.get("seed", 42)))

    x_train = []
    y_train = []
    id_train: List[str] = []
    x_test = []
    y_test = []
    id_test: List[str] = []
    miss_seq = 0
    miss_esm = 0
    miss_evo = 0
    total = 0
    max_samples = int(params.get("max_samples", -1))

    for split_name, (split_dir, label) in split_cfg.items():
        rows = _read_split_rows(split_dir, split_name)
        if max_samples > 0:
            rows = rows[:max_samples]
        for header, stem in rows:
            total += 1
            if header not in seqs:
                miss_seq += 1
                continue
            if header not in esm_map:
                miss_esm += 1
                continue
            try:
                evo_vec = _evo_mean(_find_evo_csv(split_dir, stem))
            except FileNotFoundError:
                miss_evo += 1
                continue

            prot_vec = protbert.encode_mean(header, seqs[header], max_len=1024)
            feat = np.concatenate([esm_map[header], evo_vec, prot_vec], axis=0).astype(np.float32)

            if split_name.startswith("train_"):
                x_train.append(feat)
                y_train.append(label)
                id_train.append(header)
            else:
                x_test.append(feat)
                y_test.append(label)
                id_test.append(header)

    if not x_train:
        raise RuntimeError("No training samples prepared for KD fusion")
    if not x_test:
        raise RuntimeError("No test samples prepared for KD fusion")

    x_train_arr = np.stack(x_train)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    x_test_arr = np.stack(x_test)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    idx = np.arange(len(x_train_arr))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * float(params.get("val_ratio", 0.2))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx
        val_idx = idx[:1]

    payload = {
        "train": {
            "X": torch.tensor(x_train_arr[train_idx], dtype=torch.float32),
            "y": torch.tensor(y_train_arr[train_idx], dtype=torch.float32),
            "ids": [id_train[i] for i in train_idx],
        },
        "val": {
            "X": torch.tensor(x_train_arr[val_idx], dtype=torch.float32),
            "y": torch.tensor(y_train_arr[val_idx], dtype=torch.float32),
            "ids": [id_train[i] for i in val_idx],
        },
        "test": {
            "X": torch.tensor(x_test_arr, dtype=torch.float32),
            "y": torch.tensor(y_test_arr, dtype=torch.float32),
            "ids": id_test,
        },
        "meta": {
            "feature_dim": int(x_train_arr.shape[1]),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(x_test_arr)),
            "miss_seq": int(miss_seq),
            "miss_esm": int(miss_esm),
            "miss_evo": int(miss_evo),
            "total_candidates": int(total),
        },
    }

    out_file = Path(str(params["output_file"]))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)
    print(json.dumps(payload["meta"], indent=2))
    print(f"[KD-DATA] saved: {out_file}")
    return 0


def _native_run_oracle_cnn(params: Dict[str, object]) -> int:
    _set_cuda_visible_devices(str(params.get("gpus", "")))

    out_dir = Path(str(params["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = _require_file(str(params["label_file"]))
    _require_file(str(params["train_fasta"]))
    _require_file(str(params["valid_fasta"]))

    with label_file.open("r", encoding="utf-8") as f:
        label_data = f.readlines()

    use_v2 = bool(params.get("use_v2"))
    esm2_cache_dir_raw = str(params.get("esm2_cache_dir", "")).strip()
    esm2_cache_dir = Path(esm2_cache_dir_raw) if esm2_cache_dir_raw else None
    if esm2_cache_dir is not None and not esm2_cache_dir.exists():
        print(f"Warning: missing ESM2 cache dir: {esm2_cache_dir} (fallback to on-the-fly extraction)")
        esm2_cache_dir = None
    label_dict, classes = _oracle_generate_dict(label_data, use_v2=use_v2)

    torch = _load_torch()
    torch.save(label_dict, out_dir / "label_dict.pt")
    torch.save(classes, out_dir / "classes.pt")

    max_seq_len = int(params.get("max_seq_len", 1022)) if use_v2 else None
    train_embeddings = _oracle_fasta_to_esm(
        str(params["train_fasta"]),
        label_dict,
        max_seq_len=max_seq_len,
        use_v2=use_v2,
        cache_dir=esm2_cache_dir,
    )
    valid_embeddings = _oracle_fasta_to_esm(
        str(params["valid_fasta"]),
        label_dict,
        max_seq_len=max_seq_len,
        use_v2=use_v2,
        cache_dir=esm2_cache_dir,
    )

    torch.save(ClassifierDataset(train_embeddings), out_dir / "oracle_train_dataset.pt")
    torch.save(ClassifierDataset(valid_embeddings), out_dir / "oracle_valid_dataset.pt")
    print(f"All files have been saved to {out_dir}")
    return 0


def _native_run_stage1_mibig(params: Dict[str, object]) -> int:
    input_fasta = _require_file(str(params["input_fasta"]))
    out_dir = Path(str(params["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids, seq_by_id, labels_by_id = _load_mibig_real_samples(input_fasta)
    samples_fasta = out_dir / "samples.fasta"
    labels_tsv = out_dir / "labels.tsv"
    _write_selected_fasta(samples_fasta, ordered_ids, seq_by_id)
    _write_stage1_label_tsv(labels_tsv, ordered_ids, labels_by_id)

    extract_targets = _parse_csv_option(str(params.get("extract", "esm2,protbert")), default=["esm2", "protbert"])
    invalid = [item for item in extract_targets if item not in {"esm2", "protbert", "alphagenome", "evo2"}]
    if invalid:
        raise ValueError(f"Unsupported --extract items: {invalid}")

    _set_cuda_visible_devices(str(params.get("gpus", "")))
    cache_paths = _extract_stage1_caches(
        output_dir=out_dir,
        ordered_ids=ordered_ids,
        seq_by_id=seq_by_id,
        extract_targets=extract_targets,
        esm_max_seq_len=int(params.get("esm_max_seq_len", 1022)),
        protbert_max_len=int(params.get("protbert_max_len", 1024)),
        protbert_path=str(params.get("protbert_path", "local_assets/data/ProtBERT/ProtBERT")),
        evo2_max_len=int(params.get("evo2_max_len", 16384)),
        evo2_layer_name=str(params.get("evo2_layer_name", "blocks.21.mlp.l3")),
        alphagenome_dir=str(params.get("alphagenome_dir", "")),
    )

    manifest = {
        "stage": "stage1-mibig",
        "input_fasta": str(input_fasta),
        "num_samples": len(ordered_ids),
        "samples_fasta": str(samples_fasta),
        "labels_tsv": str(labels_tsv),
        "extract": extract_targets,
        "cache": cache_paths,
        "esm_max_seq_len": int(params.get("esm_max_seq_len", 1022)),
        "protbert_max_len": int(params.get("protbert_max_len", 1024)),
        "protbert_path": str(params.get("protbert_path", "local_assets/data/ProtBERT/ProtBERT")),
        "evo2_max_len": int(params.get("evo2_max_len", 16384)),
        "evo2_layer_name": str(params.get("evo2_layer_name", "blocks.21.mlp.l3")),
        "alphagenome_dir": str(params.get("alphagenome_dir", "")),
    }
    (out_dir / "stage1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _native_run_stage2_pack(params: Dict[str, object]) -> int:
    stage1_dir = _require_dir(str(params["stage1_dir"]))
    output_dir = Path(str(params["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids, seq_by_id, labels_by_id, stage1_manifest = _load_stage1_samples(stage1_dir)
    requested_targets = _parse_csv_option(str(params.get("targets", "cnn,kd,oracle")), default=["cnn", "kd", "oracle"])
    invalid = [item for item in requested_targets if item not in {"kg", "cnn", "oracle", "kd"}]
    if invalid:
        raise ValueError(f"Unsupported --targets items: {invalid}")

    expanded_targets = set(requested_targets)
    if "oracle" in expanded_targets:
        expanded_targets.update({"kg"})

    multi_layout = len(expanded_targets) > 1 or "oracle" in requested_targets
    summary: dict[str, object] = {
        "stage": "stage2-pack",
        "stage1_dir": str(stage1_dir),
        "requested_targets": requested_targets,
        "expanded_targets": sorted(expanded_targets),
        "outputs": {},
    }

    cnn_embedding = str(params.get("cnn_embedding", "esm2")).strip() or "esm2"
    oracle_embedding = str(params.get("oracle_embedding", "protbert")).strip() or "protbert"
    kd_embeddings = _parse_csv_option(str(params.get("kd_embeddings", "esm2,protbert")), default=["esm2", "protbert"])
    val_ratio = float(params.get("val_ratio", 0.1))

    if "kg" in expanded_targets:
        kg_out = output_dir / "kg" if multi_layout else output_dir
        rc = _package_kg_from_samples(
            ordered_ids=ordered_ids,
            seq_by_id=seq_by_id,
            labels_by_id=labels_by_id,
            output_dir=kg_out,
            train_ratio=float(params.get("train_ratio", 0.8)),
            seed=int(params.get("seed", 42)),
        )
        if rc != 0:
            return rc
        summary["outputs"]["kg"] = {
            "root": str(kg_out),
            "train_dir": str(kg_out / "train"),
            "val_dir": str(kg_out / "val"),
            "kg_tsv": str(kg_out / "mibig_kg.tsv"),
        }

    if "cnn" in expanded_targets:
        cnn_root = output_dir / "cnn" if multi_layout else output_dir
        cnn_entry = _get_stage1_cache_entry(stage1_manifest, cnn_embedding)
        cnn_summary = _build_oracle_cnn_dataset_from_cache(
            ids=ordered_ids,
            seq_by_id=seq_by_id,
            labels_by_id=labels_by_id,
            cache_entry=cnn_entry,
            embedding_name=cnn_embedding,
            output_dir=cnn_root,
            train_ratio=float(params.get("train_ratio", 0.8)),
            seed=int(params.get("seed", 42)),
            add_neg=bool(params.get("add_neg", True)),
            neg_ratio=float(params.get("neg_ratio", 0.2)),
            min_seq_len=int(params.get("min_seq_len", 200)),
            swap_frac_min=float(params.get("swap_frac_min", 0.10)),
            swap_frac_max=float(params.get("swap_frac_max", 0.30)),
            neg_prefix=str(params.get("neg_prefix", "NEG")),
        )
        summary["outputs"]["cnn"] = cnn_summary

    if "kd" in expanded_targets:
        kd_root = output_dir / "kd" if multi_layout else output_dir
        kd_entries = [_get_stage1_cache_entry(stage1_manifest, name) for name in kd_embeddings]
        kd_summary = _build_kd_dataset_from_cache(
            ids=ordered_ids,
            seq_by_id=seq_by_id,
            labels_by_id=labels_by_id,
            cache_entries=kd_entries,
            embedding_names=kd_embeddings,
            output_file=kd_root / "kd_fusion_dataset.pt",
            train_ratio=float(params.get("train_ratio", 0.8)),
            val_ratio=val_ratio,
            seed=int(params.get("seed", 42)),
            add_neg=bool(params.get("add_neg", True)),
            neg_ratio=float(params.get("neg_ratio", 0.2)),
            min_seq_len=int(params.get("min_seq_len", 200)),
            swap_frac_min=float(params.get("swap_frac_min", 0.10)),
            swap_frac_max=float(params.get("swap_frac_max", 0.30)),
            neg_prefix=str(params.get("neg_prefix", "NEG")),
        )
        kd_summary["root"] = str(kd_root)
        summary["outputs"]["kd"] = kd_summary

    if "oracle" in requested_targets:
        oracle_entry = _get_stage1_cache_entry(stage1_manifest, oracle_embedding)
        summary["outputs"]["oracle"] = {
            "oracle_kg_train_dir": str((output_dir / "kg" if multi_layout else output_dir) / "train"),
            "oracle_kg_val_dir": str((output_dir / "kg" if multi_layout else output_dir) / "val"),
            "embedding_name": oracle_embedding,
            "embedding_cache_dir": str(oracle_entry["dir"]),
            "embedding_level": str(oracle_entry.get("level", "")),
            "embedding_dim": int(oracle_entry.get("dim", 0)),
        }

    (output_dir / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _native_run_mibig_kg(params: Dict[str, object]) -> int:
    input_fasta = _require_file(str(params["input_fasta"]))
    out_dir = Path(str(params["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, seq_by_id, labels_by_id = _load_mibig_real_samples(input_fasta)
    return _package_kg_from_samples(
        ordered_ids=ids,
        seq_by_id=seq_by_id,
        labels_by_id=labels_by_id,
        output_dir=out_dir,
        train_ratio=float(params.get("train_ratio", 0.8)),
        seed=int(params.get("seed", 42)),
    )


def _native_run_mibig_cnn(params: Dict[str, object]) -> int:
    out_root = Path(str(params["output_dir"]))
    stage1_dir = out_root / "stage1"
    cnn_embedding = str(params.get("cnn_embedding", "esm2"))
    extract_targets = _parse_csv_option(str(params.get("extract", cnn_embedding)), default=[cnn_embedding])
    if cnn_embedding not in extract_targets:
        extract_targets.append(cnn_embedding)

    rc = _native_run_stage1_mibig(
        {
            "input_fasta": str(params["input_fasta"]),
            "output_dir": str(stage1_dir),
            "extract": ",".join(extract_targets),
            "gpus": str(params.get("gpus", "")),
            "esm_max_seq_len": int(params.get("max_seq_len", 1022)),
            "protbert_max_len": int(params.get("protbert_max_len", 1024)),
            "protbert_path": str(params.get("protbert_path", "local_assets/data/ProtBERT/ProtBERT")),
            "evo2_max_len": int(params.get("evo2_max_len", 16384)),
            "evo2_layer_name": str(params.get("evo2_layer_name", "blocks.21.mlp.l3")),
            "alphagenome_dir": str(params.get("alphagenome_dir", "")),
        }
    )
    if rc != 0:
        return rc

    return _native_run_stage2_pack(
        {
            "stage1_dir": str(stage1_dir),
            "output_dir": str(out_root),
            "targets": "cnn",
            "train_ratio": float(params.get("train_ratio", 0.8)),
            "val_ratio": float(params.get("val_ratio", 0.1)),
            "seed": int(params.get("seed", 42)),
            "cnn_embedding": cnn_embedding,
            "add_neg": bool(params.get("add_neg", True)),
            "neg_ratio": float(params.get("neg_ratio", 0.2)),
            "min_seq_len": int(params.get("min_seq_len", 200)),
            "swap_frac_min": float(params.get("swap_frac_min", 0.10)),
            "swap_frac_max": float(params.get("swap_frac_max", 0.30)),
            "neg_prefix": str(params.get("neg_prefix", "NEG")),
            "use_v2": bool(params.get("use_v2", True)),
            "max_seq_len": int(params.get("max_seq_len", 1022)),
            "gpus": str(params.get("gpus", "")),
        }
    )


def _native_run_mibig_oracle(params: Dict[str, object]) -> int:
    out_root = Path(str(params["output_dir"]))
    stage1_dir = out_root / "stage1"
    oracle_embedding = str(params.get("oracle_embedding", "protbert"))
    extract_targets = _parse_csv_option(str(params.get("extract", "esm2,protbert")), default=["esm2", "protbert"])
    if oracle_embedding not in extract_targets:
        extract_targets.append(oracle_embedding)

    stage1_params = {
        "input_fasta": str(params["input_fasta"]),
        "output_dir": str(stage1_dir),
        "extract": ",".join(extract_targets),
        "gpus": str(params.get("gpus", "")),
        "esm_max_seq_len": int(params.get("max_seq_len", 1022)),
        "protbert_max_len": int(params.get("protbert_max_len", 1024)),
        "protbert_path": str(params.get("protbert_path", "local_assets/data/ProtBERT/ProtBERT")),
        "evo2_max_len": int(params.get("evo2_max_len", 16384)),
        "evo2_layer_name": str(params.get("evo2_layer_name", "blocks.21.mlp.l3")),
        "alphagenome_dir": str(params.get("alphagenome_dir", "")),
    }
    rc = _native_run_stage1_mibig(stage1_params)
    if rc != 0:
        return rc

    return _native_run_stage2_pack(
        {
            "stage1_dir": str(stage1_dir),
            "output_dir": str(out_root),
            "targets": "oracle",
            "train_ratio": float(params.get("train_ratio", 0.8)),
            "val_ratio": float(params.get("val_ratio", 0.1)),
            "seed": int(params.get("seed", 42)),
            "cnn_embedding": str(params.get("cnn_embedding", "esm2")),
            "oracle_embedding": oracle_embedding,
            "kd_embeddings": str(params.get("kd_embeddings", "esm2,protbert")),
            "add_neg": bool(params.get("add_neg", True)),
            "neg_ratio": float(params.get("neg_ratio", 0.2)),
            "min_seq_len": int(params.get("min_seq_len", 200)),
            "swap_frac_min": float(params.get("swap_frac_min", 0.10)),
            "swap_frac_max": float(params.get("swap_frac_max", 0.30)),
            "neg_prefix": str(params.get("neg_prefix", "NEG")),
            "use_v2": bool(params.get("use_v2", True)),
            "max_seq_len": int(params.get("max_seq_len", 1022)),
            "gpus": str(params.get("gpus", "")),
        }
    )


NATIVE_RUNNERS = {
    "assets": _native_run_assets,
    "esm2": _native_run_esm2,
    "evo2": _native_run_evo2,
    "kg": _native_run_kg,
    "ee": _native_run_ee,
    "kd-fusion": _native_run_kd_fusion,
    "oracle-cnn": _native_run_oracle_cnn,
    "stage1-mibig": _native_run_stage1_mibig,
    "stage2-pack": _native_run_stage2_pack,
    "mibig-kg": _native_run_mibig_kg,
    "mibig-cnn": _native_run_mibig_cnn,
    "mibig-oracle": _native_run_mibig_oracle,
}


def run_with_manifest(
    *,
    pipeline: str,
    cmd: List[str],
    output_dir: Path,
    outputs: List[Path],
    input_files: List[Path],
    params: Dict[str, object],
    adapter_target: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rc = 0

    try:
        if params.get("_native"):
            rc = int(NATIVE_RUNNERS[pipeline](params))
        else:
            cp = run_cmd(cmd)
            rc = cp.returncode
    except Exception:
        traceback.print_exc()
        rc = 1

    duration = time.time() - start
    manifest = build_manifest(
        pipeline=pipeline,
        command=shlex.join(cmd),
        output_dir=output_dir,
        outputs=outputs,
        input_files=input_files,
        params=_public_params(params),
        status="success" if rc == 0 else "failed",
        return_code=rc,
        duration_sec=duration,
        adapter_target=adapter_target,
    )
    write_manifest(output_dir / "manifest.json", manifest)
    return rc


def cmd_assets(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "00_prepare_assets.sh"
    cmd = _legacy_sh_cmd(
        legacy,
        [
            "--env",
            args.env,
            *(["--download", " ".join(args.download_models)] if args.download_models else []),
        ],
    )
    params = {**vars(args), "_native": True}
    return cmd, [], [], params, legacy


def cmd_esm2(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "01_prepare_esm2_data.sh"
    cmd = _legacy_sh_cmd(
        legacy,
        [
            "--env",
            args.env,
            "--out-dir",
            args.output_dir,
            "--train-pos",
            args.train_pos_fasta,
            "--train-neg",
            args.train_neg_fasta,
            "--valid-pos",
            args.valid_pos_fasta,
            "--valid-neg",
            args.valid_neg_fasta,
            "--test-pos",
            args.test_pos_fasta,
            "--test-neg",
            args.test_neg_fasta,
            *(_maybe("--gpus") + _maybe(args.gpus) if args.gpus else []),
            *(["--mock"] if args.mock else []),
            *(["--mock-dim", str(args.mock_dim)] if args.mock_dim else []),
        ],
    )
    out = Path(args.output_dir)
    outputs = [
        out / "train_dataset_esm2.pt",
        out / "valid_dataset_esm2.pt",
        out / "test_dataset_esm2.pt",
    ]
    inputs = _paths(
        [
            args.train_pos_fasta,
            args.train_neg_fasta,
            args.valid_pos_fasta,
            args.valid_neg_fasta,
            args.test_pos_fasta,
            args.test_neg_fasta,
        ]
    )
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_evo2(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "03_prepare_evo2_data.sh"
    cmd = _legacy_sh_cmd(
        legacy,
        [
            "--env",
            args.env,
            "--work-dir",
            args.output_dir,
            "--train-pos-fasta",
            args.train_pos_fasta,
            "--train-neg-fasta",
            args.train_neg_fasta,
            "--test-pos-fasta",
            args.test_pos_fasta,
            "--test-neg-fasta",
            args.test_neg_fasta,
            *(["--gpus", args.gpus] if args.gpus else []),
            *(["--layer", args.layer] if args.layer else []),
            "--max-len",
            str(args.max_len),
            "--report-every",
            str(args.report_every),
            "--samples-per-shard",
            str(args.samples_per_shard),
            "--dtype",
            args.dtype,
            "--min-free-gb",
            str(args.min_free_gb),
            *(["--stop-on-low-disk"] if args.stop_on_low_disk else []),
            *(["--mock"] if args.mock else []),
            *(["--mock-dim", str(args.mock_dim)] if args.mock_dim else []),
        ],
    )
    out = Path(args.output_dir)
    outputs = [out / "train.pt", out / "test.pt", out / "emb"]
    inputs = _paths(
        [
            args.train_pos_fasta,
            args.train_neg_fasta,
            args.test_pos_fasta,
            args.test_neg_fasta,
        ]
    )
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_kg(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "05_prepare_kg_data.sh"
    cmd = _legacy_sh_cmd(
        legacy,
        [
            "--input-tsv",
            args.input_tsv,
            "--out-dir",
            args.output_dir,
            "--train-ratio",
            str(args.train_ratio),
            "--seed",
            str(args.seed),
        ],
    )
    out = Path(args.output_dir)
    outputs = [out / "train", out / "val"]
    inputs = _paths([args.input_tsv])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_ee(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "13_prepare_ee_data.sh"
    cmd = _legacy_sh_cmd(
        legacy,
        [
            "--env",
            args.env,
            *(["--gpus", args.gpus] if args.gpus else []),
            "--evo-emb-root",
            args.evo_emb_root,
            "--esm-train-dataset",
            args.esm_train_dataset,
            "--esm-valid-dataset",
            args.esm_valid_dataset,
            "--esm-test-dataset",
            args.esm_test_dataset,
            "--out-dir",
            args.output_dir,
            *(["--allow-missing"] if args.allow_missing else []),
        ],
    )
    out = Path(args.output_dir)
    outputs = [out / "combined_protein_dna_dataset.pt"]
    inputs = _paths(
        [
            args.esm_train_dataset,
            args.esm_valid_dataset,
            args.esm_test_dataset,
        ]
    )
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_kd_fusion(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "14_prepare_kd_fusion_data.py"
    cmd = _legacy_py_cmd(
        legacy,
        [
            "--evo-emb-root",
            args.evo_emb_root,
            "--esm-train",
            args.esm_train,
            "--esm-valid",
            args.esm_valid,
            "--esm-test",
            args.esm_test,
            "--protbert",
            args.protbert,
            "--train-pos-fasta",
            args.train_pos_fasta,
            "--train-neg-fasta",
            args.train_neg_fasta,
            "--test-pos-fasta",
            args.test_pos_fasta,
            "--test-neg-fasta",
            args.test_neg_fasta,
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--max-samples",
            str(args.max_samples),
            "--out",
            args.output_file,
        ],
    )
    out_file = Path(args.output_file)
    outputs = [out_file]
    inputs = _paths(
        [
            args.esm_train,
            args.esm_valid,
            args.esm_test,
            args.train_pos_fasta,
            args.train_neg_fasta,
            args.test_pos_fasta,
            args.test_neg_fasta,
        ]
    )
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_oracle_cnn(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    legacy = "ORACLE_CNN_dataset_v2.py" if args.use_v2 else "ORACLE_CNN_dataset.py"
    extra = ["--max_seq_len", str(args.max_seq_len)] if args.use_v2 else []
    gpus = getattr(args, "gpus", "")
    esm2_cache_dir = getattr(args, "esm2_cache_dir", "")
    if gpus:
        extra.extend(["--gpus", gpus])
    if esm2_cache_dir:
        extra.extend(["--esm2-cache-dir", esm2_cache_dir])
    cmd = _legacy_py_cmd(
        legacy,
        [
            "--label",
            args.label_file,
            "--train",
            args.train_fasta,
            "--valid",
            args.valid_fasta,
            "--outdir",
            args.output_dir,
            *extra,
        ],
    )
    out = Path(args.output_dir)
    outputs = [
        out / "label_dict.pt",
        out / "classes.pt",
        out / "oracle_train_dataset.pt",
        out / "oracle_valid_dataset.pt",
    ]
    inputs = _paths([args.label_file, args.train_fasta, args.valid_fasta])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, legacy


def cmd_stage1_mibig(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    this_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(this_file),
        "stage1-mibig",
        "--input-fasta",
        args.input_fasta,
        "--output-dir",
        args.output_dir,
        "--extract",
        args.extract,
        "--esm-max-seq-len",
        str(args.esm_max_seq_len),
        "--protbert-max-len",
        str(args.protbert_max_len),
        "--protbert-path",
        args.protbert_path,
        "--evo2-max-len",
        str(args.evo2_max_len),
        "--evo2-layer-name",
        args.evo2_layer_name,
        "--alphagenome-dir",
        args.alphagenome_dir,
    ]
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    out = Path(args.output_dir)
    outputs = [out / "samples.fasta", out / "labels.tsv", out / "stage1_manifest.json", out / "cache"]
    inputs = _paths([args.input_fasta])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, "MiBiG FASTA -> stage1 caches"


def cmd_stage2_pack(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    this_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(this_file),
        "stage2-pack",
        "--stage1-dir",
        args.stage1_dir,
        "--targets",
        args.targets,
        "--output-dir",
        args.output_dir,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--cnn-embedding",
        args.cnn_embedding,
        "--oracle-embedding",
        args.oracle_embedding,
        "--kd-embeddings",
        args.kd_embeddings,
        "--neg-ratio",
        str(args.neg_ratio),
        "--min-seq-len",
        str(args.min_seq_len),
        "--swap-frac-min",
        str(args.swap_frac_min),
        "--swap-frac-max",
        str(args.swap_frac_max),
        "--neg-prefix",
        args.neg_prefix,
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    if args.add_neg:
        cmd.append("--add-neg")
    if args.use_v2:
        cmd.append("--use-v2")
    out = Path(args.output_dir)
    outputs = [out / "stage2_summary.json"]
    inputs = _paths([args.stage1_dir])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, "stage1 -> target pack"


def cmd_mibig_kg(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    this_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(this_file),
        "mibig-kg",
        "--input-fasta",
        args.input_fasta,
        "--output-dir",
        args.output_dir,
        "--train-ratio",
        str(args.train_ratio),
        "--seed",
        str(args.seed),
    ]
    out = Path(args.output_dir)
    outputs = [out / "mibig_kg.tsv", out / "train", out / "val"]
    inputs = _paths([args.input_fasta])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, "MiBiG FASTA -> KG TSV+dirs"


def cmd_mibig_cnn(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    this_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(this_file),
        "mibig-cnn",
        "--input-fasta",
        args.input_fasta,
        "--output-dir",
        args.output_dir,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--extract",
        args.extract,
        "--cnn-embedding",
        args.cnn_embedding,
        "--neg-ratio",
        str(args.neg_ratio),
        "--min-seq-len",
        str(args.min_seq_len),
        "--swap-frac-min",
        str(args.swap_frac_min),
        "--swap-frac-max",
        str(args.swap_frac_max),
        "--neg-prefix",
        args.neg_prefix,
        "--max-seq-len",
        str(args.max_seq_len),
        "--protbert-max-len",
        str(args.protbert_max_len),
        "--protbert-path",
        args.protbert_path,
        "--evo2-max-len",
        str(args.evo2_max_len),
        "--evo2-layer-name",
        args.evo2_layer_name,
        "--alphagenome-dir",
        args.alphagenome_dir,
    ]
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    if args.add_neg:
        cmd.append("--add-neg")
    if args.use_v2:
        cmd.append("--use-v2")
    out = Path(args.output_dir)
    outputs = [
        out / "stage1" / "stage1_manifest.json",
        out / "oracle_cnn" / "oracle_train_dataset.pt",
        out / "oracle_cnn" / "oracle_valid_dataset.pt",
        out / "stage2_summary.json",
    ]
    inputs = _paths([args.input_fasta])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, "MiBiG FASTA -> oracle-cnn inputs+dataset"


def cmd_mibig_oracle(args: argparse.Namespace) -> Tuple[List[str], List[Path], List[Path], Dict[str, object], str]:
    this_file = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(this_file),
        "mibig-oracle",
        "--input-fasta",
        args.input_fasta,
        "--output-dir",
        args.output_dir,
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--extract",
        args.extract,
        "--cnn-embedding",
        args.cnn_embedding,
        "--oracle-embedding",
        args.oracle_embedding,
        "--kd-embeddings",
        args.kd_embeddings,
        "--neg-ratio",
        str(args.neg_ratio),
        "--min-seq-len",
        str(args.min_seq_len),
        "--swap-frac-min",
        str(args.swap_frac_min),
        "--swap-frac-max",
        str(args.swap_frac_max),
        "--neg-prefix",
        args.neg_prefix,
        "--max-seq-len",
        str(args.max_seq_len),
        "--protbert-max-len",
        str(args.protbert_max_len),
        "--protbert-path",
        args.protbert_path,
        "--evo2-max-len",
        str(args.evo2_max_len),
        "--evo2-layer-name",
        args.evo2_layer_name,
        "--alphagenome-dir",
        args.alphagenome_dir,
    ]
    if args.gpus:
        cmd.extend(["--gpus", args.gpus])
    if args.add_neg:
        cmd.append("--add-neg")
    if args.use_v2:
        cmd.append("--use-v2")
    out = Path(args.output_dir)
    outputs = [
        out / "stage1" / "stage1_manifest.json",
        out / "kg" / "mibig_kg.tsv",
        out / "kg" / "train",
        out / "kg" / "val",
        out / "stage2_summary.json",
    ]
    inputs = _paths([args.input_fasta])
    params = {**vars(args), "_native": True}
    return cmd, outputs, inputs, params, "MiBiG FASTA -> KG + oracle-cnn bundle"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified data-prep entrypoint (native in-file mode)")
    sp = p.add_subparsers(dest="pipeline", required=True)

    s = sp.add_parser("assets")
    s.add_argument("--env", default="evo2")
    s.add_argument("--download-models", nargs="*")

    s = sp.add_parser("esm2")
    s.add_argument("--env", default="evo2")
    s.add_argument("--train-pos-fasta", required=True)
    s.add_argument("--train-neg-fasta", required=True)
    s.add_argument("--valid-pos-fasta", required=True)
    s.add_argument("--valid-neg-fasta", required=True)
    s.add_argument("--test-pos-fasta", required=True)
    s.add_argument("--test-neg-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--gpus", default="")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--mock-dim", type=int, default=1280)

    s = sp.add_parser("evo2")
    s.add_argument("--env", default="evo2")
    s.add_argument("--train-pos-fasta", required=True)
    s.add_argument("--train-neg-fasta", required=True)
    s.add_argument("--test-pos-fasta", required=True)
    s.add_argument("--test-neg-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--gpus", default="")
    s.add_argument("--layer", default="blocks.21.mlp.l3")
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--report-every", type=int, default=50)
    s.add_argument("--samples-per-shard", type=int, default=64)
    s.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    s.add_argument("--min-free-gb", type=int, default=50)
    s.add_argument("--stop-on-low-disk", action="store_true")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--mock-dim", type=int, default=1920)

    s = sp.add_parser("kg")
    s.add_argument("--input-tsv", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=42)

    s = sp.add_parser("ee")
    s.add_argument("--env", default="evo2")
    s.add_argument("--gpus", default="")
    s.add_argument("--evo-emb-root", required=True)
    s.add_argument("--esm-train-dataset", required=True)
    s.add_argument("--esm-valid-dataset", default="")
    s.add_argument("--esm-test-dataset", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--allow-missing", action="store_true")

    s = sp.add_parser("kd-fusion")
    s.add_argument("--evo-emb-root", required=True)
    s.add_argument("--esm-train", required=True)
    s.add_argument("--esm-valid", default="")
    s.add_argument("--esm-test", required=True)
    s.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    s.add_argument("--train-pos-fasta", required=True)
    s.add_argument("--train-neg-fasta", required=True)
    s.add_argument("--test-pos-fasta", required=True)
    s.add_argument("--test-neg-fasta", required=True)
    s.add_argument("--val-ratio", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--max-samples", type=int, default=-1)
    s.add_argument("--output-file", required=True)

    s = sp.add_parser("oracle-cnn")
    s.add_argument("--label-file", required=True)
    s.add_argument("--train-fasta", required=True)
    s.add_argument("--valid-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--use-v2", action="store_true")
    s.add_argument("--max-seq-len", type=int, default=1022)
    s.add_argument("--gpus", default="")
    s.add_argument("--esm2-cache-dir", default="")

    s = sp.add_parser("stage1-mibig")
    s.add_argument("--input-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--extract", default="esm2,protbert")
    s.add_argument("--esm-max-seq-len", type=int, default=1022)
    s.add_argument("--protbert-max-len", type=int, default=1024)
    s.add_argument("--protbert-path", default="local_assets/data/ProtBERT/ProtBERT")
    s.add_argument("--evo2-max-len", type=int, default=16384)
    s.add_argument("--evo2-layer-name", default="blocks.21.mlp.l3")
    s.add_argument("--alphagenome-dir", default="")
    s.add_argument("--gpus", default="")

    s = sp.add_parser("stage2-pack")
    s.add_argument("--stage1-dir", required=True)
    s.add_argument("--targets", default="cnn,kd,oracle")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--val-ratio", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--cnn-embedding", default="esm2")
    s.add_argument("--oracle-embedding", default="protbert")
    s.add_argument("--kd-embeddings", default="esm2,protbert")
    s.add_argument("--add-neg", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--neg-ratio", type=float, default=0.2)
    s.add_argument("--min-seq-len", type=int, default=200)
    s.add_argument("--swap-frac-min", type=float, default=0.10)
    s.add_argument("--swap-frac-max", type=float, default=0.30)
    s.add_argument("--neg-prefix", default="NEG")
    s.add_argument("--use-v2", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--max-seq-len", type=int, default=1022)
    s.add_argument("--gpus", default="")

    s = sp.add_parser("mibig-kg")
    s.add_argument("--input-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=42)

    s = sp.add_parser("mibig-cnn")
    s.add_argument("--input-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--val-ratio", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--extract", default="esm2")
    s.add_argument("--cnn-embedding", default="esm2")
    s.add_argument("--add-neg", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--neg-ratio", type=float, default=0.2)
    s.add_argument("--min-seq-len", type=int, default=200)
    s.add_argument("--swap-frac-min", type=float, default=0.10)
    s.add_argument("--swap-frac-max", type=float, default=0.30)
    s.add_argument("--neg-prefix", default="NEG")
    s.add_argument("--use-v2", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--max-seq-len", type=int, default=1022)
    s.add_argument("--protbert-max-len", type=int, default=1024)
    s.add_argument("--protbert-path", default="local_assets/data/ProtBERT/ProtBERT")
    s.add_argument("--evo2-max-len", type=int, default=16384)
    s.add_argument("--evo2-layer-name", default="blocks.21.mlp.l3")
    s.add_argument("--alphagenome-dir", default="")
    s.add_argument("--gpus", default="")

    s = sp.add_parser("mibig-oracle")
    s.add_argument("--input-fasta", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--train-ratio", type=float, default=0.8)
    s.add_argument("--val-ratio", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--extract", default="esm2,protbert")
    s.add_argument("--cnn-embedding", default="esm2")
    s.add_argument("--oracle-embedding", default="protbert")
    s.add_argument("--kd-embeddings", default="esm2,protbert")
    s.add_argument("--add-neg", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--neg-ratio", type=float, default=0.2)
    s.add_argument("--min-seq-len", type=int, default=200)
    s.add_argument("--swap-frac-min", type=float, default=0.10)
    s.add_argument("--swap-frac-max", type=float, default=0.30)
    s.add_argument("--neg-prefix", default="NEG")
    s.add_argument("--use-v2", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--max-seq-len", type=int, default=1022)
    s.add_argument("--protbert-max-len", type=int, default=1024)
    s.add_argument("--protbert-path", default="local_assets/data/ProtBERT/ProtBERT")
    s.add_argument("--evo2-max-len", type=int, default=16384)
    s.add_argument("--evo2-layer-name", default="blocks.21.mlp.l3")
    s.add_argument("--alphagenome-dir", default="")
    s.add_argument("--gpus", default="")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "assets": cmd_assets,
        "esm2": cmd_esm2,
        "evo2": cmd_evo2,
        "kg": cmd_kg,
        "ee": cmd_ee,
        "kd-fusion": cmd_kd_fusion,
        "oracle-cnn": cmd_oracle_cnn,
        "stage1-mibig": cmd_stage1_mibig,
        "stage2-pack": cmd_stage2_pack,
        "mibig-kg": cmd_mibig_kg,
        "mibig-cnn": cmd_mibig_cnn,
        "mibig-oracle": cmd_mibig_oracle,
    }

    cmd, outputs, inputs, params, adapter_target = dispatch[args.pipeline](args)

    if args.pipeline == "kd-fusion":
        manifest_dir = Path(args.output_file).resolve().parent
    elif args.pipeline == "assets":
        manifest_dir = Path(".").resolve() / "data_prep_unified_assets_run"
    else:
        manifest_dir = Path(getattr(args, "output_dir")).resolve()

    return run_with_manifest(
        pipeline=args.pipeline,
        cmd=cmd,
        output_dir=manifest_dir,
        outputs=outputs,
        input_files=inputs,
        params=params,
        adapter_target=adapter_target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
