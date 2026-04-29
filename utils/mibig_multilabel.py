from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def normalize_seq_id(seq_id: str) -> str:
    seq_id = seq_id.strip()
    if seq_id.startswith("BGC") or seq_id.startswith("NEG"):
        return seq_id[:12]
    return seq_id


def load_split_index_labels(path: str | Path):
    path = Path(path)
    labels_by_id: dict[str, list[str]] = {}
    classes: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"seq_id", "classes"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            seq_id = normalize_seq_id((row.get("seq_id") or ""))
            classes_raw = (row.get("classes") or "").strip()
            if not seq_id or not classes_raw:
                continue
            labels = [label.strip() for label in classes_raw.split(",") if label.strip()]
            if not labels:
                continue
            labels_by_id[seq_id] = labels
            classes.update(labels)
    if not labels_by_id:
        raise ValueError(f"No valid labels parsed from {path}")
    return labels_by_id, sorted(classes)


def build_multihot_labels(ids: list[str], labels_by_id: dict[str, list[str]], classes: list[str]) -> np.ndarray:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    label_matrix = np.zeros((len(ids), len(classes)), dtype=np.float32)
    missing: list[str] = []
    for row_idx, seq_id in enumerate(ids):
        key = normalize_seq_id(seq_id)
        labels = labels_by_id.get(key)
        if labels is None:
            missing.append(seq_id)
            continue
        for label in labels:
            if label not in class_to_idx:
                continue
            label_matrix[row_idx, class_to_idx[label]] = 1.0
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing labels for {len(missing)} ids, e.g. {preview}")
    return label_matrix
