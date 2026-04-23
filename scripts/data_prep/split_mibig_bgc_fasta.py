#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO


NAME_MAP = {
    "nrps": "NRPS",
    "other": "Other",
    "pks": "PKS",
    "ribosomal": "Ribosomal",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split MiBiG-formatted DNA BGC FASTA into train/val/test positive and synthetic negative FASTAs."
    )
    p.add_argument("--input-fasta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--add-neg", action="store_true")
    p.add_argument("--neg-ratio", type=float, default=0.2)
    p.add_argument("--min-seq-len", type=int, default=200)
    p.add_argument("--swap-frac-min", type=float, default=0.10)
    p.add_argument("--swap-frac-max", type=float, default=0.30)
    p.add_argument("--neg-prefix", default="NEG")
    p.add_argument("--prefix", default="MiBiG")
    return p.parse_args()


def extract_id_and_labels(desc: str) -> tuple[str | None, set[str]]:
    m_id = re.search(r"BGC_ID=(BGC\d+)", desc)
    m_cls = re.search(r"bgc_class=([^|]+)", desc)
    if not m_id or not m_cls:
        return None, set()

    bgc_id = m_id.group(1)
    labels: set[str] = set()
    for part in m_cls.group(1).split(";"):
        m = re.search(r"class:([^,;]+)", part)
        if not m:
            continue
        cls = m.group(1).strip().lower()
        mapped = NAME_MAP.get(cls)
        if mapped:
            labels.add(mapped)
    return bgc_id, labels


def load_real_samples(in_fasta: Path) -> tuple[list[str], dict[str, str], dict[str, set[str]]]:
    seq_by_id: dict[str, str] = {}
    labels_by_id: dict[str, set[str]] = defaultdict(set)
    for rec in SeqIO.parse(str(in_fasta), "fasta"):
        bgc_id, labels = extract_id_and_labels(rec.description)
        if not bgc_id or not labels:
            continue
        if bgc_id not in seq_by_id:
            seq_by_id[bgc_id] = str(rec.seq).upper()
        labels_by_id[bgc_id].update(labels)

    ids = [sample_id for sample_id in seq_by_id if sample_id in labels_by_id and labels_by_id[sample_id]]
    return ids, seq_by_id, labels_by_id


def make_chimera(seq_a: str, seq_b: str, frac_min: float, frac_max: float, rng: random.Random) -> str:
    max_len = min(len(seq_a), len(seq_b))
    seg_len = max(1, int(max_len * rng.uniform(frac_min, frac_max)))
    seg_len = min(seg_len, len(seq_a), len(seq_b))
    start_a = rng.randint(0, len(seq_a) - seg_len)
    start_b = rng.randint(0, len(seq_b) - seg_len)
    seg_b = seq_b[start_b : start_b + seg_len]
    return seq_a[:start_a] + seg_b + seq_a[start_a + seg_len :]


def add_negative_samples(
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

        seq_by_id[neg_id] = make_chimera(
            seq_by_id[a],
            seq_by_id[b],
            swap_frac_min,
            swap_frac_max,
            rng,
        )
        labels_by_id[neg_id] = {"neg"}
        new_ids.append(neg_id)
    return new_ids


def split_ids(ids: list[str], train_ratio: float, val_ratio: float, rng: random.Random) -> dict[str, list[str]]:
    data = list(ids)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n >= 3:
        n_test = max(1, n_test)
        if n_train + n_val + n_test > n:
            n_train = max(1, n - n_val - n_test)
        if n_val == 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_test = max(1, n_test - 1)
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def write_fasta(path: Path, ids: list[str], seq_by_id: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample_id in ids:
            f.write(f">{sample_id}\n{seq_by_id[sample_id]}\n")


def write_manifest(
    path: Path,
    *,
    input_fasta: Path,
    pos_split: dict[str, list[str]],
    neg_split: dict[str, list[str]],
    labels_by_id: dict[str, set[str]],
) -> None:
    payload = {
        "input_fasta": str(input_fasta),
        "splits": {
            split: {"pos": len(pos_split[split]), "neg": len(neg_split[split])}
            for split in ("train", "val", "test")
        },
        "positive_label_counts": dict(
            sorted(
                Counter(label for sample_id, labels in labels_by_id.items() if "neg" not in labels for label in labels).items()
            )
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_index_tsv(
    path: Path,
    *,
    pos_split: dict[str, list[str]],
    neg_split: dict[str, list[str]],
    labels_by_id: dict[str, set[str]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("seq_id\tsplit\tbinary_label\tclasses\n")
        for split in ("train", "val", "test"):
            for sample_id in pos_split[split]:
                classes = ",".join(sorted(labels_by_id[sample_id]))
                f.write(f"{sample_id}\t{split}\t1\t{classes}\n")
            for sample_id in neg_split[split]:
                f.write(f"{sample_id}\t{split}\t0\tneg\n")


def main() -> int:
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be in (0, 1)")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be < 1")

    rng = random.Random(args.seed)
    input_fasta = Path(args.input_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_ids, seq_by_id, labels_by_id = load_real_samples(input_fasta)
    if not pos_ids:
        raise ValueError(f"No usable BGC records parsed from {input_fasta}")

    neg_ids: list[str] = []
    if args.add_neg:
        neg_ids = add_negative_samples(
            pos_ids,
            seq_by_id,
            labels_by_id,
            neg_ratio=args.neg_ratio,
            min_seq_len=args.min_seq_len,
            swap_frac_min=args.swap_frac_min,
            swap_frac_max=args.swap_frac_max,
            neg_prefix=args.neg_prefix,
            rng=rng,
        )

    pos_split = split_ids(pos_ids, args.train_ratio, args.val_ratio, rng)
    neg_split = split_ids(neg_ids, args.train_ratio, args.val_ratio, rng) if neg_ids else {
        "train": [],
        "val": [],
        "test": [],
    }

    for split in ("train", "val", "test"):
        write_fasta(output_dir / f"{args.prefix}_{split}_pos.fasta", pos_split[split], seq_by_id)
        write_fasta(output_dir / f"{args.prefix}_{split}_neg.fasta", neg_split[split], seq_by_id)

    write_manifest(
        output_dir / f"{args.prefix}_split_manifest.json",
        input_fasta=input_fasta,
        pos_split=pos_split,
        neg_split=neg_split,
        labels_by_id=labels_by_id,
    )
    write_index_tsv(
        output_dir / f"{args.prefix}_split_index.tsv",
        pos_split=pos_split,
        neg_split=neg_split,
        labels_by_id=labels_by_id,
    )

    print(json.dumps({
        "input": str(input_fasta),
        "positives": len(pos_ids),
        "negatives": len(neg_ids),
        "train_pos": len(pos_split["train"]),
        "train_neg": len(neg_split["train"]),
        "val_pos": len(pos_split["val"]),
        "val_neg": len(neg_split["val"]),
        "test_pos": len(pos_split["test"]),
        "test_neg": len(neg_split["test"]),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
