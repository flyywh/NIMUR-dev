#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from genome_detect_common import WindowPred, dump_summary, make_windows, merge_regions, write_regions_csv
from oracle_predict import build_model, build_runtime_args, encode_sequences, load_trainer_module


def parse_args():
    ap = argparse.ArgumentParser(description="ORACLE genome detector")
    ap.add_argument("--model", required=True, help="best.pt")
    ap.add_argument("--genome-fna", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protbert-path", default="local_assets/data/ProtBERT/ProtBERT")
    ap.add_argument("--window", type=int, default=12000)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--merge-gap", type=int, default=1500)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--embedding-cache-dir", default="")
    ap.add_argument("--esm-max-len", type=int, default=1022)
    ap.add_argument("--protbert-max-len", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=8)
    return ap.parse_args()


def sigmoid_prob(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def score_distances(dist_row: np.ndarray, class_names: list[str], top_k: int):
    pos_indices = [idx for idx, name in enumerate(class_names) if name.lower() != "neg"]
    if not pos_indices:
        raise ValueError("ORACLE classes do not contain positive types")
    neg_idx = next((idx for idx, name in enumerate(class_names) if name.lower() == "neg"), None)
    pos_dists = dist_row[pos_indices]
    best_local = int(np.argmin(pos_dists))
    best_idx = pos_indices[best_local]
    best_pos_dist = float(pos_dists[best_local])
    neg_dist = float(dist_row[neg_idx]) if neg_idx is not None else float("nan")
    bgc_score = sigmoid_prob(neg_dist - best_pos_dist) if neg_idx is not None else sigmoid_prob(-best_pos_dist)
    ranked_pos = sorted(((class_names[idx], float(dist_row[idx])) for idx in pos_indices), key=lambda x: x[1])[:top_k]
    return bgc_score, class_names[best_idx], best_pos_dist, neg_dist, ranked_pos


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer_module()
    device = resolve_device(args.device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "type2id" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {args.model}")

    runtime_args = build_runtime_args(ckpt.get("args", {}), args)
    if not args.embedding_cache_dir:
        runtime_args.embedding_cache_dir = ""
        runtime_args.protbert_cache_dir = ""
    model, tokenizer, type2id = build_model(trainer, ckpt, runtime_args, device)
    id2type = {idx: name for name, idx in type2id.items()}
    class_names = [id2type[idx] for idx in sorted(id2type)]

    seqs = [(str(r.id), str(r.seq).upper()) for r in SeqIO.parse(str(args.genome_fna), "fasta")]
    if not seqs:
        raise ValueError("No sequences in genome fna")

    windows = []
    for cid, seq in seqs:
        windows.extend(make_windows(cid, seq, int(args.window), int(args.step)))

    preds: list[WindowPred] = []
    window_rows = []
    all_type_ids = torch.arange(model.type_emb.num_embeddings, device=device)
    all_type_vecs = model.type_emb(all_type_ids)
    rel = model.rel.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        for start_idx in range(0, len(windows), int(args.batch_size)):
            batch = windows[start_idx : start_idx + int(args.batch_size)]
            headers = [f"{cid}:{start}-{end}" for cid, start, end, _sub in batch]
            sequences = [sub for _cid, _start, _end, sub in batch]
            meta = [(cid, start, end) for cid, start, end, _sub in batch]
            prot_vecs = encode_sequences(trainer, model, tokenizer, headers, sequences, runtime_args, device)
            dist_all = torch.norm(prot_vecs.unsqueeze(1) + rel - all_type_vecs.unsqueeze(0), p=2, dim=-1).detach().cpu().numpy()
            for sid, (cid, start, end), dist_row in zip(headers, meta, dist_all):
                bgc_score, top_type, best_pos_dist, neg_dist, ranked_pos = score_distances(dist_row, class_names, int(args.top_k))
                preds.append(WindowPred(contig=cid, start=start, end=end, seq_id=sid, bgc_score=bgc_score, top_type=top_type, neg_score=neg_dist))
                window_rows.append(
                    {
                        "contig": cid,
                        "window_start": start,
                        "window_end": end,
                        "seq_id": sid,
                        "bgc_score": bgc_score,
                        "top_type": top_type,
                        "best_pos_distance": best_pos_dist,
                        "neg_distance": neg_dist,
                        "topk_json": json.dumps(ranked_pos, ensure_ascii=False),
                    }
                )
            done = min(start_idx + len(batch), len(windows))
            if done % 10 == 0 or done == len(windows):
                print(f"[ORACLE-GENOME] processed {done}/{len(windows)} windows")

    win_csv = out_dir / "window_predictions.csv"
    with win_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["contig", "window_start", "window_end", "seq_id", "bgc_score", "top_type", "best_pos_distance", "neg_distance", "topk_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)

    regions = merge_regions(preds, threshold=float(args.threshold), max_gap=int(args.merge_gap))
    reg_csv = out_dir / "bgc_regions.csv"
    write_regions_csv(reg_csv, regions)

    dump_summary(
        out_dir / "summary.json",
        {
            "model": str(Path(args.model).resolve()),
            "genome_fna": str(Path(args.genome_fna).resolve()),
            "classes": class_names,
            "num_contigs": len(seqs),
            "num_windows": len(preds),
            "num_regions": len(regions),
            "window": int(args.window),
            "step": int(args.step),
            "threshold": float(args.threshold),
            "merge_gap": int(args.merge_gap),
            "window_predictions_csv": str(win_csv.resolve()),
            "bgc_regions_csv": str(reg_csv.resolve()),
        },
    )
    print(f"[ORACLE-GENOME] done: {out_dir}")


if __name__ == "__main__":
    main()
