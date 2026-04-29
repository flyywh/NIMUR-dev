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
from oracle_cnn_predict import CNNClassifierModel, infer_num_classes_from_state_dict, load_class_names, load_state_dict, resolve_device
from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder


def parse_args():
    ap = argparse.ArgumentParser(description="CNN genome detector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--genome-fna", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    ap.add_argument("--window", type=int, default=12000)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--merge-gap", type=int, default=1500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--esm-max-len", type=int, default=1022)
    ap.add_argument("--max-seq-len", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=8)
    return ap.parse_args()


def load_model(config_path: Path, model_path: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    state_dict = load_state_dict(model_path)
    num_classes = infer_num_classes_from_state_dict(state_dict)
    model = CNNClassifierModel(**config["model"], num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    class_names = load_class_names(model_path, None, num_classes)
    return model, class_names, int(config["model"]["input_dim"])


def score_probabilities(probs: np.ndarray, class_names: list[str]):
    if probs.ndim == 0:
        probs = np.asarray([float(probs)], dtype=np.float32)
    if len(class_names) == 1:
        return float(probs[0]), class_names[0], float("nan")
    pos_indices = [idx for idx, name in enumerate(class_names) if name.lower() != "neg"]
    neg_idx = next((idx for idx, name in enumerate(class_names) if name.lower() == "neg"), None)
    if not pos_indices:
        return float(np.max(probs)), class_names[int(np.argmax(probs))], float(probs[neg_idx]) if neg_idx is not None else float("nan")
    pos_probs = probs[pos_indices]
    best_local = int(np.argmax(pos_probs))
    best_idx = pos_indices[best_local]
    return float(pos_probs[best_local]), class_names[best_idx], float(probs[neg_idx]) if neg_idx is not None else float("nan")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, class_names, feature_dim = load_model(Path(args.config), Path(args.model), device)

    esm_embedder = ESM2Embedder(device=str(device), layer=33)
    prot_embedder = ProtBERTEmbedder(args.protbert, device=str(device))

    seqs = [(str(r.id), str(r.seq).upper()) for r in SeqIO.parse(str(args.genome_fna), "fasta")]
    if not seqs:
        raise ValueError("No sequences in genome fna")

    windows = []
    for cid, seq in seqs:
        windows.extend(make_windows(cid, seq, int(args.window), int(args.step)))

    preds: list[WindowPred] = []
    window_rows = []
    for start_idx in range(0, len(windows), int(args.batch_size)):
        batch = windows[start_idx : start_idx + int(args.batch_size)]
        feats = []
        ids = []
        meta = []
        for cid, start, end, sub in batch:
            sid = f"{cid}:{start}-{end}"
            esm_vec = esm_embedder.encode_mean_chunked(sid, sub, chunk_len=int(args.esm_max_len)).astype(np.float32)
            prot_vec = prot_embedder.encode_mean(sid, sub, max_len=int(args.max_seq_len)).astype(np.float32)
            feat = np.concatenate([esm_vec, prot_vec], axis=0).astype(np.float32)
            if feat.shape[0] != feature_dim:
                raise RuntimeError(f"feature_dim mismatch {feat.shape[0]} vs {feature_dim}")
            feats.append(feat)
            ids.append(sid)
            meta.append((cid, start, end))

        x = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x).detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        for sid, (cid, start, end), prob_vec in zip(ids, meta, probs):
            bgc_score, top_type, neg_score = score_probabilities(prob_vec, class_names)
            preds.append(WindowPred(contig=cid, start=start, end=end, seq_id=sid, bgc_score=bgc_score, top_type=top_type, neg_score=neg_score))
            row = {
                "contig": cid,
                "window_start": start,
                "window_end": end,
                "seq_id": sid,
                "bgc_score": bgc_score,
                "top_type": top_type,
                "neg_score": neg_score,
            }
            for name, value in zip(class_names, prob_vec.tolist()):
                row[f"pred_{name}"] = float(value)
            window_rows.append(row)
        done = min(start_idx + len(batch), len(windows))
        if done % 10 == 0 or done == len(windows):
            print(f"[CNN-GENOME] processed {done}/{len(windows)} windows")

    win_csv = out_dir / "window_predictions.csv"
    with win_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["contig", "window_start", "window_end", "seq_id", "bgc_score", "top_type", "neg_score"] + [f"pred_{name}" for name in class_names]
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
    print(f"[CNN-GENOME] done: {out_dir}")


if __name__ == "__main__":
    main()
