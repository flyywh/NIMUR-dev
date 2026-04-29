#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kd_fusion_common import build_models, logits_to_probs
from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder, parse_fasta_rows


def parse_args():
    ap = argparse.ArgumentParser(description="Infer KD fusion checkpoint on FASTA using ESM2 + ProtBERT features")
    ap.add_argument("--model", required=True, help="Path to kd_best.pt")
    ap.add_argument("--fasta", required=True, help="Input FASTA")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--scorer", default="student", choices=["student", "teacher"])
    ap.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    ap.add_argument("--esm-chunk-len", type=int, default=1022)
    ap.add_argument("--protbert-max-len", type=int, default=1024)
    ap.add_argument("--esm-dim", type=int, default=1280)
    ap.add_argument("--prot-dim", type=int, default=1024)
    ap.add_argument("--strict-dims", action="store_true")
    return ap.parse_args()


def infer_dims(feature_dim: int, esm_dim: int, prot_dim: int, strict: bool):
    total = esm_dim + prot_dim
    if total == feature_dim:
        return esm_dim, prot_dim
    if strict:
        raise ValueError(
            f"Dim mismatch: esm_dim+prot_dim={total}, checkpoint feature_dim={feature_dim}. "
            "Adjust --esm-dim/--prot-dim."
        )
    inferred_prot = feature_dim - esm_dim
    if inferred_prot <= 0:
        raise ValueError(
            f"Cannot infer prot_dim from feature_dim={feature_dim}, esm_dim={esm_dim}"
        )
    return esm_dim, inferred_prot


def trim_or_pad(vec: np.ndarray, dim: int) -> np.ndarray:
    if vec.shape[0] >= dim:
        return vec[:dim]
    return np.pad(vec, (0, dim - vec.shape[0]))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    feature_dim = int(ckpt["feature_dim"])
    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    esm_dim, prot_dim = infer_dims(feature_dim, int(args.esm_dim), int(args.prot_dim), bool(args.strict_dims))

    teacher, student = build_models(ckpt, in_dim=feature_dim, device=device)
    scorer = teacher if args.scorer == "teacher" else student

    rows = parse_fasta_rows(Path(args.fasta))
    if not rows:
        raise ValueError(f"No valid sequences in {args.fasta}")

    esm_embedder = ESM2Embedder(device=device, layer=33)
    prot_embedder = ProtBERTEmbedder(args.protbert, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        num_labels = int(ckpt.get("num_labels", 1)) if isinstance(ckpt, dict) else 1
        class_names = classes if isinstance(classes, list) and len(classes) == num_labels else [f"class_{idx}" for idx in range(num_labels)]
        if num_labels == 1:
            writer.writerow(["seq_id", "prediction", "scorer", "feature_dim"])
        else:
            writer.writerow(["seq_id", "scorer", "feature_dim"] + [f"pred_{name}" for name in class_names])

        for sid, seq in rows:
            try:
                esm_vec = esm_embedder.encode_mean_chunked(
                    sid,
                    seq,
                    chunk_len=int(args.esm_chunk_len),
                ).astype(np.float32)
                prot_vec = prot_embedder.encode_mean(
                    sid,
                    seq,
                    max_len=int(args.protbert_max_len),
                ).astype(np.float32)
                feat = np.concatenate(
                    [trim_or_pad(esm_vec, esm_dim), trim_or_pad(prot_vec, prot_dim)],
                    axis=0,
                ).astype(np.float32)

                if feat.shape[0] != feature_dim:
                    raise RuntimeError(
                        f"Feature dim mismatch: got={feat.shape[0]} expected={feature_dim}"
                    )

                x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logit = scorer(x).detach().cpu().numpy().reshape(-1)
                probs = logits_to_probs(logit)
                if num_labels == 1:
                    writer.writerow([sid, f"{float(probs[0]):.6f}", args.scorer, feature_dim])
                else:
                    writer.writerow([sid, args.scorer, feature_dim] + [f"{float(v):.6f}" for v in probs.tolist()])
            except Exception as ex:
                if num_labels == 1:
                    writer.writerow([sid, "ERROR", args.scorer, feature_dim])
                else:
                    writer.writerow([sid, args.scorer, feature_dim] + ["ERROR" for _ in range(num_labels)])
                print(f"[KD-FASTA-PREDICT] failed for {sid}: {ex}")

    print(f"[KD-FASTA-PREDICT] done: {out_path}")


if __name__ == "__main__":
    main()
