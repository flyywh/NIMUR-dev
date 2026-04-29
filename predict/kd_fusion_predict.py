#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kd_fusion_common import build_models, logits_to_probs, metrics_from_logits


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate/infer KD fusion checkpoint on a prepared .pt dataset split")
    ap.add_argument("--model", required=True, help="Path to kd_best.pt")
    ap.add_argument("--features", required=True, help="Path to dataset .pt produced by build_esm_protbert_dataset.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--metrics-out", default="", help="Optional output JSON path for metrics")
    ap.add_argument("--scorer", default="student", choices=["student", "teacher"])
    ap.add_argument("--batch-size", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    obj = torch.load(args.features, map_location="cpu", weights_only=False)

    if args.split not in obj:
        raise KeyError(f"Split {args.split!r} not found in {args.features}")

    split_obj = obj[args.split]
    X = split_obj["X"].float()
    y = split_obj["y"].float()
    ids = split_obj.get("ids")
    classes = None
    if isinstance(obj.get("meta"), dict):
        classes = obj["meta"].get("classes")
    if not isinstance(classes, list):
        classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    if ids is None:
        ids = [f"{args.split}_{idx}" for idx in range(len(X))]

    teacher, student = build_models(ckpt, in_dim=int(X.shape[1]), device=device)
    scorer = teacher if args.scorer == "teacher" else student

    loader = DataLoader(TensorDataset(X, y), batch_size=int(args.batch_size), shuffle=False)

    logits_list = []
    y_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = scorer(xb)
            logits_list.append(out.detach().cpu().numpy())
            y_list.append(yb.detach().cpu().numpy())

    logits = np.concatenate(logits_list).astype(np.float32)
    y_true = np.concatenate(y_list).astype(np.float32)
    probs = logits_to_probs(logits)
    metrics = metrics_from_logits(y_true, logits)
    metrics.update(
        {
            "split": args.split,
            "scorer": args.scorer,
            "num_samples": int(len(y_true)),
            "feature_dim": int(X.shape[1]),
            "classes": classes,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if y_true.ndim == 1:
            writer.writerow(["seq_id", "label", "prediction", "logit", "split", "scorer"])
            for sid, yy, pp, lg in zip(ids, y_true.astype(int), probs, logits):
                writer.writerow([sid, int(yy), f"{float(pp):.6f}", f"{float(lg):.6f}", args.split, args.scorer])
        else:
            class_names = classes if isinstance(classes, list) and len(classes) == y_true.shape[1] else [f"class_{idx}" for idx in range(y_true.shape[1])]
            writer.writerow(
                ["seq_id", "split", "scorer"]
                + [f"label_{name}" for name in class_names]
                + [f"pred_{name}" for name in class_names]
                + [f"logit_{name}" for name in class_names]
            )
            pred_labels = (probs >= 0.5).astype(int)
            for sid, yy, pp, lg, pred in zip(ids, y_true.astype(int), probs, logits, pred_labels):
                writer.writerow(
                    [sid, args.split, args.scorer]
                    + yy.tolist()
                    + [f"{float(v):.6f}" for v in pp.tolist()]
                    + [f"{float(v):.6f}" for v in lg.tolist()]
                )

    metrics_path = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[KD-PREDICT] predictions: {out_path}")
    print(f"[KD-PREDICT] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
