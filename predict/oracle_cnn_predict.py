#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder, parse_fasta_rows


class ResidualDSConvSEBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.Dropout(dropout),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        out = self.depthwise(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = out * self.se(out)
        out = self.pointwise(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        return residual + out


class CNNClassifierModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
        seq_len: int = 512,
        num_classes: int = 1,
    ):
        super().__init__()
        del num_heads
        self.input_dim = input_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.feature_proj = nn.Linear(input_dim, seq_len * ffn_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, ffn_dim) * 0.02)
        self.input_proj = nn.Conv1d(ffn_dim, ffn_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualDSConvSEBlock(ffn_dim, kernel_size=7, dropout=dropout) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim // 2, num_classes),
        )

    def forward(self, x, mask=None):
        del mask
        batch_size, _ = x.size()
        x = self.feature_proj(x)
        x = x.reshape(batch_size, self.seq_len, self.ffn_dim).contiguous()
        x = x + self.pos_embed
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for ORACLE CNN classifier")
    parser.add_argument("--config", required=True, help="Path to oracle_cnn_config.json")
    parser.add_argument("--model", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fasta", default="", help="Input FASTA for online feature extraction")
    parser.add_argument("--dataset-dir", default="", help="Chunked dataset directory, e.g. data/cnn_dataset/test")
    parser.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--esm-max-len", type=int, default=1022)
    parser.add_argument("--max-seq-len", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def infer_num_classes_from_state_dict(state_dict) -> int:
    weight = state_dict.get("classifier.3.weight") if isinstance(state_dict, dict) else None
    if torch.is_tensor(weight) and weight.ndim == 2:
        return int(weight.shape[0])
    return 1


def load_class_names(model_path: Path, dataset_dir: Path | None, num_classes: int):
    candidates = []
    if dataset_dir is not None:
        candidates.append(dataset_dir.parent / "metadata.json")
    candidates.append(model_path.parent / "classes.json")
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == num_classes:
            return data
        if isinstance(data, dict):
            classes = data.get("classes")
            if isinstance(classes, list) and len(classes) == num_classes:
                return classes
    if num_classes == 1:
        return ["positive"]
    return [f"class_{idx}" for idx in range(num_classes)]


def load_model(config_path: Path, model_path: Path, device: torch.device):
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    state_dict = load_state_dict(model_path)
    model = CNNClassifierModel(**config["model"], num_classes=infer_num_classes_from_state_dict(state_dict))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


def predict_tensor(model: nn.Module, features: torch.Tensor, batch_size: int, threshold: float, device: torch.device):
    logits_list = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            logits = model(batch).squeeze(-1)
            logits_list.append(logits.cpu())
    logits = torch.cat(logits_list, dim=0).numpy().astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)
    return logits, probs, preds


def load_chunked_dataset(dataset_dir: Path):
    ids: list[str] = []
    labels: list[float] = []
    features = []
    chunk_files = sorted(dataset_dir.glob("chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.pt found in {dataset_dir}")
    for chunk_file in chunk_files:
        obj = torch.load(chunk_file, map_location="cpu", weights_only=False)
        ids.extend(obj.get("ids", []))
        features.append(obj["X"].float())
        if "y" in obj:
            labels.extend(obj["y"].float().tolist())
    feature_tensor = torch.cat(features, dim=0)
    label_array = np.asarray(labels, dtype=np.float32) if labels else None
    return ids, feature_tensor, label_array


def extract_features_from_fasta(fasta_path: Path, protbert_path: str, device: torch.device, esm_max_len: int, max_seq_len: int):
    rows = parse_fasta_rows(fasta_path)
    if not rows:
        raise ValueError(f"No valid sequences in {fasta_path}")
    esm_embedder = ESM2Embedder(device=str(device), layer=33)
    prot_embedder = ProtBERTEmbedder(protbert_path, device=str(device))
    ids: list[str] = []
    features = []
    for seq_id, seq in rows:
        esm_vec = esm_embedder.encode_mean_chunked(seq_id, seq, chunk_len=esm_max_len).astype(np.float32)
        prot_vec = prot_embedder.encode_mean(seq_id, seq, max_len=max_seq_len).astype(np.float32)
        feat = np.concatenate([esm_vec, prot_vec], axis=0).astype(np.float32)
        ids.append(seq_id)
        features.append(feat)
    return ids, torch.tensor(np.stack(features, axis=0), dtype=torch.float32)


def metrics_from_logits(labels: np.ndarray, logits: np.ndarray, threshold: float):
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
        labels_i = labels.reshape(-1).astype(np.int64)
        probs = probs.reshape(-1)
        preds = (probs >= threshold).astype(np.int64)
        acc = float((preds == labels_i).mean())
        try:
            auc = float(roc_auc_score(labels_i, probs))
        except Exception:
            auc = 0.0
        try:
            auprc = float(average_precision_score(labels_i, probs))
        except Exception:
            auprc = 0.0
        return {"acc": acc, "auc": auc, "auprc": auprc, "threshold": threshold, "num_samples": int(len(labels_i)), "num_labels": 1}

    preds = (probs >= threshold).astype(np.int64)
    labels_i = labels.astype(np.int64)
    per_class_auc = []
    per_class_auprc = []
    per_class_acc = []
    for class_idx in range(labels_i.shape[1]):
        class_true = labels_i[:, class_idx]
        class_prob = probs[:, class_idx]
        class_pred = preds[:, class_idx]
        per_class_acc.append(float((class_pred == class_true).mean()))
        try:
            per_class_auc.append(float(roc_auc_score(class_true, class_prob)))
        except Exception:
            pass
        try:
            per_class_auprc.append(float(average_precision_score(class_true, class_prob)))
        except Exception:
            pass
    return {
        "acc": float(np.mean(per_class_acc)) if per_class_acc else 0.0,
        "exact_match": float((preds == labels_i).all(axis=1).mean()) if len(labels_i) else 0.0,
        "auc_macro": float(np.mean(per_class_auc)) if per_class_auc else 0.0,
        "auprc_macro": float(np.mean(per_class_auprc)) if per_class_auprc else 0.0,
        "threshold": threshold,
        "num_samples": int(len(labels_i)),
        "num_labels": int(labels_i.shape[1]),
    }


def write_predictions(out_path: Path, ids, logits: np.ndarray, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray | None, class_names):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
            probs = probs.reshape(-1, 1)
            preds = preds.reshape(-1, 1)
            if labels is not None:
                labels = labels.reshape(-1, 1)
        if labels is None:
            if logits.shape[1] == 1:
                writer.writerow(["seq_id", "prediction", "logit", "pred_label"])
                for seq_id, prob, logit, pred in zip(ids, probs.reshape(-1), logits.reshape(-1), preds.reshape(-1)):
                    writer.writerow([seq_id, f"{float(prob):.6f}", f"{float(logit):.6f}", int(pred)])
            else:
                writer.writerow(["seq_id"] + [f"pred_{name}" for name in class_names] + [f"logit_{name}" for name in class_names] + [f"pred_label_{name}" for name in class_names])
                for seq_id, prob, logit, pred in zip(ids, probs, logits, preds):
                    writer.writerow([seq_id] + [f"{float(v):.6f}" for v in prob.tolist()] + [f"{float(v):.6f}" for v in logit.tolist()] + pred.astype(int).tolist())
        elif logits.shape[1] == 1:
            writer.writerow(["seq_id", "label", "prediction", "logit", "pred_label"])
            for seq_id, label, prob, logit, pred in zip(ids, labels.astype(int).reshape(-1), probs.reshape(-1), logits.reshape(-1), preds.reshape(-1)):
                writer.writerow([seq_id, int(label), f"{float(prob):.6f}", f"{float(logit):.6f}", int(pred)])
        else:
            writer.writerow(["seq_id"] + [f"label_{name}" for name in class_names] + [f"pred_{name}" for name in class_names] + [f"logit_{name}" for name in class_names] + [f"pred_label_{name}" for name in class_names])
            for seq_id, label, prob, logit, pred in zip(ids, labels.astype(int), probs, logits, preds):
                writer.writerow([seq_id] + label.tolist() + [f"{float(v):.6f}" for v in prob.tolist()] + [f"{float(v):.6f}" for v in logit.tolist()] + pred.astype(int).tolist())


def main():
    args = parse_args()
    if bool(args.fasta) == bool(args.dataset_dir):
        raise ValueError("Provide exactly one of --fasta or --dataset-dir")

    device = resolve_device(args.device)
    model, config = load_model(Path(args.config), Path(args.model), device)
    expected_dim = int(config["model"]["input_dim"])
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    class_names = load_class_names(Path(args.model), dataset_dir, int(model.num_classes))

    if args.fasta:
        ids, features = extract_features_from_fasta(
            Path(args.fasta),
            args.protbert,
            device,
            int(args.esm_max_len),
            int(args.max_seq_len),
        )
        labels = None
    else:
        ids, features, labels = load_chunked_dataset(Path(args.dataset_dir))

    if int(features.shape[1]) != expected_dim:
        raise ValueError(f"Feature dim mismatch: got {int(features.shape[1])}, expected {expected_dim}")

    logits, probs, preds = predict_tensor(
        model,
        features,
        batch_size=int(args.batch_size),
        threshold=float(args.threshold),
        device=device,
    )
    out_path = Path(args.out)
    write_predictions(out_path, ids, logits, probs, preds, labels, class_names)
    print(f"[ORACLE-CNN-PREDICT] predictions: {out_path}")

    if labels is not None:
        metrics = metrics_from_logits(labels, logits, float(args.threshold))
        metrics["classes"] = class_names
        metrics_path = out_path.with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"[ORACLE-CNN-PREDICT] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
