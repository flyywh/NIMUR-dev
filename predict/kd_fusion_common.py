#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


class TeacherNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class StudentNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def infer_num_labels(ckpt) -> int:
    if isinstance(ckpt, dict):
        if "num_labels" in ckpt:
            return int(ckpt["num_labels"])
        classes = ckpt.get("classes")
        if isinstance(classes, list) and classes:
            return len(classes)
        student_state = ckpt.get("student_state")
        if isinstance(student_state, dict):
            weight = student_state.get("net.3.weight")
            if torch.is_tensor(weight) and weight.ndim == 2:
                return int(weight.shape[0])
    return 1


def build_models(ckpt, in_dim: int, device: torch.device):
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    out_dim = infer_num_labels(ckpt)

    teacher = TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]), out_dim=out_dim)
    student = StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]), out_dim=out_dim)

    teacher_state = ckpt.get("teacher_state") if isinstance(ckpt, dict) else None
    student_state = ckpt.get("student_state") if isinstance(ckpt, dict) else None
    if teacher_state:
        teacher.load_state_dict(teacher_state)
    if student_state:
        student.load_state_dict(student_state)

    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    return teacher, student


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        y_true_flat = y_true.reshape(-1)
        logits_flat = logits.reshape(-1)
        probs = logits_to_probs(logits_flat)
        preds = (probs >= threshold).astype(np.int64)
        y_true_int = y_true_flat.astype(np.int64)

        try:
            auc = float(roc_auc_score(y_true_int, probs))
        except Exception:
            auc = 0.0
        try:
            auprc = float(average_precision_score(y_true_int, probs))
        except Exception:
            auprc = 0.0
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_int,
                preds,
                average="binary",
                zero_division=0,
            )
            precision = float(precision)
            recall = float(recall)
            f1 = float(f1)
        except Exception:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        acc = float((preds == y_true_int).mean())
        return {
            "auc": auc,
            "auprc": auprc,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_labels": 1,
        }

    probs = logits_to_probs(logits)
    preds = (probs >= threshold).astype(np.int64)
    y_true_int = y_true.astype(np.int64)

    per_class_auc = []
    per_class_auprc = []
    per_class_acc = []
    for class_idx in range(y_true_int.shape[1]):
        class_true = y_true_int[:, class_idx]
        class_prob = probs[:, class_idx]
        class_pred = preds[:, class_idx]
        per_class_acc.append(float((class_pred == class_true).mean()))
        try:
            score = float(roc_auc_score(class_true, class_prob))
            if np.isfinite(score):
                per_class_auc.append(score)
        except Exception:
            pass
        try:
            score = float(average_precision_score(class_true, class_prob))
            if np.isfinite(score):
                per_class_auprc.append(score)
        except Exception:
            pass

    acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0
    exact_match = float((preds == y_true_int).all(axis=1).mean()) if len(y_true_int) else 0.0
    return {
        "auc_macro": float(np.mean(per_class_auc)) if per_class_auc else 0.0,
        "auprc_macro": float(np.mean(per_class_auprc)) if per_class_auprc else 0.0,
        "acc": acc,
        "exact_match": exact_match,
        "num_labels": int(y_true_int.shape[1]),
    }
