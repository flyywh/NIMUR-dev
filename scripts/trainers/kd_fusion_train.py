#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray):
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    probs = 1 / (1 + np.exp(-logits))
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        y_true = y_true.reshape(-1)
        probs = probs.reshape(-1)
        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = 0.0
        try:
            auprc = float(average_precision_score(y_true, probs))
        except Exception:
            auprc = 0.0
        acc = float(((probs >= 0.5) == y_true).mean())
        return {"auc": auc, "auprc": auprc, "acc": acc, "num_labels": 1}

    preds = (probs >= 0.5).astype(np.int64)
    y_true = y_true.astype(np.int64)
    per_class_auc = []
    per_class_auprc = []
    per_class_acc = []
    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]
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

    return {
        "auc_macro": float(np.mean(per_class_auc)) if per_class_auc else 0.0,
        "auprc_macro": float(np.mean(per_class_auprc)) if per_class_auprc else 0.0,
        "acc": float(np.mean(per_class_acc)) if per_class_acc else 0.0,
        "exact_match": float((preds == y_true).all(axis=1).mean()) if len(y_true) else 0.0,
        "num_labels": int(y_true.shape[1]),
    }


def run_eval(student, loader, device):
    student.eval()
    ys, logits = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = student(xb)
            ys.append(yb.numpy())
            logits.append(out.cpu().numpy())
    y = np.concatenate(ys)
    lg = np.concatenate(logits)
    return metrics_from_logits(y, lg), y, lg


def main():
    ap = argparse.ArgumentParser(description="KD fusion trainer (ESM2 + Evo2 + ProtBERT)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    set_seed(int(tcfg.get("seed", 42)))

    obj = torch.load(args.dataset, map_location="cpu", weights_only=False)
    Xtr, ytr = obj["train"]["X"], obj["train"]["y"]
    Xva, yva = obj["val"]["X"], obj["val"]["y"]
    Xte, yte = obj["test"]["X"], obj["test"]["y"]
    classes = None
    if isinstance(obj.get("meta"), dict):
        classes = obj["meta"].get("classes")

    in_dim = Xtr.shape[1]
    num_labels = 1 if ytr.ndim == 1 else int(ytr.shape[1])
    teacher = TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]), out_dim=num_labels)
    student = StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]), out_dim=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    student = student.to(device)

    batch_size = int(tcfg["batch_size"])
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    params = list(teacher.parameters()) + list(student.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    alpha = float(tcfg.get("alpha", 0.5))
    T = float(tcfg.get("temperature", 2.0))
    teacher_w = float(tcfg.get("teacher_loss_weight", 0.5))
    epochs = int(tcfg["num_epochs"])

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    history = []

    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        teacher.train()
        student.train()
        ep_loss = 0.0

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            tlog = teacher(xb)
            slog = student(xb)

            loss_t = bce(tlog, yb)
            loss_hard = bce(slog, yb)

            with torch.no_grad():
                soft_t = torch.sigmoid(tlog / T)
            loss_kd = bce(slog / T, soft_t) * (T * T)

            loss = teacher_w * loss_t + alpha * loss_hard + (1 - alpha) * loss_kd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        val_m, _, _ = run_eval(student, va_loader, device)
        row = {"epoch": ep, "train_loss": ep_loss / max(1, len(tr_loader)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        val_score = float(val_m.get("auc_macro", val_m.get("auc", 0.0)))
        if not np.isfinite(val_score):
            val_score = float(val_m.get("auprc_macro", val_m.get("auprc", -1.0)))
        if val_score > best_auc:
            best_auc = val_score
            torch.save(
                {
                    "student_state": student.state_dict(),
                    "teacher_state": teacher.state_dict(),
                    "feature_dim": int(in_dim),
                    "num_labels": int(num_labels),
                    "classes": classes,
                    "config": cfg,
                    "best_val": val_m,
                },
                outdir / "models" / "kd_best.pt",
            )

    # final eval on test with best model
    best = torch.load(outdir / "models" / "kd_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(best["student_state"])
    test_m, y, lg = run_eval(student, te_loader, device)
    print("[KD] test:", json.dumps(test_m, ensure_ascii=False))

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "test": test_m, "history": history}, f, ensure_ascii=False, indent=2)

    probs = 1 / (1 + np.exp(-lg))
    with (outdir / "test_predictions.csv").open("w", encoding="utf-8") as f:
        if num_labels == 1:
            f.write("seq_id,label,prediction\n")
            for sid, yy, pp in zip(obj["test"]["ids"], y.astype(int), probs):
                f.write(f"{sid},{yy},{pp:.6f}\n")
        else:
            class_names = classes if isinstance(classes, list) and len(classes) == num_labels else [f"class_{idx}" for idx in range(num_labels)]
            header = ["seq_id"] + [f"label_{name}" for name in class_names] + [f"pred_{name}" for name in class_names]
            f.write(",".join(header) + "\n")
            for sid, yy, pp in zip(obj["test"]["ids"], y.astype(int), probs):
                row = [sid] + [str(int(v)) for v in yy.tolist()] + [f"{float(v):.6f}" for v in pp.tolist()]
                f.write(",".join(row) + "\n")

    print(f"[KD] done: {outdir}")


if __name__ == "__main__":
    main()
