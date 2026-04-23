#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embed_extract import ESM2Embedder, Evo2Embedder, ProtBERTEmbedder, parse_fasta_rows


class TeacherNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StudentNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Unified predictor with 3 modes: "
            "single-route, multi-route fusion with teacher(no-student), "
            "multi-route fusion with student."
        )
    )
    ap.add_argument("--model", required=True, help="Path to kd_best.pt")
    ap.add_argument("--fasta", required=True, help="Input FASTA")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument(
        "--mode",
        default="fusion_student",
        choices=["single", "fusion_teacher", "fusion_student"],
    )
    ap.add_argument(
        "--single-route",
        default="esm2",
        choices=["esm2", "evo2", "protbert"],
        help="Used when --mode single",
    )
    ap.add_argument(
        "--single-scorer",
        default="student",
        choices=["student", "teacher"],
        help="Scorer for --mode single",
    )
    ap.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    ap.add_argument("--evo-layer", default="blocks.21.mlp.l3")
    ap.add_argument("--evo-max-len", type=int, default=16384)
    ap.add_argument("--esm-chunk-len", type=int, default=1022)

    ap.add_argument("--esm-dim", type=int, default=1280)
    ap.add_argument("--evo-dim", type=int, default=1920)
    ap.add_argument("--prot-dim", type=int, default=1024)
    ap.add_argument(
        "--strict-dims",
        action="store_true",
        help="If set, require esm_dim+evo_dim+prot_dim == checkpoint feature_dim",
    )
    return ap.parse_args()


def infer_dims(feature_dim: int, esm_dim: int, evo_dim: int, prot_dim: int, strict: bool):
    total = esm_dim + evo_dim + prot_dim
    if total == feature_dim:
        return esm_dim, evo_dim, prot_dim
    if strict:
        raise ValueError(
            f"Dim mismatch: esm+evo+prot={total}, ckpt feature_dim={feature_dim}. "
            "Adjust --esm-dim/--evo-dim/--prot-dim."
        )
    inferred_prot = feature_dim - esm_dim - evo_dim
    if inferred_prot <= 0:
        raise ValueError(
            f"Cannot infer prot dim from feature_dim={feature_dim}, esm_dim={esm_dim}, evo_dim={evo_dim}"
        )
    return esm_dim, evo_dim, inferred_prot


def build_models(ckpt, in_dim: int, device: torch.device):
    cfg = ckpt["config"]
    mcfg = cfg["model"]

    teacher = TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]))
    student = StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]))

    if "teacher_state" in ckpt:
        teacher.load_state_dict(ckpt["teacher_state"])
    if "student_state" in ckpt:
        student.load_state_dict(ckpt["student_state"])

    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    return teacher, student


def sigmoid_prob(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit)))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    feature_dim = int(ckpt["feature_dim"])

    esm_dim, evo_dim, prot_dim = infer_dims(
        feature_dim=feature_dim,
        esm_dim=int(args.esm_dim),
        evo_dim=int(args.evo_dim),
        prot_dim=int(args.prot_dim),
        strict=bool(args.strict_dims),
    )

    teacher, student = build_models(ckpt, in_dim=feature_dim, device=device)
    scorer = teacher if args.mode == "fusion_teacher" else student
    if args.mode == "single":
        scorer = teacher if args.single_scorer == "teacher" else student

    rows = parse_fasta_rows(Path(args.fasta))
    if not rows:
        raise ValueError(f"No valid sequences in {args.fasta}")

    esm_embedder = ESM2Embedder(device=device, layer=33)
    evo_embedder = Evo2Embedder(device=device, layer_name=args.evo_layer, max_len=int(args.evo_max_len))
    prot_embedder = ProtBERTEmbedder(args.protbert, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "prediction", "mode", "scorer", "feature_dim"])

        for sid, seq in rows:
            try:
                esm_vec = esm_embedder.encode_mean_chunked(sid, seq, chunk_len=int(args.esm_chunk_len)).astype(np.float32)
                evo_vec = evo_embedder.encode_mean(sid, seq).astype(np.float32)
                prot_vec = prot_embedder.encode_mean(sid, seq, max_len=1024).astype(np.float32)

                # Ensure branch dims align with routing contract.
                esm_vec = esm_vec[:esm_dim] if esm_vec.shape[0] >= esm_dim else np.pad(esm_vec, (0, esm_dim - esm_vec.shape[0]))
                evo_vec = evo_vec[:evo_dim] if evo_vec.shape[0] >= evo_dim else np.pad(evo_vec, (0, evo_dim - evo_vec.shape[0]))
                prot_vec = prot_vec[:prot_dim] if prot_vec.shape[0] >= prot_dim else np.pad(prot_vec, (0, prot_dim - prot_vec.shape[0]))

                if args.mode == "single":
                    z_esm = np.zeros(esm_dim, dtype=np.float32)
                    z_evo = np.zeros(evo_dim, dtype=np.float32)
                    z_prot = np.zeros(prot_dim, dtype=np.float32)
                    if args.single_route == "esm2":
                        z_esm = esm_vec
                    elif args.single_route == "evo2":
                        z_evo = evo_vec
                    else:
                        z_prot = prot_vec
                    feat = np.concatenate([z_esm, z_evo, z_prot], axis=0).astype(np.float32)
                else:
                    feat = np.concatenate([esm_vec, evo_vec, prot_vec], axis=0).astype(np.float32)

                if feat.shape[0] != feature_dim:
                    raise RuntimeError(f"Feature dim mismatch: got={feat.shape[0]} expected={feature_dim}")

                x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logit = float(scorer(x).squeeze(0).item())
                prob = sigmoid_prob(logit)
                scorer_name = args.single_scorer if args.mode == "single" else ("teacher" if args.mode == "fusion_teacher" else "student")
                w.writerow([sid, f"{prob:.6f}", args.mode, scorer_name, feature_dim])
            except Exception as ex:
                w.writerow([sid, "ERROR", args.mode, "NA", feature_dim])
                print(f"[predict_v2] failed for {sid}: {ex}")

    print(f"[predict_v2] done: {out_path}")


if __name__ == "__main__":
    main()

