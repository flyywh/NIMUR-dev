#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embed_extract import ESM2Embedder, Evo2Embedder, ProtBERTEmbedder


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


@dataclass
class WindowPred:
    contig: str
    start: int
    end: int
    seq_id: str
    pred: float


def make_windows(contig_id: str, seq: str, win: int, step: int) -> List[Tuple[str, int, int, str]]:
    n = len(seq)
    if n <= win:
        return [(contig_id, 1, n, seq)]
    rows = []
    start = 0
    while start < n:
        end = min(start + win, n)
        if end - start < max(500, win // 5):
            break
        rows.append((contig_id, start + 1, end, seq[start:end]))
        if end == n:
            break
        start += step
    return rows


def merge_regions(rows: List[WindowPred], threshold: float, max_gap: int) -> List[dict]:
    hot = [r for r in rows if r.pred >= threshold]
    if not hot:
        return []
    hot.sort(key=lambda x: (x.contig, x.start))

    merged = []
    cur = None
    for r in hot:
        if cur is None:
            cur = {
                "contig": r.contig,
                "start": r.start,
                "end": r.end,
                "scores": [r.pred],
                "num_windows": 1,
            }
            continue

        same = r.contig == cur["contig"]
        close = r.start <= cur["end"] + max_gap
        if same and close:
            cur["end"] = max(cur["end"], r.end)
            cur["scores"].append(r.pred)
            cur["num_windows"] += 1
        else:
            merged.append(cur)
            cur = {
                "contig": r.contig,
                "start": r.start,
                "end": r.end,
                "scores": [r.pred],
                "num_windows": 1,
            }

    if cur is not None:
        merged.append(cur)

    out = []
    for i, m in enumerate(merged, start=1):
        out.append(
            {
                "region_id": f"R{i:04d}",
                "contig": m["contig"],
                "start": int(m["start"]),
                "end": int(m["end"]),
                "length": int(m["end"] - m["start"] + 1),
                "num_windows": int(m["num_windows"]),
                "max_score": float(np.max(m["scores"])),
                "mean_score": float(np.mean(m["scores"])),
            }
        )
    return out


def sigmoid_prob(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-logit)))


def main():
    ap = argparse.ArgumentParser(description="KD genome detector v2")
    ap.add_argument("--model", required=True, help="kd_best.pt")
    ap.add_argument("--genome-fna", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protbert", default="local_assets/data/ProtBERT/ProtBERT")
    ap.add_argument("--evo-layer", default="blocks.21.mlp.l3")
    ap.add_argument("--evo-max-len", type=int, default=16384)
    ap.add_argument("--window", type=int, default=12000)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--merge-gap", type=int, default=1500)
    ap.add_argument("--scorer", choices=["student", "teacher"], default="student")
    ap.add_argument("--esm-chunk-len", type=int, default=1022)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    in_dim = int(ckpt["feature_dim"])
    mcfg = cfg["model"]

    teacher = TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]))
    student = StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]))
    if "teacher_state" in ckpt:
        teacher.load_state_dict(ckpt["teacher_state"])
    if "student_state" in ckpt:
        student.load_state_dict(ckpt["student_state"])
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    scorer = teacher if args.scorer == "teacher" else student

    esm_embedder = ESM2Embedder(device=device, layer=33)
    evo_embedder = Evo2Embedder(device=device, layer_name=args.evo_layer, max_len=int(args.evo_max_len))
    prot_embedder = ProtBERTEmbedder(Path(args.protbert), device=device)

    seqs = [(str(r.id), str(r.seq).upper()) for r in SeqIO.parse(str(args.genome_fna), "fasta")]
    if not seqs:
        raise ValueError("No sequences in genome fna")

    windows = []
    for cid, seq in seqs:
        windows.extend(make_windows(cid, seq, int(args.window), int(args.step)))

    preds: List[WindowPred] = []
    for idx, (cid, s, e, sub) in enumerate(windows, start=1):
        sid = f"{cid}:{s}-{e}"
        try:
            esm_vec = esm_embedder.encode_mean_chunked(sid, sub, chunk_len=int(args.esm_chunk_len)).astype(np.float32)
            evo_vec = evo_embedder.encode_mean(sid, sub).astype(np.float32)
            prot_vec = prot_embedder.encode_mean(sid, sub, max_len=1024).astype(np.float32)
            feat = np.concatenate([esm_vec, evo_vec, prot_vec], axis=0).astype(np.float32)
            if feat.shape[0] != in_dim:
                raise RuntimeError(f"feature_dim mismatch {feat.shape[0]} vs {in_dim}")
            x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = float(scorer(x).squeeze(0).item())
            prob = sigmoid_prob(logit)
        except Exception as ex:
            print(f"[KD-GENOME-V2] window failed {sid}: {ex}")
            prob = 0.0
        preds.append(WindowPred(contig=cid, start=s, end=e, seq_id=sid, pred=prob))
        if idx % 10 == 0:
            print(f"[KD-GENOME-V2] processed {idx}/{len(windows)} windows")

    win_csv = out_dir / "window_predictions.csv"
    with win_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contig", "window_start", "window_end", "seq_id", "prediction"])
        for r in preds:
            w.writerow([r.contig, r.start, r.end, r.seq_id, f"{r.pred:.6f}"])

    regions = merge_regions(preds, threshold=float(args.threshold), max_gap=int(args.merge_gap))
    reg_csv = out_dir / "bgc_regions.csv"
    with reg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region_id", "contig", "start", "end", "length", "num_windows", "max_score", "mean_score"])
        for r in regions:
            w.writerow(
                [
                    r["region_id"],
                    r["contig"],
                    r["start"],
                    r["end"],
                    r["length"],
                    r["num_windows"],
                    f"{r['max_score']:.6f}",
                    f"{r['mean_score']:.6f}",
                ]
            )

    summary = {
        "model": str(Path(args.model).resolve()),
        "genome_fna": str(Path(args.genome_fna).resolve()),
        "scorer": args.scorer,
        "num_contigs": len(seqs),
        "num_windows": len(preds),
        "num_regions": len(regions),
        "window": int(args.window),
        "step": int(args.step),
        "threshold": float(args.threshold),
        "merge_gap": int(args.merge_gap),
        "window_predictions_csv": str(win_csv.resolve()),
        "bgc_regions_csv": str(reg_csv.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[KD-GENOME-V2] done: {out_dir}")


if __name__ == "__main__":
    main()

