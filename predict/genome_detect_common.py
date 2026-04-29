#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class WindowPred:
    contig: str
    start: int
    end: int
    seq_id: str
    bgc_score: float
    top_type: str = ""
    neg_score: float = float("nan")


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


def merge_regions(rows: Sequence[WindowPred], threshold: float, max_gap: int) -> List[dict]:
    hot = [r for r in rows if r.bgc_score >= threshold]
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
                "scores": [r.bgc_score],
                "types": [r.top_type] if r.top_type else [],
                "num_windows": 1,
            }
            continue

        same = r.contig == cur["contig"]
        close = r.start <= cur["end"] + max_gap
        if same and close:
            cur["end"] = max(cur["end"], r.end)
            cur["scores"].append(r.bgc_score)
            if r.top_type:
                cur["types"].append(r.top_type)
            cur["num_windows"] += 1
        else:
            merged.append(cur)
            cur = {
                "contig": r.contig,
                "start": r.start,
                "end": r.end,
                "scores": [r.bgc_score],
                "types": [r.top_type] if r.top_type else [],
                "num_windows": 1,
            }

    if cur is not None:
        merged.append(cur)

    out = []
    for i, m in enumerate(merged, start=1):
        type_counts = {}
        for item in m["types"]:
            type_counts[item] = type_counts.get(item, 0) + 1
        top_type = ""
        if type_counts:
            top_type = sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
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
                "top_type": top_type,
            }
        )
    return out


def write_regions_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "contig", "start", "end", "length", "num_windows", "max_score", "mean_score", "top_type"])
        for row in rows:
            writer.writerow(
                [
                    row["region_id"],
                    row["contig"],
                    row["start"],
                    row["end"],
                    row["length"],
                    row["num_windows"],
                    f"{row['max_score']:.6f}",
                    f"{row['mean_score']:.6f}",
                    row.get("top_type", ""),
                ]
            )


def dump_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
