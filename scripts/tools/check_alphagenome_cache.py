#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.embed_extract.alphagenome import (
    infer_ag_dim,
    infer_ag_level,
    load_all_ag_embeddings,
    summarize_ag_embeddings,
)
from utils.embed_extract.fasta import parse_fasta_rows


def extract_mibig_ids(path: Path) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    current_header = ""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_header = line[1:]
                match = re.search(r"BGC_ID=(BGC\d+)", current_header)
                if match:
                    sample_id = match.group(1)
                    if sample_id not in seen:
                        ids.append(sample_id)
                        seen.add(sample_id)
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate external AlphaGenome .npz cache directory and optional coverage against sample ids."
    )
    p.add_argument("--ag-dir", required=True, help="Directory containing AlphaGenome .npz files.")
    p.add_argument("--sample-fasta", default="", help="FASTA whose headers are already normalized sample ids.")
    p.add_argument("--mibig-fasta", default="", help="Raw MiBiG formatted FASTA with BGC_ID=... in headers.")
    p.add_argument("--out-json", default="", help="Optional output JSON path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ag_dir = Path(args.ag_dir)
    if not ag_dir.is_dir():
        raise FileNotFoundError(f"Missing AlphaGenome directory: {ag_dir}")

    ag_map = load_all_ag_embeddings(ag_dir)
    if not ag_map:
        raise ValueError(f"No AlphaGenome embeddings found under {ag_dir}")

    summary = {
        "ag_dir": str(ag_dir.resolve()),
        "cache_summary": summarize_ag_embeddings(ag_map),
    }

    sample_ids: list[str] = []
    sample_source = ""
    if args.sample_fasta:
        sample_source = str(Path(args.sample_fasta).resolve())
        sample_ids = [sample_id for sample_id, _seq in parse_fasta_rows(args.sample_fasta)]
    elif args.mibig_fasta:
        sample_source = str(Path(args.mibig_fasta).resolve())
        sample_ids = extract_mibig_ids(Path(args.mibig_fasta))

    if sample_ids:
        ag_ids = set(ag_map.keys())
        wanted = list(dict.fromkeys(sample_ids))
        matched = [sample_id for sample_id in wanted if sample_id in ag_ids]
        missing = [sample_id for sample_id in wanted if sample_id not in ag_ids]
        extra = sorted(ag_ids - set(wanted))
        summary["coverage"] = {
            "sample_source": sample_source,
            "sample_count": len(wanted),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "coverage_ratio": round(len(matched) / max(1, len(wanted)), 6),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
        }

    first_items = []
    for idx, (sample_id, emb) in enumerate(ag_map.items()):
        if idx >= 5:
            break
        first_items.append(
            {
                "sample_id": sample_id,
                "level": infer_ag_level(emb),
                "dim": infer_ag_dim(emb),
            }
        )
    summary["examples"] = first_items

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
