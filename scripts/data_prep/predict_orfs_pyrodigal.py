#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyrodigal
from Bio import SeqIO


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict protein ORFs from DNA FASTA using pyrodigal.")
    p.add_argument("--input-fasta", required=True)
    p.add_argument("--output-faa", required=True)
    p.add_argument("--output-tsv", default="")
    p.add_argument("--meta", action="store_true", default=True)
    p.add_argument("--min-aa-len", type=int, default=30)
    return p.parse_args()


def clean_protein(seq: str) -> str:
    seq = seq.upper().rstrip("*")
    return "".join(ch if ch in ALLOWED_AA else "X" for ch in seq)


def main() -> int:
    args = parse_args()
    in_fasta = Path(args.input_fasta)
    out_faa = Path(args.output_faa)
    out_faa.parent.mkdir(parents=True, exist_ok=True)
    out_tsv = Path(args.output_tsv) if args.output_tsv else out_faa.with_suffix(".tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    finder = pyrodigal.GeneFinder(meta=args.meta)
    dna_records = 0
    proteins_written = 0
    skipped_short = 0

    with out_faa.open("w", encoding="utf-8") as faa, out_tsv.open("w", encoding="utf-8") as tsv:
        tsv.write("orf_id\tparent_id\tbegin\tend\tstrand\tpartial_begin\tpartial_end\ttranslation_table\taa_len\n")
        for rec in SeqIO.parse(str(in_fasta), "fasta"):
            dna_records += 1
            genes = finder.find_genes(str(rec.seq).upper())
            parent_id = str(rec.id)
            for idx, gene in enumerate(genes, start=1):
                aa = clean_protein(gene.translate())
                if len(aa) < args.min_aa_len:
                    skipped_short += 1
                    continue
                orf_id = f"{parent_id}|orf{idx:05d}"
                desc = (
                    f"parent={parent_id} begin={gene.begin} end={gene.end} strand={gene.strand} "
                    f"partial_begin={int(gene.partial_begin)} partial_end={int(gene.partial_end)} "
                    f"translation_table={gene.translation_table}"
                )
                faa.write(f">{orf_id} {desc}\n{aa}\n")
                tsv.write(
                    f"{orf_id}\t{parent_id}\t{gene.begin}\t{gene.end}\t{gene.strand}\t"
                    f"{int(gene.partial_begin)}\t{int(gene.partial_end)}\t{gene.translation_table}\t{len(aa)}\n"
                )
                proteins_written += 1

    summary = {
        "input_fasta": str(in_fasta),
        "output_faa": str(out_faa),
        "output_tsv": str(out_tsv),
        "dna_records": dna_records,
        "proteins_written": proteins_written,
        "skipped_short": skipped_short,
        "min_aa_len": args.min_aa_len,
        "meta": args.meta,
    }
    summary_path = out_faa.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
