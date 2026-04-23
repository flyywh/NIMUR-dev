from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from Bio import SeqIO


def _normalize_seq(seq: str) -> str:
    return seq.upper().replace(" ", "")


def parse_fasta_rows(path: Path | str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq_id = str(rec.id)
        seq = _normalize_seq(str(rec.seq))
        if not seq:
            continue
        rows.append((seq_id, seq))
    return rows


def parse_fasta_dict(path: Path | str, strict_unique: bool = False) -> Dict[str, str]:
    rows = parse_fasta_rows(path)
    data: Dict[str, str] = {}
    for seq_id, seq in rows:
        if strict_unique and seq_id in data:
            raise ValueError(f"Duplicate FASTA id found: {seq_id}")
        data.setdefault(seq_id, seq)
    return data


def load_sequences(paths: Iterable[Path | str]) -> Dict[str, str]:
    all_seq: Dict[str, str] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        all_seq.update(parse_fasta_dict(path))
    return all_seq

