#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List


def repo_root_from_this_file() -> Path:
    # .../merged_scripts/data_prep_unified/adapters.py -> repo root
    return Path(__file__).resolve().parents[2]


def legacy_dir() -> Path:
    return repo_root_from_this_file() / "merged_scripts" / "data_prep"


def legacy_script(name: str) -> Path:
    p = legacy_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"Legacy script not found: {p}")
    return p


def sh_cmd(script_name: str, args: List[str]) -> List[str]:
    return ["bash", str(legacy_script(script_name)), *args]


def py_cmd(script_name: str, args: List[str]) -> List[str]:
    return ["python", str(legacy_script(script_name)), *args]

