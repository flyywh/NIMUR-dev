#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_value(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clone JSON config and override dotted keys")
    ap.add_argument("--base", required=True, help="Base JSON config path")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override in form key=value, key can be dotted (e.g. training.lr=1e-6)",
    )
    return ap


def run(base: str, out: str, pairs: list[str]) -> Path:
    with open(base, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}', expected key=value")
        key, raw = item.split("=", 1)
        set_nested(data, key, parse_value(raw))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return out_path


def main() -> int:
    args = build_parser().parse_args()
    out_path = run(args.base, args.out, args.set)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
