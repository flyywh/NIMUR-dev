#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunManifest:
    schema_version: str
    pipeline: str
    created_at_utc: str
    host_platform: str
    command: str
    output_dir: str
    outputs: List[str]
    input_files: List[str]
    params: Dict[str, Any]
    status: str
    return_code: int
    duration_sec: float
    adapter_target: Optional[str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    pipeline: str,
    command: str,
    output_dir: Path,
    outputs: List[Path],
    input_files: List[Path],
    params: Dict[str, Any],
    status: str,
    return_code: int,
    duration_sec: float,
    adapter_target: Optional[str],
) -> RunManifest:
    return RunManifest(
        schema_version="data_prep_unified/v1",
        pipeline=pipeline,
        created_at_utc=utc_now_iso(),
        host_platform=f"{platform.system()} {platform.release()}",
        command=command,
        output_dir=str(output_dir),
        outputs=[str(p) for p in outputs],
        input_files=[str(p) for p in input_files],
        params=params,
        status=status,
        return_code=return_code,
        duration_sec=duration_sec,
        adapter_target=adapter_target,
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False)

