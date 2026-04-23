from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lora import lora


def run_with_default_model(default_model: str) -> int:
    argv = sys.argv[1:]
    if "--model" not in argv:
        sys.argv = [sys.argv[0], "--model", default_model, *argv]
    return lora.main()

