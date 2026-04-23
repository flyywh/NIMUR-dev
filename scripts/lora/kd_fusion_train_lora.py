#!/usr/bin/env python3
from __future__ import annotations

from scripts.lora._legacy_wrapper import run_with_default_model


if __name__ == "__main__":
    raise SystemExit(run_with_default_model("kd-fusion"))
