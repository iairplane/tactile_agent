#!/usr/bin/env python3
"""Standalone Hand-E status/open/close entry point; no FR3 Ethernet client."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo2gripper.hande_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
