#!/usr/bin/env python3
"""Dependency-light direct Modbus RTU entry point for Robotiq Hand-E."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo2gripper.hande_modbus_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
