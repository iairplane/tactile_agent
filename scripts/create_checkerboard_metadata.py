#!/usr/bin/env python3
"""Create strict metadata for a measured, existing checkerboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

try:
    from checkerboard_calibration import checkerboard_metadata
except ModuleNotFoundError:  # Imported as a module from the repository root.
    from scripts.checkerboard_calibration import checkerboard_metadata


def square_count(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if result < 3:
        raise argparse.ArgumentTypeError("a checkerboard needs at least 3 squares per axis")
    return result


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the measured geometry of an existing checkerboard. The two square "
            "counts must have opposite parity to eliminate 180-degree ordering ambiguity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--squares-x", required=True, type=square_count)
    parser.add_argument("--squares-y", required=True, type=square_count)
    parser.add_argument(
        "--square-length-mm",
        required=True,
        type=positive_float,
        help="Measured side length of one printed square in millimetres",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing metadata: {output}")
    metadata = checkerboard_metadata(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_mm / 1000.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote checkerboard metadata to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
