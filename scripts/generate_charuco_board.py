#!/usr/bin/env python3
"""Generate a physically sized ChArUco board for camera calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import cv2
from PIL import Image

MM_PER_METRE = Decimal("1000")
METRES_PER_INCH = Decimal("0.0254")
MAX_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_SIDE = 50_000
FILE_STEM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def positive_decimal(value: str) -> Decimal:
    """Parse a finite, positive decimal command-line value."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"expected a decimal number, got {value!r}") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive finite number, got {value!r}")
    return parsed


def nonnegative_decimal(value: str) -> Decimal:
    """Parse a finite, non-negative decimal command-line value."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"expected a decimal number, got {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative finite number, got {value!r}")
    return parsed


def squares_count(value: str) -> int:
    """Parse a valid ChArUco square count."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 2:
        raise argparse.ArgumentTypeError("a ChArUco board needs at least 2 squares per axis")
    return parsed


def file_stem(value: str) -> str:
    """Reject path traversal and suffix-like output names."""
    if not FILE_STEM_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "file stem must contain only letters, digits, '.', '_' or '-' and start "
            "with a letter or digit"
        )
    return value


def available_dictionaries() -> tuple[str, ...]:
    """Return OpenCV predefined ArUco dictionary names."""
    return tuple(
        sorted(
            name
            for name in dir(cv2.aruco)
            if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
        )
    )


def build_parser() -> argparse.ArgumentParser:
    dictionaries = available_dictionaries()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dimensioned ChArUco PNG, a 100%-scale printable PDF, and JSON "
            "metadata. The output directory must be absent or empty."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--squares-x", type=squares_count, default=7)
    parser.add_argument("--squares-y", type=squares_count, default=5)
    parser.add_argument("--square-length-m", type=positive_decimal, default=Decimal("0.030"))
    parser.add_argument("--marker-length-m", type=positive_decimal, default=Decimal("0.022"))
    parser.add_argument(
        "--margin-m",
        type=nonnegative_decimal,
        default=Decimal("0.010"),
        help="white margin on each side of the board pattern in metres (default: 0.010)",
    )
    parser.add_argument("--dictionary", choices=dictionaries, default="DICT_4X4_50")
    parser.add_argument(
        "--dpi",
        type=positive_decimal,
        default=Decimal("300"),
        help=(
            "requested raster resolution; the embedded DPI may be adjusted slightly so "
            "the declared metric dimensions map to whole pixels (default: 300)"
        ),
    )
    parser.add_argument("--file-stem", type=file_stem, default="charuco_board")
    return parser


def required_pixels_per_metre_quantum(length_m: Decimal) -> int:
    """Return the PPM step that maps a decimal metric length to whole pixels."""
    fraction = Fraction(length_m)
    return fraction.denominator // math.gcd(fraction.numerator, fraction.denominator)


def choose_pixel_geometry(
    square_length_m: Decimal,
    marker_length_m: Decimal,
    margin_m: Decimal,
    requested_dpi: Decimal,
    squares_x: int,
    squares_y: int,
) -> dict[str, int | Decimal]:
    """Choose a nearby integer PPM that represents all declared lengths exactly."""
    square_quantum = required_pixels_per_metre_quantum(square_length_m)
    marker_quantum = required_pixels_per_metre_quantum(marker_length_m)
    margin_quantum = required_pixels_per_metre_quantum(margin_m) if margin_m > 0 else 1
    ppm_quantum = math.lcm(square_quantum, marker_quantum, margin_quantum)
    requested_ppm = requested_dpi / METRES_PER_INCH
    multiplier = int(
        (requested_ppm / Decimal(ppm_quantum)).to_integral_value(rounding=ROUND_HALF_UP)
    )
    pixels_per_metre = max(1, multiplier) * ppm_quantum

    square_pixels_fraction = Fraction(square_length_m) * pixels_per_metre
    marker_pixels_fraction = Fraction(marker_length_m) * pixels_per_metre
    margin_pixels_fraction = Fraction(margin_m) * pixels_per_metre
    if (
        square_pixels_fraction.denominator != 1
        or marker_pixels_fraction.denominator != 1
        or margin_pixels_fraction.denominator != 1
    ):
        raise RuntimeError("internal error: selected pixel density does not represent lengths")

    square_pixels = square_pixels_fraction.numerator
    marker_pixels = marker_pixels_fraction.numerator
    margin_pixels = margin_pixels_fraction.numerator
    board_width_pixels = square_pixels * squares_x
    board_height_pixels = square_pixels * squares_y
    page_width_pixels = board_width_pixels + 2 * margin_pixels
    page_height_pixels = board_height_pixels + 2 * margin_pixels
    pixel_count = page_width_pixels * page_height_pixels

    if marker_pixels < 12 or square_pixels < 20:
        raise ValueError(
            "requested resolution is too low for a reliably detectable marker; increase --dpi"
        )
    if (
        page_width_pixels > MAX_IMAGE_SIDE
        or page_height_pixels > MAX_IMAGE_SIDE
        or pixel_count > MAX_IMAGE_PIXELS
    ):
        raise ValueError(
            "the requested decimal lengths and DPI require an excessively large exact-size "
            f"raster ({page_width_pixels}x{page_height_pixels}); use fewer decimal places "
            "or a lower DPI"
        )

    return {
        "pixels_per_metre": pixels_per_metre,
        "square_pixels": square_pixels,
        "marker_pixels": marker_pixels,
        "margin_pixels": margin_pixels,
        "board_width_pixels": board_width_pixels,
        "board_height_pixels": board_height_pixels,
        "page_width_pixels": page_width_pixels,
        "page_height_pixels": page_height_pixels,
        "effective_dpi": Decimal(pixels_per_metre) * METRES_PER_INCH,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decimal_number(value: Decimal) -> int | float:
    """Serialize Decimal values as readable JSON numbers."""
    integral = value.to_integral_value()
    if value == integral:
        return int(integral)
    return float(value)


def prepare_output_directory(output_dir: Path) -> tuple[Path, Path]:
    """Validate the target and create a sibling staging directory."""
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    return output_dir, staging_dir


def publish_directory(staging_dir: Path, output_dir: Path) -> None:
    """Publish the complete output set with one directory rename."""
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory became non-empty: {output_dir}")
        output_dir.rmdir()
    os.replace(staging_dir, output_dir)


def generate(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    if args.marker_length_m >= args.square_length_m:
        raise ValueError("--marker-length-m must be smaller than --square-length-m")

    geometry = choose_pixel_geometry(
        args.square_length_m,
        args.marker_length_m,
        args.margin_m,
        args.dpi,
        args.squares_x,
        args.squares_y,
    )
    dictionary_id = getattr(cv2.aruco, args.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    expected_marker_count = (args.squares_x * args.squares_y) // 2
    dictionary_capacity = int(dictionary.bytesList.shape[0])
    if expected_marker_count > dictionary_capacity:
        raise ValueError(
            f"board needs {expected_marker_count} markers, but {args.dictionary} contains only "
            f"{dictionary_capacity}"
        )

    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        float(args.square_length_m),
        float(args.marker_length_m),
        dictionary,
    )
    board.setLegacyPattern(False)
    board_width_pixels = int(geometry["board_width_pixels"])
    board_height_pixels = int(geometry["board_height_pixels"])
    margin_pixels = int(geometry["margin_pixels"])
    page_width_pixels = int(geometry["page_width_pixels"])
    page_height_pixels = int(geometry["page_height_pixels"])
    board_pattern = board.generateImage(
        (board_width_pixels, board_height_pixels),
        marginSize=0,
        borderBits=1,
    )
    if board_pattern.shape != (board_height_pixels, board_width_pixels):
        raise RuntimeError(
            f"OpenCV returned unexpected board shape {board_pattern.shape}, expected "
            f"{(board_height_pixels, board_width_pixels)}"
        )
    board_image = cv2.copyMakeBorder(
        board_pattern,
        margin_pixels,
        margin_pixels,
        margin_pixels,
        margin_pixels,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    if board_image.shape != (page_height_pixels, page_width_pixels):
        raise RuntimeError(
            f"OpenCV returned unexpected page shape {board_image.shape}, expected "
            f"{(page_height_pixels, page_width_pixels)}"
        )
    if set(int(value) for value in board_image.flat) != {0, 255}:
        raise RuntimeError("OpenCV generated a non-binary board image")

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, marker_ids = detector.detectBoard(board_image)
    expected_corner_count = (args.squares_x - 1) * (args.squares_y - 1)
    detected_markers = 0 if marker_ids is None else len(marker_ids)
    detected_corners = 0 if charuco_ids is None else len(charuco_ids)
    if detected_markers != expected_marker_count or detected_corners != expected_corner_count:
        raise RuntimeError(
            "generated board failed its detection check: "
            f"detected {detected_markers}/{expected_marker_count} markers and "
            f"{detected_corners}/{expected_corner_count} ChArUco corners"
        )
    del charuco_corners

    output_dir, staging_dir = prepare_output_directory(args.output_dir)
    png_name = f"{args.file_stem}.png"
    pdf_name = f"{args.file_stem}.pdf"
    json_name = f"{args.file_stem}.json"
    png_path = staging_dir / png_name
    pdf_path = staging_dir / pdf_name
    json_path = staging_dir / json_name

    try:
        image = Image.fromarray(board_image).convert("1", dither=Image.Dither.NONE)
        effective_dpi = float(geometry["effective_dpi"])
        image.save(png_path, format="PNG", dpi=(effective_dpi, effective_dpi), optimize=True)
        image.save(
            pdf_path,
            format="PDF",
            dpi=(effective_dpi, effective_dpi),
            title="OpenCV ChArUco calibration board",
            subject="Print at Actual Size / 100%; do not fit or scale to page",
            creator="scripts/generate_charuco_board.py",
        )

        board_width_m = args.square_length_m * args.squares_x
        board_height_m = args.square_length_m * args.squares_y
        page_width_m = board_width_m + 2 * args.margin_m
        page_height_m = board_height_m + 2 * args.margin_m
        metadata: dict[str, object] = {
            "schema": "demo2gripper.charuco_board.v1",
            "schema_version": 1,
            "type": "opencv_charuco_board",
            "board": {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_length_m": decimal_number(args.square_length_m),
                "marker_length_m": decimal_number(args.marker_length_m),
                "dictionary": args.dictionary,
                "dictionary_id": dictionary_id,
                "legacy_pattern": False,
                "marker_ids": [int(value) for value in board.getIds().reshape(-1)],
                "charuco_corner_count": expected_corner_count,
                "width_m": decimal_number(board_width_m),
                "height_m": decimal_number(board_height_m),
                "outer_width_m": decimal_number(board_width_m),
                "outer_height_m": decimal_number(board_height_m),
            },
            "page": {
                "margin_each_side_m": decimal_number(args.margin_m),
                "width_m": decimal_number(page_width_m),
                "height_m": decimal_number(page_height_m),
                "width_px": page_width_pixels,
                "height_px": page_height_pixels,
            },
            "raster": {
                "requested_dpi": decimal_number(args.dpi),
                "embedded_dpi": decimal_number(geometry["effective_dpi"]),
                "pixels_per_metre": geometry["pixels_per_metre"],
                "width_px": page_width_pixels,
                "height_px": page_height_pixels,
                "board_width_px": board_width_pixels,
                "board_height_px": board_height_pixels,
                "square_length_px": geometry["square_pixels"],
                "marker_length_px": geometry["marker_pixels"],
                "margin_px": margin_pixels,
                "marker_border_bits": 1,
            },
            "print": {
                "pdf_page_width_mm": decimal_number(page_width_m * MM_PER_METRE),
                "pdf_page_height_mm": decimal_number(page_height_m * MM_PER_METRE),
                "margin_each_side_mm": decimal_number(args.margin_m * MM_PER_METRE),
                "board_outer_width_mm": decimal_number(board_width_m * MM_PER_METRE),
                "board_outer_height_mm": decimal_number(board_height_m * MM_PER_METRE),
                "scale_percent": 100,
                "instructions_zh": [
                    "打印 PDF 时选择“实际大小”或 100%，关闭“适合页面”和任何自动缩放。",
                    "PDF 页面已包含白边；请使用能容纳完整页面的纸张或输出设备。",
                    "打印后必须实测标定板外框宽高和单格边长，并记录实测值；尺寸不符时不要开始标定。",
                ],
                "post_print_measurement_required": True,
                "expected_outer_width_mm": decimal_number(board_width_m * MM_PER_METRE),
                "expected_outer_height_mm": decimal_number(board_height_m * MM_PER_METRE),
                "expected_square_length_mm": decimal_number(args.square_length_m * MM_PER_METRE),
                "expected_marker_length_mm": decimal_number(args.marker_length_m * MM_PER_METRE),
            },
            "files": {
                "png": {"name": png_name, "sha256": sha256(png_path)},
                "pdf": {"name": pdf_name, "sha256": sha256(pdf_path)},
            },
            "generator": {
                "opencv_version": cv2.__version__,
                "script": "scripts/generate_charuco_board.py",
            },
        }
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_directory(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return output_dir, metadata


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output_dir, metadata = generate(args)
    except (FileExistsError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        parser.error(str(exc))

    board = metadata["board"]
    raster = metadata["raster"]
    print(f"Wrote ChArUco board to {output_dir}")
    print(
        f"Board: {board['width_m']} m x {board['height_m']} m; "
        f"page: {raster['width_px']} x {raster['height_px']} px at "
        f"{raster['embedded_dpi']} DPI"
    )
    print("Print the PDF at Actual Size / 100%, then measure the printed board before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
