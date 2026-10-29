#!/usr/bin/env python3
"""Compile reviewed test-tube geometry evidence without dispatching hardware.

The output preserves projective/image constraints when transparent-object depth
is unavailable.  It deliberately does not synthesize an executable tube or
socket transform from neighbouring rack/background depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.test_tube_geometry import (  # noqa: E402
    GEOMETRY_CONFIG_SCHEMA,
    compile_geometry_contracts,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(config_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image {path}")
    return image


def _point(value: object) -> tuple[int, int]:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("overlay point must contain two finite coordinates")
    return int(round(point[0])), int(round(point[1]))


def _caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    for index, line in enumerate(lines):
        y = 24 + index * 21
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 3)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
    return canvas


def _draw_arrow(
    image: np.ndarray,
    start: object,
    end: object,
    colour: tuple[int, int, int],
    label: str,
) -> None:
    p0, p1 = _point(start), _point(end)
    cv2.arrowedLine(image, p0, p1, colour, 3, cv2.LINE_AA, tipLength=0.08)
    midpoint = ((p0[0] + p1[0]) // 2 + 4, (p0[1] + p1[1]) // 2 - 6)
    cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)


def _render_overlay(
    *,
    config: dict[str, Any],
    demo_dir: Path,
    live_dir: Path,
    output_path: Path,
) -> None:
    source = config["review"]["source_grasp"]
    live_source = config["review"]["live_source"]
    socket = config["review"]["socket_reference"]
    source_image = _read_image(demo_dir / "rgb" / f"{int(source['frame']):06d}.png")
    terminal_image = _read_image(
        demo_dir / "rgb" / f"{int(socket['demo_terminal_frame']):06d}.png"
    )
    live_image = _read_image(
        live_dir / "rgb" / f"{int(live_source['live_review_frame']):06d}.png"
    )

    green = (70, 220, 70)
    magenta = (255, 50, 255)
    orange = (0, 165, 255)
    cyan = (255, 255, 0)

    _draw_arrow(
        source_image,
        source["closed_end_uv"],
        source["open_end_uv"],
        green,
        "tube axis: closed -> open",
    )
    pinch = _point(source["pinch_centre_uv"])
    cv2.circle(source_image, pinch, 8, magenta, 3, cv2.LINE_AA)
    cv2.putText(
        source_image,
        "demo pinch centre",
        (pinch[0] + 10, pinch[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        magenta,
        2,
    )
    source_image = _caption(
        source_image,
        ["DEMO SOURCE GRASP", "dimensionless axial relation; cylindrical roll remains free"],
    )

    _draw_arrow(
        live_image,
        live_source["closed_end_uv"],
        live_source["open_end_uv"],
        green,
        "source tube",
    )
    live_image[:] = _caption(
        live_image,
        ["PRE-EXECUTION SOURCE REFERENCE", "matches the demonstration initial steep tube"],
    )
    distractor = live_source.get("distractor_axis_uv")
    if distractor is not None:
        _draw_arrow(live_image, distractor[0], distractor[1], orange, "rack / other tube")

    _draw_arrow(
        terminal_image,
        socket["closed_end_uv"],
        socket["open_end_uv"],
        green,
        "settled tube / socket axis",
    )
    entry = _point(socket["socket_entry_uv"])
    cv2.circle(terminal_image, entry, 10, cyan, 3, cv2.LINE_AA)
    cv2.putText(
        terminal_image,
        "2D entry seed",
        (entry[0] + 12, entry[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        cyan,
        2,
    )
    terminal_image[:] = _caption(
        terminal_image,
        ["DEMO TERMINAL SOCKET REFERENCE", "same fixed camera; exact 3D pose unavailable"],
    )

    border = np.full((source_image.shape[0], 6, 3), 255, dtype=np.uint8)
    montage = cv2.hconcat([source_image, border, live_image, border, terminal_image])
    footer_height = 52
    footer = np.full((footer_height, montage.shape[1], 3), 25, dtype=np.uint8)
    footer_lines = [
        "PLAN-ONLY: image/projective constraints reviewed; metric transforms stay null.",
        "Transparent depth requires a fresh execution-start observation before motion.",
    ]
    for index, line in enumerate(footer_lines):
        cv2.putText(
            footer,
            line,
            (10, 20 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    rendered = cv2.vconcat([montage, footer])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}{output_path.suffix}")
    if not cv2.imwrite(str(temporary), rendered):
        raise ValueError(f"could not write overlay {temporary}")
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/test_tube_geometry_20261023.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("schema") != GEOMETRY_CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {GEOMETRY_CONFIG_SCHEMA!r}")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be an object")

    demo_dir = _resolve(config_path, paths.get("demo_dir"), name="paths.demo_dir")
    live_dir = _resolve(
        config_path, paths.get("live_reference_dir"), name="paths.live_reference_dir"
    )
    K_path = _resolve(config_path, paths.get("K"), name="paths.K")
    transform_path = _resolve(
        config_path, paths.get("T_base_camera"), name="paths.T_base_camera"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _resolve(config_path, paths.get("output_dir"), name="paths.output_dir")
    )
    names = {
        "tube_grasp_relation": output_dir / "tube_grasp_relation.json",
        "live_tube_pose": output_dir / "live_tube_pose.json",
        "socket_frame": output_dir / "socket_frame.json",
    }
    overlay_path = output_dir / "test_tube_geometry_overlay.png"
    report_path = output_dir / "geometry_compile_report.json"
    outputs = [*names.values(), overlay_path, report_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(
            "refusing to overwrite existing outputs; pass --overwrite: "
            + ", ".join(map(str, existing))
        )

    K = np.load(K_path)
    T_base_camera = np.load(transform_path)
    depth_paths = sorted((live_dir / "depth").glob("*.npy"))
    if not depth_paths:
        raise ValueError(f"no live reference depth frames in {live_dir / 'depth'}")
    depths = [np.load(path) for path in depth_paths]
    source_frame = int(config["review"]["source_grasp"]["frame"])
    terminal_frame = int(config["review"]["socket_reference"]["demo_terminal_frame"])
    live_frame = int(config["review"]["live_source"]["live_review_frame"])
    source_hand_frame = int(
        config["review"]["live_source"]["open_hand_depth_proxy"]["demo_frame"]
    )
    socket_hand_frame = int(
        config["review"]["socket_reference"]["open_hand_depth_proxy"]["demo_frame"]
    )
    source_image_path = demo_dir / "rgb" / f"{source_frame:06d}.png"
    terminal_image_path = demo_dir / "rgb" / f"{terminal_frame:06d}.png"
    terminal_depth_path = demo_dir / "depth" / f"{terminal_frame:06d}.npy"
    source_hand_depth_path = demo_dir / "depth" / f"{source_hand_frame:06d}.npy"
    socket_hand_depth_path = demo_dir / "depth" / f"{socket_hand_frame:06d}.npy"
    live_image_path = live_dir / "rgb" / f"{live_frame:06d}.png"
    manifest_path = live_dir / "input_manifest.json"
    source_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "src/demo2gripper/test_tube_geometry.py",
        config_path,
        K_path,
        transform_path,
        source_image_path,
        terminal_image_path,
        terminal_depth_path,
        source_hand_depth_path,
        socket_hand_depth_path,
        live_image_path,
        manifest_path,
        *depth_paths,
    ]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise ValueError("missing input: " + ", ".join(map(str, missing)))
    provenance = {
        "compiler": "scripts/compile_test_tube_geometry.py",
        "inputs": [
            {"path": _repo_path(path), "sha256": _sha256(path)} for path in source_paths
        ],
        "calibration_limit": config["assumptions"],
        "depth_interpretation": (
            "Depth near the live source tip and terminal 2D entry is retained only as an "
            "adjacent cup/support/rack proxy. It is not assigned to the transparent tube "
            "or socket centre."
        ),
    }
    contracts = compile_geometry_contracts(
        config=config,
        K=K,
        T_base_camera=T_base_camera,
        live_depth_frames=depths,
        socket_depth_frames=[np.load(terminal_depth_path)],
        source_hand_depth_frames=[np.load(source_hand_depth_path)],
        socket_hand_depth_frames=[np.load(socket_hand_depth_path)],
        provenance=provenance,
    )
    for key, path in names.items():
        _atomic_json(path, contracts[key])
    _render_overlay(config=config, demo_dir=demo_dir, live_dir=live_dir, output_path=overlay_path)
    report = {
        "schema": "demo2gripper.test_tube_geometry_compile_report.v1",
        "task_id": config.get("task_id"),
        "status": "plan_only_geometry_evidence_complete",
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "artifacts": {
            key: {
                "path": _repo_path(path),
                "schema": contracts[key]["schema"],
                "sha256": _sha256(path),
                "reviewed": True,
            }
            for key, path in names.items()
        }
        | {
            "overlay": {
                "path": _repo_path(overlay_path),
                "sha256": _sha256(overlay_path),
            }
        },
        "plan_only_ready": True,
        "metric_execution_geometry_ready": False,
        "single_execution_geometry_blocker": (
            "At execution start, reacquire the moved source tube and measure a metric "
            "T_base_tube plus rack-entry origin/axis/insertion depth; current transparent "
            "RGB-D supplies only image/projective references."
        ),
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
