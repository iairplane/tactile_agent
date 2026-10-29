#!/usr/bin/env python3
"""Generate non-executable Hand-E candidates around a live circular cup rim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from demo2gripper.circular_rim_candidates import generate_circular_rim_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--live-t-base-rim", type=Path, required=True)
    parser.add_argument(
        "--cup-rim-relation",
        type=Path,
        required=True,
        help="cup_rim_relation.json or cup_rim_geometry.npz",
    )
    parser.add_argument("--t-link8-tcp", type=Path, required=True)
    parser.add_argument("--demo-t-base-rim", type=Path)
    parser.add_argument("--reference-radial-base", type=float, nargs=3)
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        nargs="+",
        default=[0.0, 90.0, -90.0, 180.0],
    )
    parser.add_argument("--approach-distance-m", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_transform(path: Path, *, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(f"{path} does not contain {key}")
            return np.asarray(archive[key], dtype=float)
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        if key not in payload:
            raise ValueError(f"{path} does not contain {key}")
        return np.asarray(payload[key], dtype=float)
    raise ValueError(f"unsupported transform file extension: {path}")


def _load_relation(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        if payload.get("schema") != "demo2gripper.cup_rim_relation.v1":
            raise ValueError("cup rim relation has an unsupported schema")
        if payload.get("status") != "review_only" or payload.get(
            "execution_authorized"
        ) is not False:
            raise ValueError("cup rim relation must retain its review-only boundary")
        return np.asarray(payload.get("T_rim_human_pinch_tcp"), dtype=float)
    return _load_transform(path, key="T_rim_human_pinch_tcp")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite {args.output}; pass --overwrite")
    if args.demo_t_base_rim is not None and args.reference_radial_base is not None:
        raise ValueError(
            "supply at most one of --demo-t-base-rim and --reference-radial-base"
        )
    payload = generate_circular_rim_candidates(
        T_base_live_rim=_load_transform(
            args.live_t_base_rim, key="T_base_rim"
        ),
        T_rim_human_pinch_tcp=_load_relation(args.cup_rim_relation),
        T_link8_tcp=_load_transform(args.t_link8_tcp, key="T_link8_tcp"),
        approach_distance_m=args.approach_distance_m,
        azimuth_offsets_deg=args.azimuth_deg,
        T_base_demo_rim=(
            None
            if args.demo_t_base_rim is None
            else _load_transform(args.demo_t_base_rim, key="T_base_rim")
        ),
        reference_radial_base=(
            None
            if args.reference_radial_base is None
            else np.asarray(args.reference_radial_base, dtype=float)
        ),
    )
    payload["source_files"] = {
        "live_T_base_rim": str(args.live_t_base_rim.resolve()),
        "cup_rim_relation": str(args.cup_rim_relation.resolve()),
        "T_link8_tcp": str(args.t_link8_tcp.resolve()),
        "demo_T_base_rim": (
            None
            if args.demo_t_base_rim is None
            else str(args.demo_t_base_rim.resolve())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output}")
    print(
        "status: review_only; MoveIt plan-only evaluation required; "
        "execution_authorized=false"
    )


if __name__ == "__main__":
    main()
