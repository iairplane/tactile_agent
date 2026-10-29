#!/usr/bin/env python3
"""Recover an auditable top-down test-tube grasp relation from a human demo.

The command is offline and never opens a camera, robot, ROS graph, or gripper.
It always writes the observable constraint set.  A full ``T_tube_tcp`` is
emitted only when independent metric tube-pose evidence and a one-time visual
review satisfy the explicit qualification contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.test_tube_demo_relation import (  # noqa: E402
    DEMO_RELATION_RECEIPT_SCHEMA,
    DEMO_RELATION_REVIEW_SCHEMA,
    DEMO_TUBE_POSE_RECEIPT_SCHEMA,
    estimate_pinch_3d,
    fit_tube_from_pinch_and_length,
    instantiate_top_down_relation,
    interpolate_pinch_keypoints,
    make_tube_frame,
    parse_human_review,
    parse_qualified_pose_receipt,
    relation_uncertainty,
    runner_geometry_fragment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--demo-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grasp-frame", type=int)
    parser.add_argument("--tube-pose-receipt", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument("--depth-radius-px", type=int, default=3)
    parser.add_argument("--max-keypoint-frame-gap", type=int, default=10)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, str]:
    try:
        display = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": _sha256(path)}


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_image(path: Path, image: np.ndarray, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise ValueError(f"could not write image {temporary}")
    temporary.replace(path)


def _draw_overlay(
    image: np.ndarray,
    *,
    closed_uv: np.ndarray,
    open_uv: np.ndarray,
    thumb_uv: np.ndarray,
    index_uv: np.ndarray,
    runtime_qualified: bool,
) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 96), (18, 18, 18), -1)

    def point(value: np.ndarray) -> tuple[int, int]:
        return tuple(np.rint(value).astype(int))

    cv2.arrowedLine(
        canvas,
        point(closed_uv),
        point(open_uv),
        (40, 220, 40),
        3,
        cv2.LINE_AA,
        tipLength=0.08,
    )
    cv2.putText(
        canvas,
        "tube +Z: closed -> open",
        point(0.55 * closed_uv + 0.45 * open_uv),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(
        canvas,
        point(index_uv),
        point(thumb_uv),
        (255, 80, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.12,
    )
    midpoint = 0.5 * (thumb_uv + index_uv)
    cv2.circle(canvas, point(midpoint), 8, (0, 210, 255), 3, cv2.LINE_AA)
    status = "QUALIFIED T_tube_tcp" if runtime_qualified else "PARTIAL: T_tube_tcp = null"
    lines = [
        "DEMO RELATION REVIEW - OFFLINE ONLY",
        "magenta: RTMPose index->thumb; orange: pinch midpoint",
        "top approach: TCP +Z and final path follow base -Z",
        status,
    ]
    for index, line in enumerate(lines):
        y = 18 + index * 22
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
        )
    return canvas


def _review_template(*, frame: int, relation_id: str, overlay_path: Path) -> dict[str, Any]:
    return {
        "schema": DEMO_RELATION_REVIEW_SCHEMA,
        "relation_id": relation_id,
        "frame": frame,
        "review_overlay": str(overlay_path),
        "accepted": False,
        "closed_open_labels_accepted": False,
        "pinch_keypoints_accepted": False,
        "top_approach_accepted": False,
        "reviewer": "",
        "note": (
            "Set the four booleans only after checking the overlay. This review does "
            "not replace the independent metric tube-pose receipt."
        ),
    }


def _pose_template(*, frame: int) -> dict[str, Any]:
    return {
        "schema": DEMO_TUBE_POSE_RECEIPT_SCHEMA,
        "frame": frame,
        "runtime_qualified": False,
        "T_camera_tube": None,
        "tube_frame": {
            "origin": "geometric_centre",
            "z_axis": "closed_to_open",
            "roll": "canonical_gauge",
        },
        "tube_length_m": None,
        "tube_length_uncertainty_m": None,
        "position_std_m": None,
        "orientation_std_deg": None,
        "provenance": {},
        "note": (
            "Populate from an independently qualified same-frame pose/marker result; "
            "do not copy the diagnostic candidate from the relation receipt."
        ),
    }


def run(args: argparse.Namespace) -> int:
    demo_dir = args.demo_dir.expanduser().resolve()
    analysis_dir = demo_dir / "analysis"
    relation_path = analysis_dir / "tube_grasp_relation.json"
    keypoints_path = analysis_dir / "rtmpose_grasp_review" / "review.json"
    K_path = demo_dir / "K.npy"
    T_base_camera_path = REPO_ROOT / "calibration" / "T_base_camera.npy"
    for path in (relation_path, keypoints_path, K_path, T_base_camera_path):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    relation = _read_json(relation_path)
    keypoint_review = _read_json(keypoints_path)
    measurement = relation.get("demo_measurement")
    if not isinstance(measurement, dict):
        raise ValueError("tube_grasp_relation.demo_measurement must be an object")
    frame = int(args.grasp_frame if args.grasp_frame is not None else measurement["frame"])
    rgb_path = demo_dir / "rgb" / f"{frame:06d}.png"
    depth_path = demo_dir / "depth" / f"{frame:06d}.npy"
    if not rgb_path.is_file() or not depth_path.is_file():
        raise ValueError(f"RGB-D frame {frame} is missing")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else analysis_dir / "tube_demo_relation_receipt.json"
    )
    overlay_path = output_path.with_name(f"{output_path.stem}_review.png")
    review_template_path = output_path.with_name("tube_demo_relation_review.template.json")
    pose_template_path = output_path.with_name("tube_demo_pose_receipt.template.json")

    K = np.asarray(np.load(K_path, allow_pickle=False), dtype=float)
    T_base_camera = np.asarray(
        np.load(T_base_camera_path, allow_pickle=False), dtype=float
    )
    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=float)
    records = keypoint_review.get("records")
    if not isinstance(records, list):
        raise ValueError("RTMPose review records must be an array")
    pinch_2d = interpolate_pinch_keypoints(
        records,
        frame=frame,
        max_frame_gap=args.max_keypoint_frame_gap,
    )
    pinch_3d = estimate_pinch_3d(
        K=K,
        depth_m=depth,
        pinch=pinch_2d,
        depth_radius_px=args.depth_radius_px,
    )

    closed_uv = np.asarray(measurement.get("closed_end_uv_px"), dtype=float)
    open_uv = np.asarray(measurement.get("open_end_uv_px"), dtype=float)
    fraction = float(measurement.get("pinch_axial_fraction_closed_to_open"))
    fraction_uncertainty = float(measurement.get("pinch_axial_fraction_uncertainty"))
    assumptions = relation.get("dimension_assumptions")
    if not isinstance(assumptions, dict):
        raise ValueError("tube relation dimension_assumptions must be an object")
    nominal_length = float(assumptions.get("tube_length_m"))
    nominal_length_uncertainty = float(assumptions.get("tube_length_uncertainty_m"))
    candidate = fit_tube_from_pinch_and_length(
        K=K,
        closed_uv=closed_uv,
        open_uv=open_uv,
        pinch_camera_m=pinch_3d.midpoint_camera_m,
        axial_fraction_closed_to_open=fraction,
        tube_length_m=nominal_length,
        pinch_position_sigma_m=max(
            0.004,
            0.5 * max(pinch_3d.thumb_depth.spread_m, pinch_3d.index_depth.spread_m),
        ),
        tube_length_sigma_m=nominal_length_uncertainty,
    )
    camera_base = np.linalg.inv(T_base_camera)
    T_camera_tube_candidate = make_tube_frame(
        centre_parent_m=candidate.centre_camera_m,
        axis_closed_to_open_parent=candidate.axis_camera_unit,
        transverse_reference_parent=camera_base[:3, :3] @ np.array([1.0, 0.0, 0.0]),
    )
    candidate_relation, candidate_detail = instantiate_top_down_relation(
        T_base_camera=T_base_camera,
        T_camera_tube=T_camera_tube_candidate,
        tube_length_m=nominal_length,
        axial_fraction_closed_to_open=fraction,
        human_closing_axis_camera=pinch_3d.closing_axis_camera_unit,
    )

    blockers: list[str] = []
    if pinch_2d.confidence < args.min_keypoint_confidence:
        blockers.append("pinch_keypoint_confidence_too_low")
    if relation.get("T_object_gripper_tcp") is not None:
        blockers.append("legacy_relation_unexpectedly_contains_transform")
    pose_payload = (
        None
        if args.tube_pose_receipt is None
        else _read_json(args.tube_pose_receipt.expanduser().resolve())
    )
    T_camera_tube, qualified_length, pose_blockers, pose_detail = (
        parse_qualified_pose_receipt(pose_payload, expected_frame=frame)
    )
    blockers.extend(pose_blockers)
    review_payload = (
        None
        if args.human_review is None
        else _read_json(args.human_review.expanduser().resolve())
    )
    review_blockers, review_detail = parse_human_review(
        review_payload, expected_frame=frame
    )
    blockers.extend(review_blockers)
    blockers = sorted(set(blockers))

    full_transform: np.ndarray | None = None
    full_detail: dict[str, Any] | None = None
    uncertainty: dict[str, float] | None = None
    if not blockers and T_camera_tube is not None and qualified_length is not None:
        full_transform, full_detail = instantiate_top_down_relation(
            T_base_camera=T_base_camera,
            T_camera_tube=T_camera_tube,
            tube_length_m=qualified_length,
            axial_fraction_closed_to_open=fraction,
            human_closing_axis_camera=pinch_3d.closing_axis_camera_unit,
        )
        uncertainty = relation_uncertainty(
            tube_length_m=qualified_length,
            tube_length_uncertainty_m=float(
                pose_payload["tube_length_uncertainty_m"]  # type: ignore[index]
            ),
            axial_fraction=fraction,
            axial_fraction_uncertainty=fraction_uncertainty,
            pose_orientation_std_deg=float(
                pose_payload["orientation_std_deg"]  # type: ignore[index]
            ),
        )

    relation_id = f"{relation.get('task_id', demo_dir.name)}-top-down-demo-relation"
    provenance_paths = [
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "demo2gripper" / "test_tube_demo_relation.py",
        relation_path,
        keypoints_path,
        K_path,
        T_base_camera_path,
        rgb_path,
        depth_path,
    ]
    if args.tube_pose_receipt is not None:
        provenance_paths.append(args.tube_pose_receipt.expanduser().resolve())
    if args.human_review is not None:
        provenance_paths.append(args.human_review.expanduser().resolve())
    runtime_qualified = full_transform is not None and not blockers
    receipt: dict[str, Any] = {
        "schema": DEMO_RELATION_RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "relation_id": relation_id,
        "task_id": relation.get("task_id", demo_dir.name),
        "status": "qualified_full_relation" if runtime_qualified else "partial_constraints_only",
        "runtime_qualified": runtime_qualified,
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "transform_convention": "T_parent_child; metres; right-handed",
        "tube_frame": {
            "origin": "geometric_centre",
            "z_axis": "closed_to_open",
            "roll": "canonical gauge; no task meaning for a circular tube",
        },
        "tcp_frame": {
            "origin": "tube centreline at demonstrated axial fraction",
            "x_axis": "parallel-jaw closing axis",
            "z_axis": "base -Z top-approach direction",
        },
        "provenance": {
            "compiler": "scripts/recover_test_tube_demo_relation.py",
            "inputs": [_path_record(path) for path in provenance_paths],
        },
        "observable_demo_constraints": {
            "frame": frame,
            "tube_closed_end_uv_px": closed_uv.tolist(),
            "tube_open_end_uv_px": open_uv.tolist(),
            "tube_end_sign_source": "reviewed image labels in legacy relation",
            "pinch_axial_fraction_closed_to_open": fraction,
            "pinch_axial_fraction_uncertainty": fraction_uncertainty,
            "pinch_2d": pinch_2d.as_dict(),
            "pinch_3d": pinch_3d.as_dict(),
            "closing_axis_constraint": (
                "jaw axis perpendicular to tube axis; thumb/index direction selects "
                "only the symmetric jaw-axis sign"
            ),
            "top_approach_direction_base_unit": [0.0, 0.0, -1.0],
        },
        "deterministic_constrained_candidate": {
            "status": "diagnostic_not_runtime_qualified",
            "method": (
                "fit endpoint ray depths from RGB-D fingertip midpoint, nominal tube "
                "length, and reviewed axial fraction"
            ),
            "unverified_assumption": (
                "the midpoint of anatomical fingertip keypoints lies on the tube centreline"
            ),
            "tube_length_prior_m": nominal_length,
            "tube_length_prior_uncertainty_m": nominal_length_uncertainty,
            "metric_fit": candidate.as_dict(),
            "T_camera_tube_candidate": T_camera_tube_candidate.tolist(),
            "T_tube_tcp_candidate": candidate_relation.tolist(),
            "instantiation": candidate_detail,
            "must_not_be_promoted_to_execution": True,
        },
        "independent_metric_pose_evidence": pose_detail,
        "human_review": review_detail,
        "T_tube_tcp_demo_grasp": None if full_transform is None else full_transform.tolist(),
        "qualified_instantiation": full_detail,
        "uncertainty": uncertainty,
        "runner_preflight_blockers": blockers,
        "runner_geometry_fragment": None,
        "post_pivot_contract": {
            "update": (
                "freeze measured inv(T_base_tube) @ T_base_tcp after tactile-confirmed "
                "reclamp; preserve the demonstrated axial coordinate unless axial slip "
                "is observed"
            ),
            "angle_source": (
                "qualified front-RGB relative tube-axis change or calibrated tactile "
                "rotation primitive; never invent an angle from force direction"
            ),
            "does_not_require_unoccluded_f2g_after_contact": True,
        },
        "next_action": (
            "Review the overlay once and provide an independently qualified same-frame "
            "metric tube-pose receipt with measured length."
            if not runtime_qualified
            else "Copy runner_geometry_fragment into the task configuration."
        ),
    }
    if runtime_qualified:
        receipt["runner_geometry_fragment"] = runner_geometry_fragment(receipt)

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read {rgb_path}")
    overlay = _draw_overlay(
        image,
        closed_uv=closed_uv,
        open_uv=open_uv,
        thumb_uv=pinch_2d.thumb_uv,
        index_uv=pinch_2d.index_uv,
        runtime_qualified=runtime_qualified,
    )
    _atomic_json(output_path, receipt, overwrite=args.overwrite)
    _atomic_image(overlay_path, overlay, overwrite=args.overwrite)
    if not review_template_path.exists() or args.overwrite:
        _atomic_json(
            review_template_path,
            _review_template(frame=frame, relation_id=relation_id, overlay_path=overlay_path),
            overwrite=args.overwrite,
        )
    if not pose_template_path.exists() or args.overwrite:
        _atomic_json(
            pose_template_path,
            _pose_template(frame=frame),
            overwrite=args.overwrite,
        )
    print(json.dumps({
        "status": receipt["status"],
        "runtime_qualified": runtime_qualified,
        "output": str(output_path),
        "review_overlay": str(overlay_path),
        "blockers": blockers,
    }, indent=2))
    return 0 if runtime_qualified else 2


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
