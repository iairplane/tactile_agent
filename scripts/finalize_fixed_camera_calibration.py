#!/usr/bin/env python3
"""Bind an accepted eye-to-hand result to the live test-tube configuration.

``solve_eye_to_hand.py`` is the authority that publishes calibration matrices:
it changes its output directory only after all training, held-out validation,
method-consistency, and Park cross-check gates pass.  This tool performs the
small second publication step needed by the live stack.  It verifies the
accepted report against ``T_base_camera.npy``, emits a digest-bound fixed-camera
qualification receipt, and atomically updates one or more task configs.

It never opens a camera, ROS node, serial device, or robot connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from demo2gripper.franka import validate_rigid_transform  # noqa: E402

CALIBRATION_SCHEMA = "demo2gripper.eye_to_hand_calibration.v1"
LIVE_QUALIFICATION_SCHEMA = "demo2gripper.test_tube_live_io_qualification.v1"
TASK_CONFIG_SCHEMA = "demo2gripper.test_tube_closed_loop_config.v1"
QUALIFICATION_KIND = "fixed_camera_extrinsic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=REPOSITORY_ROOT / "calibration",
        help="Accepted solver output containing T_base_camera.npy and report.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help=(
            "Qualification receipt path; defaults to "
            "<calibration-dir>/T_base_camera.qualification.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Live task config to update; repeat for multiple configs",
    )
    parser.add_argument(
        "--camera-serial",
        help="Expected camera serial; defaults to each config's declared serial",
    )
    return parser


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _repo_display(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def qualification_receipt(
    calibration_dir: Path, *, expected_camera_serial: str
) -> dict[str, Any]:
    calibration_dir = calibration_dir.expanduser().resolve()
    transform_path = calibration_dir / "T_base_camera.npy"
    report_path = calibration_dir / "report.json"
    if not transform_path.is_file():
        raise ValueError(f"T_base_camera.npy is missing: {transform_path}")
    report = _read_object(report_path, label="calibration report")
    if report.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("calibration report schema is unsupported")
    acceptance = _mapping(report.get("acceptance"), label="report.acceptance")
    if acceptance.get("accepted") is not True or acceptance.get("failures") not in ([], None):
        raise ValueError("calibration report is not accepted")

    generation_id = _nonempty(report.get("generation_id"), label="report.generation_id")
    provenance = _mapping(
        report.get("sample_provenance"), label="report.sample_provenance"
    )
    if (
        provenance.get("verification_policy") != "strict_manifests"
        or provenance.get("all_samples_verified") is not True
    ):
        raise ValueError("calibration samples are not all strict-manifest verified")
    camera = _mapping(provenance.get("camera"), label="report.sample_provenance.camera")
    report_serial = _nonempty(camera.get("serial"), label="report camera serial")
    if report_serial != expected_camera_serial:
        raise ValueError(
            f"report camera serial {report_serial!r} does not match "
            f"expected {expected_camera_serial!r}"
        )

    training = report.get("training_indices")
    validation = report.get("validation_indices")
    if not isinstance(training, list) or not isinstance(validation, list):
        raise ValueError("calibration report sample partitions are invalid")
    thresholds = _mapping(report.get("thresholds"), label="report.thresholds")
    minimum_training = int(thresholds.get("min_training_samples", 6))
    minimum_validation = int(thresholds.get("min_validation_samples", 3))
    if len(training) < minimum_training:
        raise ValueError("accepted report contains too few training samples")
    if len(validation) < minimum_validation or len(validation) < 3:
        raise ValueError("fixed-camera publication requires at least three holdout samples")

    selected_method = _nonempty(
        report.get("selected_method"), label="report.selected_method"
    )
    solutions = _mapping(report.get("solutions"), label="report.solutions")
    selected = _mapping(solutions.get(selected_method), label="selected solution")
    reported_transform = validate_rigid_transform(
        np.asarray(selected.get("T_base_camera"), dtype=float),
        name="reported T_base_camera",
    )
    artifact_transform = validate_rigid_transform(
        np.load(transform_path, allow_pickle=False), name="published T_base_camera"
    )
    if not np.allclose(reported_transform, artifact_transform, atol=1e-10, rtol=0.0):
        raise ValueError("T_base_camera.npy does not match the accepted selected solution")

    validation_residual = _mapping(
        selected.get("validation_residual"), label="selected validation_residual"
    )
    if int(validation_residual.get("count", -1)) != len(validation):
        raise ValueError("validation residual count does not match holdout partition")
    translation = _mapping(
        validation_residual.get("translation_mm"),
        label="validation translation residual",
    )
    rotation = _mapping(
        validation_residual.get("rotation_deg"),
        label="validation rotation residual",
    )
    # The live stack treats these fields as propagated uncertainty.  The
    # observed maximum holdout closure errors are a conservative bound and are
    # therefore preferable here to the smaller RMS values.
    position_bound_m = _nonnegative(
        translation.get("max"), label="maximum validation translation"
    ) / 1000.0
    orientation_bound_deg = _nonnegative(
        rotation.get("max"), label="maximum validation rotation"
    )
    if position_bound_m > float(thresholds["max_validation_translation_mm"]) / 1000.0:
        raise ValueError("validation translation exceeds the published threshold")
    if orientation_bound_deg > float(thresholds["max_validation_rotation_deg"]):
        raise ValueError("validation rotation exceeds the published threshold")

    park = _mapping(report.get("park_cross_check"), label="report.park_cross_check")
    if park.get("enabled") is not True:
        raise ValueError("fixed-camera publication requires the Park cross-check")

    qualification_id = f"fixed-camera-extrinsic-{generation_id}"
    return {
        "schema": LIVE_QUALIFICATION_SCHEMA,
        "kind": QUALIFICATION_KIND,
        "qualification_id": qualification_id,
        "runtime_qualified": True,
        "workspace_calibration_id": f"eye-to-hand-{generation_id}",
        "created_utc": datetime.now(UTC).isoformat(),
        "camera_serial": report_serial,
        "frame_convention": report.get("frame_convention"),
        "artifact": _repo_display(transform_path),
        "artifact_sha256": _sha256(transform_path),
        "source_report": _repo_display(report_path),
        "source_report_sha256": _sha256(report_path),
        "source_samples_dir": report.get("samples_dir"),
        "solver_generation_id": generation_id,
        "selected_method": selected_method,
        "training_sample_count": len(training),
        "validation_sample_count": len(validation),
        "position_std_m": position_bound_m,
        "orientation_std_deg": orientation_bound_deg,
        "uncertainty_basis": "maximum held-out closure residual",
        "park_cross_check_enabled": True,
    }


def _config_camera(config: Mapping[str, Any]) -> Mapping[str, Any]:
    live = _mapping(config.get("live_adapter"), label="config.live_adapter")
    return _mapping(live.get("camera"), label="config.live_adapter.camera")


def _update_config(
    path: Path,
    *,
    transform_path: Path,
    receipt_path: Path,
    expected_camera_serial: str,
) -> None:
    config = _read_object(path, label="task config")
    if config.get("schema") != TASK_CONFIG_SCHEMA:
        raise ValueError(f"unsupported task config schema: {path}")
    live = dict(_mapping(config.get("live_adapter"), label="config.live_adapter"))
    camera = dict(_mapping(live.get("camera"), label="config.live_adapter.camera"))
    declared_serial = _nonempty(camera.get("serial"), label="config camera serial")
    if declared_serial != expected_camera_serial:
        raise ValueError(
            f"config camera serial {declared_serial!r} does not match "
            f"calibration camera {expected_camera_serial!r}: {path}"
        )
    camera["camera_extrinsic"] = _repo_display(transform_path)
    camera["camera_extrinsic_qualification_receipt"] = _repo_display(receipt_path)
    live["camera"] = camera
    config["live_adapter"] = live
    _atomic_json(path, config)


def run(args: argparse.Namespace) -> int:
    calibration_dir = args.calibration_dir.expanduser().resolve()
    transform_path = calibration_dir / "T_base_camera.npy"
    receipt_path = (
        args.receipt.expanduser().resolve()
        if args.receipt is not None
        else calibration_dir / "T_base_camera.qualification.json"
    )
    configs = [path.expanduser().resolve() for path in args.config]
    if not configs:
        raise ValueError("at least one --config is required")

    declared_serials = {
        _nonempty(_config_camera(_read_object(path, label="task config")).get("serial"),
                  label="config camera serial")
        for path in configs
    }
    if len(declared_serials) != 1:
        raise ValueError("all task configs must declare the same camera serial")
    declared_serial = next(iter(declared_serials))
    expected_serial = args.camera_serial or declared_serial
    if expected_serial != declared_serial:
        raise ValueError("--camera-serial does not match the task config")

    receipt = qualification_receipt(
        calibration_dir, expected_camera_serial=expected_serial
    )
    _atomic_json(receipt_path, receipt)
    for config in configs:
        _update_config(
            config,
            transform_path=transform_path,
            receipt_path=receipt_path,
            expected_camera_serial=expected_serial,
        )

    print(f"wrote {receipt_path}")
    for config in configs:
        print(f"updated {config}")
    print(
        "held-out bound: "
        f"{1000.0 * float(receipt['position_std_m']):.3f} mm, "
        f"{float(receipt['orientation_std_deg']):.3f} deg"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
