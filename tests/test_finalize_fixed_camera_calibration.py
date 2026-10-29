from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import finalize_fixed_camera_calibration as finalize

SERIAL = "254622073804"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    transform = np.eye(4)
    transform[:3, 3] = [0.6, 0.5, 0.2]
    np.save(calibration / "T_base_camera.npy", transform)
    report = {
        "schema": finalize.CALIBRATION_SCHEMA,
        "generation_id": "unit-generation",
        "frame_convention": "T_parent_child, metres",
        "samples_dir": str(tmp_path / "samples"),
        "sample_provenance": {
            "verification_policy": "strict_manifests",
            "all_samples_verified": True,
            "camera": {"serial": SERIAL},
        },
        "training_indices": list(range(9)),
        "validation_indices": [9, 10, 11],
        "selected_method": "shah",
        "thresholds": {
            "min_training_samples": 6,
            "min_validation_samples": 3,
            "max_validation_translation_mm": 5.0,
            "max_validation_rotation_deg": 1.0,
        },
        "solutions": {
            "shah": {
                "T_base_camera": transform.tolist(),
                "validation_residual": {
                    "count": 3,
                    "translation_mm": {"max": 2.5, "rms": 1.5},
                    "rotation_deg": {"max": 0.4, "rms": 0.2},
                },
            }
        },
        "park_cross_check": {"enabled": True},
        "acceptance": {"accepted": True, "failures": []},
    }
    (calibration / "report.json").write_text(json.dumps(report), encoding="utf-8")
    config = tmp_path / "task.json"
    config.write_text(
        json.dumps(
            {
                "schema": finalize.TASK_CONFIG_SCHEMA,
                "live_adapter": {
                    "camera": {
                        "serial": SERIAL,
                        "camera_extrinsic": None,
                        "camera_extrinsic_qualification_receipt": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return calibration, config


def test_finalize_binds_digest_and_conservative_holdout_bound(tmp_path: Path) -> None:
    calibration, config = _write_inputs(tmp_path)
    receipt = tmp_path / "camera-receipt.json"
    args = finalize.build_parser().parse_args(
        [
            "--calibration-dir",
            str(calibration),
            "--receipt",
            str(receipt),
            "--config",
            str(config),
        ]
    )

    assert finalize.run(args) == 0

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["runtime_qualified"] is True
    assert value["position_std_m"] == 0.0025
    assert value["orientation_std_deg"] == 0.4
    assert value["validation_sample_count"] == 3
    assert value["artifact_sha256"] == hashlib.sha256(
        (calibration / "T_base_camera.npy").read_bytes()
    ).hexdigest()
    task = json.loads(config.read_text(encoding="utf-8"))
    assert task["live_adapter"]["camera"]["camera_extrinsic"] == str(
        calibration / "T_base_camera.npy"
    )
    assert task["live_adapter"]["camera"][
        "camera_extrinsic_qualification_receipt"
    ] == str(receipt)


def test_finalize_rejects_nonaccepted_report(tmp_path: Path) -> None:
    calibration, _ = _write_inputs(tmp_path)
    report_path = calibration / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["acceptance"] = {"accepted": False, "failures": ["unit failure"]}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    try:
        finalize.qualification_receipt(
            calibration, expected_camera_serial=SERIAL
        )
    except ValueError as exc:
        assert "not accepted" in str(exc)
    else:
        raise AssertionError("nonaccepted calibration must not receive a receipt")
