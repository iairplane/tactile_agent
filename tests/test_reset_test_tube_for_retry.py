from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _load_script():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/reset_test_tube_for_retry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "reset_test_tube_for_retry", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contact() -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = [0.58, 0.19, 0.173]
    return value


def test_targets_restore_orientation_then_lower_and_retreat() -> None:
    module = _load_script()
    contact = _contact()
    current = contact.copy()
    current[:3, :3] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    current[:3, 3] += [0.001, -0.001, 0.041]

    targets = module.build_reset_targets(contact, current, retreat_mm=80.0)

    restored = targets["restore_contact_orientation_at_high_pose"]
    assert restored[:3, :3] == pytest.approx(contact[:3, :3])
    assert restored[:2, 3] == pytest.approx(contact[:2, 3])
    assert restored[2, 3] == pytest.approx(current[2, 3])
    assert targets["lower_to_supported_contact"] == pytest.approx(contact)
    assert targets["retreat_after_release"][:3, 3] == pytest.approx(
        contact[:3, 3] + [0.0, 0.0, 0.080]
    )


def test_live_start_accepts_current_partial_level_envelope() -> None:
    module = _load_script()
    contact = _contact()
    current = contact.copy()
    current[:3, :3] = Rotation.from_euler("y", 24.5, degrees=True).as_matrix()
    current[:3, 3] += [0.001, -0.001, 0.0408]

    evidence = module.validate_live_start(contact, current)

    assert evidence["lift_mm"] == pytest.approx(40.8)
    assert evidence["xy_error_mm"] == pytest.approx(2**0.5)
    assert evidence["rotation_error_deg"] == pytest.approx(24.5)


@pytest.mark.parametrize(
    "delta, match",
    [
        ([0.0, 0.0, 0.010], "outside the reviewed reset envelope"),
        ([0.010, 0.0, 0.040], "XY drift"),
    ],
)
def test_live_start_rejects_pose_outside_reset_envelope(delta, match) -> None:
    module = _load_script()
    contact = _contact()
    current = contact.copy()
    current[:3, 3] += delta
    with pytest.raises(RuntimeError, match=match):
        module.validate_live_start(contact, current)


def test_load_contact_reference_accepts_old_and_new_strict_status(tmp_path: Path) -> None:
    module = _load_script()
    for status in ("bilateral_contact", "grasp_ready"):
        directory = tmp_path / status
        directory.mkdir()
        payload = {
            "schema": "demo2gripper.test_tube_fast_probe.v1",
            "status": status,
            "T_base_tcp_final": _contact().tolist(),
            "final_gripper": {"width_m": 0.0175},
        }
        (directory / "run_report.json").write_text(json.dumps(payload))
        pose, width, report = module.load_contact_reference(directory)
        assert pose == pytest.approx(_contact())
        assert width == pytest.approx(0.0175)
        assert report["status"] == status


def test_execute_requires_collision_model_ack() -> None:
    module = _load_script()
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                "unused.json",
                "--close-run",
                "unused",
                "--output-dir",
                "unused-output",
                "--execute",
            ]
        )
