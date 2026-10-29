from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from demo2gripper.gf220_live import (
    GF220Calibration,
    GF220ClassifierWindows,
)
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.test_tube_closed_loop import GripperFeedback
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactBootstrapConfig as ContactBootstrapConfig,
)
from demo2gripper.test_tube_preflight import TubeState
from demo2gripper.test_tube_tactile_controller import TestTubeTactileMode as TactileMode
from demo2gripper.test_tube_tactile_interpreter import (
    AUDIT_SCHEMA,
    QUALIFICATION_SCHEMA,
    FailClosedTestTubeTactileInterpreter,
    QualifiedTestTubeTactileInterpreter,
    create_test_tube_tactile_interpreter,
    mode_for_tube_state,
)

SERIALS = ("tube-left", "tube-right")
SHAPE = (48, 64, 3)


def _calibrations() -> tuple[GF220Calibration, GF220Calibration]:
    baseline = np.full(SHAPE, 100, dtype=np.uint8)
    values = tuple(
        GF220Calibration.create(
            serial=serial,
            warp_sha256=("a" if index == 0 else "b") * 64,
            unloaded_warped=baseline,
        )
        for index, serial in enumerate(SERIALS)
    )
    return values[0], values[1]


def _timed(values: np.ndarray, start: float) -> TimedArrayWindow:
    count = len(values)
    times = start + np.arange(count, dtype=float) * 0.01
    return TimedArrayWindow(values, times, times, np.ones(count, dtype=bool))


def _patch(center: tuple[float, float] = (32.0, 24.0)) -> np.ndarray:
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    image = np.full(SHAPE, 100, dtype=np.uint8)
    image[(np.abs(xx - center[0]) <= 18.0) & (np.abs(yy - center[1]) <= 3.0)] = 190
    return image


def _windows(
    calibrations: tuple[GF220Calibration, GF220Calibration],
    *,
    start: float,
    forces: tuple[float, float] = (4.0, 4.0),
    centers: tuple[tuple[float, float], tuple[float, float]] = (
        (32.0, 24.0),
        (32.0, 24.0),
    ),
) -> GF220ClassifierWindows:
    fingers = []
    for index, calibration in enumerate(calibrations):
        images = np.repeat(_patch(centers[index])[None], 4, axis=0)
        force = np.zeros((4, 6), dtype=float)
        force[:, 2] = forces[index]
        fingers.append(
            FingerRimClassifierWindow(
                calibration_id=calibration.calibration_id,
                warped=_timed(images, start),
                diff=_timed(images, start),
                force6d=_timed(force, start),
            )
        )
    return GF220ClassifierWindows(
        sensor_names=("left", "right"),
        serials=SERIALS,
        windows=(fingers[0], fingers[1]),
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=start + 0.03,
    )


def _bundle(
    tmp_path: Path,
    calibrations: tuple[GF220Calibration, GF220Calibration],
    *,
    axis_qualified: bool = True,
    slip_qualified: bool = True,
    jam_qualified: bool = False,
) -> Path:
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, np.ones(SHAPE[:2], dtype=bool), allow_pickle=False)
    bootstrap = ContactBootstrapConfig(
        min_unique_samples=4,
        min_calibration_windows_per_stage=1,
        contact_confirmation_windows=1,
        loss_confirmation_windows=1,
    )
    contact_fingers = [
        {
            "serial": calibration.serial,
            "calibration_id": calibration.calibration_id,
            "unloaded_fz_sensor_units": 0.0,
            "unloaded_noise_sensor_units": 0.02,
            "positive_fz_sensor_units": 4.0,
            "positive_delta_sensor_units": 4.0,
            "positive_support_windows": 3,
            "positive_total_windows": 3,
            "contact_response_qualified": True,
            "last_bootstrap_sdk_time_s": 2.0,
            "last_bootstrap_capture_time_s": 2.0,
        }
        for calibration in calibrations
    ]
    image_fingers = [
        {
            "serial": calibration.serial,
            "calibration_id": calibration.calibration_id,
            "roi_mask": roi_path.name,
            "R_tcp_pad": np.eye(3).tolist(),
            "residual_threshold": 20.0,
            "min_contact_area_px": 80,
            "min_orientation_elongation": 1.6,
            "min_orientation_concentration": 0.35,
            "min_contact_frame_fraction": 0.6,
        }
        for calibration in calibrations
    ]
    jam = None
    wrench = None
    if jam_qualified:
        jam = {
            "profile_id": "tube-insert-wrench-v1",
            "task_frame_id": "tube_insert",
            "insertion_axis_task": [0.0, 0.0, 1.0],
            "reaction_sign": 1,
            "resistance_baseline_signal": 0.0,
            "resistance_noise_signal": 0.1,
            "minimum_resistance_delta_signal": 5.0,
            "runtime_qualified": True,
            "resistance_noise_multiplier": 8.0,
            "minimum_progress_ratio": 0.25,
            "confirmation_windows": 1,
        }
        wrench = {
            "profile_id": "tube-insert-wrench-v1",
            "task_frame_id": "tube_insert",
            "min_samples": 3,
            "max_age_s": 0.2,
            "max_pair_skew_s": 0.08,
            "future_tolerance_s": 1e-6,
            "sensors": [
                {
                    "serial": calibration.serial,
                    "logical_side": side,
                    "calibration_id": calibration.calibration_id,
                    "R_task_sensor": np.eye(3).tolist(),
                    "contact_position_task_m": [0.02 if side == "left" else -0.02, 0, 0],
                    "force6d_bias": [0, 0, 0, 0, 0, 0],
                    "provenance": {
                        "serial_side_mapping_id": "reviewed-side-map",
                        "task_sensor_rotation_id": "reviewed-task-rotation",
                        "contact_position_id": "measured-contact-position",
                        "force6d_bias_id": "unloaded-force-bias",
                        "force6d_axes_units_id": "gf220-force6d-axes-review",
                    },
                }
                for calibration, side in zip(calibrations, ("left", "right"), strict=True)
            ],
        }
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "accepted",
        "runtime_qualified": True,
        "qualification_id": "tube-tactile-qualified-test",
        "session_id": "tube-session-test",
        "serial_side_mapping": {SERIALS[0]: "left", SERIALS[1]: "right"},
        "target_axis_base": [1.0, 0.0, 0.0],
        "rotation_stop": {
            "runtime_qualified": axis_qualified,
            "maximum_axis_delta_deg": 1.0,
            "confirmation_windows": 1,
        },
        "contact_profile": {
            "profile_id": "tube-contact-test",
            "session_evidence_sha256": "c" * 64,
            "engage_threshold_sensor_units": 1.0,
            "retain_threshold_sensor_units": 0.4,
            "config": asdict(bootstrap),
            "fingers": contact_fingers,
        },
        "controller": {
            "profile_id": "tube-controller-test",
            "retained_width_min_m": 0.010,
            "retained_width_max_m": 0.025,
            "axis_runtime_qualified": axis_qualified,
            "slip_runtime_qualified": slip_qualified,
            "min_unique_image_samples": 4,
            "max_age_s": 0.25,
            "max_pair_skew_s": 0.08,
            "future_tolerance_s": 1e-6,
            "bilateral_confirmation_windows": 1,
            "target_axis_tolerance_deg": 5.0,
            "axis_confirmation_windows": 1,
            "max_pad_axis_disagreement_deg": 10.0,
            "slip_center_shift_threshold_px": 4.0,
            "slip_confirmation_windows": 1,
            "fingers": image_fingers,
            "jam": jam,
        },
        "wrench_adapter": wrench,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evaluate(
    interpreter: QualifiedTestTubeTactileInterpreter | FailClosedTestTubeTactileInterpreter,
    windows: GF220ClassifierWindows,
    state: TubeState,
    *,
    command: float | None = None,
    progress: float | None = None,
):
    return interpreter.evaluate(
        windows=windows,
        state=state,
        now_s=windows.collected_monotonic_s + 0.01,
        T_base_tcp=np.eye(4),
        gripper=GripperFeedback(0.015, False, True),
        commanded_insertion_step_m=command,
        measured_insertion_progress_m=progress,
    )


def test_missing_bundle_is_fail_closed_and_audited(tmp_path: Path) -> None:
    calibrations = _calibrations()
    interpreter = create_test_tube_tactile_interpreter(
        config={}, calibrations=calibrations, output_dir=tmp_path
    )
    result = _evaluate(interpreter, _windows(calibrations, start=3.0), TubeState.SIDE_GRASP)

    assert isinstance(interpreter, FailClosedTestTubeTactileInterpreter)
    assert result.valid is False
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["blockers"] == ["qualification_bundle_missing"]
    assert audit["hardware_commands_dispatched"] is False


def test_factory_promotes_bilateral_contact_and_maps_single_side(tmp_path: Path) -> None:
    calibrations = _calibrations()
    path = _bundle(tmp_path, calibrations)
    interpreter = create_test_tube_tactile_interpreter(
        config={"qualification_bundle": str(path)},
        calibrations=calibrations,
        output_dir=tmp_path,
    )
    first = _evaluate(
        interpreter,
        _windows(calibrations, start=3.0),
        TubeState.SIDE_GRASP,
    )
    second = _evaluate(
        interpreter,
        _windows(
            calibrations,
            start=3.1,
            forces=(4.0, 0.0),
            centers=((39.0, 24.0), (32.0, 24.0)),
        ),
        TubeState.SIDE_GRASP,
    )

    assert isinstance(interpreter, QualifiedTestTubeTactileInterpreter)
    assert first.valid and first.contact_present and first.bilateral_contact
    assert second.valid and second.contact_present
    assert not second.bilateral_contact
    assert second.single_contact_side == "left"
    assert second.slip_detected
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["controller"]["slip_candidate"] is True
    assert audit["controller"]["promoted_predicates"] == [
        "contact_present",
        "slip_detected",
    ]


def test_axis_and_rotation_stop_need_explicit_qualification(tmp_path: Path) -> None:
    calibrations = _calibrations()
    path = _bundle(tmp_path, calibrations, axis_qualified=False)
    interpreter = create_test_tube_tactile_interpreter(
        config={"qualification_bundle": str(path)},
        calibrations=calibrations,
        output_dir=tmp_path,
    )
    result = _evaluate(
        interpreter,
        _windows(calibrations, start=3.0),
        TubeState.PIVOT_EVALUATE,
    )

    assert not result.valid
    audit = json.loads((tmp_path / "tactile_interpreter/latest.json").read_text())
    assert audit["blockers"] == [
        "tactile_axis_not_runtime_qualified",
        "rotation_stop_not_runtime_qualified",
    ]


def test_reclamp_rotation_stop_requires_a_second_stable_axis_window(tmp_path: Path) -> None:
    calibrations = _calibrations()
    path = _bundle(tmp_path, calibrations)
    interpreter = create_test_tube_tactile_interpreter(
        config={"qualification_bundle": str(path)},
        calibrations=calibrations,
        output_dir=tmp_path,
    )
    first = _evaluate(
        interpreter,
        _windows(calibrations, start=3.0),
        TubeState.RECLAMP,
    )
    second = _evaluate(
        interpreter,
        _windows(calibrations, start=3.1),
        TubeState.RECLAMP,
    )

    assert first.valid and first.relative_angle_target_reached
    assert not first.rotation_stopped
    assert second.valid and second.rotation_stopped


def test_insert_requires_jam_profile_and_reports_resistance(tmp_path: Path) -> None:
    calibrations = _calibrations()
    missing_path = _bundle(tmp_path, calibrations, jam_qualified=False)
    missing = create_test_tube_tactile_interpreter(
        config={"qualification_bundle": str(missing_path)},
        calibrations=calibrations,
        output_dir=tmp_path / "missing",
    )
    missing_result = _evaluate(
        missing,
        _windows(calibrations, start=3.0),
        TubeState.INSERT_EVALUATE,
        command=0.01,
        progress=0.001,
    )
    assert not missing_result.valid

    qualified_dir = tmp_path / "qualified"
    qualified_dir.mkdir()
    qualified_path = _bundle(qualified_dir, calibrations, jam_qualified=True)
    qualified = create_test_tube_tactile_interpreter(
        config={"qualification_bundle": str(qualified_path)},
        calibrations=calibrations,
        output_dir=qualified_dir,
    )
    result = _evaluate(
        qualified,
        _windows(calibrations, start=3.0, forces=(6.0, 6.0)),
        TubeState.INSERT_EVALUATE,
        command=0.01,
        progress=0.001,
    )

    assert result.valid
    assert result.insertion_resistance_high
    assert result.insertion_jam
    audit = json.loads(
        (qualified_dir / "tactile_interpreter/latest.json").read_text()
    )
    assert audit["controller"]["insertion_resistance_delta_signal"] == 12.0
    assert audit["controller"]["insertion_jam_candidate"] is True


def test_mode_mapping_covers_all_states() -> None:
    assert {mode_for_tube_state(state) for state in TubeState} == set(TactileMode)

