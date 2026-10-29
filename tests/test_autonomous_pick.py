import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo2gripper.autonomous_pick import (
    PickStage,
    PickThresholds,
    assert_stage_prefix,
    cartesian_measured_end_tcp_transform,
    cartesian_result_succeeded,
    evaluate_descend_tcp_residual,
    hande_contact_verified,
    hande_open_succeeded,
    hande_width_snapshot_valid,
    measured_pose_within_tolerance,
    pose_result_succeeded,
    prepare_execution_candidate,
    rotation_matrix_to_quaternion_xyzw,
    validate_pick_geometry,
)
from scripts.run_autonomous_pick import (
    StageFailure,
    _configured_ee_pose_command,
    _registration_command,
    _validate_config,
    _validated_live_pose_summary,
)


def anchor() -> dict:
    return {
        "schema": "demo2gripper.live_supported_object_anchor.v1",
        "status": "accepted",
        "quality": {"accepted": True},
        "center_base_m": [0.65965, 0.03672, 0.04760],
        "robust_bounds_p01_p99_base_m": {
            "min": [0.64280, 0.02500, 0.04000],
            "max": [0.67620, 0.04720, 0.05770],
        },
        "support": {
            "mode": "band_upper_edge",
            "band_base_z_m": [-0.015, 0.029],
            "effective_top_base_z_m": 0.029,
        },
        "stability": {
            "center_max_deviation_m": 0.0008,
            "center_rms_deviation_m": 0.0004,
            "point_count_cv": 0.03,
        },
    }


def grounded_candidate() -> dict:
    rotation = np.array(
        [
            [0.9899052973, 0.1417303861, 0.0],
            [0.1417303861, -0.9899052973, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    grasp = np.eye(4)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = [0.658074, 0.037652, 0.03663]
    pregrasp = grasp.copy()
    pregrasp[2, 3] += 0.08
    return {
        "schema": "demo2gripper.grounded_live_grasp_candidate.v1",
        "live_object_anchor_base_m": anchor()["center_base_m"],
        "candidate": {
            "T_base_pregrasp_tcp": pregrasp.tolist(),
            "T_base_grasp_tcp": grasp.tolist(),
            "approach_distance_m": 0.08,
        },
        "evidence": {"closing_axis_stable": True},
        "phase_targets": {
            "APPROACH": {"poses_base": [pregrasp.tolist(), grasp.tolist()]},
            "GRASP": {
                "T_base_gripper_tcp": grasp.tolist(),
                "source": "legacy_base_aligned_planar_demonstration",
            },
            "LIFT": {
                "positions_base_m": [
                    grasp[:3, 3].tolist(),
                    (grasp[:3, 3] + [0.0, 0.0, 0.08]).tolist(),
                ]
            },
        },
    }


def link8_to_canonical_tcp() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = [0.0, 0.0, 0.19]
    return transform


def execution_candidate() -> dict:
    return prepare_execution_candidate(
        grounded_candidate(),
        anchor_z_m=anchor()["center_base_m"][2],
        grasp_tcp_vertical_offset_m=-0.0053,
        lift_delta_base_m=np.array([0.0, 0.0, 0.04]),
    )


def test_execution_override_keeps_demo_xy_angle_and_uses_robot_z_and_short_lift() -> None:
    original = grounded_candidate()
    prepared = execution_candidate()
    original_grasp = np.asarray(original["candidate"]["T_base_grasp_tcp"])
    grasp = np.asarray(prepared["candidate"]["T_base_grasp_tcp"])
    pregrasp = np.asarray(prepared["candidate"]["T_base_pregrasp_tcp"])

    np.testing.assert_allclose(grasp[:2, 3], original_grasp[:2, 3])
    np.testing.assert_allclose(grasp[:3, :3], original_grasp[:3, :3])
    assert grasp[2, 3] == pytest.approx(0.0423)
    assert pregrasp[2, 3] == pytest.approx(0.1223)
    lift = np.asarray(prepared["phase_targets"]["LIFT"]["positions_base_m"])
    np.testing.assert_allclose(lift[1] - lift[0], [0.0, 0.0, 0.04])


def test_execution_preparation_never_overwrites_selected_se3_grasp() -> None:
    payload = grounded_candidate()
    payload["schema"] = "demo2gripper.grounded_live_grasp_candidate.v2"
    payload["object_frame_id"] = "die_v1"
    payload["gripper_name"] = "robotiq_hande"
    payload["evidence"] = {
        "grasp_relation_mode": "se3",
        "selected_grasp_frame_validated": True,
        "object_frame_id_matched": True,
        "agent_approach_override_allowed": False,
    }
    payload["phase_targets"]["GRASP"].update(
        source="selected_object_relative_se3", endpoint_immutable=True
    )
    original_grasp = np.asarray(payload["candidate"]["T_base_grasp_tcp"]).copy()
    original_pregrasp = np.asarray(payload["candidate"]["T_base_pregrasp_tcp"]).copy()

    prepared = prepare_execution_candidate(
        payload,
        anchor_z_m=None,
        grasp_tcp_vertical_offset_m=None,
        lift_delta_base_m=np.array([0.0, 0.0, 0.04]),
    )

    np.testing.assert_allclose(prepared["candidate"]["T_base_grasp_tcp"], original_grasp)
    np.testing.assert_allclose(prepared["candidate"]["T_base_pregrasp_tcp"], original_pregrasp)
    np.testing.assert_allclose(
        prepared["phase_targets"]["GRASP"]["T_base_gripper_tcp"],
        original_grasp,
    )
    assert prepared["robot_execution_overrides"]["grasp_endpoint_modified"] is False


def test_geometry_converts_canonical_tcp_to_link8_and_applies_all_gates() -> None:
    geometry = validate_pick_geometry(
        anchor(), execution_candidate(), link8_to_canonical_tcp(), PickThresholds()
    )

    np.testing.assert_allclose(geometry.T_base_pregrasp_link8[:3, 3], [0.658074, 0.037652, 0.3123])
    quaternion = rotation_matrix_to_quaternion_xyzw(geometry.T_base_pregrasp_link8[:3, :3])
    np.testing.assert_allclose(quaternion, [0.75555621, -0.65508382, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(geometry.approach_delta_base_m, [0.0, 0.0, -0.08])
    np.testing.assert_allclose(geometry.lift_deltas_base_m, [[0.0, 0.0, 0.04]])


def test_oblique_se3_approach_needs_an_explicit_experiment_threshold() -> None:
    downward_alignment = 0.66
    lateral_alignment = float(np.sqrt(1.0 - downward_alignment**2))
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -downward_alignment, lateral_alignment],
            [0.0, -lateral_alignment, -downward_alignment],
        ]
    )
    grasp = np.eye(4)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = [0.658, 0.038, 0.045]
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= 0.08 * rotation[:, 2]
    candidate = {
        "schema": "demo2gripper.grounded_live_grasp_candidate.v2",
        "live_object_anchor_base_m": anchor()["center_base_m"],
        "candidate": {
            "T_base_pregrasp_tcp": pregrasp.tolist(),
            "T_base_grasp_tcp": grasp.tolist(),
            "approach_distance_m": 0.08,
        },
        "evidence": {
            "grasp_relation_mode": "se3",
            "selected_grasp_frame_validated": True,
            "object_frame_id_matched": True,
        },
        "phase_targets": {
            "APPROACH": {"poses_base": [pregrasp.tolist(), grasp.tolist()]},
            "GRASP": {"T_base_gripper_tcp": grasp.tolist()},
            "LIFT": {
                "positions_base_m": [
                    grasp[:3, 3].tolist(),
                    (grasp[:3, 3] + [0.0, 0.0, 0.04]).tolist(),
                ]
            },
        },
    }
    T_link8_tcp = np.eye(4)
    T_link8_tcp[2, 3] = 0.190

    assert PickThresholds().min_downward_approach_alignment == pytest.approx(0.95)
    with pytest.raises(ValueError, match="not sufficiently downward"):
        validate_pick_geometry(anchor(), candidate, T_link8_tcp, PickThresholds())

    thresholds = PickThresholds(min_downward_approach_alignment=0.60)
    geometry = validate_pick_geometry(anchor(), candidate, T_link8_tcp, thresholds)
    np.testing.assert_allclose(
        geometry.approach_delta_base_m,
        0.08 * rotation[:, 2],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        geometry.T_base_pregrasp_link8 @ T_link8_tcp,
        pregrasp,
        atol=1e-12,
    )


def test_pregrasp_pose_command_passes_configured_ee_and_full_tool_mapping() -> None:
    target = np.eye(4)
    target[:3, 3] = [0.688, -0.022, 0.099]
    T_link8_tcp = np.eye(4)
    T_link8_tcp[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T_link8_tcp[:3, 3] = [0.01, -0.02, 0.19]
    normalized = {
        "commands": {"pose": ["run_ros", "moveit_pose_goal.py"]},
        "max_pregrasp_translation_m": 0.5,
        "max_pregrasp_rotation_rad": np.pi,
        "pose_position_tolerance_m": 0.001,
        "pose_orientation_tolerance_rad": 0.005,
        "min_joint_limit_margin_rad": 0.05,
        "max_joint_sweep_rad": 2.0,
        "max_total_joint_sweep_rad": 7.0,
        "velocity_scale": 0.03,
        "acceleration_scale": 0.03,
    }

    command = _configured_ee_pose_command(normalized, target, T_link8_tcp)

    def option_values(name: str, count: int) -> list[float]:
        start = command.index(name) + 1
        return [float(value) for value in command[start : start + count]]

    assert command[command.index("--target-frame") + 1] == "configured-ee"
    assert option_values("--target-position-m", 3) == pytest.approx(target[:3, 3])
    assert option_values("--link8-to-configured-ee-position-m", 3) == pytest.approx(
        T_link8_tcp[:3, 3]
    )
    assert option_values("--link8-to-configured-ee-orientation-xyzw", 4) == pytest.approx(
        rotation_matrix_to_quaternion_xyzw(T_link8_tcp[:3, :3])
    )
    assert option_values("--min-joint-limit-margin-rad", 1) == pytest.approx([0.05])
    assert option_values("--max-joint-sweep-rad", 1) == pytest.approx([2.0])
    assert option_values("--max-total-joint-sweep-rad", 1) == pytest.approx([7.0])


def test_geometry_uses_direct_projected_width_when_available() -> None:
    wide_aabb = anchor()
    wide_aabb["robust_bounds_p01_p99_base_m"]["max"][1] = 0.060
    geometry = validate_pick_geometry(
        wide_aabb,
        execution_candidate(),
        link8_to_canonical_tcp(),
        PickThresholds(),
        object_width_along_closing_axis_m=0.042,
    )
    assert geometry.object_width_along_closing_axis_m == pytest.approx(0.042)

    with pytest.raises(ValueError, match="exceeds usable gripper"):
        validate_pick_geometry(
            wide_aabb,
            execution_candidate(),
            link8_to_canonical_tcp(),
            PickThresholds(),
            object_width_along_closing_axis_m=0.048,
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("center_max_deviation_m", 0.003, "center_max_deviation_m"),
        ("center_rms_deviation_m", 0.002, "center_rms_deviation_m"),
        ("point_count_cv", 0.2, "point_count_cv"),
    ],
)
def test_geometry_rejects_unstable_live_anchor(field: str, bad_value: float, message: str) -> None:
    bad_anchor = anchor()
    bad_anchor["stability"][field] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_pick_geometry(
            bad_anchor, execution_candidate(), link8_to_canonical_tcp(), PickThresholds()
        )


def test_geometry_rejects_wrong_object_extent_and_tcp_workspace() -> None:
    wrong_size = anchor()
    wrong_size["robust_bounds_p01_p99_base_m"]["max"][0] = 0.700
    with pytest.raises(ValueError, match="extents"):
        validate_pick_geometry(
            wrong_size,
            execution_candidate(),
            link8_to_canonical_tcp(),
            PickThresholds(),
        )

    outside = execution_candidate()
    for key in ("T_base_pregrasp_tcp", "T_base_grasp_tcp"):
        outside["candidate"][key][0][3] = 0.700
    outside["phase_targets"]["LIFT"]["positions_base_m"] = [
        [0.700, 0.037652, 0.0423],
        [0.700, 0.037652, 0.0823],
    ]
    with pytest.raises(ValueError, match="inner workspace"):
        validate_pick_geometry(anchor(), outside, link8_to_canonical_tcp(), PickThresholds())


def hande_payload(*, opening_mm: float, code: int, fault: int = 0) -> dict:
    return {
        "ok": True,
        "state": {
            "opening_width_mm": opening_mm,
            "fault_code": fault,
            "object_status": {"code": code, "grasped": code == 2},
        },
    }


def test_hande_transition_predicates_use_close_gobj_then_reconnect_width() -> None:
    thresholds = PickThresholds()
    assert hande_open_succeeded(hande_payload(opening_mm=49.2, code=3), thresholds)
    close = hande_payload(opening_mm=27.5, code=2)
    assert hande_contact_verified(close, thresholds)
    reconnect = hande_payload(opening_mm=27.7, code=0)
    assert hande_width_snapshot_valid(reconnect, thresholds, reference_opening_m=0.0275)
    assert not hande_contact_verified(reconnect, thresholds)
    assert not hande_width_snapshot_valid(
        hande_payload(opening_mm=30.0, code=0),
        thresholds,
        reference_opening_m=0.0275,
    )


def test_moveit_result_predicates_and_stage_order() -> None:
    pose = {
        "mode": "plan_only",
        "moveit_error_code": 1,
        "action_status": 4,
        "planning_succeeded": True,
        "trajectory_audit": {"accepted": True},
        "trajectory": {"point_count": 99},
    }
    assert pose_result_succeeded(pose, executed=False)
    pose.update(
        mode="execute",
        execution_policy="plan_then_audit_then_execute_exact_trajectory",
        execution_dispatched=True,
        execution_status=4,
        execution_error_code=1,
        execution_succeeded=True,
    )
    assert pose_result_succeeded(pose, executed=True)
    pose["trajectory_audit"]["accepted"] = False
    assert not pose_result_succeeded(pose, executed=True)
    cartesian = {
        "mode": "execute",
        "moveit_error_code": 1,
        "path_fraction": 1.0,
        "trajectory": {"point_count": 20},
        "trajectory_audit": {"accepted": True},
        "planning_succeeded": True,
        "execution_policy": "plan_then_audit_then_execute_exact_trajectory",
        "execution_dispatched": True,
        "execution_status": 4,
        "execution_error_code": 1,
        "execution_succeeded": True,
    }
    assert cartesian_result_succeeded(cartesian, executed=True)
    cartesian["trajectory_audit"]["accepted"] = False
    assert not cartesian_result_succeeded(cartesian, executed=True)

    measured = {
        "input_target_frame": "configured-ee",
        "input_target_position_base_m": [0.658, 0.038, 0.122],
        "input_target_orientation_xyzw": [0.7556, -0.6551, 0.0, 0.0],
        "measured_end_configured_ee_position_base_m": [0.660, 0.038, 0.122],
        "measured_end_configured_ee_orientation_xyzw": [
            0.7556,
            -0.6551,
            0.0,
            0.0,
        ],
    }
    accepted, errors = measured_pose_within_tolerance(measured, PickThresholds())
    assert accepted
    assert errors["position_error_m"] == pytest.approx(0.002)

    assert_stage_prefix([{"name": stage.value} for stage in list(PickStage)[:4]])
    with pytest.raises(ValueError, match="ordered prefix"):
        assert_stage_prefix([{"name": PickStage.OBSERVE.value}, {"name": PickStage.PLAN.value}])


def test_cartesian_end_link8_pose_is_composed_with_tcp_transform() -> None:
    half_angle = np.deg2rad(45.0)
    payload = {
        "measured_end_position_base_m": [1.0, 2.0, 3.0],
        "measured_end_orientation_xyzw": [
            0.0,
            0.0,
            np.sin(half_angle),
            np.cos(half_angle),
        ],
    }
    T_link8_tcp = np.eye(4)
    T_link8_tcp[:3, 3] = [0.1, 0.0, 0.0]

    measured = cartesian_measured_end_tcp_transform(payload, T_link8_tcp)

    np.testing.assert_allclose(measured[:3, 3], [1.0, 2.1, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        measured[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_descend_residual_requests_bounded_downward_absolute_correction() -> None:
    thresholds = PickThresholds()
    target = np.array([0.657878, 0.038114, 0.042391])
    measured = np.array([0.659093, 0.036969, 0.047202])

    correction, details = evaluate_descend_tcp_residual(
        measured,
        target,
        support_top_base_z_m=0.029,
        thresholds=thresholds,
    )

    np.testing.assert_allclose(correction, target - measured)
    assert details["status"] == "correction_required"
    assert details["residual_norm_m"] == pytest.approx(0.005092, abs=1e-6)
    assert details["residual_xy_norm_m"] < thresholds.max_descend_correction_xy_m

    correction, details = evaluate_descend_tcp_residual(
        target + [0.0003, 0.0, 0.0004],
        target,
        support_top_base_z_m=0.029,
        thresholds=thresholds,
    )
    assert correction is None
    assert details["status"] == "within_tolerance"


def test_descend_residual_rejects_unsafe_direction_extent_and_clearance() -> None:
    thresholds = PickThresholds()
    target = np.array([0.658, 0.038, 0.042])

    with pytest.raises(ValueError, match="upward"):
        evaluate_descend_tcp_residual(
            target - [0.0, 0.0, 0.003],
            target,
            support_top_base_z_m=0.029,
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="XY residual"):
        evaluate_descend_tcp_residual(
            target + [0.004, 0.0, 0.003],
            target,
            support_top_base_z_m=0.029,
            thresholds=thresholds,
        )
    with pytest.raises(ValueError, match="support-surface clearance"):
        evaluate_descend_tcp_residual(
            [target[0], target[1], 0.031],
            target,
            support_top_base_z_m=0.029,
            thresholds=thresholds,
        )


def test_final_grasp_tcp_tolerance_has_two_millimetre_hard_cap() -> None:
    assert PickThresholds().max_final_grasp_tcp_error_m == pytest.approx(0.001)
    PickThresholds(max_final_grasp_tcp_error_m=0.002)
    with pytest.raises(ValueError, match="capped at 0.002"):
        PickThresholds(max_final_grasp_tcp_error_m=0.0021)


def test_anchor_z_bounds_allow_finite_negative_range_and_reject_non_finite_values() -> None:
    thresholds = PickThresholds(min_anchor_z_m=-0.080, max_anchor_z_m=-0.010)
    assert (thresholds.min_anchor_z_m, thresholds.max_anchor_z_m) == pytest.approx(
        (-0.080, -0.010)
    )

    for field in ("min_anchor_z_m", "max_anchor_z_m"):
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="anchor Z bounds must be finite"):
                PickThresholds(**{field: bad_value})


def test_runner_config_fails_closed_without_live_pose_or_registration_template(
    tmp_path: Path,
) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "autonomous_pick.example.json"
    config = json.loads(example_path.read_text(encoding="utf-8"))
    config["repo_root"] = str(Path(__file__).resolve().parents[1])
    for name in ("compiler_report", "T_base_camera", "T_link8_tcp"):
        artifact = tmp_path / f"{name}.json"
        artifact.write_text("{}", encoding="utf-8")
        config["inputs"][name] = str(artifact)
    config["inputs"].pop("object_template_dir")
    config_path = tmp_path / "config.json"

    with pytest.raises(ValueError, match="configure exactly one"):
        _validate_config(config, config_path=config_path, execute=False)


def test_runner_config_rejects_external_pose_and_registration_template_together(
    tmp_path: Path,
) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "autonomous_pick.example.json"
    config = json.loads(example_path.read_text(encoding="utf-8"))
    config["inputs"]["live_object_pose"] = str(tmp_path / "pose.json")

    with pytest.raises(ValueError, match="configure exactly one"):
        _validate_config(config, config_path=tmp_path / "config.json", execute=False)


def test_runner_automatic_mode_requires_register_command(tmp_path: Path) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "autonomous_pick.example.json"
    config = json.loads(example_path.read_text(encoding="utf-8"))
    config["commands"].pop("register")

    with pytest.raises(ValueError, match=r"config\.commands\.register"):
        _validate_config(config, config_path=tmp_path / "config.json", execute=False)


def test_runner_automatic_registration_config_and_command_use_current_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs" / "autonomous_pick.example.json").read_text(encoding="utf-8")
    )
    config["repo_root"] = str(repo_root)
    for name in ("compiler_report", "T_base_camera", "T_link8_tcp"):
        artifact = tmp_path / f"{name}.json"
        artifact.write_text("{}", encoding="utf-8")
        config["inputs"][name] = str(artifact)
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "template_manifest.json").write_text("{}", encoding="utf-8")
    np.save(template_dir / "object_points_m.npy", np.zeros((3, 3)))
    config["inputs"]["object_template_dir"] = str(template_dir)

    normalized = _validate_config(config, config_path=tmp_path / "config.json", execute=False)
    live_points = tmp_path / "run" / "object_localization" / "points.npz"
    anchor_path = tmp_path / "run" / "object_localization" / "anchor.json"
    output = tmp_path / "run" / "registration" / "live_object_pose.json"
    command = _registration_command(
        normalized,
        live_points=live_points,
        initial_anchor=anchor_path,
        output=output,
    )

    assert normalized["live_object_pose"] is None
    assert normalized["object_template_dir"] == template_dir.resolve()
    assert command[:2] == [".venv/bin/python", "scripts/register_live_object_pose.py"]
    assert command[command.index("--template-dir") + 1] == str(template_dir.resolve())
    assert command[command.index("--live-points") + 1] == str(live_points.resolve())
    assert command[command.index("--initial-anchor") + 1] == str(anchor_path.resolve())
    assert command[command.index("--output") + 1] == str(output.resolve())


def test_runner_external_live_pose_remains_read_only_alternative(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo_root / "configs" / "autonomous_pick.example.json").read_text(encoding="utf-8")
    )
    config["repo_root"] = str(repo_root)
    for name in ("compiler_report", "T_base_camera", "T_link8_tcp"):
        artifact = tmp_path / f"{name}.json"
        artifact.write_text("{}", encoding="utf-8")
        config["inputs"][name] = str(artifact)
    pose_path = tmp_path / "external_T_base_object.npy"
    np.save(pose_path, np.eye(4))
    config["inputs"].pop("object_template_dir")
    config["inputs"]["live_object_pose"] = str(pose_path)
    config["inputs"]["object_frame_id"] = "object-v1"

    normalized = _validate_config(config, config_path=tmp_path / "config.json", execute=False)
    summary = _validated_live_pose_summary(
        normalized["live_object_pose"],
        expected_object_frame_id=normalized["object_frame_id"],
        source="explicit_external_read_only_input",
    )

    assert normalized["object_template_dir"] is None
    assert "register" not in normalized["commands"]
    assert summary["source"] == "explicit_external_read_only_input"
    assert summary["quality_contract"] == "explicit_external_read_only_matrix"


def test_runner_rejects_registration_without_fresh_point_provenance(
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    template_manifest = template_dir / "template_manifest.json"
    template_points = template_dir / "object_points_m.npy"
    template_manifest.write_text("{}", encoding="utf-8")
    np.save(template_points, np.zeros((3, 3)))
    fresh_points = tmp_path / "run" / "object_localization" / "points.npz"
    fresh_points.parent.mkdir(parents=True)
    np.savez(fresh_points, frame_0=np.zeros((3, 3)))
    pose_path = tmp_path / "run" / "registration" / "live_object_pose.json"
    pose_path.parent.mkdir()
    payload = {
        "schema": "demo2gripper.live_object_pose.v1",
        "status": "accepted",
        "frame_convention": "T_parent_child, right-handed, column vectors, metres",
        "base_frame_id": "base",
        "object_frame_id": "object-v1",
        "T_base_object": np.eye(4).tolist(),
        "quality": {"accepted": True, "fit": {"rmse_m": 0.001}},
        "orientation_prior": {
            "source": "explicit test prior",
            "path": "/tmp/prior.npy",
        },
        "provenance": {
            "template_manifest": str(template_manifest.resolve()),
            "template_points": str(template_points.resolve()),
            "live_points": str((tmp_path / "stale_points.npz").resolve()),
        },
    }
    pose_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageFailure, match="fresh localization points"):
        _validated_live_pose_summary(
            pose_path,
            expected_object_frame_id="object-v1",
            source="automatic_registration_from_current_observation",
            expected_live_points=fresh_points,
            expected_template_dir=template_dir,
        )

    payload["provenance"]["live_points"] = str(fresh_points.resolve())
    pose_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = _validated_live_pose_summary(
        pose_path,
        expected_object_frame_id="object-v1",
        source="automatic_registration_from_current_observation",
        expected_live_points=fresh_points,
        expected_template_dir=template_dir,
    )
    assert summary["quality"] == payload["quality"]
    assert summary["provenance"] == payload["provenance"]
    assert summary["orientation_prior"] == payload["orientation_prior"]

    payload["quality"]["accepted"] = False
    pose_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageFailure, match="quality gates"):
        _validated_live_pose_summary(
            pose_path,
            expected_object_frame_id="object-v1",
            source="automatic_registration_from_current_observation",
            expected_live_points=fresh_points,
            expected_template_dir=template_dir,
        )


def test_runner_npy_live_pose_requires_object_frame_id(tmp_path: Path) -> None:
    example_path = Path(__file__).resolve().parents[1] / "configs" / "autonomous_pick.example.json"
    config = json.loads(example_path.read_text(encoding="utf-8"))
    config["repo_root"] = str(Path(__file__).resolve().parents[1])
    for name in ("compiler_report", "T_base_camera", "T_link8_tcp"):
        artifact = tmp_path / f"{name}.json"
        artifact.write_text("{}", encoding="utf-8")
        config["inputs"][name] = str(artifact)
    pose_path = tmp_path / "T_base_object.npy"
    np.save(pose_path, np.eye(4))
    config["inputs"].pop("object_template_dir")
    config["inputs"]["live_object_pose"] = str(pose_path)
    config["inputs"].pop("object_frame_id")

    with pytest.raises(ValueError, match="inputs.object_frame_id"):
        _validate_config(config, config_path=tmp_path / "config.json", execute=False)
