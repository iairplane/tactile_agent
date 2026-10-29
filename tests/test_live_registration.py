from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.live_registration import (
    RegistrationConfig,
    cube_rotation_group,
    register_partial_template,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.register_live_object_pose import (  # noqa: E402
    FRAME_CONVENTION,
    SCHEMA,
    _object_frame_sha256,
    run,
)


def _partial_cube(rng: np.random.Generator, *, points_per_face: int = 180) -> np.ndarray:
    half_extent = 0.016
    faces = []
    for axis in range(3):
        points = rng.uniform(-half_extent, half_extent, (points_per_face, 3))
        points[:, axis] = half_extent
        faces.append(points)
    return np.concatenate(faces)


def _scene() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    template = _partial_cube(rng)
    reference_rotation = Rotation.from_euler("xyz", [0.20, -0.10, 0.40]).as_matrix()
    object_rotation = reference_rotation @ Rotation.from_euler(
        "xyz", [0.08, -0.12, 0.15]
    ).as_matrix()
    translation = np.asarray([0.62, 0.07, 0.05])
    frames = []
    for _ in range(5):
        retained = rng.random(len(template)) > 0.18
        frame = (
            template[retained] @ object_rotation.T
            + translation
            + rng.normal(0.0, 0.00045, (int(np.count_nonzero(retained)), 3))
        )
        # Sparse segmentation speckles must not move the fit.
        frame = np.concatenate(
            (frame, translation + rng.uniform(-0.028, 0.028, (35, 3)))
        )
        frames.append(frame)
    reference = np.eye(4)
    reference[:3, :3] = reference_rotation
    reference[:3, 3] = translation
    expected = np.eye(4)
    expected[:3, :3] = object_rotation
    expected[:3, 3] = translation
    return template, frames, reference, expected


def _rotation_error_modulo_cube_deg(first: np.ndarray, second: np.ndarray) -> float:
    return min(
        Rotation.from_matrix(first.T @ second @ symmetry).magnitude() * 180.0 / np.pi
        for symmetry in cube_rotation_group()
    )


def test_partial_cube_registration_recovers_pose_and_validates_held_out_frames() -> None:
    template, frames, reference, expected = _scene()

    result = register_partial_template(
        template,
        frames,
        reference,
        config=RegistrationConfig(min_points_per_frame=80, max_iterations=35),
    )

    assert result.fit_frame_indices == (0, 2, 4)
    assert result.held_out_frame_indices == (1, 3)
    np.testing.assert_allclose(
        result.T_base_object[:3, 3], expected[:3, 3], atol=0.001
    )
    assert (
        _rotation_error_modulo_cube_deg(
            result.T_base_object[:3, :3], expected[:3, :3]
        )
        < 1.0
    )
    assert result.fit_metrics.rmse_m < 0.0015
    assert max(result.frame_translation_deviation_m) < 0.001
    assert result.initial_hypothesis_count == 24
    assert result.orientation_prior_used is False


def test_explicit_orientation_prior_selects_one_cube_branch() -> None:
    template, frames, reference, expected = _scene()

    result = register_partial_template(
        template,
        frames,
        reference,
        config=RegistrationConfig(min_points_per_frame=80, max_iterations=35),
        orientation_prior_base_object=expected[:3, :3],
        max_orientation_prior_error_deg=5.0,
        orientation_prior_uniqueness_margin_deg=5.0,
    )

    assert result.orientation_prior_used is True
    assert result.orientation_prior_error_deg is not None
    assert result.orientation_prior_error_deg < 1.0
    if result.orientation_prior_second_best_error_deg is not None:
        assert (
            result.orientation_prior_second_best_error_deg
            - result.orientation_prior_error_deg
            >= 5.0
        )
    np.testing.assert_allclose(result.T_base_object[:3, 3], expected[:3, 3], atol=0.001)


def test_locked_orientation_uses_exact_prior_and_estimates_only_translation() -> None:
    template, frames, reference, expected = _scene()
    prior = expected[:3, :3].copy()

    result = register_partial_template(
        template,
        frames,
        reference,
        config=RegistrationConfig(min_points_per_frame=80, max_iterations=35),
        orientation_prior_base_object=prior,
        max_orientation_prior_error_deg=5.0,
        orientation_prior_uniqueness_margin_deg=5.0,
        lock_orientation_to_prior=True,
    )

    np.testing.assert_allclose(result.T_base_object[:3, :3], prior, atol=1e-12)
    np.testing.assert_allclose(
        result.T_base_object[:3, 3], expected[:3, 3], atol=0.001
    )
    assert result.orientation_prior_used is True
    assert result.orientation_locked_to_prior is True
    assert result.orientation_lock_error_deg == pytest.approx(0.0, abs=1e-12)
    assert result.unlocked_fit_metrics is not None
    assert result.fit_metrics.rmse_m < 0.0015
    assert result.held_out_frame_indices == (1, 3)
    assert max(result.frame_translation_deviation_m) < 0.001


def test_locked_orientation_requires_an_explicit_prior() -> None:
    template, frames, reference, _ = _scene()

    with pytest.raises(ValueError, match="requires an explicit orientation prior"):
        register_partial_template(
            template,
            frames,
            reference,
            config=RegistrationConfig(min_points_per_frame=80),
            lock_orientation_to_prior=True,
        )


def test_locked_orientation_still_rejects_a_disagreeing_held_out_frame() -> None:
    template, frames, reference, expected = _scene()
    frames[1] = frames[1] + np.asarray([0.025, 0.0, 0.0])

    with pytest.raises(ValueError, match="held-out|inconsistent"):
        register_partial_template(
            template,
            frames,
            reference,
            config=RegistrationConfig(min_points_per_frame=80, max_iterations=35),
            orientation_prior_base_object=expected[:3, :3],
            max_orientation_prior_error_deg=5.0,
            orientation_prior_uniqueness_margin_deg=5.0,
            lock_orientation_to_prior=True,
        )


def test_orientation_prior_still_rejects_without_unique_separation() -> None:
    template, frames, reference, expected = _scene()

    with pytest.raises(ValueError, match="cannot uniquely disambiguate"):
        register_partial_template(
            template,
            frames,
            reference,
            config=RegistrationConfig(
                min_points_per_frame=80,
                max_iterations=1,
                ambiguity_score_margin_m=1.0,
            ),
            orientation_prior_base_object=expected,
            max_orientation_prior_error_deg=180.0,
            orientation_prior_uniqueness_margin_deg=100.0,
        )


def test_registration_rejects_a_disagreeing_held_out_frame() -> None:
    template, frames, reference, _ = _scene()
    frames[1] = frames[1] + np.asarray([0.025, 0.0, 0.0])

    with pytest.raises(ValueError, match="held-out|inconsistent"):
        register_partial_template(
            template,
            frames,
            reference,
            config=RegistrationConfig(min_points_per_frame=80, max_iterations=35),
        )


def test_registration_requires_multiple_frames_for_independent_validation() -> None:
    template, frames, reference, _ = _scene()

    with pytest.raises(ValueError, match="at least 3 live frames"):
        register_partial_template(
            template,
            frames[:2],
            reference,
            config=RegistrationConfig(min_points_per_frame=80),
        )


def test_script_writes_grounder_compatible_pose_with_locked_prior_evidence(
    tmp_path: Path,
) -> None:
    template, frames, reference, expected = _scene()
    template_dir = tmp_path / "object_frame_intent"
    template_dir.mkdir()
    points_path = template_dir / "object_points_m.npy"
    np.save(points_path, template, allow_pickle=False)
    digest = hashlib.sha256(points_path.read_bytes()).hexdigest()
    frame_convention = "origin=test; axes=test; right_handed; metres"
    semantic_digest = _object_frame_sha256(template, frame_convention)
    frame_id = f"object_template_camera_axes:sha256:{semantic_digest}"
    manifest = {
        "schema": "demo2gripper.no_cad_object_template.v1",
        "frame_id": frame_id,
        "object_points_file": "object_points_m.npy",
        "object_points_file_sha256": digest,
        "object_template_sha256": semantic_digest,
        "point_count": len(template),
        "units": "metres",
        "frame_convention": frame_convention,
        "T_demo_camera_object_template": np.eye(4).tolist(),
    }
    (template_dir / "template_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    live_path = tmp_path / "selected_object_points_base_m.npz"
    np.savez_compressed(
        live_path,
        **{f"frame_{index:06d}": frame for index, frame in enumerate(frames)},
    )
    camera_path = tmp_path / "T_base_camera.npy"
    camera = np.eye(4)
    camera[:3, :3] = reference[:3, :3]
    np.save(camera_path, camera, allow_pickle=False)
    prior_path = tmp_path / "R_base_object_prior.npy"
    np.save(prior_path, expected[:3, :3], allow_pickle=False)
    output = tmp_path / "live_object_pose.json"
    args = Namespace(
        template_dir=template_dir,
        live_points=live_path,
        t_base_camera=camera_path,
        initial_anchor=None,
        initial_t_base_object=None,
        orientation_prior=prior_path,
        orientation_prior_source="fixture records that the cube was translated only",
        lock_orientation_to_prior=True,
        max_orientation_prior_error_deg=5.0,
        orientation_prior_uniqueness_margin_deg=5.0,
        output=output,
        symmetry="cube",
        min_frames=3,
        min_points_per_frame=80,
        voxel_size_m=0.001,
        max_correspondence_distance_m=0.012,
        inlier_distance_m=0.005,
        max_fit_rmse_m=0.0045,
        max_held_out_rmse_m=0.0055,
        max_frame_translation_deviation_m=0.006,
        max_frame_rotation_deviation_deg=12.0,
        overwrite=False,
    )

    written = run(args)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA
    assert payload["status"] == "accepted"
    assert payload["authorized_for_robot_execution"] is False
    assert payload["frame_convention"] == FRAME_CONVENTION
    assert payload["base_frame_id"] == "base"
    assert payload["object_frame_id"] == frame_id
    assert payload["quality"]["accepted"] is True
    assert payload["quality"]["pose_identifiability"] == (
        "rotation_fixed_by_explicit_prior"
    )
    assert payload["quality"]["held_out_frame_indices"] == [1, 3]
    assert payload["quality"]["orientation_prior"]["used"] is True
    assert payload["quality"]["orientation_prior"]["rotation_locked"] is True
    assert payload["quality"]["orientation_prior"]["error_deg"] < 1.0
    assert payload["quality"]["rotation_lock"]["enabled"] is True
    assert (
        payload["quality"]["rotation_lock"][
            "output_rotation_estimated_from_live_points"
        ]
        is False
    )
    assert payload["quality"]["rotation_lock"]["free_rotation_fit"] is not None
    np.testing.assert_allclose(
        np.asarray(payload["T_base_object"])[:3, :3], expected[:3, :3], atol=1e-12
    )
    assert payload["orientation_prior"]["source"].startswith("fixture records")
    assert payload["orientation_prior"]["file_sha256"]
    assert payload["orientation_prior"]["rotation_locked"] is True
    assert payload["provenance"]["registration_mode"] == (
        "translation_only_with_locked_explicit_orientation_prior"
    )
    assert payload["initialization"]["mode"].startswith("calibrated_camera")


def test_script_rejects_tampered_template_points(tmp_path: Path) -> None:
    template, frames, reference, _ = _scene()
    template_dir = tmp_path / "object_frame_intent"
    template_dir.mkdir()
    points_path = template_dir / "object_points_m.npy"
    np.save(points_path, template, allow_pickle=False)
    frame_convention = "origin=test; axes=test; right_handed; metres"
    semantic_digest = _object_frame_sha256(template, frame_convention)
    manifest = {
        "schema": "demo2gripper.no_cad_object_template.v1",
        "frame_id": f"object_template_camera_axes:sha256:{semantic_digest}",
        "object_points_file": "object_points_m.npy",
        "object_points_file_sha256": "0" * 64,
        "object_template_sha256": semantic_digest,
        "point_count": len(template),
        "units": "metres",
        "frame_convention": frame_convention,
        "T_demo_camera_object_template": np.eye(4).tolist(),
    }
    (template_dir / "template_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    live_path = tmp_path / "points.npz"
    np.savez_compressed(live_path, **{f"frame_{i:06d}": x for i, x in enumerate(frames)})
    camera_path = tmp_path / "camera.npy"
    np.save(camera_path, reference, allow_pickle=False)
    args = Namespace(
        template_dir=template_dir,
        live_points=live_path,
        t_base_camera=camera_path,
        initial_anchor=None,
        initial_t_base_object=None,
        output=tmp_path / "pose.json",
        symmetry="cube",
        min_frames=3,
        min_points_per_frame=80,
        voxel_size_m=0.001,
        max_correspondence_distance_m=0.012,
        inlier_distance_m=0.005,
        max_fit_rmse_m=0.0045,
        max_held_out_rmse_m=0.0055,
        max_frame_translation_deviation_m=0.006,
        max_frame_rotation_deviation_deg=12.0,
        overwrite=False,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        run(args)
    assert not args.output.exists()

