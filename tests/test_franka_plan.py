import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from demo2gripper.franka import (
    canonical_tcp_to_panda_hand,
    make_tcp_targets,
)


def test_franka_targets_use_local_approach_and_base_vertical_lift() -> None:
    base_object = np.eye(4)
    base_object[:3, :3] = Rotation.from_euler("y", np.pi / 2).as_matrix()
    base_object[:3, 3] = [0.5, -0.1, 0.2]
    object_tcp = np.eye(4)
    object_tcp[:3, 3] = [0.02, 0.0, 0.0]

    targets = make_tcp_targets(
        base_object,
        object_tcp,
        approach_distance_m=0.1,
        lift_height_m=0.03,
    )

    np.testing.assert_allclose(
        targets["grasp"][:3, 3] - targets["pregrasp"][:3, 3],
        targets["grasp"][:3, 2] * 0.1,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        targets["lift"][:3, 3] - targets["grasp"][:3, 3],
        [0.0, 0.0, 0.03],
        atol=1e-12,
    )


def test_stock_panda_hand_mapping_and_plan_cli(tmp_path: Path) -> None:
    base_tcp = np.eye(4)
    hand = canonical_tcp_to_panda_hand(base_tcp)
    np.testing.assert_allclose(hand[:3, 3], [0.0, 0.0, -0.1034])
    np.testing.assert_allclose(
        hand[:3, :3],
        Rotation.from_euler("z", -90, degrees=True).as_matrix(),
        atol=1e-12,
    )

    grasp_path = tmp_path / "grasp.json"
    base_camera_path = tmp_path / "T_base_camera.npy"
    camera_object_path = tmp_path / "T_camera_object.npy"
    output_path = tmp_path / "plan.json"
    grasp_path.write_text(
        json.dumps(
            {
                "candidate_index": 62,
                "T_object_gripper_tcp": np.eye(4).tolist(),
                "approach_distance_m": 0.1,
                "pregrasp_width_m": 0.08,
                "demonstrated_contact_width_m": 0.034,
            }
        ),
        encoding="utf-8",
    )
    base_camera = np.eye(4)
    base_camera[:3, 3] = [0.4, 0.0, 0.5]
    camera_object = np.eye(4)
    camera_object[:3, 3] = [0.1, 0.2, 0.3]
    np.save(base_camera_path, base_camera)
    np.save(camera_object_path, camera_object[None])

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_franka_panda_plan.py",
            "--grasp-json",
            str(grasp_path),
            "--t-base-camera",
            str(base_camera_path),
            "--t-camera-object",
            str(camera_object_path),
            "--output",
            str(output_path),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    plan = json.loads(output_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(plan["T_base_object"][:3][0], [1.0, 0.0, 0.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(plan["canonical_tcp_targets"]["grasp"])[:3, 3],
        [0.5, 0.2, 0.8],
    )
    assert plan["candidate_index"] == 62
