from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.circular_rim_candidates import (
    CIRCULAR_RIM_CANDIDATE_SCHEMA,
    generate_circular_rim_candidates,
)


def _transform(
    *, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


def _relation() -> np.ndarray:
    relation = np.eye(4)
    relation[:3, :3] = np.diag([1.0, -1.0, -1.0])
    relation[:3, 3] = [0.05, 0.0, 0.002]
    return relation


def _generate(**overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "T_base_live_rim": _transform(translation=(0.55, 0.03, 0.2)),
        "T_rim_human_pinch_tcp": _relation(),
        "T_link8_tcp": _transform(translation=(0.0, 0.0, 0.19)),
        "approach_distance_m": 0.06,
        "azimuth_offsets_deg": (0.0, 90.0),
        "reference_radial_base": np.array([0.0, 1.0, 0.0]),
    }
    inputs.update(overrides)
    return generate_circular_rim_candidates(**inputs)


def test_explicit_radials_compose_tcp_pregrasp_and_link8() -> None:
    payload = _generate()

    assert payload["schema"] == CIRCULAR_RIM_CANDIDATE_SCHEMA
    assert payload["status"] == "review_only"
    assert payload["execution_authorized"] is False
    assert payload["hardware_execution_enabled"] is False
    assert payload["candidate_count"] == 2
    symmetry = payload["symmetry_handling"]
    assert symmetry["live_rim_transverse_axes_consumed"] is False
    assert symmetry["reference_is_agent_preference_not_object_yaw"] is True

    first = payload["candidates"][0]
    np.testing.assert_allclose(first["radial_axis_base"], [0.0, 1.0, 0.0])
    contact = np.asarray(first["T_base_contact_tcp"])
    pregrasp = np.asarray(first["T_base_pregrasp_tcp"])
    np.testing.assert_allclose(contact[:3, 3], [0.55, 0.08, 0.202])
    np.testing.assert_allclose(
        pregrasp[:3, 3], contact[:3, 3] - 0.06 * contact[:3, 2]
    )
    link8_tcp = _transform(translation=(0.0, 0.0, 0.19))
    np.testing.assert_allclose(
        np.asarray(first["T_base_contact_link8"]) @ link8_tcp, contact
    )
    assert first["moveit_plan_only"]["execute"] is False
    assert first["copied_live_object_yaw"] is False

    second = payload["candidates"][1]
    np.testing.assert_allclose(
        second["radial_axis_base"], [-1.0, 0.0, 0.0], atol=1e-12
    )


def test_live_rim_input_yaw_is_intentionally_ignored() -> None:
    identity = _transform(translation=(0.55, 0.03, 0.2))
    yawed = identity.copy()
    yawed[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    first = _generate(T_base_live_rim=identity, reference_radial_base=None)
    second = _generate(T_base_live_rim=yawed, reference_radial_base=None)

    assert first["candidates"] == second["candidates"]


def test_demo_interaction_radial_is_parallel_transported_to_live_axis() -> None:
    live = np.eye(4)
    live[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    demo = np.eye(4)

    payload = _generate(
        T_base_live_rim=live,
        T_base_demo_rim=demo,
        reference_radial_base=None,
        azimuth_offsets_deg=(0.0,),
    )

    assert (
        payload["symmetry_handling"]["reference_radial_source"]
        == "parallel_transported_demonstration_interaction_radial"
    )
    np.testing.assert_allclose(
        payload["candidates"][0]["radial_axis_base"], [1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        payload["candidates"][0]["cup_axis_base"], [0.0, -1.0, 0.0]
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"approach_distance_m": 0.0}, "approach_distance_m"),
        ({"azimuth_offsets_deg": (0.0, 360.0)}, "duplicate"),
        ({"reference_radial_base": np.array([0.0, 0.0, 1.0])}, "parallel"),
        ({"T_rim_human_pinch_tcp": np.eye(4)}, "radial contact"),
    ],
)
def test_invalid_or_ambiguous_inputs_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _generate(**overrides)


def test_cli_writes_plan_only_json(tmp_path: Path) -> None:
    live_path = tmp_path / "T_base_rim.npy"
    relation_path = tmp_path / "cup_rim_relation.json"
    link8_path = tmp_path / "T_link8_tcp.npy"
    output_path = tmp_path / "candidates.json"
    np.save(live_path, _transform(translation=(0.5, 0.0, 0.2)))
    np.save(link8_path, _transform(translation=(0.0, 0.0, 0.19)))
    relation_path.write_text(
        json.dumps(
            {
                "schema": "demo2gripper.cup_rim_relation.v1",
                "status": "review_only",
                "execution_authorized": False,
                "T_rim_human_pinch_tcp": _relation().tolist(),
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_circular_rim_candidates.py",
            "--live-t-base-rim",
            str(live_path),
            "--cup-rim-relation",
            str(relation_path),
            "--t-link8-tcp",
            str(link8_path),
            "--reference-radial-base",
            "1",
            "0",
            "0",
            "--azimuth-deg",
            "0",
            "120",
            "240",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == CIRCULAR_RIM_CANDIDATE_SCHEMA
    assert payload["candidate_count"] == 3
    assert payload["execution_authorized"] is False
    assert all(
        candidate["moveit_plan_only"]["execute"] is False
        for candidate in payload["candidates"]
    )
