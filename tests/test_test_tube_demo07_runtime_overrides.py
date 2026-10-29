from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo2gripper.test_tube_closed_loop import parse_geometry, parse_policy


def _config() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/test_tube_closed_loop.demo07.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_demo07_operator_depth_override_is_explicit_and_consistent() -> None:
    config = _config()
    normalized = config["geometry"]["normalized_demo_grasp"]
    override = normalized["operator_depth_override"]

    assert normalized["axial_fraction_closed_to_open"] == pytest.approx(0.85)
    assert override["previous_axial_fraction_closed_to_open"] == pytest.approx(0.94)
    assert override["new_axial_fraction_closed_to_open"] == pytest.approx(0.85)
    assert override["direction"] == "toward_closed_end"
    effective = (
        override["previous_axial_fraction_closed_to_open"]
        - override["new_axial_fraction_closed_to_open"]
    ) * override["nominal_tube_length_m"]
    assert effective == pytest.approx(override["effective_deeper_m"])
    assert effective == pytest.approx(0.0099)

    geometry = parse_geometry(config["geometry"])
    assert geometry.normalized_demo_grasp is not None
    assert geometry.normalized_demo_grasp.axial_fraction_closed_to_open == pytest.approx(
        0.85
    )


def test_demo07_resident_extract_clearance_is_eighty_mm() -> None:
    config = _config()
    policy = parse_policy(config["policy"])
    assert policy.extract_clearance_m == pytest.approx(0.08)


def test_demo07_sixty_five_mm_committed_descent_has_measured_time_budget() -> None:
    config = _config()
    policy = parse_policy(config["policy"])
    assert policy.final_approach_commit_distance_m == pytest.approx(0.065)
    assert policy.final_approach_timeout_s == pytest.approx(15.0)
