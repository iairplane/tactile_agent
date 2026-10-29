from __future__ import annotations

import pytest

from demo2gripper.trajectory_safety import (
    audit_fr3_joint_trajectory,
    validate_fr3_trajectory_audit_thresholds,
)

JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]


def trajectory(*positions: list[float]) -> dict:
    return {
        "joint_names": JOINT_NAMES,
        "point_count": len(positions),
        "points": [
            {
                "positions": values,
                "velocities": [0.0] * 7,
                "accelerations": [0.0] * 7,
                "time_from_start_s": float(index),
            }
            for index, values in enumerate(positions)
        ],
    }


def test_accepts_bounded_fr3_plan_and_reports_per_joint_evidence() -> None:
    start = [0.0, 0.2, 0.0, -1.5, 0.0, 1.5, 0.0]
    end = [0.3, -0.2, 0.4, -1.2, -0.5, 1.8, 0.6]

    audit = audit_fr3_joint_trajectory(trajectory(start, end))

    assert audit["accepted"] is True
    assert audit["violations"] == []
    assert audit["total_joint_sweep_rad"] == pytest.approx(2.8)
    assert audit["joint_metrics"]["fr3_joint2"]["sweep_rad"] == pytest.approx(0.4)
    assert audit["joint_metrics"]["fr3_joint6"]["limit_margin_passed"] is True


def test_rejects_observed_branch_flip_and_near_limit_plan() -> None:
    start = [
        -0.52475977,
        1.02275062,
        0.53890181,
        -1.18499243,
        0.26857889,
        1.67259896,
        2.59015608,
    ]
    end = [
        2.72415882,
        -1.83358899,
        -1.04763488,
        -1.14888979,
        -2.35078261,
        1.10350549,
        -2.02601846,
    ]

    audit = audit_fr3_joint_trajectory(trajectory(start, end))

    assert audit["accepted"] is False
    assert audit["joint_metrics"]["fr3_joint2"][
        "minimum_hardware_limit_margin_rad"
    ] == pytest.approx(0.00251101)
    assert audit["joint_metrics"]["fr3_joint7"]["sweep_rad"] == pytest.approx(
        4.61617454
    )
    assert audit["total_joint_sweep_rad"] == pytest.approx(15.53252703)
    assert any("fr3_joint2 hardware-limit margin" in item for item in audit["violations"])
    assert any("fr3_joint7 sweep" in item for item in audit["violations"])
    assert any("total joint sweep" in item for item in audit["violations"])


def test_task_override_disables_only_joint_sweep_magnitude_guards() -> None:
    start = [-1.2, -1.0, -1.2, -2.5, -1.2, 1.0, -1.2]
    end = [1.2, 1.0, 1.2, -0.5, 1.2, 3.0, 1.2]

    guarded = audit_fr3_joint_trajectory(trajectory(start, end))
    overridden = audit_fr3_joint_trajectory(
        trajectory(start, end), enforce_joint_sweep_guards=False
    )

    assert guarded["accepted"] is False
    assert any("sweep" in item for item in guarded["violations"])
    assert overridden["accepted"] is True
    assert overridden["violations"] == []
    assert overridden["joint_sweep_guards_enabled"] is False
    assert overridden["total_joint_sweep_rad"] == pytest.approx(15.6)
    assert all(
        metrics["limit_margin_passed"] is True
        and metrics["sweep_passed"] is None
        for metrics in overridden["joint_metrics"].values()
    )


def test_sweep_override_keeps_hardware_limit_margin_blocking() -> None:
    start = [0.0, 1.80, 0.0, -1.5, 0.0, 1.5, 0.0]
    end = [0.1, 1.81, 0.1, -1.4, 0.1, 1.6, 0.1]

    audit = audit_fr3_joint_trajectory(
        trajectory(start, end), enforce_joint_sweep_guards=False
    )

    assert audit["accepted"] is False
    assert any("fr3_joint2 hardware-limit margin" in item for item in audit["violations"])
    assert not any("sweep" in item for item in audit["violations"])


def test_sweep_override_keeps_non_finite_trajectory_blocking() -> None:
    invalid = trajectory(
        [0.0, 0.2, 0.0, -1.5, 0.0, 1.5, 0.0],
        [0.3, -0.2, float("nan"), -1.2, -0.5, 1.8, 0.6],
    )

    audit = audit_fr3_joint_trajectory(
        invalid, enforce_joint_sweep_guards=False
    )

    assert audit["accepted"] is False
    assert any("positions are not finite" in item for item in audit["violations"])


@pytest.mark.parametrize(
    "bad_trajectory",
    [
        {"joint_names": JOINT_NAMES, "points": []},
        trajectory([0.0] * 7),
        {
            "joint_names": JOINT_NAMES[:-1],
            "points": [
                {"positions": [0.0] * 6, "time_from_start_s": 0.0},
                {"positions": [0.1] * 6, "time_from_start_s": 1.0},
            ],
        },
        {
            "joint_names": JOINT_NAMES,
            "points": [
                {"positions": [0.0] * 7, "time_from_start_s": 0.0},
                {"positions": [0.1] * 7, "time_from_start_s": 0.0},
            ],
        },
    ],
)
def test_malformed_trajectory_fails_closed(bad_trajectory: dict) -> None:
    audit = audit_fr3_joint_trajectory(bad_trajectory)

    assert audit["accepted"] is False
    assert audit["violations"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_joint_limit_margin_rad": 0.049}, "cannot weaken"),
        ({"max_joint_sweep_rad": 2.001}, "cannot weaken"),
        ({"max_total_joint_sweep_rad": 7.001}, "cannot weaken"),
    ],
)
def test_reviewed_thresholds_can_only_be_made_stricter(
    overrides: dict[str, float], message: str
) -> None:
    values = {
        "min_joint_limit_margin_rad": 0.05,
        "max_joint_sweep_rad": 2.0,
        "max_total_joint_sweep_rad": 7.0,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_fr3_trajectory_audit_thresholds(**values)
