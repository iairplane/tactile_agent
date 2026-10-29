import numpy as np
import pytest

from demo2gripper.gf220_live import GF220ClassifierWindows
from demo2gripper.tactile_contact_classifier import FingerRimClassifierWindow
from demo2gripper.tactile_rim_probe import TimedArrayWindow
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactBootstrapConfig as BootstrapConfig,
)
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactBootstrapError as ContactBootstrapError,
)
from demo2gripper.test_tube_contact_bootstrap import (
    TestTubeContactGate as ContactGate,
)
from demo2gripper.test_tube_contact_bootstrap import bootstrap_test_tube_contact

SESSION = "test-tube-session-001"


def _timed(values: np.ndarray, start: float) -> TimedArrayWindow:
    count = len(values)
    times = start + np.arange(count, dtype=float) * 0.01
    return TimedArrayWindow(values, times, times, np.ones(count, dtype=bool))


def _pair(
    *,
    start: float,
    first_fz: float,
    second_fz: float,
    noise: float = 0.01,
    calibration_suffix: str = "",
) -> GF220ClassifierWindows:
    windows = []
    for index, fz in enumerate((first_fz, second_fz)):
        count = 6
        force = np.zeros((count, 6), dtype=float)
        force[:, 2] = fz + np.linspace(-noise, noise, count)
        images = np.zeros((count, 8, 8, 3), dtype=np.uint8)
        windows.append(
            FingerRimClassifierWindow(
                calibration_id=f"cal-{index}{calibration_suffix}",
                warped=_timed(images, start),
                diff=_timed(images, start),
                force6d=_timed(force, start),
            )
        )
    return GF220ClassifierWindows(
        sensor_names=("A", "B"),
        serials=("serial-a", "serial-b"),
        windows=(windows[0], windows[1]),
        start_skew_s=0.0,
        end_skew_s=0.0,
        collected_monotonic_s=start + 0.05,
    )


def _series(
    *,
    start: float,
    first_fz: float,
    second_fz: float,
    count: int = 3,
    noise: float = 0.01,
) -> tuple[GF220ClassifierWindows, ...]:
    return tuple(
        _pair(
            start=start + index * 0.10,
            first_fz=first_fz,
            second_fz=second_fz,
            noise=noise,
        )
        for index in range(count)
    )


def _profile():
    return bootstrap_test_tube_contact(
        _series(start=1.0, first_fz=10.0, second_fz=20.0),
        _series(start=2.0, first_fz=14.0, second_fz=20.0),
        session_id=SESSION,
    )


def test_bootstrap_requires_repeated_windows() -> None:
    with pytest.raises(ContactBootstrapError, match="at least 3 windows"):
        bootstrap_test_tube_contact(
            _pair(start=1.0, first_fz=10.0, second_fz=20.0),
            _pair(start=2.0, first_fz=14.0, second_fz=20.0),
            session_id=SESSION,
        )


def test_bootstrap_builds_noise_aware_any_pad_presence_profile() -> None:
    result = _profile()
    assert result.report["status"] == "session_contact_presence_qualified"
    assert result.profile.session_id == SESSION
    assert result.profile.engage_threshold_sensor_units == pytest.approx(1.0)
    assert result.profile.retain_threshold_sensor_units == pytest.approx(0.4)
    assert result.report["policy"]["contact_logic"] == (
        "any_qualified_pad_contact_present"
    )
    assert [finger.contact_response_qualified for finger in result.profile.fingers] == [
        True,
        False,
    ]
    assert "grasp established" in result.report["policy"]["explicitly_not_claimed"]


def test_bootstrap_rejects_response_inside_unloaded_noise_margin() -> None:
    config = BootstrapConfig(
        minimum_fz_change_sensor_units=0.001,
        minimum_threshold_floor_sensor_units=0.001,
    )
    with pytest.raises(ContactBootstrapError, match="above unloaded noise"):
        bootstrap_test_tube_contact(
            _series(
                start=1.0,
                first_fz=10.0,
                second_fz=20.0,
                noise=0.5,
            ),
            _series(
                start=2.0,
                first_fz=10.1,
                second_fz=20.1,
                noise=0.5,
            ),
            session_id=SESSION,
            config=config,
        )


def test_bootstrap_rejects_inconsistent_positive_windows() -> None:
    positive = (
        _pair(start=2.0, first_fz=14.0, second_fz=20.0),
        _pair(start=2.1, first_fz=6.0, second_fz=20.0),
        _pair(start=2.2, first_fz=10.0, second_fz=20.0),
    )
    with pytest.raises(ContactBootstrapError, match="above unloaded noise"):
        bootstrap_test_tube_contact(
            _series(start=1.0, first_fz=10.0, second_fz=20.0),
            positive,
            session_id=SESSION,
        )


def test_profile_is_cryptographically_bound_to_session_id() -> None:
    first = _profile().profile
    second = bootstrap_test_tube_contact(
        _series(start=1.0, first_fz=10.0, second_fz=20.0),
        _series(start=2.0, first_fz=14.0, second_fz=20.0),
        session_id="test-tube-session-002",
    ).profile
    assert first.profile_id != second.profile_id
    assert first.session_evidence_sha256 != second.session_evidence_sha256


def test_gate_confirms_contact_but_never_claims_a_grasp() -> None:
    gate = ContactGate(_profile().profile)
    first = gate.update(
        _pair(start=3.0, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.06,
    )
    assert first.valid and not first.contact_present
    assert first.decision == "CONTACT_ABSENT"
    assert first.action == "CONTACT_ABSENT"
    second = gate.update(
        _pair(start=3.1, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.16,
        state=first.next_state,
    )
    assert second.valid and second.contact_present
    assert second.decision == "CONTACT_PRESENT"
    assert second.action == "CONTACT_PRESENT"
    assert "does not establish a grasp" in second.reason


def test_gate_requires_confirmed_loss_and_only_reports_absence() -> None:
    gate = ContactGate(_profile().profile)
    contact_1 = gate.update(
        _pair(start=3.0, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.06,
    )
    contact_2 = gate.update(
        _pair(start=3.1, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.16,
        state=contact_1.next_state,
    )
    loss_1 = gate.update(
        _pair(start=4.0, first_fz=10.01, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=4.06,
        state=contact_2.next_state,
    )
    assert loss_1.contact_present
    loss_2 = gate.update(
        _pair(start=4.1, first_fz=10.01, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=4.16,
        state=loss_1.next_state,
    )
    assert loss_2.valid and not loss_2.contact_present
    assert loss_2.action == "CONTACT_ABSENT"


def test_gate_rejects_wrong_session_and_changed_provenance() -> None:
    gate = ContactGate(_profile().profile)
    wrong_session = gate.update(
        _pair(start=3.0, first_fz=13.0, second_fz=20.0),
        session_id="some-other-session",
        now_monotonic_s=3.06,
    )
    assert not wrong_session.valid and wrong_session.action == "INVALID"
    changed_calibration = gate.update(
        _pair(
            start=3.0,
            first_fz=13.0,
            second_fz=20.0,
            calibration_suffix="-changed",
        ),
        session_id=SESSION,
        now_monotonic_s=3.06,
    )
    assert not changed_calibration.valid
    assert "provenance changed" in changed_calibration.reason


def test_gate_rejects_replay_and_any_older_nonidentical_window() -> None:
    gate = ContactGate(_profile().profile)
    first_windows = _pair(start=3.0, first_fz=13.0, second_fz=20.0)
    first = gate.update(
        first_windows,
        session_id=SESSION,
        now_monotonic_s=3.06,
    )
    replay = gate.update(
        first_windows,
        session_id=SESSION,
        now_monotonic_s=3.06,
        state=first.next_state,
    )
    assert not replay.valid
    assert "replays" in replay.reason
    older = gate.update(
        _pair(start=2.9, first_fz=13.5, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.06,
        state=first.next_state,
    )
    assert not older.valid
    assert "backwards" in older.reason


def test_gate_rejects_stale_external_state_even_with_new_evidence() -> None:
    gate = ContactGate(_profile().profile)
    initial_state = gate.state
    first = gate.update(
        _pair(start=3.0, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.06,
        state=initial_state,
    )
    assert first.valid
    rewound = gate.update(
        _pair(start=3.1, first_fz=13.0, second_fz=20.0),
        session_id=SESSION,
        now_monotonic_s=3.16,
        state=initial_state,
    )
    assert not rewound.valid
    assert "rewound" in rewound.reason


def test_bootstrap_rejects_empty_or_whitespace_session_id() -> None:
    for session_id in ("", " bad "):
        with pytest.raises(ValueError, match="session_id"):
            bootstrap_test_tube_contact(
                _series(start=1.0, first_fz=10.0, second_fz=20.0),
                _series(start=2.0, first_fz=14.0, second_fz=20.0),
                session_id=session_id,
            )
