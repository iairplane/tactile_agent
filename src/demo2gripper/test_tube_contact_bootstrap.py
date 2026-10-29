"""Session-bound GF220 contact-presence evidence for the test-tube task.

The human demonstration did not load either robot finger, so it cannot provide
a positive tactile label. At robot-session start a caller may collect several
unloaded windows, perform one bounded close with the tube between the fingers,
and collect several positive windows. This module derives a session-local,
raw-SDK ``Fz`` contact-presence threshold from that evidence.

The result answers one question only: does at least one qualified pad still
show the contact response learned in this session? It does not establish a
two-sided grasp, stable grasp, physical force, tube angle, slip, or insertion
jam. The module owns no device and emits no robot command.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .gf220_live import GF220ClassifierWindows

BOOTSTRAP_SCHEMA = "demo2gripper.test_tube_contact_bootstrap.v2"


class TestTubeContactBootstrapError(ValueError):
    """The supplied evidence cannot establish a session contact gate."""


@dataclass(frozen=True)
class TestTubeContactBootstrapConfig:
    """Quality gates for a raw-SDK, session-local presence detector."""

    min_unique_samples: int = 5
    min_calibration_windows_per_stage: int = 3
    max_age_s: float = 0.20
    max_pair_skew_s: float = 0.08
    # Raw SDK units are not physical force units, so both floors are explicit.
    minimum_fz_change_sensor_units: float = 1e-3
    minimum_threshold_floor_sensor_units: float = 1e-3
    positive_margin_noise_multiplier: float = 8.0
    threshold_noise_multiplier: float = 6.0
    minimum_consistent_positive_fraction: float = 2.0 / 3.0
    engage_fraction_of_positive: float = 0.25
    retain_fraction_of_positive: float = 0.10
    contact_confirmation_windows: int = 2
    loss_confirmation_windows: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_unique_samples",
            "min_calibration_windows_per_stage",
            "contact_confirmation_windows",
            "loss_confirmation_windows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_unique_samples < 2:
            raise ValueError("min_unique_samples must be at least two")
        for name in (
            "max_age_s",
            "max_pair_skew_s",
            "minimum_fz_change_sensor_units",
            "minimum_threshold_floor_sensor_units",
            "positive_margin_noise_multiplier",
            "threshold_noise_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "minimum_consistent_positive_fraction",
            "engage_fraction_of_positive",
            "retain_fraction_of_positive",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.retain_fraction_of_positive >= self.engage_fraction_of_positive:
            raise ValueError("retain fraction must be smaller than engage fraction")


@dataclass(frozen=True)
class TestTubeFingerContactCalibration:
    serial: str
    calibration_id: str
    unloaded_fz_sensor_units: float
    unloaded_noise_sensor_units: float
    positive_fz_sensor_units: float
    positive_delta_sensor_units: float
    positive_support_windows: int
    positive_total_windows: int
    contact_response_qualified: bool
    last_bootstrap_sdk_time_s: float
    last_bootstrap_capture_time_s: float


@dataclass(frozen=True)
class TestTubeContactProfile:
    """Session-only any-pad contact-presence thresholds in raw SDK units."""

    profile_id: str
    session_id: str
    session_evidence_sha256: str
    fingers: tuple[TestTubeFingerContactCalibration, TestTubeFingerContactCalibration]
    engage_threshold_sensor_units: float
    retain_threshold_sensor_units: float
    config: TestTubeContactBootstrapConfig


@dataclass(frozen=True)
class TestTubeContactState:
    """Owned state of one gate; callers may persist but may not rewind it."""

    session_id: str | None = None
    profile_id: str | None = None
    contact_present: bool = False
    contact_windows: int = 0
    loss_windows: int = 0
    last_sdk_time_s_by_serial: tuple[tuple[str, float], ...] = ()
    last_capture_time_s_by_serial: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TestTubeContactDecision:
    """A contact-presence observation, never a grasp or motion decision."""

    valid: bool
    contact_present: bool
    decision: str
    max_abs_fz_delta_sensor_units: float | None
    active_serials: tuple[str, ...]
    reason: str
    next_state: TestTubeContactState

    @property
    def action(self) -> str:
        """Compatibility alias; the value is an observation, not a command."""

        return self.decision


@dataclass(frozen=True)
class TestTubeContactBootstrapResult:
    profile: TestTubeContactProfile
    report: dict[str, Any]


@dataclass(frozen=True)
class _FingerWindow:
    calibration_id: str
    fz: np.ndarray
    sdk_time_s: np.ndarray
    capture_time_s: np.ndarray


def _validated_session_id(session_id: str) -> str:
    if not isinstance(session_id, str):
        raise TypeError("session_id must be a string")
    if not session_id or session_id != session_id.strip():
        raise ValueError("session_id must be non-empty and have no surrounding whitespace")
    if len(session_id) > 128 or any(ord(character) < 0x20 for character in session_id):
        raise ValueError("session_id is invalid")
    return session_id


def _as_window_sequence(
    value: GF220ClassifierWindows | Sequence[GF220ClassifierWindows],
    *,
    stage: str,
) -> tuple[GF220ClassifierWindows, ...]:
    if isinstance(value, GF220ClassifierWindows):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{stage} must be a window or a sequence of windows")
    result = tuple(value)
    if not result or not all(isinstance(item, GF220ClassifierWindows) for item in result):
        raise TypeError(f"{stage} must contain GF220ClassifierWindows")
    return result


def _extract(
    pair: GF220ClassifierWindows,
    *,
    minimum_samples: int,
    maximum_pair_skew_s: float,
    stage: str,
) -> dict[str, _FingerWindow]:
    if len(pair.serials) != 2 or len(set(pair.serials)) != 2:
        raise TestTubeContactBootstrapError(f"{stage} must contain two unique serials")
    if pair.endpoint_skew_s > maximum_pair_skew_s:
        raise TestTubeContactBootstrapError(f"{stage} exceeds the pair-skew limit")

    result: dict[str, _FingerWindow] = {}
    for serial, window in zip(pair.serials, pair.windows, strict=True):
        force = window.force6d
        if force is None:
            raise TestTubeContactBootstrapError(f"{stage}/{serial}: FORCE6D is missing")
        values = np.asarray(force.values, dtype=float)
        sdk = np.asarray(force.sdk_time_s, dtype=float)
        host = np.asarray(force.capture_monotonic_s, dtype=float)
        valid = np.asarray(force.valid, dtype=bool)
        if values.ndim != 2 or values.shape[1] != 6:
            raise TestTubeContactBootstrapError(
                f"{stage}/{serial}: FORCE6D must have shape (N, 6)"
            )
        indices = np.flatnonzero(valid & np.isfinite(values).all(axis=1))
        if indices.size:
            _, unique = np.unique(sdk[indices], return_index=True)
            indices = indices[np.sort(unique)]
        if indices.size < minimum_samples:
            raise TestTubeContactBootstrapError(
                f"{stage}/{serial}: only {indices.size} valid unique samples"
            )
        if np.any(np.diff(sdk[indices]) <= 0.0) or np.any(np.diff(host[indices]) <= 0.0):
            raise TestTubeContactBootstrapError(
                f"{stage}/{serial}: timestamps must strictly increase"
            )
        result[serial] = _FingerWindow(
            calibration_id=window.calibration_id,
            fz=values[indices, 2],
            sdk_time_s=sdk[indices],
            capture_time_s=host[indices],
        )
    return result


def _extract_many(
    value: GF220ClassifierWindows | Sequence[GF220ClassifierWindows],
    *,
    config: TestTubeContactBootstrapConfig,
    stage: str,
) -> dict[str, tuple[_FingerWindow, ...]]:
    pairs = _as_window_sequence(value, stage=stage)
    if len(pairs) < config.min_calibration_windows_per_stage:
        raise TestTubeContactBootstrapError(
            f"{stage} needs at least {config.min_calibration_windows_per_stage} windows"
        )
    result: dict[str, list[_FingerWindow]] = {}
    expected_serials: set[str] | None = None
    expected_calibrations: dict[str, str] = {}
    last_sdk: dict[str, float] = {}
    last_host: dict[str, float] = {}
    for index, pair in enumerate(pairs):
        extracted = _extract(
            pair,
            minimum_samples=config.min_unique_samples,
            maximum_pair_skew_s=config.max_pair_skew_s,
            stage=f"{stage}[{index}]",
        )
        serials = set(extracted)
        if expected_serials is None:
            expected_serials = serials
        elif serials != expected_serials:
            raise TestTubeContactBootstrapError(f"{stage}: sensor serials changed")
        for serial in sorted(extracted):
            window = extracted[serial]
            previous_calibration = expected_calibrations.setdefault(
                serial, window.calibration_id
            )
            if window.calibration_id != previous_calibration:
                raise TestTubeContactBootstrapError(
                    f"{stage}/{serial}: calibration provenance changed"
                )
            if serial in last_sdk and (
                window.sdk_time_s[0] <= last_sdk[serial]
                or window.capture_time_s[0] <= last_host[serial]
            ):
                raise TestTubeContactBootstrapError(
                    f"{stage}/{serial}: windows overlap, replay, or are out of order"
                )
            last_sdk[serial] = float(window.sdk_time_s[-1])
            last_host[serial] = float(window.capture_time_s[-1])
            result.setdefault(serial, []).append(window)
    return {serial: tuple(windows) for serial, windows in result.items()}


def _robust_noise(windows: tuple[_FingerWindow, ...], baseline: float) -> float:
    values = np.concatenate([window.fz for window in windows])
    sample_mad = float(np.median(np.abs(values - baseline)))
    medians = np.asarray([np.median(window.fz) for window in windows], dtype=float)
    between_window_mad = float(np.median(np.abs(medians - np.median(medians))))
    return 1.4826 * max(sample_mad, between_window_mad)


def _digest(
    *,
    session_id: str,
    stages: dict[str, dict[str, tuple[_FingerWindow, ...]]],
) -> str:
    digest = hashlib.sha256((BOOTSTRAP_SCHEMA + "\0" + session_id + "\0").encode())
    for stage in ("unloaded", "positive"):
        for serial in sorted(stages[stage]):
            digest.update(stage.encode())
            digest.update(serial.encode())
            for window in stages[stage][serial]:
                digest.update(window.calibration_id.encode())
                for value in (window.fz, window.sdk_time_s, window.capture_time_s):
                    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def bootstrap_test_tube_contact(
    unloaded: GF220ClassifierWindows | Sequence[GF220ClassifierWindows],
    positive_contact: GF220ClassifierWindows | Sequence[GF220ClassifierWindows],
    *,
    session_id: str,
    config: TestTubeContactBootstrapConfig = TestTubeContactBootstrapConfig(),
) -> TestTubeContactBootstrapResult:
    """Fit an any-pad contact-presence gate from repeated session evidence."""

    session_id = _validated_session_id(session_id)
    stages = {
        "unloaded": _extract_many(unloaded, config=config, stage="unloaded"),
        "positive": _extract_many(
            positive_contact, config=config, stage="positive_contact"
        ),
    }
    if set(stages["unloaded"]) != set(stages["positive"]):
        raise TestTubeContactBootstrapError("sensor serials changed during bootstrap")

    calibrations: list[TestTubeFingerContactCalibration] = []
    for serial in sorted(stages["unloaded"]):
        before_windows = stages["unloaded"][serial]
        after_windows = stages["positive"][serial]
        if before_windows[0].calibration_id != after_windows[0].calibration_id:
            raise TestTubeContactBootstrapError(
                f"{serial}: calibration id changed during bootstrap"
            )
        before_last = before_windows[-1]
        after_first = after_windows[0]
        if (
            before_last.sdk_time_s[-1] >= after_first.sdk_time_s[0]
            or before_last.capture_time_s[-1] >= after_first.capture_time_s[0]
        ):
            raise TestTubeContactBootstrapError(
                f"{serial}: unloaded and positive windows overlap or are out of order"
            )
        unloaded_values = np.concatenate([window.fz for window in before_windows])
        unloaded_fz = float(np.median(unloaded_values))
        unloaded_noise = _robust_noise(before_windows, unloaded_fz)
        positive_medians = np.asarray(
            [np.median(window.fz) for window in after_windows], dtype=float
        )
        signed_deltas = positive_medians - unloaded_fz
        signed_response = float(np.median(signed_deltas))
        positive_delta = abs(signed_response)
        required_margin = max(
            config.minimum_fz_change_sensor_units,
            config.positive_margin_noise_multiplier * unloaded_noise,
        )
        if signed_response == 0.0:
            support = 0
        else:
            direction = math.copysign(1.0, signed_response)
            support = int(
                np.count_nonzero(
                    (np.sign(signed_deltas) == direction)
                    & (np.abs(signed_deltas) >= required_margin)
                )
            )
        required_support = math.ceil(
            config.minimum_consistent_positive_fraction * len(after_windows)
        )
        qualified = positive_delta >= required_margin and support >= required_support
        last_positive = after_windows[-1]
        calibrations.append(
            TestTubeFingerContactCalibration(
                serial=serial,
                calibration_id=before_windows[0].calibration_id,
                unloaded_fz_sensor_units=unloaded_fz,
                unloaded_noise_sensor_units=unloaded_noise,
                positive_fz_sensor_units=float(np.median(positive_medians)),
                positive_delta_sensor_units=positive_delta,
                positive_support_windows=support,
                positive_total_windows=len(after_windows),
                contact_response_qualified=qualified,
                last_bootstrap_sdk_time_s=float(last_positive.sdk_time_s[-1]),
                last_bootstrap_capture_time_s=float(last_positive.capture_time_s[-1]),
            )
        )

    qualified = [item for item in calibrations if item.contact_response_qualified]
    if not qualified:
        raise TestTubeContactBootstrapError(
            "neither tactile pad has a repeatable Fz margin above unloaded noise"
        )
    positive_response = max(item.positive_delta_sensor_units for item in qualified)
    maximum_noise = max(item.unloaded_noise_sensor_units for item in qualified)
    noise_floor = config.threshold_noise_multiplier * maximum_noise
    engage = max(
        config.minimum_threshold_floor_sensor_units,
        noise_floor,
        config.engage_fraction_of_positive * positive_response,
    )
    retain = max(
        config.minimum_threshold_floor_sensor_units,
        noise_floor,
        config.retain_fraction_of_positive * positive_response,
    )
    if retain >= engage:
        raise TestTubeContactBootstrapError(
            "positive response leaves no valid hysteresis above unloaded noise/floor"
        )

    evidence_digest = _digest(session_id=session_id, stages=stages)
    profile = TestTubeContactProfile(
        profile_id=f"test-tube-contact-{evidence_digest}",
        session_id=session_id,
        session_evidence_sha256=evidence_digest,
        fingers=(calibrations[0], calibrations[1]),
        engage_threshold_sensor_units=engage,
        retain_threshold_sensor_units=retain,
        config=config,
    )
    report = {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "session_contact_presence_qualified",
        "profile_id": profile.profile_id,
        "session_id": session_id,
        "session_evidence_sha256": evidence_digest,
        "policy": {
            "signal": "maximum absolute raw SDK Fz change across qualified pads",
            "contact_logic": "any_qualified_pad_contact_present",
            "hysteresis": (
                "engage threshold while absent; lower retain threshold while present"
            ),
            "explicitly_not_claimed": [
                "two-sided contact",
                "grasp established",
                "grasp stability",
                "physical force units",
                "tube angle",
                "slip classification",
                "insertion jam classification",
                "logical left/right identity",
            ],
        },
        "thresholds_sensor_units": {
            "engage": engage,
            "retain": retain,
            "noise_floor": noise_floor,
        },
        "quality_gates": asdict(config),
        "fingers": [asdict(item) for item in calibrations],
        "hardware_commands_dispatched": False,
    }
    return TestTubeContactBootstrapResult(profile=profile, report=report)


class TestTubeContactGate:
    """Evaluate fresh windows and emit only session contact-presence evidence."""

    def __init__(self, profile: TestTubeContactProfile) -> None:
        if not isinstance(profile, TestTubeContactProfile):
            raise TypeError("profile must be TestTubeContactProfile")
        self.profile = profile
        self._state = TestTubeContactState(
            session_id=profile.session_id,
            profile_id=profile.profile_id,
            last_sdk_time_s_by_serial=tuple(
                (item.serial, item.last_bootstrap_sdk_time_s)
                for item in sorted(profile.fingers, key=lambda item: item.serial)
            ),
            last_capture_time_s_by_serial=tuple(
                (item.serial, item.last_bootstrap_capture_time_s)
                for item in sorted(profile.fingers, key=lambda item: item.serial)
            ),
        )

    @property
    def state(self) -> TestTubeContactState:
        return self._state

    def _invalid(self, reason: str) -> TestTubeContactDecision:
        return TestTubeContactDecision(
            valid=False,
            contact_present=self._state.contact_present,
            decision="INVALID",
            max_abs_fz_delta_sensor_units=None,
            active_serials=(),
            reason=reason,
            next_state=self._state,
        )

    def update(
        self,
        windows: GF220ClassifierWindows,
        *,
        session_id: str,
        now_monotonic_s: float,
        state: TestTubeContactState | None = None,
    ) -> TestTubeContactDecision:
        """Consume one strictly newer window from the profile's exact session."""

        try:
            supplied_session = _validated_session_id(session_id)
        except (TypeError, ValueError) as error:
            return self._invalid(str(error))
        if supplied_session != self.profile.session_id:
            return self._invalid("runtime session_id differs from the contact profile")
        if state is not None and state != self._state:
            return self._invalid("contact state is stale, foreign, or rewound")
        if not math.isfinite(now_monotonic_s):
            raise ValueError("now_monotonic_s must be finite")
        try:
            extracted = _extract(
                windows,
                minimum_samples=self.profile.config.min_unique_samples,
                maximum_pair_skew_s=self.profile.config.max_pair_skew_s,
                stage="runtime",
            )
        except (TypeError, TestTubeContactBootstrapError) as error:
            return self._invalid(str(error))

        expected = {item.serial: item for item in self.profile.fingers}
        if set(extracted) != set(expected):
            return self._invalid("runtime sensor serials differ from the session profile")

        previous_sdk = dict(self._state.last_sdk_time_s_by_serial)
        previous_host = dict(self._state.last_capture_time_s_by_serial)
        deltas: dict[str, float] = {}
        endpoints_sdk: list[tuple[str, float]] = []
        endpoints_host: list[tuple[str, float]] = []
        for serial in sorted(extracted):
            window = extracted[serial]
            calibration = expected[serial]
            if window.calibration_id != calibration.calibration_id:
                return self._invalid(f"{serial}: calibration provenance changed")
            if (
                window.sdk_time_s[0] <= previous_sdk[serial]
                or window.capture_time_s[0] <= previous_host[serial]
            ):
                return self._invalid(
                    f"{serial}: tactile evidence overlaps, replays, or moves backwards"
                )
            if window.capture_time_s[-1] < now_monotonic_s - self.profile.config.max_age_s:
                return self._invalid(f"{serial}: tactile window is stale")
            if window.capture_time_s[-1] > now_monotonic_s + 1e-6:
                return self._invalid(f"{serial}: tactile window is from the future")
            if calibration.contact_response_qualified:
                deltas[serial] = abs(
                    float(np.median(window.fz))
                    - calibration.unloaded_fz_sensor_units
                )
            endpoints_sdk.append((serial, float(window.sdk_time_s[-1])))
            endpoints_host.append((serial, float(window.capture_time_s[-1])))

        maximum = max(deltas.values())
        threshold = (
            self.profile.retain_threshold_sensor_units
            if self._state.contact_present
            else self.profile.engage_threshold_sensor_units
        )
        seen = maximum >= threshold
        contact_windows = self._state.contact_windows + 1 if seen else 0
        loss_windows = self._state.loss_windows + 1 if not seen else 0
        present = self._state.contact_present
        if not present and (
            contact_windows >= self.profile.config.contact_confirmation_windows
        ):
            present = True
            loss_windows = 0
        elif present and loss_windows >= self.profile.config.loss_confirmation_windows:
            present = False
            contact_windows = 0
        next_state = TestTubeContactState(
            session_id=self.profile.session_id,
            profile_id=self.profile.profile_id,
            contact_present=present,
            contact_windows=contact_windows,
            loss_windows=loss_windows,
            last_sdk_time_s_by_serial=tuple(endpoints_sdk),
            last_capture_time_s_by_serial=tuple(endpoints_host),
        )
        self._state = next_state
        active = tuple(
            serial for serial, value in sorted(deltas.items()) if value >= threshold
        )
        return TestTubeContactDecision(
            valid=True,
            contact_present=present,
            decision="CONTACT_PRESENT" if present else "CONTACT_ABSENT",
            max_abs_fz_delta_sensor_units=maximum,
            active_serials=active,
            reason=(
                "at least one qualified pad has the session contact response; "
                "this does not establish a grasp"
                if present
                else "no qualified pad has confirmed the session contact response"
            ),
            next_state=next_state,
        )


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "TestTubeContactBootstrapConfig",
    "TestTubeContactBootstrapError",
    "TestTubeContactBootstrapResult",
    "TestTubeContactDecision",
    "TestTubeContactGate",
    "TestTubeContactProfile",
    "TestTubeContactState",
    "TestTubeFingerContactCalibration",
    "bootstrap_test_tube_contact",
]
