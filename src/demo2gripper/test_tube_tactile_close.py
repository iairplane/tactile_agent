"""Pure staged Hand-E closing schedule and dual-pad tactile state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gf220_live import GF220ClassifierWindows
from .test_tube_contact_approach import (
    ContactApproachMemory,
    ContactApproachProfile,
    FingerContactObservation,
    evaluate_contact_window,
)

TACTILE_CLOSE_SCHEMA = "demo2gripper.test_tube_tactile_close.v1"


@dataclass(frozen=True)
class TactileCloseMemory:
    last_sdk_time_s_by_serial: tuple[tuple[str, float], ...] = ()
    bilateral_windows: int = 0
    single_serial: str | None = None
    single_windows: int = 0


@dataclass(frozen=True)
class TactileCloseDecision:
    valid: bool
    action: str
    active_serials: tuple[str, ...]
    bilateral_contact: bool
    grasp_ready: bool
    needs_centering: bool
    reason: str
    observations: tuple[FingerContactObservation, FingerContactObservation]
    next_memory: TactileCloseMemory


def make_staged_close_schedule(
    initial_opening_mm: float,
    *,
    coarse_until_mm: float = 25.0,
    coarse_step_mm: float = 5.0,
    fine_step_mm: float = 2.0,
    minimum_opening_mm: float = 10.0,
) -> tuple[float, ...]:
    """Return decreasing absolute opening targets without crossing the floor."""

    initial = _finite(initial_opening_mm, "initial_opening_mm")
    coarse_until = _finite(coarse_until_mm, "coarse_until_mm")
    coarse_step = _finite(coarse_step_mm, "coarse_step_mm")
    fine_step = _finite(fine_step_mm, "fine_step_mm")
    minimum = _finite(minimum_opening_mm, "minimum_opening_mm")
    if not 0.0 <= minimum < initial <= 50.0:
        raise ValueError(
            "openings must satisfy 0 <= minimum < initial <= 50 mm"
        )
    if not minimum < coarse_until <= 50.0:
        raise ValueError("coarse_until_mm must exceed the minimum and not exceed 50 mm")
    if not 0.0 < coarse_step <= 5.0:
        raise ValueError("coarse_step_mm must lie in (0, 5]")
    if not 0.0 < fine_step <= 2.0:
        raise ValueError("fine_step_mm must lie in (0, 2]")
    targets: list[float] = []
    current = initial
    while current > coarse_until + 1e-9:
        current = max(coarse_until, current - coarse_step)
        targets.append(current)
    while current > minimum + 1e-9:
        current = max(minimum, current - fine_step)
        targets.append(current)
    return tuple(targets)


def evaluate_tactile_close_window(
    windows: GF220ClassifierWindows,
    profile: ContactApproachProfile,
    memory: TactileCloseMemory = TactileCloseMemory(),
) -> TactileCloseDecision:
    """Promote two bilateral windows or reject two persistent one-pad windows."""

    classifier_memory = ContactApproachMemory(
        last_sdk_time_s_by_serial=memory.last_sdk_time_s_by_serial
    )
    classified = evaluate_contact_window(windows, profile, classifier_memory)
    endpoints = classified.next_memory.last_sdk_time_s_by_serial
    if not classified.valid:
        return TactileCloseDecision(
            valid=False,
            action="hold_invalid",
            active_serials=(),
            bilateral_contact=False,
            grasp_ready=False,
            needs_centering=False,
            reason=classified.reason,
            observations=classified.observations,
            next_memory=TactileCloseMemory(last_sdk_time_s_by_serial=endpoints),
        )

    active = classified.active_serials
    bilateral = len(active) == 2
    if bilateral:
        bilateral_windows = memory.bilateral_windows + 1
        single_serial = None
        single_windows = 0
    elif len(active) == 1:
        bilateral_windows = 0
        single_serial = active[0]
        single_windows = (
            memory.single_windows + 1 if memory.single_serial == single_serial else 1
        )
    else:
        bilateral_windows = 0
        single_serial = None
        single_windows = 0

    ready = bilateral_windows >= 2
    needs_centering = single_windows >= 2
    if ready:
        action = "grasp_ready"
        reason = "bilateral image contact persisted for two fresh windows"
    elif needs_centering:
        action = "needs_centering"
        reason = f"single-pad contact persisted on {single_serial} for two fresh windows"
    elif active:
        action = "hold_confirm"
        reason = "contact observed; hold opening and collect stationary confirmation"
    else:
        action = "continue_closing"
        reason = "neither pad reports contact"
    next_memory = TactileCloseMemory(
        last_sdk_time_s_by_serial=endpoints,
        bilateral_windows=bilateral_windows,
        single_serial=single_serial,
        single_windows=single_windows,
    )
    return TactileCloseDecision(
        valid=True,
        action=action,
        active_serials=active,
        bilateral_contact=bilateral,
        grasp_ready=ready,
        needs_centering=needs_centering,
        reason=reason,
        observations=classified.observations,
        next_memory=next_memory,
    )


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "TACTILE_CLOSE_SCHEMA",
    "TactileCloseDecision",
    "TactileCloseMemory",
    "evaluate_tactile_close_window",
    "make_staged_close_schedule",
]
