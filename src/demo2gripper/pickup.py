"""Fail-closed event checks for a CAD-free RGB-D pickup demonstration.

The grasp geometry is estimated from a stationary, pre-contact object cloud.
This module only validates temporal evidence that the demonstrated object was
subsequently lifted together with the hand; it does not estimate a reusable
object pose or authorize robot execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed_object import make_fixed_camera_observation
from .geometry import longest_true_run, normalize, robust_direction_mean, robust_mean
from .intent import INDEX_TIP, THUMB_TIP, WRIST, IntentConfig, stable_contact_mask


@dataclass(frozen=True)
class PickupEvents:
    """Detected lift events in the calibrated robot-base frame.

    Intervals are half-open.  ``vertical_displacement_m`` is relative to the
    robust median object center in the clean reference prefix.
    """

    reference_center_base_m: np.ndarray
    reference_vertical_sigma_m: float
    reference_motion_p95_m: float
    effective_lift_onset_m: float
    lift_onset_frame: int
    airborne_interval: tuple[int, int]
    peak_frame: int
    peak_lift_m: float
    airborne_duration_s: float
    vertical_displacement_m: np.ndarray


@dataclass(frozen=True)
class HandObjectCoupling:
    """Evidence that the tracked object moved rigidly with the pinch center."""

    valid_frames: int
    median_offset_base_m: np.ndarray
    median_center_distance_m: float
    relative_motion_median_m: float
    relative_motion_p95_m: float
    moving_step_pairs: int
    velocity_cosine_median: float
    step_displacement_mismatch_p95_m: float


@dataclass(frozen=True)
class PickupIntent:
    """Human pinch geometry inferred before the object starts to move.

    The numeric pose is ``T_camera_gripper_tcp``.  The two points are observed
    3-D fingertips and are deliberately not labelled as force contacts.  Pose
    orientation comes from stable-contact hand geometry; ``approach_*`` fields
    describe the earlier free-space path for phase/replay review only.
    """

    T_camera_gripper_tcp: np.ndarray
    demonstrated_pinch_width_m: float
    observed_fingertip_points_camera_m: np.ndarray
    stable_interval: tuple[int, int]
    support_frames: int
    approach_interval: tuple[int, int] | None
    approach_trajectory_displacement_m: float | None
    approach_trajectory_residual_p95_m: float | None
    approach_window_angle_p95_deg: float | None
    approach_window_angle_max_deg: float | None
    approach_window_count: int


@dataclass(frozen=True)
class ApproachReplayAssessment:
    """Replay-only assessment of the demonstrated free-space approach.

    These diagnostics may withhold a direct fixed-object replay target.  They
    do not invalidate the independently observed contact, lift, or hand/object
    coupling evidence of a pickup demonstration.
    """

    signal_to_residual_ratio: float | None
    window_stable: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class PreliftObjectMotion:
    """Object motion between the clean reference prefix and contact.

    A demonstrated pickup can remain valid after the demonstrator slides the
    object, but a grasp pose anchored to the original object location cannot.
    ``replay_target_valid`` therefore describes only direct replay at the
    reference location; it is independent of pickup-event validation.
    """

    contact_center_base_m: np.ndarray
    shift_vector_base_m: np.ndarray
    contact_center_shift_m: float
    contact_shift_p95_m: float
    effective_shift_limit_m: float
    valid_frames: int
    replay_target_valid: bool


def align_hand_joints_to_reference_translation(
    hand_joints_camera_m: np.ndarray,
    object_centers_camera_m: np.ndarray,
    reference_center_camera_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align hand joints with a static reference cloud using object translation.

    ``object_centers_camera_m`` is allowed to contain non-finite rows because
    RGB-D tracking can be unavailable under occlusion.  Those rows are copied
    unchanged into the returned joint array and marked false in ``track_valid``;
    callers must exclude them from contact and trajectory inference (for
    example, by setting their joint confidences to zero).  The corresponding
    rows in ``translations_camera_m`` remain NaN so they cannot be mistaken for
    measured zero motion.

    This is deliberately translation-only.  It does not infer or compensate
    object rotation.
    """

    joints = np.asarray(hand_joints_camera_m, dtype=float)
    centers = np.asarray(object_centers_camera_m, dtype=float)
    reference = np.asarray(reference_center_camera_m, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"hand joints must have shape (F, 21, 3), got {joints.shape}")
    if centers.shape != (len(joints), 3):
        raise ValueError(
            f"object centers must have shape ({len(joints)}, 3), got {centers.shape}"
        )
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference center must be a finite 3-vector")
    if not np.isfinite(joints).all():
        raise ValueError("hand joints contain NaN or infinity")

    track_valid = np.isfinite(centers).all(axis=1)
    translations = np.full_like(centers, np.nan, dtype=float)
    translations[track_valid] = centers[track_valid] - reference
    aligned = joints.copy()
    aligned[track_valid] -= translations[track_valid, None, :]
    return aligned, track_valid, translations


def _fit_approach_trajectory(
    pinch_centers: np.ndarray,
    timestamps_s: np.ndarray,
    valid: np.ndarray,
    start: int,
    *,
    lookback_frames: int,
    guard_frames: int,
    min_frames: int,
) -> tuple[np.ndarray, tuple[int, int], float, float]:
    if guard_frames < 0:
        raise ValueError("approach guard_frames must be non-negative")
    approach_end = start - guard_frames
    approach_start = max(0, approach_end - lookback_frames)
    if approach_end <= approach_start:
        raise ValueError("no frames remain for the pre-contact approach trajectory")
    indices = np.flatnonzero(valid[approach_start:approach_end]) + approach_start
    if len(indices) < min_frames:
        raise ValueError(
            f"only {len(indices)} valid pre-contact trajectory frames, need {min_frames}"
        )
    times = timestamps_s[indices]
    times = times - np.mean(times)
    points = pinch_centers[indices]
    design = np.column_stack((times, np.ones(len(times))))
    coefficients = np.linalg.lstsq(design, points, rcond=None)[0]
    fitted = design @ coefficients
    residuals = np.linalg.norm(points - fitted, axis=1)
    direction = normalize(coefficients[0], name="pre-contact trajectory")
    displacement = float(np.linalg.norm(coefficients[0]) * (times.max() - times.min()))
    return (
        direction,
        (int(indices[0]), int(indices[-1]) + 1),
        displacement,
        float(np.quantile(residuals, 0.95)),
    )


def infer_pre_lift_pickup_intent(
    hand_joints_camera_m: np.ndarray,
    hand_joint_confidence: np.ndarray,
    timestamps_s: np.ndarray,
    object_points_camera_m: np.ndarray,
    config: IntentConfig = IntentConfig(),
    *,
    approach_lookback_frames: int = 30,
    approach_guard_frames: int = 4,
    min_approach_frames: int = 15,
) -> PickupIntent:
    """Infer a canonical TCP from the longest stationary pre-lift pinch.

    Fingertip proximity to the clean object cloud selects the interval.  Pose
    orientation comes from contact hand geometry: thumb-to-index defines
    canonical ``+X`` and wrist-to-pinch defines canonical ``+Z``.  The fitted
    free-space trajectory is retained only as an optional phase/replay
    diagnostic.  Retaining raw hand directions avoids unstable nearest-surface
    assignments on a partial single-view cloud.
    """

    observation = make_fixed_camera_observation(
        hand_joints_camera_m,
        hand_joint_confidence,
        timestamps_s,
        object_points_camera_m,
    )
    contact_mask = stable_contact_mask(observation, config)
    start, end = longest_true_run(contact_mask)
    if end - start < config.min_stable_frames:
        raise ValueError(
            f"no stable pre-lift contact interval with at least {config.min_stable_frames} frames"
        )

    joints = np.asarray(hand_joints_camera_m, dtype=float)
    tips = joints[start:end, [THUMB_TIP, INDEX_TIP]]
    midpoints = np.mean(tips, axis=1)
    closing_directions = tips[:, 1] - tips[:, 0]
    x_reference = robust_direction_mean(closing_directions, name="closing axis")
    contact_approach = midpoints - joints[start:end, WRIST]
    contact_approach = contact_approach[
        np.linalg.norm(contact_approach, axis=1) > 1e-9
    ]
    if len(contact_approach) == 0:
        raise ValueError("wrist and pinch center do not define a contact approach axis")
    z_axis = robust_direction_mean(
        contact_approach, name="contact wrist-to-pinch approach axis"
    )
    x_axis = normalize(
        x_reference - np.dot(x_reference, z_axis) * z_axis,
        name="closing axis",
    )
    y_axis = normalize(np.cross(z_axis, x_axis), name="lateral axis")
    x_axis = normalize(np.cross(y_axis, z_axis), name="closing axis")
    if np.dot(x_axis, x_reference) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis

    tip_confidence = np.min(
        np.asarray(hand_joint_confidence, dtype=float)[:, [THUMB_TIP, INDEX_TIP]], axis=1
    )
    approach_interval: tuple[int, int] | None = None
    approach_displacement: float | None = None
    approach_residual_p95: float | None = None
    trajectory_z_axis: np.ndarray | None = None
    try:
        approach, approach_interval, approach_displacement, approach_residual_p95 = (
            _fit_approach_trajectory(
                np.mean(joints[:, [THUMB_TIP, INDEX_TIP]], axis=1),
                np.asarray(timestamps_s, dtype=float),
                tip_confidence >= config.min_joint_confidence,
                start,
                lookback_frames=approach_lookback_frames,
                guard_frames=approach_guard_frames,
                min_frames=min_approach_frames,
            )
        )
        trajectory_z_axis = normalize(
            approach - np.dot(approach, x_reference) * x_reference,
            name="trajectory approach axis",
        )
    except ValueError:
        # Free-space motion is not pickup evidence.  A missing or degenerate
        # fit is represented explicitly and later withholds direct replay.
        pass

    lookbacks = list(range(min_approach_frames, approach_lookback_frames + 1, 5))
    if approach_lookback_frames not in lookbacks:
        lookbacks.append(approach_lookback_frames)
    window_axes: list[np.ndarray] = []
    if trajectory_z_axis is not None:
        for lookback in sorted(set(lookbacks)):
            try:
                window_direction, _, _, _ = _fit_approach_trajectory(
                    np.mean(joints[:, [THUMB_TIP, INDEX_TIP]], axis=1),
                    np.asarray(timestamps_s, dtype=float),
                    tip_confidence >= config.min_joint_confidence,
                    start,
                    lookback_frames=lookback,
                    guard_frames=approach_guard_frames,
                    min_frames=min_approach_frames,
                )
                window_axes.append(
                    normalize(
                        window_direction
                        - np.dot(window_direction, x_reference) * x_reference,
                        name="windowed trajectory approach axis",
                    )
                )
            except ValueError:
                continue
    if window_axes and trajectory_z_axis is not None:
        window_angles = np.rad2deg(
            np.arccos(
                np.clip(np.asarray(window_axes) @ trajectory_z_axis, -1.0, 1.0)
            )
        )
        window_angle_p95 = float(np.quantile(window_angles, 0.95))
        window_angle_max = float(np.max(window_angles))
    else:
        window_angles = np.empty(0, dtype=float)
        window_angle_p95 = None
        window_angle_max = None

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = robust_mean(midpoints)
    widths = np.linalg.norm(closing_directions, axis=1)
    return PickupIntent(
        T_camera_gripper_tcp=transform,
        demonstrated_pinch_width_m=float(np.median(widths)),
        observed_fingertip_points_camera_m=np.stack(
            [robust_mean(tips[:, 0]), robust_mean(tips[:, 1])]
        ),
        stable_interval=(start, end),
        support_frames=end - start,
        approach_interval=approach_interval,
        approach_trajectory_displacement_m=approach_displacement,
        approach_trajectory_residual_p95_m=approach_residual_p95,
        approach_window_angle_p95_deg=window_angle_p95,
        approach_window_angle_max_deg=window_angle_max,
        approach_window_count=len(window_angles),
    )


def assess_approach_for_fixed_object_replay(
    intent: PickupIntent,
    *,
    min_trajectory_displacement_m: float,
    max_trajectory_residual_p95_m: float,
    min_signal_to_residual: float,
    min_window_count: int,
    max_window_angle_p95_deg: float,
) -> ApproachReplayAssessment:
    """Assess free-space approach evidence for direct fixed-object replay only."""

    failures: list[str] = []
    displacement = intent.approach_trajectory_displacement_m
    residual = intent.approach_trajectory_residual_p95_m
    window_angle = intent.approach_window_angle_p95_deg
    if displacement is None or residual is None or intent.approach_interval is None:
        return ApproachReplayAssessment(
            signal_to_residual_ratio=None,
            window_stable=False,
            gate_failures=("approach_trajectory_unavailable",),
        )

    if displacement < min_trajectory_displacement_m:
        failures.append("approach_trajectory_too_short")
    if residual > max_trajectory_residual_p95_m:
        failures.append("approach_trajectory_fit_residual")
    ratio = displacement / max(residual, np.finfo(float).eps)
    if ratio < min_signal_to_residual:
        failures.append("approach_signal_to_residual")
    window_stable = (
        window_angle is not None
        and intent.approach_window_count >= min_window_count
        and window_angle <= max_window_angle_p95_deg
    )
    if not window_stable:
        failures.append("approach_window_instability")
    return ApproachReplayAssessment(
        signal_to_residual_ratio=ratio,
        window_stable=window_stable,
        gate_failures=tuple(failures),
    )


def measure_prelift_object_motion(
    object_centers_base_m: np.ndarray,
    reference_center_base_m: np.ndarray,
    contact_interval: tuple[int, int],
    *,
    reference_motion_p95_m: float,
    max_shift_m: float = 0.005,
    noise_multiplier: float = 6.0,
    min_frames: int = 5,
) -> PreliftObjectMotion:
    """Gate direct replay when the object moved before the demonstrated lift.

    The limit adapts to measured reference-track noise but never falls below
    ``max_shift_m``.  The p95 distance is used for the gate, while the robust
    contact center and its shift are retained for review.
    """

    centers = np.asarray(object_centers_base_m, dtype=float)
    reference = np.asarray(reference_center_base_m, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"object centers must have shape (F, 3), got {centers.shape}")
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference center must be a finite 3-vector")
    start, end = contact_interval
    if not 0 <= start < end <= len(centers):
        raise ValueError("contact interval must be a non-empty half-open frame interval")
    for name, value in (
        ("reference_motion_p95_m", reference_motion_p95_m),
        ("max_shift_m", max_shift_m),
        ("noise_multiplier", noise_multiplier),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")

    selected = centers[start:end]
    selected = selected[np.isfinite(selected).all(axis=1)]
    if len(selected) < min_frames:
        raise ValueError(
            f"only {len(selected)} valid pre-lift object centers, need {min_frames}"
        )
    contact_center = robust_mean(selected)
    shift_vector = contact_center - reference
    distances = np.linalg.norm(selected - reference, axis=1)
    shift_p95 = float(np.quantile(distances, 0.95))
    effective_limit = max(max_shift_m, noise_multiplier * reference_motion_p95_m)
    return PreliftObjectMotion(
        contact_center_base_m=contact_center,
        shift_vector_base_m=shift_vector,
        contact_center_shift_m=float(np.linalg.norm(shift_vector)),
        contact_shift_p95_m=shift_p95,
        effective_shift_limit_m=effective_limit,
        valid_frames=len(selected),
        replay_target_valid=shift_p95 <= effective_limit,
    )


def _validate_track(
    object_centers_base_m: np.ndarray, timestamps_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(object_centers_base_m, dtype=float)
    timestamps = np.asarray(timestamps_s, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"object centers must have shape (F, 3), got {centers.shape}")
    if timestamps.shape != (len(centers),):
        raise ValueError(f"timestamps must have shape ({len(centers)},), got {timestamps.shape}")
    if not np.isfinite(timestamps).all() or (
        len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("timestamps must be finite and strictly increasing")
    return centers, timestamps


def _true_runs(mask: np.ndarray, *, start: int = 0) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    padded = np.pad(values, (1, 1), constant_values=False).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [
        (max(int(first), start), int(end))
        for first, end in zip(starts, ends, strict=True)
        if end > start
    ]


def detect_pickup_events(
    object_centers_base_m: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    reference_frame_count: int,
    min_reference_frames: int = 15,
    max_reference_motion_p95_m: float = 0.005,
    min_lift_onset_m: float = 0.008,
    onset_noise_sigma: float = 6.0,
    min_onset_frames: int = 5,
    min_airborne_height_m: float = 0.030,
    min_airborne_frames: int = 15,
    min_airborne_duration_s: float = 0.5,
) -> PickupEvents:
    """Detect a sustained vertical pickup using the calibrated base ``+Z`` axis.

    The onset threshold is the larger of ``min_lift_onset_m`` and a robust
    noise-derived threshold from the stationary reference prefix.  A second,
    higher threshold validates a sustained airborne interval.  Missing 3-D
    observations break a run, so occlusion cannot silently count as lift.
    """

    centers, timestamps = _validate_track(object_centers_base_m, timestamps_s)
    frame_count = len(centers)
    if not 1 <= reference_frame_count < frame_count:
        raise ValueError("reference_frame_count must lie in [1, frame_count)")
    for name, value in (
        ("min_reference_frames", min_reference_frames),
        ("min_onset_frames", min_onset_frames),
        ("min_airborne_frames", min_airborne_frames),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("max_reference_motion_p95_m", max_reference_motion_p95_m),
        ("min_lift_onset_m", min_lift_onset_m),
        ("onset_noise_sigma", onset_noise_sigma),
        ("min_airborne_height_m", min_airborne_height_m),
        ("min_airborne_duration_s", min_airborne_duration_s),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if min_airborne_height_m <= min_lift_onset_m:
        raise ValueError("min_airborne_height_m must exceed min_lift_onset_m")

    valid = np.isfinite(centers).all(axis=1)
    reference_valid = valid[:reference_frame_count]
    if int(np.count_nonzero(reference_valid)) < min_reference_frames:
        raise ValueError(
            "insufficient valid object centers in reference prefix: "
            f"{np.count_nonzero(reference_valid)} < {min_reference_frames}"
        )
    reference_centers = centers[:reference_frame_count][reference_valid]
    reference_center = np.median(reference_centers, axis=0)
    reference_residuals = np.linalg.norm(reference_centers - reference_center, axis=1)
    reference_motion_p95 = float(np.quantile(reference_residuals, 0.95))
    if reference_motion_p95 > max_reference_motion_p95_m:
        raise ValueError(
            "object is not stationary in the reference prefix: p95 center motion "
            f"{reference_motion_p95 * 1000.0:.1f} mm"
        )

    vertical_reference = reference_centers[:, 2]
    vertical_mad = float(
        np.median(np.abs(vertical_reference - np.median(vertical_reference)))
    )
    vertical_sigma = 1.4826 * vertical_mad
    effective_onset = max(min_lift_onset_m, onset_noise_sigma * vertical_sigma)

    vertical_displacement = np.full(frame_count, np.nan, dtype=float)
    vertical_displacement[valid] = centers[valid, 2] - reference_center[2]
    onset_mask = valid & (vertical_displacement >= effective_onset)
    onset_runs = [
        run
        for run in _true_runs(onset_mask, start=reference_frame_count)
        if run[1] - run[0] >= min_onset_frames and run[1] > reference_frame_count
    ]
    if not onset_runs:
        raise ValueError(
            "no sustained lift onset: object never remains at least "
            f"{effective_onset * 1000.0:.1f} mm above the reference height for "
            f"{min_onset_frames} frames"
        )
    lift_onset = max(onset_runs[0][0], reference_frame_count)

    airborne_mask = valid & (vertical_displacement >= min_airborne_height_m)
    airborne_runs = [
        run
        for run in _true_runs(airborne_mask, start=lift_onset)
        if run[0] >= lift_onset and run[1] - run[0] >= min_airborne_frames
    ]
    if not airborne_runs:
        raise ValueError(
            "pickup is not validated: no sustained airborne interval at least "
            f"{min_airborne_height_m * 1000.0:.1f} mm high for "
            f"{min_airborne_frames} frames"
        )
    airborne_start, airborne_end = airborne_runs[0]
    airborne_duration = float(timestamps[airborne_end - 1] - timestamps[airborne_start])
    if airborne_duration < min_airborne_duration_s:
        raise ValueError(
            f"airborne interval lasts {airborne_duration:.3f}s, below "
            f"{min_airborne_duration_s:.3f}s"
        )

    peak_frame = int(
        airborne_start
        + np.nanargmax(vertical_displacement[airborne_start:airborne_end])
    )
    return PickupEvents(
        reference_center_base_m=reference_center,
        reference_vertical_sigma_m=vertical_sigma,
        reference_motion_p95_m=reference_motion_p95,
        effective_lift_onset_m=effective_onset,
        lift_onset_frame=lift_onset,
        airborne_interval=(airborne_start, airborne_end),
        peak_frame=peak_frame,
        peak_lift_m=float(vertical_displacement[peak_frame]),
        airborne_duration_s=airborne_duration,
        vertical_displacement_m=vertical_displacement,
    )


def measure_hand_object_coupling(
    object_centers_base_m: np.ndarray,
    pinch_centers_base_m: np.ndarray,
    pinch_confidence: np.ndarray,
    interval: tuple[int, int],
    *,
    min_confidence: float = 0.5,
    min_frames: int = 15,
    max_offset_residual_m: float | None = None,
) -> HandObjectCoupling:
    """Measure whether the lifted object retains a fixed offset to the pinch.

    When ``max_offset_residual_m`` is provided, a component-wise median of all
    otherwise-valid offsets defines a robust consensus.  Offset rows farther
    than the threshold are discarded before every reported metric is computed.
    This rejects isolated hand-pose jumps without allowing them to satisfy the
    minimum-frame requirement.
    """

    objects = np.asarray(object_centers_base_m, dtype=float)
    pinch = np.asarray(pinch_centers_base_m, dtype=float)
    confidence = np.asarray(pinch_confidence, dtype=float)
    if objects.ndim != 2 or objects.shape[1] != 3:
        raise ValueError(f"object centers must have shape (F, 3), got {objects.shape}")
    if pinch.shape != objects.shape:
        raise ValueError(f"pinch centers must have shape {objects.shape}, got {pinch.shape}")
    if confidence.shape != (len(objects),):
        raise ValueError(f"pinch confidence must have shape ({len(objects)},)")
    start, end = interval
    if not 0 <= start < end <= len(objects):
        raise ValueError("interval must be a non-empty half-open frame interval")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must lie in [0, 1]")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    if max_offset_residual_m is not None and (
        not np.isfinite(max_offset_residual_m) or max_offset_residual_m < 0.0
    ):
        raise ValueError("max_offset_residual_m must be finite and non-negative")

    valid = (
        np.isfinite(objects).all(axis=1)
        & np.isfinite(pinch).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= min_confidence)
    )
    interval_valid = valid[start:end].copy()
    candidate_frames = int(np.count_nonzero(interval_valid))
    if candidate_frames < min_frames:
        raise ValueError(
            f"only {candidate_frames} valid hand/object coupling frames, need {min_frames}"
        )

    candidate_offsets = (
        objects[start:end][interval_valid] - pinch[start:end][interval_valid]
    )
    if max_offset_residual_m is not None:
        consensus = np.median(candidate_offsets, axis=0)
        candidate_residuals = np.linalg.norm(candidate_offsets - consensus, axis=1)
        candidate_inliers = candidate_residuals <= max_offset_residual_m
        candidate_indices = np.flatnonzero(interval_valid)
        interval_valid[candidate_indices[~candidate_inliers]] = False

    valid_frames = int(np.count_nonzero(interval_valid))
    if valid_frames < min_frames:
        raise ValueError(
            f"only {valid_frames} inlier hand/object coupling frames after "
            f"offset-residual filtering, need {min_frames}"
        )
    offsets = objects[start:end][interval_valid] - pinch[start:end][interval_valid]
    median_offset = np.median(offsets, axis=0)
    relative_motion = np.linalg.norm(offsets - median_offset, axis=1)
    center_distance = np.linalg.norm(offsets, axis=1)

    interval_indices = np.flatnonzero(interval_valid) + start
    adjacent = np.diff(interval_indices) == 1
    first_indices = interval_indices[:-1][adjacent]
    second_indices = interval_indices[1:][adjacent]
    object_steps = objects[second_indices] - objects[first_indices]
    pinch_steps = pinch[second_indices] - pinch[first_indices]
    object_step_norm = np.linalg.norm(object_steps, axis=1)
    pinch_step_norm = np.linalg.norm(pinch_steps, axis=1)
    moving = (object_step_norm >= 0.0005) & (pinch_step_norm >= 0.0005)
    moving_pairs = int(np.count_nonzero(moving))
    if moving_pairs:
        velocity_cosines = np.einsum(
            "ij,ij->i", object_steps[moving], pinch_steps[moving]
        ) / (object_step_norm[moving] * pinch_step_norm[moving])
        step_mismatch = np.linalg.norm(object_steps[moving] - pinch_steps[moving], axis=1)
        velocity_cosine_median = float(np.median(velocity_cosines))
        step_mismatch_p95 = float(np.quantile(step_mismatch, 0.95))
    else:
        velocity_cosine_median = float("nan")
        step_mismatch_p95 = float("nan")
    return HandObjectCoupling(
        valid_frames=valid_frames,
        median_offset_base_m=median_offset,
        median_center_distance_m=float(np.median(center_distance)),
        relative_motion_median_m=float(np.median(relative_motion)),
        relative_motion_p95_m=float(np.quantile(relative_motion, 0.95)),
        moving_step_pairs=moving_pairs,
        velocity_cosine_median=velocity_cosine_median,
        step_displacement_mismatch_p95_m=step_mismatch_p95,
    )


def validate_contact_before_lift(
    stable_interval: tuple[int, int],
    lift_onset_frame: int,
    *,
    max_gap_frames: int = 3,
) -> None:
    """Require inferred contact to persist until immediately before the lift."""

    start, end = stable_interval
    if not 0 <= start < end <= lift_onset_frame:
        raise ValueError("stable contact interval must end no later than lift onset")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    gap = lift_onset_frame - end
    if gap > max_gap_frames:
        raise ValueError(
            f"stable contact ends {gap} frames before lift onset; allowed gap is "
            f"{max_gap_frames} frames"
        )
