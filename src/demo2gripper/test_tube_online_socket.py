"""Temporal RGB-D localization of a test-tube rack socket entry.

The detected frame is the *entry frame itself*: its origin is the centre of
the selected opening and local +Z points out of the rack.  Consequently the
controller-side ``T_socket_tube_entry`` is identity in this runtime mode.
No 2-D circle is promoted to metric space without valid annulus depth, a
plane fit, unique candidate selection, and temporal SE(3) stability.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .franka import validate_rigid_transform


class OnlineSocketError(RuntimeError):
    pass


@dataclass(frozen=True)
class SocketEntryCandidate:
    T_camera_entry: np.ndarray
    score: float
    position_std_m: float
    orientation_std_deg: float
    centre_uv: tuple[float, float]
    evidence_id: str

    def __post_init__(self) -> None:
        transform = validate_rigid_transform(
            self.T_camera_entry, name="socket candidate T_camera_entry"
        ).copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_camera_entry", transform)
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("socket candidate score must lie in [0,1]")
        for name in ("position_std_m", "orientation_std_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if len(self.centre_uv) != 2 or not np.isfinite(self.centre_uv).all():
            raise ValueError("centre_uv must contain two finite pixels")


@dataclass(frozen=True)
class OnlineSocketTrackerConfig:
    min_score: float = 0.55
    min_stable_frames: int = 4
    max_translation_jump_m: float = 0.010
    max_orientation_jump_deg: float = 8.0
    max_window_position_std_m: float = 0.004
    max_window_orientation_std_deg: float = 4.0
    target_hint_uv: tuple[float, float] | None = None
    max_target_hint_distance_px: float = 80.0
    max_age_s: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must lie in [0,1]")
        if (
            isinstance(self.min_stable_frames, bool)
            or not isinstance(self.min_stable_frames, int)
            or self.min_stable_frames < 2
        ):
            raise ValueError("min_stable_frames must be at least two")
        for name in (
            "max_translation_jump_m",
            "max_orientation_jump_deg",
            "max_window_position_std_m",
            "max_window_orientation_std_deg",
            "max_target_hint_distance_px",
            "max_age_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.target_hint_uv is not None:
            hint = np.asarray(self.target_hint_uv, dtype=float)
            if hint.shape != (2,) or not np.isfinite(hint).all():
                raise ValueError("target_hint_uv must contain two finite pixels")


@dataclass(frozen=True)
class QualifiedSocketEntry:
    T_camera_entry: np.ndarray
    position_std_m: float
    orientation_std_deg: float
    sample_count: int
    observed_at_s: float
    evidence_id: str


class OnlineSocketEntryTracker:
    def __init__(self, config: OnlineSocketTrackerConfig | None = None) -> None:
        self.config = config or OnlineSocketTrackerConfig()
        self._window: deque[tuple[float, SocketEntryCandidate]] = deque(
            maxlen=self.config.min_stable_frames
        )
        self.last_reasons: tuple[str, ...] = ("socket_not_observed",)

    def update(
        self, *, timestamp_s: float, candidates: Sequence[SocketEntryCandidate]
    ) -> QualifiedSocketEntry | None:
        if not math.isfinite(float(timestamp_s)):
            self._window.clear()
            self.last_reasons = ("socket_timestamp_invalid",)
            return None
        if self._window:
            elapsed = float(timestamp_s) - self._window[-1][0]
            if elapsed <= 0.0:
                self._window.clear()
                self.last_reasons = ("socket_timestamp_not_monotonic",)
                return None
            if elapsed > self.config.max_age_s:
                self._window.clear()
        eligible = [
            item
            for item in candidates
            if item.score >= self.config.min_score
            and item.position_std_m <= self.config.max_window_position_std_m
            and item.orientation_std_deg <= self.config.max_window_orientation_std_deg
        ]
        selected = self._select(eligible)
        if selected is None:
            self._window.clear()
            self.last_reasons = ("socket_candidate_missing_or_ambiguous",)
            return None
        if self._window:
            previous = self._window[-1][1].T_camera_entry
            translation = float(np.linalg.norm(selected.T_camera_entry[:3, 3] - previous[:3, 3]))
            angle = math.degrees(
                Rotation.from_matrix(
                    selected.T_camera_entry[:3, :3] @ previous[:3, :3].T
                ).magnitude()
            )
            if translation > self.config.max_translation_jump_m:
                self._window.clear()
                self.last_reasons = ("socket_translation_jump",)
                return None
            if angle > self.config.max_orientation_jump_deg:
                self._window.clear()
                self.last_reasons = ("socket_orientation_jump",)
                return None
        self._window.append((float(timestamp_s), selected))
        if len(self._window) < self.config.min_stable_frames:
            self.last_reasons = ("socket_temporal_window_not_full",)
            return None
        transforms = [item[1].T_camera_entry for item in self._window]
        translations = np.stack([item[:3, 3] for item in transforms])
        position = translations.mean(axis=0)
        position_std = float(np.sqrt(np.mean(np.sum(np.square(translations - position), axis=1))))
        rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in transforms]))
        mean_rotation = rotations.mean()
        residual = mean_rotation.inv() * rotations
        orientation_std = float(math.degrees(np.sqrt(np.mean(np.square(residual.magnitude())))))
        if position_std > self.config.max_window_position_std_m:
            self.last_reasons = ("socket_position_dispersion_too_high",)
            return None
        if orientation_std > self.config.max_window_orientation_std_deg:
            self.last_reasons = ("socket_orientation_dispersion_too_high",)
            return None
        transform = np.eye(4)
        transform[:3, :3] = mean_rotation.as_matrix()
        transform[:3, 3] = position
        evidence = "+".join(item[1].evidence_id for item in self._window)
        evidence_digest = hashlib.sha256(evidence.encode()).hexdigest()[:12]
        self.last_reasons = ()
        return QualifiedSocketEntry(
            T_camera_entry=validate_rigid_transform(
                transform, name="filtered T_camera_socket_entry"
            ),
            position_std_m=max(
                position_std,
                max(item[1].position_std_m for item in self._window),
            ),
            orientation_std_deg=max(
                orientation_std,
                max(item[1].orientation_std_deg for item in self._window),
            ),
            sample_count=len(self._window),
            observed_at_s=float(timestamp_s),
            evidence_id=f"online-socket:{evidence_digest}",
        )

    def _select(self, candidates: Sequence[SocketEntryCandidate]) -> SocketEntryCandidate | None:
        if not candidates:
            return None
        if self.config.target_hint_uv is not None:
            hint = np.asarray(self.config.target_hint_uv, dtype=float)
            ranked = sorted(
                candidates,
                key=lambda item: np.linalg.norm(np.asarray(item.centre_uv) - hint),
            )
            if (
                np.linalg.norm(np.asarray(ranked[0].centre_uv) - hint)
                > self.config.max_target_hint_distance_px
            ):
                return None
            if len(ranked) > 1:
                d0 = float(np.linalg.norm(np.asarray(ranked[0].centre_uv) - hint))
                d1 = float(np.linalg.norm(np.asarray(ranked[1].centre_uv) - hint))
                if d1 - d0 < 5.0:
                    return None
            return ranked[0]
        # A generic circle score carries no semantic target identity.  Without
        # a demo-derived image hint or a tight ROI, accepting the visually
        # cleanest of several rack holes could choose the wrong socket.
        if len(candidates) != 1:
            return None
        return candidates[0]


@dataclass(frozen=True)
class HoughSocketConfig:
    roi_uvxy: tuple[int, int, int, int]
    min_radius_px: int = 5
    max_radius_px: int = 40
    hough_dp: float = 1.2
    hough_min_distance_px: float = 20.0
    hough_param1: float = 80.0
    hough_param2: float = 18.0
    annulus_inner_fraction: float = 0.75
    annulus_outer_fraction: float = 1.35
    min_annulus_depth_points: int = 80
    max_plane_residual_m: float = 0.004

    def __post_init__(self) -> None:
        roi = tuple(self.roi_uvxy)
        if len(roi) != 4 or any(isinstance(value, bool) for value in roi):
            raise ValueError("roi_uvxy must contain four integer pixels")
        if not all(isinstance(value, int) for value in roi):
            raise ValueError("roi_uvxy must contain four integer pixels")
        if not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
            raise ValueError("roi_uvxy must be a valid image box")
        for name in ("min_radius_px", "max_radius_px", "min_annulus_depth_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_radius_px < self.min_radius_px:
            raise ValueError("max_radius_px must be at least min_radius_px")
        for name in (
            "hough_dp",
            "hough_min_distance_px",
            "hough_param1",
            "hough_param2",
            "annulus_inner_fraction",
            "annulus_outer_fraction",
            "max_plane_residual_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.annulus_outer_fraction <= self.annulus_inner_fraction:
            raise ValueError("annulus_outer_fraction must exceed annulus_inner_fraction")


def extract_hough_socket_candidates(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    config: HoughSocketConfig,
) -> tuple[SocketEntryCandidate, ...]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise OnlineSocketError("OpenCV is required for socket detection") from error
    image = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=float)
    intrinsics = np.asarray(K, dtype=float)
    u0, v0, u1, v1 = config.roi_uvxy
    if not (0 <= u0 < u1 <= image.shape[1] and 0 <= v0 < v1 <= image.shape[0]):
        raise ValueError("socket ROI is outside the image")
    gray = cv2.cvtColor(image[v0:v1, u0:u1], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=config.hough_dp,
        minDist=config.hough_min_distance_px,
        param1=config.hough_param1,
        param2=config.hough_param2,
        minRadius=config.min_radius_px,
        maxRadius=config.max_radius_px,
    )
    if circles is None:
        return ()
    fy, fx = float(intrinsics[1, 1]), float(intrinsics[0, 0])
    cy, cx = float(intrinsics[1, 2]), float(intrinsics[0, 2])
    rows_grid, cols_grid = np.indices(depth.shape)
    result: list[SocketEntryCandidate] = []
    for index, (local_u, local_v, radius) in enumerate(circles[0]):
        u, v = float(local_u + u0), float(local_v + v0)
        radial_px = np.hypot(cols_grid - u, rows_grid - v)
        annulus = (
            (radial_px >= config.annulus_inner_fraction * radius)
            & (radial_px <= config.annulus_outer_fraction * radius)
            & np.isfinite(depth)
            & (depth > 0.05)
        )
        rows, columns = np.nonzero(annulus)
        if len(rows) < config.min_annulus_depth_points:
            continue
        z = depth[rows, columns]
        points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z))
        centroid = points.mean(axis=0)
        _, singular, vh = np.linalg.svd(points - centroid, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        residual = float(np.sqrt(np.mean(np.square((points - centroid) @ normal))))
        if residual > config.max_plane_residual_m:
            continue
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        denominator = float(normal @ ray)
        if abs(denominator) < 1e-5:
            continue
        centre = ray * float(normal @ centroid) / denominator
        if centre[2] <= 0.0:
            continue
        # The outward normal is the sign facing free space/camera.
        if float(normal @ (-centre)) < 0.0:
            normal *= -1.0
        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis -= normal * float(x_axis @ normal)
        if np.linalg.norm(x_axis) < 1e-5:
            x_axis = np.array([0.0, 1.0, 0.0])
            x_axis -= normal * float(x_axis @ normal)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, normal)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((x_axis, y_axis, normal))
        transform[:3, 3] = centre
        planarity = float(singular[-1] / max(singular[-2], 1e-9))
        coverage = min(1.0, len(rows) / (2.0 * math.pi * radius * max(2.0, 0.3 * radius)))
        score = float(np.clip(0.5 * coverage + 0.5 * (1.0 - min(planarity, 1.0)), 0.0, 1.0))
        result.append(
            SocketEntryCandidate(
                T_camera_entry=transform,
                score=score,
                position_std_m=max(residual, 1e-4),
                orientation_std_deg=math.degrees(math.atan2(residual, max(centre[2], 1e-6))),
                centre_uv=(u, v),
                evidence_id=f"hough-depth-frame-circle-{index}",
            )
        )
    return tuple(result)


class OnlineSocketSceneInterpreter:
    """Wrap another scene interpreter and replace its static socket online."""

    def __init__(
        self,
        *,
        T_base_camera: np.ndarray,
        expected_camera_serial: str,
        tracker: OnlineSocketEntryTracker,
        candidate_extractor: Callable[
            [np.ndarray, np.ndarray, np.ndarray], Sequence[SocketEntryCandidate]
        ],
        downstream: object,
    ) -> None:
        self._T_base_camera = validate_rigid_transform(T_base_camera, name="T_base_camera").copy()
        self._serial = expected_camera_serial
        self._tracker = tracker
        self._extract = candidate_extractor
        self._downstream = downstream
        self._latest = None
        self._latest_sequence: int | None = None

    @property
    def last_hold_reasons(self) -> tuple[str, ...]:
        return self._tracker.last_reasons

    def observe(self, *, frame: object, state: object, T_base_tcp: np.ndarray) -> object:
        self._observe_socket_frame(frame)
        return self._downstream.observe(frame=frame, state=state, T_base_tcp=T_base_tcp)

    def observe_burst(
        self,
        *,
        first_frame: object,
        read_frame: Callable[[], object],
        state: object,
        T_base_tcp: np.ndarray,
    ) -> tuple[object, object]:
        """Qualify the static socket from one camera-rate RGB-D burst.

        The outer task loop also performs remote source-tube inference, tactile
        collection, and robot-state reads, so consecutive calls to ``observe``
        can be seconds apart.  Feeding those sparse frames to a tracker whose
        age limit is intentionally camera-rate would clear its temporal window
        on every call.  During the initial OBSERVE state this method consumes
        exactly the tracker's required number of fresh RGB-D pairs, but runs
        the expensive downstream scene interpreter only on the final pair.
        """

        frame = first_frame
        for index in range(self._tracker.config.min_stable_frames):
            if index:
                frame = read_frame()
            self._observe_socket_frame(frame)
        evidence = self._downstream.observe(
            frame=frame,
            state=state,
            T_base_tcp=T_base_tcp,
        )
        return frame, evidence

    def _observe_socket_frame(self, frame: object) -> None:
        if str(getattr(frame, "serial")) != self._serial:
            raise OnlineSocketError("camera serial changed during socket tracking")
        timestamp = float(getattr(frame, "host_received_monotonic_s"))
        candidates = self._extract(
            np.asarray(getattr(frame, "rgb")),
            np.asarray(getattr(frame, "depth_m")),
            np.asarray(getattr(getattr(frame, "intrinsics"), "matrix")),
        )
        qualified = self._tracker.update(timestamp_s=timestamp, candidates=candidates)
        self._latest = qualified
        self._latest_sequence = int(getattr(frame, "sequence"))
        if qualified is not None:
            setter = getattr(self._downstream, "set_socket_pose", None)
            if callable(setter):
                setter(self._T_base_camera @ qualified.T_camera_entry)

    def socket_pose(self, *, frame: object, **_: object) -> object | None:
        if self._latest is None or int(getattr(frame, "sequence")) != self._latest_sequence:
            return None
        from .test_tube_live_io import QualifiedBasePose

        value = self._latest
        return QualifiedBasePose(
            T_base_object=self._T_base_camera @ value.T_camera_entry,
            position_std_m=value.position_std_m,
            orientation_std_deg=value.orientation_std_deg,
            tracking_id="online-rack-socket-entry",
            evidence_id=value.evidence_id,
            axis_sign_source="rim_detector",
            cylindrical_roll_canonicalized=False,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._downstream, name)


def create_online_socket_scene_interpreter(
    *, config: Mapping[str, object], output_dir: object
) -> OnlineSocketSceneInterpreter:
    if not isinstance(config, Mapping):
        raise ValueError("online socket config must be an object")
    raw_camera_transform = config.get("T_base_camera")
    if raw_camera_transform is None:
        camera_path = Path(str(config.get("camera_extrinsic", ""))).expanduser().resolve()
        if not camera_path.is_file():
            raise ValueError(
                "T_base_camera or camera_extrinsic is required for online socket localization"
            )
        T_base_camera = np.load(camera_path, allow_pickle=False)
    else:
        T_base_camera = np.asarray(raw_camera_transform, dtype=float)
    serial = str(config.get("camera_serial", ""))
    if not serial:
        raise ValueError("camera_serial is required")
    downstream_spec = config.get("downstream_factory")
    if not isinstance(downstream_spec, str) or ":" not in downstream_spec:
        raise ValueError("downstream_factory must use module:function syntax")
    module, function = downstream_spec.split(":", 1)
    factory = getattr(importlib.import_module(module), function, None)
    if not callable(factory):
        raise ValueError("downstream_factory is not callable")
    downstream = factory(config=config.get("downstream_config", {}), output_dir=output_dir)
    raw_tracker = config.get("tracker", {})
    raw_hough = config.get("hough", {})
    if not isinstance(raw_tracker, Mapping) or not isinstance(raw_hough, Mapping):
        raise ValueError("tracker and hough must be objects")
    tracker_options = dict(raw_tracker)
    if tracker_options.get("target_hint_uv") is not None:
        tracker_options["target_hint_uv"] = tuple(tracker_options["target_hint_uv"])
    hough_options = dict(raw_hough)
    hough_options["roi_uvxy"] = tuple(hough_options["roi_uvxy"])
    hough = HoughSocketConfig(**hough_options)

    def extractor(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray):
        return extract_hough_socket_candidates(rgb, depth, K, hough)

    return OnlineSocketSceneInterpreter(
        T_base_camera=T_base_camera,
        expected_camera_serial=serial,
        tracker=OnlineSocketEntryTracker(OnlineSocketTrackerConfig(**tracker_options)),
        candidate_extractor=extractor,
        downstream=downstream,
    )


__all__ = [
    "HoughSocketConfig",
    "OnlineSocketEntryTracker",
    "OnlineSocketError",
    "OnlineSocketSceneInterpreter",
    "OnlineSocketTrackerConfig",
    "QualifiedSocketEntry",
    "SocketEntryCandidate",
    "create_online_socket_scene_interpreter",
    "extract_hough_socket_candidates",
]
