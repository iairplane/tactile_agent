"""Fail-closed geometry contracts for the transparent test-tube task.

The D435 does not return the test-tube surface reliably.  This module therefore
keeps image evidence, projective constraints, and metric measurements separate.
In particular, it never turns an adjacent rack/background depth return into an
exact ``T_base_tube`` transform.

There is no robot, camera, ROS, or gripper I/O in this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

GEOMETRY_CONFIG_SCHEMA = "demo2gripper.test_tube_geometry_config.v1"
GRASP_RELATION_SCHEMA = "demo2gripper.test_tube_grasp_relation.v1"
LIVE_POSE_SCHEMA = "demo2gripper.test_tube_live_pose_observation.v1"
SOCKET_FRAME_SCHEMA = "demo2gripper.test_tube_socket_frame_observation.v1"


def _uv(value: object, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain two finite image coordinates")
    return point


def _positive(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def image_axis_angle_deg(start_uv: object, end_uv: object) -> float:
    """Return the undirected line angle in image coordinates in ``[0, 180)``."""

    start = _uv(start_uv, name="start_uv")
    end = _uv(end_uv, name="end_uv")
    delta = end - start
    if np.linalg.norm(delta) <= 1e-12:
        raise ValueError("axis endpoints must be distinct")
    return float(math.degrees(math.atan2(delta[1], delta[0])) % 180.0)


def axial_fraction(start_uv: object, end_uv: object, point_uv: object) -> tuple[float, float]:
    """Project an image point onto an oriented axis.

    Returns ``(fraction, signed_orthogonal_distance_px)``.  A fraction of zero
    is at ``start_uv`` and one is at ``end_uv``.  The fraction is not clipped;
    callers can therefore detect a point outside the reviewed segment.
    """

    start = _uv(start_uv, name="start_uv")
    end = _uv(end_uv, name="end_uv")
    point = _uv(point_uv, name="point_uv")
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        raise ValueError("axis endpoints must be distinct")
    offset = point - start
    fraction = float(offset @ delta / denominator)
    orthogonal = float((delta[0] * offset[1] - delta[1] * offset[0]) / math.sqrt(denominator))
    return fraction, orthogonal


def camera_ray(K: np.ndarray, uv: object) -> np.ndarray:
    """Return a unit OpenCV-camera ray through an image coordinate."""

    intrinsics = np.asarray(K, dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("K must be a finite 3x3 matrix")
    point = _uv(uv, name="uv")
    ray = np.linalg.solve(intrinsics, np.array([point[0], point[1], 1.0]))
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-12:
        raise ValueError("image coordinate produced a degenerate camera ray")
    return ray / norm


def projective_axis_plane(K: np.ndarray, start_uv: object, end_uv: object) -> dict[str, Any]:
    """Describe the camera-origin plane induced by a two-dimensional line.

    Every possible metric 3-D line that projects onto the reviewed image axis
    lies in this plane.  The constraint is useful while making the missing
    depth/elevation degree of freedom explicit.
    """

    start_ray = camera_ray(K, start_uv)
    end_ray = camera_ray(K, end_uv)
    normal = np.cross(start_ray, end_ray)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("axis endpoints produce a degenerate projective plane")
    normal /= norm
    return {
        "equation": "normal_camera dot p_camera = 0",
        "normal_camera": normal.tolist(),
        "start_ray_camera_unit": start_ray.tolist(),
        "end_ray_camera_unit": end_ray.tolist(),
        "remaining_degrees_of_freedom": [
            "metric position along either camera ray",
            "metric axis elevation within the projective plane",
            "cylindrical roll about the tube axis",
        ],
    }


def backproject(K: np.ndarray, uv: object, depth_m: float) -> np.ndarray:
    """Backproject an aligned-depth sample into OpenCV camera coordinates."""

    depth = _positive(depth_m, name="depth_m")
    point = _uv(uv, name="uv")
    ray_z_one = np.linalg.solve(
        np.asarray(K, dtype=float), np.array([point[0], point[1], 1.0])
    )
    return depth * ray_z_one


def fixed_length_endpoint_solutions(
    K: np.ndarray,
    *,
    known_uv: object,
    known_depth_m: float,
    unknown_uv: object,
    length_m: float,
) -> list[dict[str, Any]]:
    """Solve the two-ray fixed-length ambiguity in optical-camera coordinates.

    ``known_depth_m`` and each returned depth use aligned-D435 optical ``Z``,
    rather than Euclidean range.  A single known endpoint and object length can
    yield two positive roots.  Both are retained for review.
    """

    intrinsics = np.asarray(K, dtype=float)
    known_depth = _positive(known_depth_m, name="known_depth_m")
    length = _positive(length_m, name="length_m")
    known = _uv(known_uv, name="known_uv")
    unknown = _uv(unknown_uv, name="unknown_uv")
    known_ray = np.linalg.solve(intrinsics, np.array([known[0], known[1], 1.0]))
    unknown_ray = np.linalg.solve(intrinsics, np.array([unknown[0], unknown[1], 1.0]))
    known_point = known_depth * known_ray
    coefficients = [
        float(unknown_ray @ unknown_ray),
        float(-2.0 * unknown_ray @ known_point),
        float(known_point @ known_point - length**2),
    ]
    discriminant = coefficients[1] ** 2 - 4.0 * coefficients[0] * coefficients[2]
    if discriminant < -1e-12:
        return []
    discriminant = max(discriminant, 0.0)
    denominator = 2.0 * coefficients[0]
    roots = [
        (-coefficients[1] - math.sqrt(discriminant)) / denominator,
        (-coefficients[1] + math.sqrt(discriminant)) / denominator,
    ]
    solutions: list[dict[str, Any]] = []
    for root in sorted(set(roots)):
        if root <= 0:
            continue
        unknown_point = root * unknown_ray
        solutions.append(
            {
                "unknown_depth_z_m": root,
                "known_xyz_camera_m": known_point.tolist(),
                "unknown_xyz_camera_m": unknown_point.tolist(),
                "endpoint_distance_m": float(np.linalg.norm(unknown_point - known_point)),
            }
        )
    return solutions


def transform_point(T_parent_child: np.ndarray, point_child: object) -> np.ndarray:
    transform = np.asarray(T_parent_child, dtype=float)
    point = np.asarray(point_child, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_parent_child must be a finite 4x4 matrix")
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_child must contain three finite coordinates")
    homogeneous = transform @ np.append(point, 1.0)
    return homogeneous[:3] / homogeneous[3]


def transform_direction(T_parent_child: np.ndarray, direction_child: object) -> np.ndarray:
    transform = np.asarray(T_parent_child, dtype=float)
    direction = np.asarray(direction_child, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_parent_child must be a finite 4x4 matrix")
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("direction_child must contain three finite coordinates")
    result = transform[:3, :3] @ direction
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("direction transform is degenerate")
    return result / norm


def depth_proxy_summary(
    depth_frames: Sequence[np.ndarray],
    *,
    uv: object,
    radius_px: int,
    accepted_range_m: tuple[float, float],
) -> dict[str, Any]:
    """Summarize valid depth near an image seed without assigning object identity."""

    point = _uv(uv, name="uv")
    if isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("radius_px must be a non-negative integer")
    lower, upper = map(float, accepted_range_m)
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 < lower < upper:
        raise ValueError("accepted_range_m must be a positive increasing pair")

    x, y = (int(round(point[0])), int(round(point[1])))
    per_frame: list[dict[str, Any]] = []
    all_values: list[np.ndarray] = []
    for index, source in enumerate(depth_frames):
        depth = np.asarray(source, dtype=float)
        if depth.ndim != 2:
            raise ValueError("each depth frame must be two-dimensional")
        y0, y1 = max(0, y - radius_px), min(depth.shape[0], y + radius_px + 1)
        x0, x1 = max(0, x - radius_px), min(depth.shape[1], x + radius_px + 1)
        values = depth[y0:y1, x0:x1]
        valid = values[np.isfinite(values) & (values >= lower) & (values <= upper)]
        if valid.size:
            all_values.append(valid)
        per_frame.append(
            {
                "frame_index": index,
                "window_pixels": int(values.size),
                "accepted_pixels": int(valid.size),
                "median_m": float(np.median(valid)) if valid.size else None,
                "p10_m": float(np.percentile(valid, 10)) if valid.size else None,
                "p90_m": float(np.percentile(valid, 90)) if valid.size else None,
            }
        )
    aggregate = np.concatenate(all_values) if all_values else np.empty(0)
    return {
        "uv_px": point.tolist(),
        "window_radius_px": radius_px,
        "accepted_range_m": [lower, upper],
        "per_frame": per_frame,
        "aggregate": {
            "accepted_pixels": int(aggregate.size),
            "median_m": float(np.median(aggregate)) if aggregate.size else None,
            "p10_m": float(np.percentile(aggregate, 10)) if aggregate.size else None,
            "p90_m": float(np.percentile(aggregate, 90)) if aggregate.size else None,
        },
    }


def compile_geometry_contracts(
    *,
    config: dict[str, Any],
    K: np.ndarray,
    T_base_camera: np.ndarray,
    live_depth_frames: Sequence[np.ndarray],
    socket_depth_frames: Sequence[np.ndarray],
    source_hand_depth_frames: Sequence[np.ndarray],
    socket_hand_depth_frames: Sequence[np.ndarray],
    provenance: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compile the three review artifacts without creating an execution pose."""

    if config.get("schema") != GEOMETRY_CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {GEOMETRY_CONFIG_SCHEMA!r}")
    review = config.get("review")
    assumptions = config.get("assumptions")
    if not isinstance(review, dict) or not isinstance(assumptions, dict):
        raise ValueError("config review and assumptions must be objects")

    source = review.get("source_grasp")
    live_source = review.get("live_source")
    socket_reference = review.get("socket_reference")
    if (
        not isinstance(source, dict)
        or not isinstance(live_source, dict)
        or not isinstance(socket_reference, dict)
    ):
        raise ValueError(
            "review.source_grasp, review.live_source, and review.socket_reference "
            "must be objects"
        )

    source_start = _uv(source.get("closed_end_uv"), name="source_grasp.closed_end_uv")
    source_end = _uv(source.get("open_end_uv"), name="source_grasp.open_end_uv")
    pinch = _uv(source.get("pinch_centre_uv"), name="source_grasp.pinch_centre_uv")
    fraction, orthogonal_px = axial_fraction(source_start, source_end, pinch)
    pinch_uncertainty_px = _positive(
        source.get("pinch_centre_uncertainty_px"),
        name="source_grasp.pinch_centre_uncertainty_px",
    )
    source_span = float(np.linalg.norm(source_end - source_start))
    fraction_uncertainty = min(0.5, pinch_uncertainty_px / source_span)

    length = _positive(assumptions.get("tube_length_nominal_m"), name="tube length")
    length_uncertainty = _positive(
        assumptions.get("tube_length_uncertainty_m"), name="tube length uncertainty"
    )
    diameter = _positive(assumptions.get("tube_diameter_nominal_m"), name="tube diameter")
    diameter_uncertainty = _positive(
        assumptions.get("tube_diameter_uncertainty_m"), name="tube diameter uncertainty"
    )
    axial_from_open = (1.0 - fraction) * length
    axial_from_open_uncertainty = math.sqrt(
        (length * fraction_uncertainty) ** 2 + ((1.0 - fraction) * length_uncertainty) ** 2
    )

    live_start = _uv(live_source.get("closed_end_uv"), name="live_source.closed_end_uv")
    live_end = _uv(live_source.get("open_end_uv"), name="live_source.open_end_uv")
    live_plane = projective_axis_plane(K, live_start, live_end)
    live_normal_base = transform_direction(T_base_camera, live_plane["normal_camera"])
    live_depth_cfg = live_source.get("depth_proxy")
    if not isinstance(live_depth_cfg, dict):
        raise ValueError("live_source.depth_proxy must be an object")
    live_depth_summary = depth_proxy_summary(
        live_depth_frames,
        uv=live_start,
        radius_px=int(live_depth_cfg.get("window_radius_px")),
        accepted_range_m=tuple(live_depth_cfg.get("accepted_range_m")),
    )
    live_depth_m = live_depth_summary["aggregate"]["median_m"]
    live_point_camera = (
        backproject(K, live_start, live_depth_m) if live_depth_m is not None else None
    )
    live_point_base = (
        transform_point(T_base_camera, live_point_camera)
        if live_point_camera is not None
        else None
    )
    source_hand_cfg = live_source.get("open_hand_depth_proxy")
    if not isinstance(source_hand_cfg, dict):
        raise ValueError("live_source.open_hand_depth_proxy must be an object")
    source_hand_summary = depth_proxy_summary(
        source_hand_depth_frames,
        uv=source_hand_cfg.get("uv"),
        radius_px=int(source_hand_cfg.get("window_radius_px")),
        accepted_range_m=tuple(source_hand_cfg.get("accepted_range_m")),
    )
    source_solutions = (
        fixed_length_endpoint_solutions(
            K,
            known_uv=live_start,
            known_depth_m=live_depth_m,
            unknown_uv=live_end,
            length_m=length,
        )
        if live_depth_m is not None
        else []
    )
    source_hand_depth_m = source_hand_summary["aggregate"]["median_m"]
    source_candidates: list[dict[str, Any]] = []
    for index, solution in enumerate(source_solutions):
        closed_camera = np.asarray(solution["known_xyz_camera_m"])
        open_camera = np.asarray(solution["unknown_xyz_camera_m"])
        closed_base = transform_point(T_base_camera, closed_camera)
        open_base = transform_point(T_base_camera, open_camera)
        axis_base = open_base - closed_base
        axis_base /= np.linalg.norm(axis_base)
        source_candidates.append(
            {
                "id": f"source_branch_{index}",
                **solution,
                "closed_xyz_base_m": closed_base.tolist(),
                "open_xyz_base_m": open_base.tolist(),
                "axis_closed_to_open_base_unit": axis_base.tolist(),
                "open_depth_difference_to_demo_hand_proxy_m": (
                    abs(solution["unknown_depth_z_m"] - source_hand_depth_m)
                    if source_hand_depth_m is not None
                    else None
                ),
            }
        )
    preferred_source = (
        min(
            source_candidates,
            key=lambda candidate: candidate["open_depth_difference_to_demo_hand_proxy_m"],
        )["id"]
        if source_candidates and source_hand_depth_m is not None
        else None
    )

    socket_start = _uv(
        socket_reference.get("closed_end_uv"), name="socket_reference.closed_end_uv"
    )
    socket_end = _uv(
        socket_reference.get("open_end_uv"), name="socket_reference.open_end_uv"
    )
    entry = _uv(
        socket_reference.get("socket_entry_uv"), name="socket_reference.socket_entry_uv"
    )
    socket_plane = projective_axis_plane(K, socket_start, socket_end)
    socket_normal_base = transform_direction(T_base_camera, socket_plane["normal_camera"])
    socket_depth_cfg = socket_reference.get("depth_proxy")
    if not isinstance(socket_depth_cfg, dict):
        raise ValueError("socket_reference.depth_proxy must be an object")
    socket_depth_summary = depth_proxy_summary(
        socket_depth_frames,
        uv=entry,
        radius_px=int(socket_depth_cfg.get("window_radius_px")),
        accepted_range_m=tuple(socket_depth_cfg.get("accepted_range_m")),
    )
    socket_depth_m = socket_depth_summary["aggregate"]["median_m"]
    socket_point_camera = (
        backproject(K, entry, socket_depth_m) if socket_depth_m is not None else None
    )
    socket_point_base = (
        transform_point(T_base_camera, socket_point_camera)
        if socket_point_camera is not None
        else None
    )
    socket_hand_cfg = socket_reference.get("open_hand_depth_proxy")
    if not isinstance(socket_hand_cfg, dict):
        raise ValueError("socket_reference.open_hand_depth_proxy must be an object")
    socket_hand_summary = depth_proxy_summary(
        socket_hand_depth_frames,
        uv=socket_hand_cfg.get("uv"),
        radius_px=int(socket_hand_cfg.get("window_radius_px")),
        accepted_range_m=tuple(socket_hand_cfg.get("accepted_range_m")),
    )
    socket_open_depth_m = socket_hand_summary["aggregate"]["median_m"]
    socket_solutions = (
        fixed_length_endpoint_solutions(
            K,
            known_uv=socket_end,
            known_depth_m=socket_open_depth_m,
            unknown_uv=socket_start,
            length_m=length,
        )
        if socket_open_depth_m is not None
        else []
    )
    socket_candidates: list[dict[str, Any]] = []
    for index, solution in enumerate(socket_solutions):
        open_camera = np.asarray(solution["known_xyz_camera_m"])
        closed_camera = np.asarray(solution["unknown_xyz_camera_m"])
        open_base = transform_point(T_base_camera, open_camera)
        closed_base = transform_point(T_base_camera, closed_camera)
        axis_base = open_base - closed_base
        axis_base /= np.linalg.norm(axis_base)
        socket_candidates.append(
            {
                "id": f"socket_branch_{index}",
                **solution,
                "closed_xyz_base_m": closed_base.tolist(),
                "open_xyz_base_m": open_base.tolist(),
                "axis_closed_to_open_base_unit": axis_base.tolist(),
            }
        )
    preferred_socket = None
    if socket_candidates:
        # The physically useful branch has the closed/inserted tip below the
        # open end and nearest the rack support height.  This is a ranking
        # heuristic, not an execution qualification.
        eligible = [
            candidate
            for candidate in socket_candidates
            if candidate["closed_xyz_base_m"][2] < candidate["open_xyz_base_m"][2]
        ]
        ranked = eligible or socket_candidates
        preferred_socket = min(
            ranked,
            key=lambda candidate: abs(candidate["closed_xyz_base_m"][2] - 0.10),
        )["id"]
    camera_origin_base = np.asarray(T_base_camera, dtype=float)[:3, 3]

    common = {
        "task_id": config.get("task_id"),
        "status": "reviewed_plan_only_evidence",
        "execution_authorized": False,
        "hardware_commands_dispatched": False,
        "transform_convention": "T_parent_child; metres; right-handed",
        "provenance": provenance,
    }
    grasp = {
        "schema": GRASP_RELATION_SCHEMA,
        **common,
        "relation_type": "cylindrical_side_pinch_candidate_family",
        "demo_measurement": {
            "frame": int(source.get("frame")),
            "closed_end_uv_px": source_start.tolist(),
            "open_end_uv_px": source_end.tolist(),
            "pinch_centre_uv_px": pinch.tolist(),
            "tube_axis_angle_deg_image_y_down": image_axis_angle_deg(
                source_start, source_end
            ),
            "pinch_axial_fraction_closed_to_open": fraction,
            "pinch_axial_fraction_uncertainty": fraction_uncertainty,
            "pinch_orthogonal_offset_px": orthogonal_px,
        },
        "dimension_assumptions": {
            "tube_length_m": length,
            "tube_length_uncertainty_m": length_uncertainty,
            "tube_diameter_m": diameter,
            "tube_diameter_uncertainty_m": diameter_uncertainty,
            "source": "nominal visual estimate; not physically measured",
            "used_to_infer_metric_pose": False,
        },
        "nominal_axial_grasp_band": {
            "fraction_closed_to_open": [
                max(0.0, fraction - fraction_uncertainty),
                min(1.0, fraction + fraction_uncertainty),
            ],
            "centre_offset_below_open_end_m": axial_from_open,
            "centre_offset_uncertainty_m": axial_from_open_uncertainty,
            "metric_values_are_dimension_scaled_estimates": True,
        },
        "robot_candidate_constraints": {
            "tcp_origin": "on the tube centreline inside the axial grasp band",
            "closing_axis": "hande_tcp local +X perpendicular to the tube axis",
            "approach_axis": "hande_tcp local +Z perpendicular to the tube axis",
            "cylindrical_roll": "free S1 candidate family; rank by IK and collision clearance",
            "jaw_opening": "tube diameter plus pregrasp clearance, then tactile contact close",
        },
        "T_object_gripper_tcp": None,
        "why_transform_is_null": (
            "The monocular demonstration constrains axial location and side-pinch topology, "
            "but cannot observe cylindrical roll or a metric tube pose."
        ),
        "execution_start_requirement": (
            "Instantiate and collision-filter the candidate family only after a fresh metric "
            "source-tube observation."
        ),
    }

    live_pose = {
        "schema": LIVE_POSE_SCHEMA,
        **common,
        "observation_role": (
            "pre-execution source-tube reference matching the demonstration initial scene"
        ),
        "capture": {
            "frame_indices": list(range(len(live_depth_frames))),
            "review_frame": int(live_source.get("live_review_frame")),
        },
        "target_identity": {
            "basis": "continuity with the demonstrated initial/source steep-axis tube",
            "target_axis_uv_px": [live_start.tolist(), live_end.tolist()],
            "distractor_axis_uv_px": live_source.get("distractor_axis_uv"),
            "multi_tube_ambiguity_present": True,
        },
        "image_axis": {
            "closed_end_uv_px": live_start.tolist(),
            "open_end_uv_px": live_end.tolist(),
            "angle_deg_image_y_down": image_axis_angle_deg(live_start, live_end),
            "acute_angle_from_image_horizontal_deg": min(
                image_axis_angle_deg(live_start, live_end),
                180.0 - image_axis_angle_deg(live_start, live_end),
            ),
            "endpoint_uncertainty_px": float(
                live_source.get("axis_endpoint_uncertainty_px")
            ),
        },
        "projective_axis_plane": {
            **live_plane,
            "base_equation": "normal_base dot (p_base - camera_origin_base) = 0",
            "normal_base": live_normal_base.tolist(),
            "camera_origin_base_m": camera_origin_base.tolist(),
        },
        "adjacent_surface_depth_proxy": {
            **live_depth_summary,
            "xyz_camera_m": (
                live_point_camera.tolist() if live_point_camera is not None else None
            ),
            "xyz_base_m": live_point_base.tolist() if live_point_base is not None else None,
            "identity": "nearby opaque cup/support return; not verified tube-centre depth",
            "may_be_used_as_T_base_tube_translation": False,
        },
        "metric_candidate_hypotheses": {
            "model": "known closed-end Z proxy + nominal tube length + open-end image ray",
            "tube_length_m": length,
            "tube_length_uncertainty_m": length_uncertainty,
            "open_endpoint_demo_hand_depth_proxy": source_hand_summary,
            "candidates": source_candidates,
            "preferred_candidate_id": preferred_source,
            "preference_basis": "closest open-end depth to the f246 hand/contact proxy",
            "execution_eligible": False,
            "why_not_execution_eligible": (
                "The closed-end depth is an adjacent cup/support return and length is nominal."
            ),
        },
        "T_base_tube": None,
        "metric_pose_available": False,
        "execution_start_requirement": {
            "fresh_observation_required": True,
            "required_outputs": [
                "target-identity mask or marker separated from other tubes",
                "metric centreline or opaque-tip anchor in the current source location",
                "T_base_tube axis and uncertainty",
            ],
            "fail_closed_if_missing": True,
        },
    }

    socket = {
        "schema": SOCKET_FRAME_SCHEMA,
        **common,
        "reference_role": "demonstration-terminal settled rack-entry image template",
        "frame_convention": {
            "+Z": "outward from socket entry toward the tube open end",
            "insertion_direction": "-Z",
            "roll": "unobservable and irrelevant for a circular tube/socket pair",
        },
        "image_template": {
            "entry_uv_px": entry.tolist(),
            "entry_uncertainty_px": float(
                socket_reference.get("socket_entry_uncertainty_px")
            ),
            "outward_axis_uv_px": [socket_start.tolist(), socket_end.tolist()],
            "angle_deg_image_y_down": image_axis_angle_deg(socket_start, socket_end),
            "acute_angle_from_image_horizontal_deg": min(
                image_axis_angle_deg(socket_start, socket_end),
                180.0 - image_axis_angle_deg(socket_start, socket_end),
            ),
            "same_fixed_camera_alignment_eligible": True,
        },
        "projective_axis_plane": {
            **socket_plane,
            "base_equation": "normal_base dot (p_base - camera_origin_base) = 0",
            "normal_base": socket_normal_base.tolist(),
            "camera_origin_base_m": camera_origin_base.tolist(),
        },
        "adjacent_surface_depth_proxy": {
            **socket_depth_summary,
            "xyz_camera_m": (
                socket_point_camera.tolist() if socket_point_camera is not None else None
            ),
            "xyz_base_m": (
                socket_point_base.tolist() if socket_point_base is not None else None
            ),
            "identity": "sparse rack/background return at the terminal 2D entry seed",
            "may_be_used_as_socket_origin": False,
        },
        "metric_candidate_hypotheses": {
            "model": (
                "f397 open-end hand/contact Z proxy + nominal tube length + "
                "closed-end image ray"
            ),
            "tube_length_m": length,
            "tube_length_uncertainty_m": length_uncertainty,
            "open_endpoint_hand_depth_proxy": socket_hand_summary,
            "candidates": socket_candidates,
            "preferred_candidate_id": preferred_socket,
            "preference_basis": (
                "closed tip below open end and nearest the observed rack support height"
            ),
            "execution_eligible": False,
            "why_not_execution_eligible": (
                "The open-end depth is hand/contact depth, tube length is nominal, and the "
                "two-ray construction remains branch-ambiguous."
            ),
        },
        "T_base_socket": None,
        "insertion_target_depth_m": None,
        "metric_socket_pose_available": False,
        "plan_only_contract": {
            "alignment": "match the live tube axis and closed tip to the reviewed image template",
            "insertion": "microstep state machine remains reviewable but has no metric endpoint",
        },
        "execution_start_requirement": {
            "required_measurement": (
                "a metric rack-entry origin, outward axis, and insertion depth from a fiducial, "
                "opaque rack geometry, touch-off, or direct physical measurement"
            ),
            "fail_closed_if_missing": True,
        },
    }
    return {
        "tube_grasp_relation": grasp,
        "live_tube_pose": live_pose,
        "socket_frame": socket,
    }


__all__ = [
    "GEOMETRY_CONFIG_SCHEMA",
    "GRASP_RELATION_SCHEMA",
    "LIVE_POSE_SCHEMA",
    "SOCKET_FRAME_SCHEMA",
    "axial_fraction",
    "backproject",
    "camera_ray",
    "compile_geometry_contracts",
    "depth_proxy_summary",
    "fixed_length_endpoint_solutions",
    "image_axis_angle_deg",
    "projective_axis_plane",
    "transform_direction",
    "transform_point",
]
