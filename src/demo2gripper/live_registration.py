"""Fail-closed registration of a no-CAD demonstration template to live points.

The template is a partial single-view cloud.  Registration therefore uses multiple
frames, keeps held-out frames out of fitting, and treats the rotational symmetry of a
cube as an explicit pose gauge instead of pretending that depth identifies die faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree

from .franka import validate_rigid_transform


@dataclass(frozen=True)
class RegistrationConfig:
    """Quality limits for partial-cloud rigid registration."""

    symmetry: str = "cube"
    min_frames: int = 3
    min_points_per_frame: int = 100
    voxel_size_m: float = 0.001
    max_iterations: int = 50
    max_correspondence_distance_m: float = 0.012
    inlier_distance_m: float = 0.005
    template_trim_fraction: float = 0.85
    live_trim_fraction: float = 0.55
    min_template_inlier_fraction: float = 0.55
    min_live_inlier_fraction: float = 0.25
    max_fit_rmse_m: float = 0.0045
    max_held_out_rmse_m: float = 0.0055
    max_frame_translation_deviation_m: float = 0.006
    max_frame_rotation_deviation_deg: float = 12.0
    ambiguity_score_margin_m: float = 0.00035
    ambiguity_translation_tolerance_m: float = 0.004
    ambiguity_rotation_tolerance_deg: float = 10.0


@dataclass(frozen=True)
class AlignmentMetrics:
    rmse_m: float
    template_rmse_m: float
    live_rmse_m: float
    template_inlier_fraction: float
    live_inlier_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse_m": self.rmse_m,
            "template_rmse_m": self.template_rmse_m,
            "live_rmse_m": self.live_rmse_m,
            "template_inlier_fraction": self.template_inlier_fraction,
            "live_inlier_fraction": self.live_inlier_fraction,
        }


@dataclass(frozen=True)
class RegistrationResult:
    T_base_object: np.ndarray
    fit_frame_indices: tuple[int, ...]
    held_out_frame_indices: tuple[int, ...]
    fit_metrics: AlignmentMetrics
    frame_metrics: tuple[AlignmentMetrics, ...]
    frame_translation_deviation_m: np.ndarray
    frame_rotation_deviation_deg: np.ndarray
    near_optimal_hypothesis_count: int
    symmetry_equivalent_hypothesis_count: int
    initial_hypothesis_count: int
    cube_axes_object: np.ndarray | None
    orientation_prior_used: bool
    orientation_locked_to_prior: bool
    unlocked_fit_metrics: AlignmentMetrics | None
    orientation_lock_error_deg: float | None
    orientation_prior_error_deg: float | None
    orientation_prior_second_best_error_deg: float | None
    orientation_prior_max_error_deg: float | None
    orientation_prior_uniqueness_margin_deg: float | None


@dataclass(frozen=True)
class _Hypothesis:
    transform: np.ndarray
    metrics: AlignmentMetrics


def cube_rotation_group() -> tuple[np.ndarray, ...]:
    """Return the 24 proper signed-permutation rotations of a cube."""

    rotations: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = permutation_matrix @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    rotations.sort(key=lambda matrix: tuple(matrix.ravel()))
    return tuple(rotations)


_CUBE_ROTATIONS = cube_rotation_group()


def estimate_cube_axes_object(
    points_object: np.ndarray,
    *,
    plane_distance_m: float = 0.001,
    trials: int = 2500,
) -> np.ndarray:
    """Estimate an unoriented cube-axis basis from two visible orthogonal faces.

    The template frame follows the demonstration camera, so the standard XYZ cube
    group cannot be applied directly.  Two face normals determine the conjugating
    basis; signs and column order do not affect the resulting cube symmetry group.
    """

    points = _validate_points(points_object, name="cube template", minimum=30)
    if not np.isfinite(plane_distance_m) or plane_distance_m <= 0.0:
        raise ValueError("plane_distance_m must be positive and finite")
    if trials < 100:
        raise ValueError("cube-axis estimation needs at least 100 RANSAC trials")
    rng = np.random.default_rng(0)
    hypotheses: list[tuple[int, np.ndarray]] = []
    minimum_support = max(15, int(np.ceil(0.06 * len(points))))
    for _ in range(trials):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= plane_distance_m
        if int(np.count_nonzero(inliers)) < minimum_support:
            continue
        centered = points[inliers] - np.mean(points[inliers], axis=0)
        _, _, right_t = np.linalg.svd(centered, full_matrices=False)
        refined = right_t[-1]
        refined /= np.linalg.norm(refined)
        plane_center = np.median(points[inliers], axis=0)
        distances = np.abs((points - plane_center) @ refined)
        support = int(np.count_nonzero(distances <= plane_distance_m))
        if support < minimum_support:
            continue
        for index, (old_support, old_normal) in enumerate(hypotheses):
            if abs(float(old_normal @ refined)) >= np.cos(np.radians(8.0)):
                if support > old_support:
                    hypotheses[index] = (support, refined)
                break
        else:
            hypotheses.append((support, refined))
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    pair: tuple[np.ndarray, np.ndarray] | None = None
    pair_score = -1
    for first_index, (first_support, first) in enumerate(hypotheses):
        for second_support, second in hypotheses[first_index + 1 :]:
            if abs(float(first @ second)) > 0.30:
                continue
            score = first_support + second_support
            if score > pair_score:
                pair = first, second
                pair_score = score
    if pair is None:
        raise ValueError(
            "template does not contain two sufficiently supported orthogonal cube faces"
        )
    first, second = pair
    second = second - float(first @ second) * first
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    basis = np.column_stack((first, second, third))
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    return basis


def _cube_group_in_object_frame(cube_axes_object: np.ndarray) -> tuple[np.ndarray, ...]:
    basis = np.asarray(cube_axes_object, dtype=float)
    if basis.shape != (3, 3) or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
        raise ValueError("cube_axes_object must be an orthonormal 3x3 basis")
    return tuple(basis @ symmetry @ basis.T for symmetry in _CUBE_ROTATIONS)


def _validate_config(config: RegistrationConfig) -> None:
    if config.symmetry not in {"cube", "none"}:
        raise ValueError("symmetry must be 'cube' or 'none'")
    if config.min_frames < 3:
        raise ValueError("min_frames must be at least three for held-out validation")
    if config.min_points_per_frame < 6:
        raise ValueError("min_points_per_frame must be at least six")
    if config.voxel_size_m < 0.0:
        raise ValueError("voxel_size_m must be non-negative")
    positive = {
        "max_iterations": config.max_iterations,
        "max_correspondence_distance_m": config.max_correspondence_distance_m,
        "inlier_distance_m": config.inlier_distance_m,
        "max_fit_rmse_m": config.max_fit_rmse_m,
        "max_held_out_rmse_m": config.max_held_out_rmse_m,
        "max_frame_translation_deviation_m": (
            config.max_frame_translation_deviation_m
        ),
        "max_frame_rotation_deviation_deg": config.max_frame_rotation_deviation_deg,
        "ambiguity_score_margin_m": config.ambiguity_score_margin_m,
        "ambiguity_translation_tolerance_m": (
            config.ambiguity_translation_tolerance_m
        ),
        "ambiguity_rotation_tolerance_deg": (
            config.ambiguity_rotation_tolerance_deg
        ),
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    for name, value in {
        "template_trim_fraction": config.template_trim_fraction,
        "live_trim_fraction": config.live_trim_fraction,
        "min_template_inlier_fraction": config.min_template_inlier_fraction,
        "min_live_inlier_fraction": config.min_live_inlier_fraction,
    }.items():
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie in (0, 1]")
    if config.inlier_distance_m > config.max_correspondence_distance_m:
        raise ValueError(
            "inlier_distance_m cannot exceed max_correspondence_distance_m"
        )


def _validate_points(points: np.ndarray, *, name: str, minimum: int) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (3,) or len(values) < minimum:
        raise ValueError(f"{name} must have shape (N, 3), N >= {minimum}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if np.linalg.matrix_rank(values - np.mean(values, axis=0), tol=1e-7) < 2:
        raise ValueError(f"{name} is geometrically degenerate")
    return values


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if voxel_size_m == 0.0:
        return points.copy()
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    downsampled = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    return downsampled


def _reject_sparse_outliers(points: np.ndarray, *, minimum: int) -> np.ndarray:
    """Reject only isolated depth speckles while retaining cube edges."""

    if len(points) < 12:
        return points
    neighbor_count = min(8, len(points))
    distances, _ = cKDTree(points).query(points, k=neighbor_count)
    local_spacing = np.mean(distances[:, 1:], axis=1)
    median = float(np.median(local_spacing))
    mad = float(np.median(np.abs(local_spacing - median)))
    cutoff = median + 6.0 * max(mad, 1e-5)
    retained = points[local_spacing <= cutoff]
    return retained if len(retained) >= minimum else points


def _prepare_cloud(
    points: np.ndarray, *, name: str, minimum: int, voxel_size_m: float
) -> np.ndarray:
    values = _validate_points(points, name=name, minimum=minimum)
    values = _voxel_downsample(values, voxel_size_m)
    values = _reject_sparse_outliers(values, minimum=minimum)
    return _validate_points(values, name=f"prepared {name}", minimum=minimum)


def _trimmed(values: np.ndarray, fraction: float, *, minimum: int = 6) -> np.ndarray:
    count = max(minimum, int(np.ceil(fraction * len(values))))
    count = min(count, len(values))
    return np.partition(values, count - 1)[:count]


def _alignment_metrics(
    template: np.ndarray,
    live: np.ndarray,
    transform: np.ndarray,
    config: RegistrationConfig,
) -> AlignmentMetrics:
    transformed = template @ transform[:3, :3].T + transform[:3, 3]
    template_distances = cKDTree(live).query(transformed, k=1)[0]
    live_distances = cKDTree(transformed).query(live, k=1)[0]
    robust_template = _trimmed(
        template_distances, config.template_trim_fraction
    )
    robust_live = _trimmed(live_distances, config.live_trim_fraction)
    template_rmse = float(np.sqrt(np.mean(np.square(robust_template))))
    live_rmse = float(np.sqrt(np.mean(np.square(robust_live))))
    return AlignmentMetrics(
        rmse_m=float(np.sqrt(0.5 * (template_rmse**2 + live_rmse**2))),
        template_rmse_m=template_rmse,
        live_rmse_m=live_rmse,
        template_inlier_fraction=float(
            np.mean(template_distances <= config.inlier_distance_m)
        ),
        live_inlier_fraction=float(
            np.mean(live_distances <= config.inlier_distance_m)
        ),
    )


def _rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _correspondence_indices(
    distances: np.ndarray,
    *,
    fraction: float,
    max_distance: float,
    minimum: int = 6,
) -> np.ndarray:
    order = np.argsort(distances)
    count = max(minimum, int(np.ceil(fraction * len(order))))
    selected = order[: min(count, len(order))]
    selected = selected[distances[selected] <= max_distance]
    return selected


def _icp(
    template: np.ndarray,
    live: np.ndarray,
    initial: np.ndarray,
    config: RegistrationConfig,
) -> np.ndarray | None:
    transform = initial.copy()
    live_tree = cKDTree(live)
    previous_error = np.inf
    for _ in range(config.max_iterations):
        transformed = template @ transform[:3, :3].T + transform[:3, 3]
        forward_distances, forward_neighbors = live_tree.query(transformed, k=1)
        transformed_tree = cKDTree(transformed)
        reverse_distances, reverse_neighbors = transformed_tree.query(live, k=1)
        forward = _correspondence_indices(
            forward_distances,
            fraction=config.template_trim_fraction,
            max_distance=config.max_correspondence_distance_m,
        )
        reverse = _correspondence_indices(
            reverse_distances,
            fraction=config.live_trim_fraction,
            max_distance=config.max_correspondence_distance_m,
        )
        if len(forward) < 6 or len(reverse) < 6:
            return None
        # Equalize the two directions so dense live depth does not dominate.
        pair_count = min(len(forward), len(reverse))
        forward = forward[:pair_count]
        reverse = reverse[:pair_count]
        source = np.concatenate((template[forward], template[reverse_neighbors[reverse]]))
        target = np.concatenate((live[forward_neighbors[forward]], live[reverse]))
        if np.linalg.matrix_rank(source - np.mean(source, axis=0), tol=1e-8) < 2:
            return None
        updated = _rigid_svd(source, target)
        error = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.concatenate(
                            (forward_distances[forward], reverse_distances[reverse])
                        )
                    )
                )
            )
        )
        translation_delta = np.linalg.norm(updated[:3, 3] - transform[:3, 3])
        rotation_delta = _rotation_angle_deg(
            updated[:3, :3] @ transform[:3, :3].T
        )
        transform = updated
        if (
            abs(previous_error - error) < 1e-7
            and translation_delta < 1e-6
            and rotation_delta < 0.01
        ):
            break
        previous_error = error
    return validate_rigid_transform(transform, name="ICP result")


def _translation_only_icp(
    template: np.ndarray,
    live: np.ndarray,
    initial: np.ndarray,
    config: RegistrationConfig,
) -> np.ndarray | None:
    """Refine only translation while preserving ``initial`` rotation exactly."""

    transform = initial.copy()
    rotation = transform[:3, :3].copy()
    rotated_template = template @ rotation.T
    live_tree = cKDTree(live)
    previous_error = np.inf
    for _ in range(config.max_iterations):
        transformed = rotated_template + transform[:3, 3]
        forward_distances, forward_neighbors = live_tree.query(transformed, k=1)
        transformed_tree = cKDTree(transformed)
        reverse_distances, reverse_neighbors = transformed_tree.query(live, k=1)
        forward = _correspondence_indices(
            forward_distances,
            fraction=config.template_trim_fraction,
            max_distance=config.max_correspondence_distance_m,
        )
        reverse = _correspondence_indices(
            reverse_distances,
            fraction=config.live_trim_fraction,
            max_distance=config.max_correspondence_distance_m,
        )
        if len(forward) < 6 or len(reverse) < 6:
            return None
        pair_count = min(len(forward), len(reverse))
        forward = forward[:pair_count]
        reverse = reverse[:pair_count]
        source = np.concatenate(
            (rotated_template[forward], rotated_template[reverse_neighbors[reverse]])
        )
        target = np.concatenate((live[forward_neighbors[forward]], live[reverse]))
        updated_translation = np.mean(target - source, axis=0)
        error = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.concatenate(
                            (forward_distances[forward], reverse_distances[reverse])
                        )
                    )
                )
            )
        )
        translation_delta = float(
            np.linalg.norm(updated_translation - transform[:3, 3])
        )
        transform[:3, 3] = updated_translation
        if abs(previous_error - error) < 1e-7 and translation_delta < 1e-6:
            break
        previous_error = error
    transform[:3, :3] = rotation
    return validate_rigid_transform(transform, name="translation-only ICP result")


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_distance_modulo_symmetry_deg(
    first: np.ndarray,
    second: np.ndarray,
    symmetry_group: tuple[np.ndarray, ...],
) -> float:
    return min(
        _rotation_angle_deg(first.T @ second @ symmetry_rotation)
        for symmetry_rotation in symmetry_group
    )


def _poses_equivalent(
    first: np.ndarray,
    second: np.ndarray,
    config: RegistrationConfig,
    symmetry_group: tuple[np.ndarray, ...],
) -> bool:
    translation_distance = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation_distance = _rotation_distance_modulo_symmetry_deg(
        first[:3, :3], second[:3, :3], symmetry_group
    )
    return (
        translation_distance <= config.ambiguity_translation_tolerance_m
        and rotation_distance <= config.ambiguity_rotation_tolerance_deg
    )


def _initial_rotations(
    reference_rotation: np.ndarray,
    search_group: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        reference_rotation @ symmetry_rotation
        for symmetry_rotation in search_group
    )


def _make_initial_transform(
    template: np.ndarray, live: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.median(live, axis=0) - rotation @ np.median(
        template, axis=0
    )
    return transform


def _validate_orientation_prior(rotation_or_transform: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_or_transform, dtype=float)
    if values.shape == (4, 4):
        values = validate_rigid_transform(
            values, name="orientation_prior_T_base_object"
        )[:3, :3]
    elif values.shape == (3, 3):
        if not np.isfinite(values).all():
            raise ValueError("orientation prior rotation contains NaN or infinity")
        if not np.allclose(values.T @ values, np.eye(3), atol=1e-5):
            raise ValueError("orientation prior rotation is not orthonormal")
        if not np.isclose(np.linalg.det(values), 1.0, atol=1e-5):
            raise ValueError("orientation prior rotation determinant must be +1")
    else:
        raise ValueError("orientation prior must have shape (3, 3) or (4, 4)")
    return values


def _absolute_pose_clusters(
    hypotheses: list[_Hypothesis], config: RegistrationConfig
) -> list[list[_Hypothesis]]:
    clusters: list[list[_Hypothesis]] = []
    identity_group = (np.eye(3),)
    for hypothesis in hypotheses:
        for cluster in clusters:
            if _poses_equivalent(
                cluster[0].transform,
                hypothesis.transform,
                config,
                identity_group,
            ):
                cluster.append(hypothesis)
                break
        else:
            clusters.append([hypothesis])
    return clusters


def register_partial_template(
    template_points_object_m: np.ndarray,
    live_points_base_by_frame: list[np.ndarray] | tuple[np.ndarray, ...],
    reference_T_base_object: np.ndarray,
    *,
    config: RegistrationConfig | None = None,
    orientation_prior_base_object: np.ndarray | None = None,
    max_orientation_prior_error_deg: float | None = None,
    orientation_prior_uniqueness_margin_deg: float | None = None,
    lock_orientation_to_prior: bool = False,
) -> RegistrationResult:
    """Register a partial template and return a pose only after all gates pass.

    ``reference_T_base_object`` supplies a real rotation gauge (normally the calibrated
    ``T_base_camera`` rotation because the template axes equal the demonstration camera
    axes).  Its translation is only an initialization; point evidence estimates the
    returned translation.  When ``lock_orientation_to_prior`` is true, the explicit
    orientation prior is copied to the result exactly and live geometry estimates only
    translation while retaining the same fit and held-out quality gates.
    """

    config = config or RegistrationConfig()
    _validate_config(config)
    reference = validate_rigid_transform(
        reference_T_base_object, name="reference_T_base_object"
    )
    if orientation_prior_base_object is None:
        if lock_orientation_to_prior:
            raise ValueError(
                "lock_orientation_to_prior requires an explicit orientation prior"
            )
        if (
            max_orientation_prior_error_deg is not None
            or orientation_prior_uniqueness_margin_deg is not None
        ):
            raise ValueError(
                "orientation-prior thresholds require an explicit orientation prior"
            )
        orientation_prior = None
    else:
        orientation_prior = _validate_orientation_prior(
            orientation_prior_base_object
        )
        if (
            max_orientation_prior_error_deg is None
            or not np.isfinite(max_orientation_prior_error_deg)
            or max_orientation_prior_error_deg <= 0.0
        ):
            raise ValueError(
                "max_orientation_prior_error_deg must be positive when a prior is used"
            )
        if (
            orientation_prior_uniqueness_margin_deg is None
            or not np.isfinite(orientation_prior_uniqueness_margin_deg)
            or orientation_prior_uniqueness_margin_deg <= 0.0
        ):
            raise ValueError(
                "orientation_prior_uniqueness_margin_deg must be positive when a prior is used"
            )
    template = _prepare_cloud(
        template_points_object_m,
        name="template points",
        minimum=config.min_points_per_frame,
        voxel_size_m=config.voxel_size_m,
    )
    if config.symmetry == "cube":
        cube_axes_object = estimate_cube_axes_object(
            template,
            plane_distance_m=max(0.0008, 1.5 * config.voxel_size_m),
        )
        symmetry_group = _cube_group_in_object_frame(cube_axes_object)
    else:
        cube_axes_object = None
        symmetry_group = (np.eye(3),)
    if len(live_points_base_by_frame) < config.min_frames:
        raise ValueError(
            f"at least {config.min_frames} live frames are required for held-out validation"
        )
    frames = tuple(
        _prepare_cloud(
            points,
            name=f"live frame {index}",
            minimum=config.min_points_per_frame,
            voxel_size_m=config.voxel_size_m,
        )
        for index, points in enumerate(live_points_base_by_frame)
    )
    held_out_indices = tuple(range(1, len(frames), 2))
    fit_indices = tuple(index for index in range(len(frames)) if index not in held_out_indices)
    if not held_out_indices or len(fit_indices) < 2:
        raise ValueError("registration needs at least two fit frames and one held-out frame")
    fit_cloud = np.concatenate([frames[index] for index in fit_indices])

    hypotheses: list[_Hypothesis] = []
    search_group = symmetry_group if config.symmetry == "cube" else _CUBE_ROTATIONS
    initial_rotations = _initial_rotations(reference[:3, :3], search_group)
    for rotation in initial_rotations:
        initial = _make_initial_transform(template, fit_cloud, rotation)
        fitted = _icp(template, fit_cloud, initial, config)
        if fitted is None:
            continue
        metrics = _alignment_metrics(template, fit_cloud, fitted, config)
        hypotheses.append(_Hypothesis(fitted, metrics))
    if not hypotheses:
        raise ValueError("registration produced no valid ICP hypothesis")
    hypotheses.sort(key=lambda hypothesis: hypothesis.metrics.rmse_m)
    best_score = hypotheses[0].metrics.rmse_m
    near_optimal = [
        hypothesis
        for hypothesis in hypotheses
        if hypothesis.metrics.rmse_m
        <= best_score + config.ambiguity_score_margin_m
    ]
    prior_error: float | None = None
    prior_second_best_error: float | None = None
    if orientation_prior is None:
        best = hypotheses[0]
        nonequivalent = [
            hypothesis
            for hypothesis in near_optimal
            if not _poses_equivalent(
                best.transform, hypothesis.transform, config, symmetry_group
            )
        ]
        if nonequivalent:
            raise ValueError(
                "registration is ambiguous: near-optimal poses are not explained by the "
                f"declared {config.symmetry!r} symmetry; provide an explicit, "
                "traceable orientation prior only if the object's orientation is known"
            )
        # Gauge fixing within a geometry-equivalent symmetry class is not evidence of
        # printed die-face identity; the report records this limitation.
        selected = min(
            near_optimal,
            key=lambda hypothesis: (
                _rotation_angle_deg(
                    reference[:3, :3].T @ hypothesis.transform[:3, :3]
                ),
                hypothesis.metrics.rmse_m,
            ),
        )
    else:
        absolute_clusters = _absolute_pose_clusters(near_optimal, config)
        ranked: list[tuple[float, _Hypothesis]] = []
        for cluster in absolute_clusters:
            representative = min(
                cluster,
                key=lambda hypothesis: (
                    _rotation_angle_deg(
                        orientation_prior.T @ hypothesis.transform[:3, :3]
                    ),
                    hypothesis.metrics.rmse_m,
                ),
            )
            ranked.append(
                (
                    _rotation_angle_deg(
                        orientation_prior.T @ representative.transform[:3, :3]
                    ),
                    representative,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1].metrics.rmse_m))
        prior_error, selected = ranked[0]
        if prior_error > max_orientation_prior_error_deg:
            raise ValueError(
                f"best near-optimal orientation is {prior_error:.3f} deg from the "
                f"explicit prior, exceeding {max_orientation_prior_error_deg:.3f} deg"
            )
        if len(ranked) > 1:
            prior_second_best_error = ranked[1][0]
            separation = prior_second_best_error - prior_error
            if separation < orientation_prior_uniqueness_margin_deg:
                raise ValueError(
                    "explicit orientation prior cannot uniquely disambiguate "
                    f"near-optimal poses: best/second errors are {prior_error:.3f}/"
                    f"{prior_second_best_error:.3f} deg, below the required "
                    f"{orientation_prior_uniqueness_margin_deg:.3f} deg separation"
                )
    unlocked_selected = selected
    unlocked_fit_metrics: AlignmentMetrics | None = None
    orientation_lock_error: float | None = None
    if lock_orientation_to_prior:
        assert orientation_prior is not None
        locked_initial = selected.transform.copy()
        locked_initial[:3, :3] = orientation_prior
        locked_transform = _translation_only_icp(
            template, fit_cloud, locked_initial, config
        )
        if locked_transform is None:
            raise ValueError(
                "translation-only registration produced no valid correspondence fit"
            )
        unlocked_fit_metrics = selected.metrics
        selected = _Hypothesis(
            locked_transform,
            _alignment_metrics(template, fit_cloud, locked_transform, config),
        )
        orientation_lock_error = _rotation_angle_deg(
            orientation_prior.T @ selected.transform[:3, :3]
        )
        if orientation_lock_error > 1e-6:
            raise ValueError("translation-only registration did not preserve prior rotation")
    if selected.metrics.rmse_m > config.max_fit_rmse_m:
        raise ValueError(
            f"fit RMSE {selected.metrics.rmse_m:.6f} m exceeds "
            f"{config.max_fit_rmse_m:.6f} m"
        )
    if (
        selected.metrics.template_inlier_fraction
        < config.min_template_inlier_fraction
    ):
        raise ValueError("fit template inlier fraction is below threshold")
    if selected.metrics.live_inlier_fraction < config.min_live_inlier_fraction:
        raise ValueError("fit live inlier fraction is below threshold")

    frame_metrics: list[AlignmentMetrics] = []
    frame_translations: list[float] = []
    frame_rotations: list[float] = []
    for index, frame in enumerate(frames):
        metrics = _alignment_metrics(template, frame, selected.transform, config)
        frame_metrics.append(metrics)
        refined = (
            _translation_only_icp(template, frame, selected.transform, config)
            if lock_orientation_to_prior
            else _icp(template, frame, selected.transform, config)
        )
        if refined is None:
            raise ValueError(f"live frame {index} could not refine from consensus pose")
        frame_translations.append(
            float(np.linalg.norm(refined[:3, 3] - selected.transform[:3, 3]))
        )
        rotation_refined = (
            _icp(template, frame, unlocked_selected.transform, config)
            if lock_orientation_to_prior
            else refined
        )
        if rotation_refined is None:
            raise ValueError(
                f"live frame {index} could not refine rotation for consistency check"
            )
        frame_rotations.append(
            _rotation_distance_modulo_symmetry_deg(
                (
                    unlocked_selected.transform[:3, :3]
                    if lock_orientation_to_prior
                    else selected.transform[:3, :3]
                ),
                rotation_refined[:3, :3],
                symmetry_group,
            )
        )

    held_out_rmse = max(frame_metrics[index].rmse_m for index in held_out_indices)
    held_out_template_inliers = min(
        frame_metrics[index].template_inlier_fraction for index in held_out_indices
    )
    held_out_live_inliers = min(
        frame_metrics[index].live_inlier_fraction for index in held_out_indices
    )
    if held_out_rmse > config.max_held_out_rmse_m:
        raise ValueError(
            f"held-out RMSE {held_out_rmse:.6f} m exceeds "
            f"{config.max_held_out_rmse_m:.6f} m"
        )
    if held_out_template_inliers < config.min_template_inlier_fraction:
        raise ValueError("held-out template inlier fraction is below threshold")
    if held_out_live_inliers < config.min_live_inlier_fraction:
        raise ValueError("held-out live inlier fraction is below threshold")
    translation_deviation = np.asarray(frame_translations)
    rotation_deviation = np.asarray(frame_rotations)
    if float(np.max(translation_deviation)) > config.max_frame_translation_deviation_m:
        raise ValueError("per-frame registration translation is inconsistent")
    if float(np.max(rotation_deviation)) > config.max_frame_rotation_deviation_deg:
        raise ValueError("per-frame registration rotation is inconsistent")

    hypothesis_reference = (
        unlocked_selected.transform if lock_orientation_to_prior else selected.transform
    )
    return RegistrationResult(
        T_base_object=selected.transform,
        fit_frame_indices=fit_indices,
        held_out_frame_indices=held_out_indices,
        fit_metrics=selected.metrics,
        frame_metrics=tuple(frame_metrics),
        frame_translation_deviation_m=translation_deviation,
        frame_rotation_deviation_deg=rotation_deviation,
        near_optimal_hypothesis_count=len(near_optimal),
        symmetry_equivalent_hypothesis_count=sum(
            _poses_equivalent(
                hypothesis_reference,
                hypothesis.transform,
                config,
                symmetry_group,
            )
            for hypothesis in near_optimal
        ),
        initial_hypothesis_count=len(initial_rotations),
        cube_axes_object=cube_axes_object,
        orientation_prior_used=orientation_prior is not None,
        orientation_locked_to_prior=lock_orientation_to_prior,
        unlocked_fit_metrics=unlocked_fit_metrics,
        orientation_lock_error_deg=orientation_lock_error,
        orientation_prior_error_deg=prior_error,
        orientation_prior_second_best_error_deg=prior_second_best_error,
        orientation_prior_max_error_deg=max_orientation_prior_error_deg,
        orientation_prior_uniqueness_margin_deg=(
            orientation_prior_uniqueness_margin_deg
        ),
    )

