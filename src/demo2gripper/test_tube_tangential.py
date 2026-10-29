"""Force-direction observation for tube pivot experiments; no hardware commands.

Inputs are simultaneous, bias-corrected forces with qualified sensor rotations.
The output is a FORCE direction, not an estimate of the tube axis angle.
"""

from dataclasses import dataclass
import math

import numpy as np


def _rotation(value):
    r = np.asarray(value, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(np.linalg.det(r), 1.0, abs_tol=1e-6)):
        raise ValueError("Expected a proper calibrated rotation")
    return r


def _vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("Expected a finite three-vector")
    return v


@dataclass(frozen=True)
class TangentialDirection:
    force_base: tuple[float, float, float]
    magnitude: float
    elevation_deg: float | None
    cancellation_ratio: float
    valid: bool


def tangential_direction(
    forces_sensor, normals_sensor, rotations_tcp_sensor, rotation_base_tcp,
    *, min_magnitude: float, min_cancellation_ratio: float,
) -> TangentialDirection:
    """Project EACH finger force before rotating and summing in base coordinates.

Normals must use each sensor's own coordinate system. All forces must have
the same physical sign convention (e.g. object-on-sensor). Caller qualifies
timestamps, unloaded bias, units and calibration before calling this function.
Thresholds must come from measured unloaded noise and the task experiment.
"""
    if not math.isfinite(min_magnitude) or min_magnitude <= 0:
        raise ValueError("min_magnitude must be positive")
    if not 0 < min_cancellation_ratio <= 1:
        raise ValueError("min_cancellation_ratio must be in (0, 1]")
    if not (len(forces_sensor) == len(normals_sensor) == len(rotations_tcp_sensor) == 2):
        raise ValueError("Exactly two fingers required")
    r_base = _rotation(rotation_base_tcp)
    components = []
    for force, normal, rotation in zip(
        forces_sensor, normals_sensor, rotations_tcp_sensor, strict=True
    ):
        f, n = _vector(force), _vector(normal)
        length = np.linalg.norm(n)
        if length < 1e-12:
            raise ValueError("Contact normal cannot be zero")
        n = n / length
        components.append(r_base @ _rotation(rotation) @ (f - n * np.dot(f, n)))
    total = np.sum(components, axis=0)
    magnitude = float(np.linalg.norm(total))
    scale = sum(float(np.linalg.norm(v)) for v in components)
    ratio = magnitude / scale if scale > 0 else 0.0
    valid = magnitude >= min_magnitude and ratio >= min_cancellation_ratio
    elevation = math.degrees(math.atan2(total[2], math.hypot(total[0], total[1])))
    return TangentialDirection(
        tuple(float(v) for v in total), magnitude,
        elevation if valid else None, ratio, valid,
    )
