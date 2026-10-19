"""Load the pipeline's compact NPZ and JSON interchange formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import GraspCandidates, GraspIntent, ObservationSequence


def load_observation_npz(
    path: str | Path, object_points_path: str | Path
) -> ObservationSequence:
    with np.load(path, allow_pickle=False) as data:
        return ObservationSequence(
            T_camera_object=np.asarray(data["T_camera_object"], dtype=float),
            hand_joints_camera_m=np.asarray(data["hand_joints_camera_m"], dtype=float),
            hand_joint_confidence=np.asarray(
                data["hand_joint_confidence"], dtype=float
            ),
            timestamps_s=np.asarray(data["timestamps_s"], dtype=float),
            object_points_m=np.asarray(
                np.load(object_points_path, allow_pickle=False), dtype=float
            ),
        )


def load_candidates_npz(path: str | Path) -> GraspCandidates:
    with np.load(path, allow_pickle=False) as data:
        return GraspCandidates(
            poses=np.asarray(data["poses"], dtype=float),
            scores=np.asarray(data["scores"], dtype=float),
            widths_m=np.asarray(data["widths_m"], dtype=float),
        )


def save_grasp_intent_json(intent: GraspIntent, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(intent.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_grasp_intent_json(path: str | Path) -> GraspIntent:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GraspIntent.from_dict(data)
