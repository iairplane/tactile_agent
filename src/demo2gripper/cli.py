"""Command line entry points for the object-centric grasp pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .intent import IntentConfig, infer_grasp_intent
from .io import (
    load_candidates_npz,
    load_grasp_intent_json,
    load_observation_npz,
    save_grasp_intent_json,
)
from .selection import SelectionConfig, rank_grasp_candidates
from .synthetic import write_synthetic_inputs


def _symmetry_axis(value: str | int | None) -> int | None:
    if value is None or value == "none":
        return None
    if isinstance(value, int):
        if value in (0, 1, 2):
            return value
        raise ValueError("continuous symmetry axis must be x, y, z, or none")
    try:
        return {"x": 0, "y": 1, "z": 2}[value.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("continuous symmetry axis must be x, y, z, or none") from error


def _ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _optional_npz_scalar(archive: Any, key: str) -> str | None:
    """Read an optional scalar string without enabling pickled arrays."""

    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"candidate metadata {key} must be a scalar string")
    text = str(value.item()).strip()
    if not text:
        raise ValueError(f"candidate metadata {key} must be non-empty")
    return text


def _write_selected(
    ranked: list[Any],
    output: str | Path,
    *,
    selection_config: SelectionConfig,
    approach_distance_m: float,
    demonstrated_contact_width_m: float,
    top_k: int,
    gripper_name: str | None = None,
    object_frame_id: str | None = None,
    intent_provenance: dict[str, str] | None = None,
) -> None:
    selected = ranked[0]

    def threshold(value: float) -> float | None:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    payload = {
        "schema_version": 1,
        "frame_convention": "T_object_gripper_tcp; +X closing, +Z approach",
        "T_object_gripper_tcp": selected.T_object_gripper_tcp.tolist(),
        "pregrasp_width_m": selected.width_m,
        "close_target_width_m": 0.0,
        "demonstrated_contact_width_m": demonstrated_contact_width_m,
        "approach_distance_m": approach_distance_m,
        "candidate_index": selected.index,
        "model_score": selected.model_score,
        "center_error_m": selected.center_error_m,
        "closing_axis_error_rad": selected.closing_axis_error_rad,
        "approach_axis_error_rad": selected.approach_axis_error_rad,
        "ranking_cost": selected.ranking_cost,
        "ranked_candidates": [candidate.to_dict() for candidate in ranked[:top_k]],
        "selection": {
            "hard_gate_thresholds": {
                "min_gripper_width_m": threshold(
                    selection_config.min_gripper_width_m
                ),
                "max_gripper_width_m": threshold(
                    selection_config.max_gripper_width_m
                ),
                "max_center_distance_m": threshold(
                    selection_config.max_center_distance_m
                ),
                "max_closing_axis_error_rad": threshold(
                    selection_config.max_closing_axis_error_rad
                ),
                "max_approach_axis_error_rad": threshold(
                    selection_config.max_approach_axis_error_rad
                ),
            },
            "hard_gates_passed": True,
        },
    }
    if gripper_name is not None:
        payload["gripper_name"] = gripper_name
    if object_frame_id is not None:
        payload["object_frame_id"] = object_frame_id
    if intent_provenance:
        payload["intent_provenance"] = intent_provenance
    _ensure_parent(output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _intent_config(values: argparse.Namespace) -> IntentConfig:
    return IntentConfig(
        max_tip_object_distance_m=values.max_contact_distance_m,
        min_joint_confidence=values.min_joint_confidence,
        min_pinch_distance_m=values.min_pinch_width_m,
        max_pinch_distance_m=values.max_pinch_width_m,
        min_stable_frames=values.min_stable_frames,
        continuous_symmetry_axis=_symmetry_axis(values.continuous_symmetry_axis),
    )


def _selection_config(values: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        min_gripper_width_m=values.min_jaw_width_m,
        max_gripper_width_m=values.max_jaw_width_m,
        max_center_distance_m=values.max_center_distance_m,
        max_closing_axis_error_rad=np.deg2rad(values.max_closing_axis_error_deg),
        max_approach_axis_error_rad=np.deg2rad(values.max_approach_axis_error_deg),
    )


def _learn_intent(values: argparse.Namespace) -> None:
    observation = load_observation_npz(values.observations, values.object_points)
    intent = infer_grasp_intent(observation, _intent_config(values))
    output = _ensure_parent(values.output)
    save_grasp_intent_json(intent, output)
    print(
        f"intent: {output} (stable frames {intent.stable_interval[0]}:"
        f"{intent.stable_interval[1]}, width {intent.width_m:.4f} m)"
    )


def _select_grasp(values: argparse.Namespace) -> None:
    intent_payload = json.loads(Path(values.intent).read_text(encoding="utf-8"))
    intent = load_grasp_intent_json(values.intent)
    candidates = load_candidates_npz(values.candidates)
    with np.load(values.candidates, allow_pickle=False) as archive:
        candidate_gripper = _optional_npz_scalar(archive, "gripper_name")
        candidate_frame = _optional_npz_scalar(archive, "object_frame_id")
        candidate_convention = _optional_npz_scalar(archive, "frame_convention")
    expected_convention = "T_object_gripper_tcp; +X closing, +Z approach"
    if candidate_convention not in (None, expected_convention):
        raise ValueError(
            "candidate frame_convention does not match the canonical grasp contract"
        )
    intent_frame = intent_payload.get("object_frame_id")
    if (
        intent_frame is not None
        and candidate_frame is not None
        and intent_frame != candidate_frame
    ):
        raise ValueError("intent and candidates use different object_frame_id values")
    gripper_name = values.gripper_name or candidate_gripper
    object_frame_id = values.object_frame_id or candidate_frame or intent_frame
    if values.gripper_name and candidate_gripper not in (None, values.gripper_name):
        raise ValueError("--gripper-name conflicts with candidates metadata")
    if values.object_frame_id and candidate_frame not in (None, values.object_frame_id):
        raise ValueError("--object-frame-id conflicts with candidates metadata")
    if values.object_frame_id and intent_frame not in (None, values.object_frame_id):
        raise ValueError("--object-frame-id conflicts with intent metadata")
    selection_config = _selection_config(values)
    ranked = rank_grasp_candidates(intent, candidates, selection_config)
    intent_provenance = {
        key: str(intent_payload[key])
        for key in ("schema", "intent_role")
        if isinstance(intent_payload.get(key), str) and intent_payload[key].strip()
    }
    _write_selected(
        ranked,
        values.output,
        selection_config=selection_config,
        approach_distance_m=values.approach_distance_m,
        demonstrated_contact_width_m=intent.width_m,
        top_k=values.top_k,
        gripper_name=gripper_name,
        object_frame_id=object_frame_id,
        intent_provenance=intent_provenance,
    )
    print(
        f"selected candidate {ranked[0].index}: {values.output} "
        f"(cost {ranked[0].ranking_cost:.4f})"
    )


def _make_synthetic(values: argparse.Namespace) -> None:
    observations, object_points, candidates = write_synthetic_inputs(values.output_dir)
    print(f"observations: {observations}")
    print(f"object points: {object_points}")
    print(f"candidates: {candidates}")


def _run_config(values: argparse.Namespace) -> None:
    config_path = Path(values.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sequence = config["sequence"]
    intent_options = config["intent"]
    selection_options = config["selection"]
    outputs = config["outputs"]

    observation = load_observation_npz(
        sequence["observations_npz"], sequence["object_points_npy"]
    )
    intent = infer_grasp_intent(
        observation,
        IntentConfig(
            max_tip_object_distance_m=float(intent_options["max_contact_distance_m"]),
            min_joint_confidence=float(intent_options["min_joint_confidence"]),
            min_pinch_distance_m=float(intent_options["min_pinch_width_m"]),
            max_pinch_distance_m=float(intent_options["max_pinch_width_m"]),
            min_stable_frames=int(intent_options["min_stable_frames"]),
            continuous_symmetry_axis=_symmetry_axis(
                intent_options.get("continuous_symmetry_axis")
            ),
        ),
    )
    intent_output = _ensure_parent(outputs["intent_json"])
    save_grasp_intent_json(intent, intent_output)

    candidates = load_candidates_npz(selection_options["candidates_npz"])
    selection_config = SelectionConfig(
        min_gripper_width_m=float(selection_options["min_jaw_width_m"]),
        max_gripper_width_m=float(selection_options["max_jaw_width_m"]),
        max_center_distance_m=float(selection_options["max_center_distance_m"]),
        max_closing_axis_error_rad=np.deg2rad(
            float(selection_options["max_closing_axis_error_deg"])
        ),
        max_approach_axis_error_rad=np.deg2rad(
            float(selection_options["max_approach_axis_error_deg"])
        ),
    )
    ranked = rank_grasp_candidates(
        intent,
        candidates,
        selection_config,
    )
    selected_output = outputs["selected_grasp_json"]
    _write_selected(
        ranked,
        selected_output,
        selection_config=selection_config,
        approach_distance_m=float(intent_options["approach_distance_m"]),
        demonstrated_contact_width_m=intent.width_m,
        top_k=int(selection_options["top_k"]),
        gripper_name=selection_options.get("gripper_name"),
        object_frame_id=selection_options.get("object_frame_id"),
    )
    print(f"intent: {intent_output}")
    print(f"selected candidate {ranked[0].index}: {selected_output}")


def _add_intent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-contact-distance-m", type=float, default=0.018)
    parser.add_argument("--min-joint-confidence", type=float, default=0.5)
    parser.add_argument("--min-pinch-width-m", type=float, default=0.005)
    parser.add_argument("--max-pinch-width-m", type=float, default=0.11)
    parser.add_argument("--min-stable-frames", type=int, default=3)
    parser.add_argument(
        "--continuous-symmetry-axis",
        choices=("none", "x", "y", "z"),
        default="none",
        help="CAD-frame axis for continuous rotational symmetry",
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-jaw-width-m", type=float, default=0.0)
    parser.add_argument("--max-jaw-width-m", type=float, default=0.08)
    parser.add_argument("--max-center-distance-m", type=float, default=0.06)
    parser.add_argument("--max-closing-axis-error-deg", type=float, default=45.0)
    parser.add_argument("--max-approach-axis-error-deg", type=float, default=60.0)
    parser.add_argument("--approach-distance-m", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--gripper-name",
        help="Robot gripper identity recorded in selected_grasp.json",
    )
    parser.add_argument(
        "--object-frame-id",
        help="Stable identity of the object/template frame shared by intent and candidates",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo2gripper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "make-synthetic", help="write deterministic observations and candidates"
    )
    synthetic.add_argument("--output-dir", required=True)
    synthetic.set_defaults(handler=_make_synthetic)

    intent = subparsers.add_parser(
        "learn-intent", help="infer an object-frame pinch intent"
    )
    intent.add_argument("--observations", required=True)
    intent.add_argument("--object-points", required=True)
    intent.add_argument("--output", required=True)
    _add_intent_arguments(intent)
    intent.set_defaults(handler=_learn_intent)

    selection = subparsers.add_parser(
        "select-grasp", help="rank robot grasp candidates against the intent"
    )
    selection.add_argument("--intent", required=True)
    selection.add_argument("--candidates", required=True)
    selection.add_argument("--output", required=True)
    _add_selection_arguments(selection)
    selection.set_defaults(handler=_select_grasp)

    run = subparsers.add_parser("run", help="run intent learning and selection from JSON")
    run.add_argument("--config", required=True)
    run.set_defaults(handler=_run_config)
    return parser


def main() -> None:
    values = build_parser().parse_args()
    values.handler(values)


if __name__ == "__main__":
    main()
