#!/usr/bin/env python3
"""Render the perception-to-grasp outputs as a four-panel annotated video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
MASK_COLORS = (
    (40, 180, 255),
    (50, 220, 80),
    (220, 80, 220),
    (255, 180, 30),
    (60, 120, 255),
)
BBOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--segmentation-dir", required=True, type=Path)
    parser.add_argument(
        "--mask-label",
        action="append",
        required=True,
        metavar="ID:LABEL",
        help="Semantic label for a segmentation object ID; repeat for each mask.",
    )
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--hand-manifest", type=Path)
    parser.add_argument("--hand-keypoints-2d", type=Path)
    parser.add_argument("--hand-scores-2d", type=Path)
    parser.add_argument("--object-points", required=True, type=Path)
    parser.add_argument("--object-manifest", type=Path)
    parser.add_argument("--object-confidence", type=Path)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--selected-grasp", required=True, type=Path)
    parser.add_argument("--isaac-result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--min-joint-confidence", type=float, default=0.5)
    return parser.parse_args()


def numeric_frames(directory: Path) -> list[Path]:
    paths = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"RGB frame stems must be integers in {directory}") from error


def parse_mask_labels(values: list[str]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for value in values:
        try:
            raw_id, label = value.split(":", 1)
            obj_id = int(raw_id)
        except ValueError as error:
            raise ValueError(f"mask label must have form ID:LABEL, got {value!r}") from error
        if not label:
            raise ValueError(f"mask label must not be empty: {value!r}")
        labels[obj_id] = label
    return labels


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_camera = np.asarray(points_camera, dtype=float)
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-6)
    pixels = np.full((len(points_camera), 2), np.nan, dtype=float)
    if np.any(valid):
        homogeneous = points_camera[valid] @ intrinsics.T
        pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, valid


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def point(pixel: np.ndarray) -> tuple[int, int]:
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def title(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (15, 15, 15), -1)
    put_text(image, text, (12, 24), scale=0.62, thickness=2)


def overlay_masks(
    image: np.ndarray,
    mask_paths: dict[str, str],
    segmentation_dir: Path,
    labels: dict[int, str],
) -> None:
    for color_index, obj_id in enumerate(sorted(labels)):
        relative_path = mask_paths.get(str(obj_id))
        if relative_path is None:
            continue
        mask = cv2.imread(str(segmentation_dir / relative_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"invalid mask for object {obj_id}: {relative_path}")
        active = mask > 0
        color = np.asarray(MASK_COLORS[color_index % len(MASK_COLORS)], dtype=np.float32)
        image[active] = np.clip(0.55 * image[active] + 0.45 * color, 0, 255).astype(
            np.uint8
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, tuple(int(value) for value in color), 2)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            if moments["m00"] > 0:
                center = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )
                put_text(image, f"{obj_id}: {labels[obj_id]}", center, scale=0.48, thickness=2)


def draw_hand(
    image: np.ndarray,
    joints_camera: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    threshold: float,
) -> None:
    pixels, valid = project(joints_camera, intrinsics)
    draw_hand_pixels(image, pixels, confidence, valid, threshold)


def draw_hand_pixels(
    image: np.ndarray,
    pixels: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> None:
    for first, second in HAND_EDGES:
        if not (valid[first] and valid[second]):
            continue
        confident = confidence[first] >= threshold and confidence[second] >= threshold
        color = (30, 230, 255) if confident else (130, 130, 130)
        cv2.line(image, point(pixels[first]), point(pixels[second]), color, 3, cv2.LINE_AA)
    for joint_index, pixel in enumerate(pixels):
        if not valid[joint_index]:
            continue
        color = (40, 40, 255) if joint_index in (4, 8) else (255, 230, 80)
        if confidence[joint_index] < threshold:
            color = (130, 130, 130)
        cv2.circle(image, point(pixel), 4, color, -1, cv2.LINE_AA)
    for joint_index, label in ((4, "thumb"), (8, "index")):
        if valid[joint_index]:
            xy = point(pixels[joint_index])
            put_text(image, label, (xy[0] + 6, xy[1] - 5), scale=0.42)


def bbox_corners(points: np.ndarray) -> np.ndarray:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )


def draw_pose_axes(
    image: np.ndarray,
    transform_camera: np.ndarray,
    intrinsics: np.ndarray,
    length_m: float,
    *,
    thickness: int = 3,
) -> None:
    axes = np.vstack(
        [
            transform_camera[:3, 3],
            transform_camera[:3, 3] + length_m * transform_camera[:3, 0],
            transform_camera[:3, 3] + length_m * transform_camera[:3, 1],
            transform_camera[:3, 3] + length_m * transform_camera[:3, 2],
        ]
    )
    pixels, valid = project(axes, intrinsics)
    if not valid[0]:
        return
    origin = point(pixels[0])
    for axis_index, color in enumerate(((0, 0, 255), (0, 220, 0), (255, 80, 40)), start=1):
        if valid[axis_index]:
            cv2.arrowedLine(
                image,
                origin,
                point(pixels[axis_index]),
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.22,
            )


def draw_object_pose(
    image: np.ndarray,
    transform_camera_object: np.ndarray,
    object_points: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    step = max(1, len(object_points) // 650)
    sampled_camera = transform_points(transform_camera_object, object_points[::step])
    pixels, valid = project(sampled_camera, intrinsics)
    for pixel in pixels[valid]:
        cv2.circle(image, point(pixel), 1, (255, 210, 40), -1, cv2.LINE_AA)

    corners_camera = transform_points(transform_camera_object, bbox_corners(object_points))
    corner_pixels, corner_valid = project(corners_camera, intrinsics)
    for first, second in BBOX_EDGES:
        if corner_valid[first] and corner_valid[second]:
            cv2.line(
                image,
                point(corner_pixels[first]),
                point(corner_pixels[second]),
                (255, 150, 30),
                2,
                cv2.LINE_AA,
            )
    scale = max(0.02, 0.45 * float(np.max(np.ptp(object_points, axis=0))))
    draw_pose_axes(image, transform_camera_object, intrinsics, scale)


def draw_intent(
    image: np.ndarray,
    frame_index: int,
    transform_camera_object: np.ndarray,
    intrinsics: np.ndarray,
    intent: dict[str, object],
    selected: dict[str, object],
) -> None:
    start, end = (int(value) for value in intent["stable_interval"])
    stable = start <= frame_index < end
    power_grip = intent.get("intent_type") == "power"
    contact_color = (50, 230, 70) if stable else (150, 150, 150)
    contact_points_object = np.asarray(intent["contact_points_object_m"], dtype=float)
    contact_points_camera = transform_points(transform_camera_object, contact_points_object)
    contact_pixels, contact_valid = project(contact_points_camera, intrinsics)
    if np.all(contact_valid):
        cv2.line(
            image,
            point(contact_pixels[0]),
            point(contact_pixels[1]),
            contact_color,
            4,
            cv2.LINE_AA,
        )
        for pixel in contact_pixels:
            cv2.circle(image, point(pixel), 7, contact_color, -1, cv2.LINE_AA)

    if power_grip:
        human_center = np.asarray(intent["human_grip_center_object_m"], dtype=float)
        human_center_camera = transform_points(
            transform_camera_object, human_center[None]
        )
        human_pixels, human_valid = project(human_center_camera, intrinsics)
        if human_valid[0]:
            cv2.circle(
                image, point(human_pixels[0]), 9, (255, 80, 230), 3, cv2.LINE_AA
            )
    else:
        intent_pose = transform_camera_object @ np.asarray(
            intent["T_object_gripper_tcp"], dtype=float
        )
        intent_center = intent_pose[:3, 3]
        intent_approach = intent_center - 0.07 * intent_pose[:3, 2]
        intent_pixels, intent_valid = project(
            np.vstack((intent_approach, intent_center)), intrinsics
        )
        if np.all(intent_valid):
            cv2.arrowedLine(
                image,
                point(intent_pixels[0]),
                point(intent_pixels[1]),
                (255, 80, 230),
                3,
                cv2.LINE_AA,
                tipLength=0.2,
            )

    selected_pose = transform_camera_object @ np.asarray(
        selected["T_object_gripper_tcp"], dtype=float
    )
    width_m = float(selected["pregrasp_width_m"])
    approach_m = float(selected["approach_distance_m"])
    center = selected_pose[:3, 3]
    closing = selected_pose[:3, 0]
    approach = selected_pose[:3, 2]
    jaw_a = center - 0.5 * width_m * closing
    jaw_b = center + 0.5 * width_m * closing
    finger_length = 0.045
    gripper_points = np.vstack(
        (
            center - approach_m * approach,
            center,
            jaw_a,
            jaw_b,
            jaw_a - finger_length * approach,
            jaw_b - finger_length * approach,
        )
    )
    gripper_pixels, gripper_valid = project(gripper_points, intrinsics)
    if np.all(gripper_valid):
        pregrasp, tcp, first_jaw, second_jaw, first_back, second_back = map(
            point, gripper_pixels
        )
        cv2.arrowedLine(image, pregrasp, tcp, (40, 255, 120), 3, cv2.LINE_AA, tipLength=0.15)
        cv2.line(image, first_jaw, second_jaw, (40, 255, 120), 4, cv2.LINE_AA)
        cv2.line(image, first_jaw, first_back, (40, 255, 120), 5, cv2.LINE_AA)
        cv2.line(image, second_jaw, second_back, (40, 255, 120), 5, cv2.LINE_AA)
        cv2.circle(image, tcp, 5, (40, 255, 120), -1, cv2.LINE_AA)

    if power_grip:
        state = "HELD POWER GRIP" if stable else "NO GRIP EVIDENCE"
    else:
        state = "CONTACT WINDOW" if stable else "SEARCH / APPROACH"
    put_text(image, state, (12, 58), color=contact_color, scale=0.62, thickness=2)
    candidate = selected.get("candidate_index")
    model_score = selected.get("model_score")
    if candidate is not None and model_score is not None:
        put_text(
            image,
            f"GraspGenX candidate {candidate} | score {float(model_score):.3f}",
            (12, 82),
            scale=0.48,
        )
    legend = (
        "magenta: human grip center | green: CAD-valid gripper"
        if power_grip
        else "magenta: human approach | green: selected gripper"
    )
    put_text(image, legend, (12, 106), scale=0.43)

    bar_left, bar_right = 16, image.shape[1] - 16
    bar_y = image.shape[0] - 20
    cv2.line(image, (bar_left, bar_y), (bar_right, bar_y), (100, 100, 100), 6)
    frame_count = int(intent.get("frame_count", max(end, frame_index + 1)))
    denominator = max(1, frame_count - 1)
    stable_left = bar_left + int((bar_right - bar_left) * start / denominator)
    stable_right = bar_left + int((bar_right - bar_left) * (end - 1) / denominator)
    cv2.line(image, (stable_left, bar_y), (stable_right, bar_y), (50, 220, 70), 6)
    marker = bar_left + int((bar_right - bar_left) * frame_index / denominator)
    cv2.circle(image, (marker, bar_y), 6, (255, 255, 255), -1)


def draw_isaac_status(image: np.ndarray, result: dict[str, object] | None) -> None:
    if result is None:
        return
    checks = (
        ("IK", "ik_reachable"),
        ("CONTACT", "bilateral_contact"),
        ("LIFT", "lifted"),
        ("HOLD", "held"),
    )
    words = [f"{label}:{'PASS' if result.get(key) else 'FAIL'}" for label, key in checks]
    passed = all(bool(result.get(key)) for _, key in checks)
    color = (70, 240, 90) if passed else (60, 60, 255)
    put_text(image, "Isaac 5.1 | " + "  ".join(words), (12, 130), color=color, scale=0.46)


def resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    labels = parse_mask_labels(args.mask_label)
    rgb_paths = numeric_frames(args.rgb_dir)
    if not rgb_paths:
        raise ValueError(f"no RGB frames found in {args.rgb_dir}")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    with np.load(args.observations, allow_pickle=False) as data:
        transforms = np.asarray(data["T_camera_object"], dtype=float)
        hand_joints = np.asarray(data["hand_joints_camera_m"], dtype=float)
        hand_confidence = np.asarray(data["hand_joint_confidence"], dtype=float)
        timestamps = np.asarray(data["timestamps_s"], dtype=float)
    object_points = np.asarray(np.load(args.object_points, allow_pickle=False), dtype=float)
    object_confidence = (
        np.asarray(np.load(args.object_confidence, allow_pickle=False), dtype=float)
        if args.object_confidence
        else None
    )
    hand_keypoints_2d = (
        np.asarray(np.load(args.hand_keypoints_2d, allow_pickle=False), dtype=float)
        if args.hand_keypoints_2d
        else None
    )
    hand_scores_2d = (
        np.asarray(np.load(args.hand_scores_2d, allow_pickle=False), dtype=float)
        if args.hand_scores_2d
        else None
    )
    frame_count = len(rgb_paths)
    if transforms.shape != (frame_count, 4, 4):
        raise ValueError(
            f"T_camera_object must have shape ({frame_count}, 4, 4), got {transforms.shape}"
        )
    if hand_joints.shape != (frame_count, 21, 3):
        raise ValueError(
            f"hand_joints_camera_m must have shape ({frame_count}, 21, 3), "
            f"got {hand_joints.shape}"
        )
    if hand_confidence.shape != (frame_count, 21):
        raise ValueError(
            f"hand_joint_confidence must have shape ({frame_count}, 21), "
            f"got {hand_confidence.shape}"
        )
    if (hand_keypoints_2d is None) != (hand_scores_2d is None):
        raise ValueError("2D hand keypoints and scores must be provided together")
    if hand_keypoints_2d is not None and hand_keypoints_2d.shape != (frame_count, 21, 2):
        raise ValueError(
            f"hand keypoints must have shape ({frame_count}, 21, 2), "
            f"got {hand_keypoints_2d.shape}"
        )
    if hand_scores_2d is not None and hand_scores_2d.shape != (frame_count, 21):
        raise ValueError(
            f"hand scores must have shape ({frame_count}, 21), got {hand_scores_2d.shape}"
        )
    if object_confidence is not None and object_confidence.shape != (frame_count,):
        raise ValueError(
            f"object confidence must have shape ({frame_count},), got {object_confidence.shape}"
        )

    segmentation = load_json(args.segmentation_dir / "manifest.json")
    input_manifest = load_json(args.input_manifest) if args.input_manifest else {}
    dataset = str(input_manifest.get("dataset", "calibrated"))
    camera_view = input_manifest.get("camera_view")
    input_label = f"{dataset} real RGB-D"
    if camera_view:
        input_label += f" | {camera_view}"
    depth_registration = input_manifest.get("depth_registration", {})
    if isinstance(depth_registration, dict) and depth_registration.get("method") == "image_translation":
        input_label += " | approx depth align"
    hand_manifest = load_json(args.hand_manifest) if args.hand_manifest else {}
    object_manifest = load_json(args.object_manifest) if args.object_manifest else {}
    segmentation_backend = str(
        segmentation.get("display_name", segmentation.get("backend", "segmentation"))
    )
    hand_backend = str(
        hand_manifest.get("display_name", hand_manifest.get("backend", "21-keypoint hand pose"))
    )
    object_backend = str(
        object_manifest.get("display_name", object_manifest.get("backend", "CAD 6D pose"))
    )
    segmentation_frames = {
        str(record["source"]): record["masks"] for record in segmentation["frames"]
    }
    intent = load_json(args.intent)
    intent["frame_count"] = frame_count
    selected = load_json(args.selected_grasp)
    isaac_result = load_json(args.isaac_result) if args.isaac_result else None

    first_image = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"failed to read {rgb_paths[0]}")
    source_height, source_width = first_image.shape[:2]
    panel_width = args.panel_width
    panel_height = int(round(panel_width * source_height / source_width))
    panel_height += panel_height % 2
    output_size = (2 * panel_width, 2 * panel_height)
    fps = args.fps
    if fps is None:
        fps = 1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {args.output}")

    stable_start, stable_end = (int(value) for value in intent["stable_interval"])
    preview_index = (stable_start + stable_end - 1) // 2
    preview_frame: np.ndarray | None = None
    synthetic_fixture = str(segmentation.get("backend", "")).startswith("synthetic")

    try:
        for frame_index, rgb_path in enumerate(rgb_paths):
            source = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if source is None or source.shape[:2] != (source_height, source_width):
                raise ValueError(f"invalid RGB frame: {rgb_path}")
            mask_paths = segmentation_frames.get(rgb_path.name)
            if mask_paths is None:
                raise ValueError(f"segmentation manifest has no masks for {rgb_path.name}")

            input_panel = source.copy()
            title(input_panel, f"INPUT | {input_label}")
            put_text(
                input_panel,
                f"frame {frame_index:04d}/{frame_count - 1:04d}  t={timestamps[frame_index]:.2f}s",
                (12, 58),
                scale=0.5,
            )

            segmentation_panel = source.copy()
            overlay_masks(segmentation_panel, mask_paths, args.segmentation_dir, labels)
            title(segmentation_panel, f"M1 SEGMENT | {segmentation_backend}")

            pose_panel = source.copy()
            if hand_keypoints_2d is not None and hand_scores_2d is not None:
                pixels = hand_keypoints_2d[frame_index]
                valid = np.isfinite(pixels).all(axis=1)
                draw_hand_pixels(
                    pose_panel,
                    pixels,
                    hand_scores_2d[frame_index],
                    valid,
                    args.min_joint_confidence,
                )
            else:
                draw_hand(
                    pose_panel,
                    hand_joints[frame_index],
                    hand_confidence[frame_index],
                    intrinsics,
                    args.min_joint_confidence,
                )
            draw_object_pose(
                pose_panel, transforms[frame_index], object_points, intrinsics
            )
            title(pose_panel, f"M2 {hand_backend} + M3 {object_backend}")
            translation = transforms[frame_index, :3, 3]
            score_text = (
                f"score {object_confidence[frame_index]:.3f} | "
                if object_confidence is not None
                else ""
            )
            put_text(
                pose_panel,
                score_text
                + f"t=({translation[0]:+.3f}, {translation[1]:+.3f}, "
                f"{translation[2]:+.3f})m",
                (12, 58),
                scale=0.48,
            )

            grasp_panel = source.copy()
            draw_intent(
                grasp_panel,
                frame_index,
                transforms[frame_index],
                intrinsics,
                intent,
                selected,
            )
            draw_isaac_status(grasp_panel, isaac_result)
            if intent.get("intent_type") == "power":
                title(grasp_panel, "M4 POWER GRIP + M5 PANDA CANDIDATE")
            else:
                title(grasp_panel, "M4 CONTACT + M5 GRIPPER")

            top = np.hstack(
                (
                    resize_panel(input_panel, panel_width, panel_height),
                    resize_panel(segmentation_panel, panel_width, panel_height),
                )
            )
            bottom = np.hstack(
                (
                    resize_panel(pose_panel, panel_width, panel_height),
                    resize_panel(grasp_panel, panel_width, panel_height),
                )
            )
            grid = np.vstack((top, bottom))
            if synthetic_fixture:
                banner = "SYNTHETIC CONTRACT DEMO - NOT MODEL OUTPUT"
                cv2.rectangle(
                    grid,
                    (0, output_size[1] - 32),
                    (output_size[0], output_size[1]),
                    (20, 20, 20),
                    -1,
                )
                text_size = cv2.getTextSize(
                    banner, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                )[0]
                put_text(
                    grid,
                    banner,
                    ((output_size[0] - text_size[0]) // 2, output_size[1] - 9),
                    color=(80, 210, 255),
                    scale=0.65,
                    thickness=2,
                )
            writer.write(grid)
            if frame_index == preview_index:
                preview_frame = grid.copy()
    finally:
        writer.release()

    if args.preview:
        if preview_frame is None:
            preview_frame = grid
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), preview_frame):
            raise RuntimeError(f"failed to write preview {args.preview}")
    print(f"Wrote {frame_count} annotated frames to {args.output}")
    if args.preview:
        print(f"Preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
