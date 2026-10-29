import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import record_front_rgbd_tactile as recorder  # noqa: E402


def _pair(frame_number: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        color_frame_number=frame_number,
        depth_frame_number=frame_number + 100,
        color_sensor_timestamp_s=12.0,
        depth_sensor_timestamp_s=12.001,
        color_sdk_timestamp_s=22.0,
        depth_sdk_timestamp_s=22.001,
        sdk_timestamp_domain="hardware_clock",
        sensor_skew_ms=1.0,
        sdk_skew_ms=1.0,
    )


def test_recording_session_publishes_lossless_raw_and_metric_data(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    session = recorder.RecordingSession(
        output_dir=output,
        depth_scale_m=0.001,
        intrinsics=object(),
        d435_device={"serial": "254622073804"},
        gf_a_path=Path("/dev/video0"),
        gf_b_path=Path("/dev/video10"),
        fps=30,
        max_tactile_age_ms=250.0,
        max_d435_frame_gap=3,
        max_pending_writes=2,
        writer_threads=1,
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 17
    depth_raw = np.full((480, 640), 1234, dtype=np.uint16)
    tactile_a = np.zeros((480, 640, 3), dtype=np.uint8)
    tactile_a[..., 1] = 31
    tactile_b = np.zeros((480, 640, 3), dtype=np.uint8)
    tactile_b[..., 2] = 47
    session.append(
        recorder.Observation(
            rgb=rgb,
            depth_raw=depth_raw,
            pair=_pair(),
            d435_host_monotonic_s=100.0,
            tactile_a=recorder.TactileSnapshot(tactile_a, 99.99, 5),
            tactile_b=recorder.TactileSnapshot(tactile_b, 100.01, 8),
        )
    )
    k = np.asarray([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    session.finalize({"fx": 600.0, "fy": 600.0}, k)

    manifest = json.loads((output / "input_manifest.json").read_text())
    assert manifest["schema"] == recorder.SCHEMA
    assert len(manifest["recording_session_id"]) == 32
    assert set(manifest["recording_session_id"]) <= set("0123456789abcdef")
    assert manifest["frames"][0]["tactile_a_sequence"] == 5
    assert manifest["frames"][0]["tactile_b_sequence"] == 8
    assert np.array_equal(np.load(output / "depth_raw/000000.npy"), depth_raw)
    assert np.allclose(np.load(output / "depth/000000.npy"), 1.234)
    assert np.array_equal(cv2.imread(str(output / "tactile_a/000000.png")), tactile_a)
    assert np.array_equal(cv2.imread(str(output / "tactile_b/000000.png")), tactile_b)
    assert np.array_equal(np.load(output / "K.npy"), k)


def test_nonempty_output_target_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    with pytest.raises(ValueError, match="must be empty"):
        recorder.validate_output_target(output)
