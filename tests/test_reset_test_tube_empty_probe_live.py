from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reset_test_tube_empty_probe_live",
    ROOT / "scripts" / "reset_test_tube_empty_probe_live.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source() -> object:
    return MODULE.EmptyProbeSource(
        report_path=Path("/tmp/failed/run_report.json"),
        session_id="failed",
        deepest_tcp=np.eye(4),
        closed_width_m=0.01015625,
        report_sha256="0" * 64,
    )


def test_retreat_target_is_exact_base_plus_z_with_fixed_orientation() -> None:
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 1.0]).as_matrix()
    start[:3, 3] = [0.55, 0.19, 0.158]
    target = MODULE.retreat_target(start)
    np.testing.assert_allclose(target[:3, :3], start[:3, :3])
    np.testing.assert_allclose(target[:3, 3], start[:3, 3] + [0.0, 0.0, 0.08])


def test_live_gate_accepts_exact_stationary_empty_close() -> None:
    feedback = {
        "width_m": 0.01015625,
        "moving": False,
        "fault_free": True,
        "object_detected": False,
    }
    validated = MODULE.validate_live_empty_probe(source(), np.eye(4), feedback)
    assert validated["width_m"] == pytest.approx(0.01015625)
    assert validated["object_detected"] is False


@pytest.mark.parametrize(
    "update",
    (
        {"width_m": 0.02},
        {"moving": True},
        {"fault_free": False},
        {"object_detected": True},
    ),
)
def test_live_gate_rejects_nonempty_or_mismatched_state(
    update: dict[str, object],
) -> None:
    feedback = {
        "width_m": 0.01015625,
        "moving": False,
        "fault_free": True,
        "object_detected": False,
    }
    feedback.update(update)
    with pytest.raises(RuntimeError, match="recorded empty close"):
        MODULE.validate_live_empty_probe(source(), np.eye(4), feedback)


def test_current_failed_run_is_exact_empty_two_probe_contract() -> None:
    failed = ROOT / "runs/live/test_tube_shallow_live_resume_20261024_202819"
    config = ROOT / "configs/test_tube_closed_loop.demo07.json"
    if not failed.is_dir():
        pytest.skip("reviewed failed run is unavailable")
    evidence = MODULE.load_empty_probe_source(
        failed,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert evidence.closed_width_m == pytest.approx(0.01015625)
    assert evidence.deepest_tcp[2, 3] == pytest.approx(0.15776336980624128)

