from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    script = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_fast_probe.py"
    spec = importlib.util.spec_from_file_location("run_test_tube_fast_probe", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(*, active: tuple[str, ...], ready: bool):
    return SimpleNamespace(valid=True, active_serials=active, grasp_ready=ready)


def test_default_probe_descends_twenty_mm() -> None:
    module = _load_script()
    args = module.parse_args(["--config", "unused.json", "--output-dir", "unused"])
    assert args.down_mm == pytest.approx(20.0)
    target = module.descent_target(np.eye(4), args.down_mm)
    assert target[2, 3] == pytest.approx(-0.020)


def test_probe_depth_is_bounded() -> None:
    module = _load_script()
    with pytest.raises(ValueError, match=r"\(0, 25\]"):
        module.descent_target(np.eye(4), 25.1)


def test_stopped_grasp_requires_two_bilateral_windows_and_grasp_ready() -> None:
    module = _load_script()
    both = ("left", "right")
    assert module.stopped_contact_kind(
        [_decision(active=both, ready=False), _decision(active=both, ready=True)]
    ) == "grasp_ready"
    assert module.stopped_contact_kind(
        [_decision(active=both, ready=False), _decision(active=both, ready=False)]
    ) == "bilateral_contact_not_confirmed"
    assert module.stopped_contact_kind(
        [
            _decision(active=("right",), ready=False),
            _decision(active=("right",), ready=False),
        ]
    ) == "bilateral_contact_not_confirmed"
