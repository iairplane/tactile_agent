import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_saved_warped_reference_distinguishes_local_contact_from_lighting():
    path = Path(__file__).resolve().parents[1] / "scripts/monitor_test_tube_held_contact.py"
    spec = importlib.util.spec_from_file_location("held_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = np.full((240, 240, 3), 100, dtype=np.uint8)
    roi = np.ones((240, 240), dtype=bool)
    classify = module.classify_warped_contact
    assert not classify(reference, reference, roi, 18., 70)["contact"]
    assert not classify(reference + 10, reference, roi, 18., 70)["contact"]
    loaded = reference.copy()
    loaded[100:120, 100:120] = 170
    result = classify(loaded, reference, roi, 18., 70)
    assert result["contact"]
    assert result["component_area_px"] >= 70
    with pytest.raises(ValueError, match="shape"):
        classify(loaded[:100], reference, roi, 18., 70)
