import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_converts_graspgenx_wxyz_pose(tmp_path: Path) -> None:
    input_path = tmp_path / "grasps.yml"
    output_path = tmp_path / "candidates.npz"
    input_path.write_text(
        yaml.safe_dump(
            {
                "format": "isaac_grasp",
                "format_version": "1.0",
                "grasps": {
                    "grasp_0": {
                        "confidence": 0.8,
                        "position": [0.1, 0.2, 0.3],
                        "orientation": {
                            "w": np.sqrt(0.5).item(),
                            "xyz": [0.0, 0.0, np.sqrt(0.5).item()],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/convert_graspgenx.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--pregrasp-width-m",
            "0.06",
            "--tcp-offset-m",
            "0.10",
        ],
        check=True,
    )

    with np.load(output_path) as candidates:
        np.testing.assert_allclose(candidates["poses"][0, :3, 3], [0.1, 0.2, 0.4])
        np.testing.assert_allclose(
            candidates["poses"][0, :3, :3],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            atol=1e-12,
        )
        np.testing.assert_allclose(candidates["scores"], [0.8])
        np.testing.assert_allclose(candidates["widths_m"], [0.06])
