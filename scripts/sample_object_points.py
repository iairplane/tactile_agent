#!/usr/bin/env python3
"""Sample a CAD surface in its original object frame for contact projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-unit-scale", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=50_000)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"mesh has no triangle faces: {args.mesh}")
    points, _ = trimesh.sample.sample_surface(mesh, args.count, seed=7)
    points = np.asarray(points, dtype=np.float64) * args.mesh_unit_scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points)


if __name__ == "__main__":
    main()
