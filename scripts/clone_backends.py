#!/usr/bin/env python3
"""Clone the upstream backends at the revisions recorded in the lock file."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "backends.lock.json"
DEFAULT_ROOT = PROJECT_ROOT / "third_party"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--names",
        nargs="+",
        help="backend keys from the lock file; omit to clone every locked backend",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list locked backend keys and revisions without cloning",
    )
    return parser.parse_args()


def load_backends(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        backends = data["backends"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(f"cannot read backend lock {path}: {exc}") from exc
    if not isinstance(backends, dict):
        raise SystemExit(f"{path}: 'backends' must be an object")
    for key, backend in backends.items():
        if not isinstance(backend, dict):
            raise SystemExit(f"{path}: backend {key!r} must be an object")
        for field in ("name", "repository", "revision"):
            if not isinstance(backend.get(field), str) or not backend[field]:
                raise SystemExit(f"{path}: backend {key!r} has no valid {field!r}")
    return backends


def run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        raise SystemExit(
            f"command failed with exit code {exc.returncode}: {joined}"
        ) from exc


def main() -> int:
    args = parse_args()
    backends = load_backends(args.lock.resolve())

    if args.list:
        for key, backend in backends.items():
            print(f"{key:28} {backend['revision']}  {backend['name']}")
        return 0

    selected = args.names or list(backends)
    unknown = sorted(set(selected) - set(backends))
    if unknown:
        available = ", ".join(backends)
        raise SystemExit(f"unknown backend key(s): {', '.join(unknown)}; available: {available}")

    destination_root = args.root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    for key in selected:
        backend = backends[key]
        destination = destination_root / key
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise SystemExit(
                    f"refusing to overwrite existing path: {destination}"
                )

        print(f"cloning {backend['name']} into {destination}")
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                backend["repository"],
                str(destination),
            ]
        )
        run(["git", "-C", str(destination), "checkout", "--detach", backend["revision"]])
        if backend.get("submodules"):
            run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
