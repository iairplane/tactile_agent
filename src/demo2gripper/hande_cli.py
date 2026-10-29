"""Fail-closed standalone control for the USB/Modbus Robotiq Hand-E.

This module deliberately imports only LeRobot's Hand-E implementation.  It does
not construct a Franka robot object and therefore never opens an FCI connection
to the FR3.  The ``status`` path uses the bundled SDK directly because closing a
``RobotiqHandE`` worker sends a stop register; status must remain strictly
read-only.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_WIDTH_MM = 50.0
DEFAULT_SPEED_REGISTER = 50
DEFAULT_FORCE_REGISTER = 40
MAX_SPEED_REGISTER = 100
MAX_FORCE_REGISTER = 70
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MODBUS_TIMEOUT_S = 0.25
DEFAULT_POLL_INTERVAL_S = 0.025
DEFAULT_ENDPOINT_TOLERANCE_MM = 2.0
DEFAULT_CLOSED_ENDPOINT_TOLERANCE_MM = 2.5

OBJECT_STATUS_NAMES = {
    0: "moving_no_object",
    1: "object_detected_opening",
    2: "grasped_object_detected_closing",
    3: "motion_complete_no_object",
}


class HandECliError(RuntimeError):
    """Expected configuration, connection, feedback, or motion failure."""


def _default_lerobot_src() -> Path:
    configured = os.environ.get("LEROBOT_SRC")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "fhh" / "lerobot" / "src"


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _register_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer register value") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read or command the USB/Modbus Hand-E without constructing an FR3/FCI client. "
            "Every command requires an explicit serial port."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_connection_options(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--port",
            required=True,
            help="Explicit serial device, preferably /dev/serial/by-id/...",
        )
        command_parser.add_argument(
            "--lerobot-src",
            type=Path,
            default=_default_lerobot_src(),
            help="LeRobot source directory containing the corrected Hand-E driver",
        )
        command_parser.add_argument(
            "--modbus-timeout-s",
            type=_positive_finite,
            default=DEFAULT_MODBUS_TIMEOUT_S,
        )

    status = commands.add_parser(
        "status",
        help="Read one feedback sample; never activate, move, or stop the fingers",
    )
    add_connection_options(status)

    for name in ("open", "close", "move"):
        motion = commands.add_parser(name, help=f"Command the Hand-E to {name}")
        add_connection_options(motion)
        motion.add_argument(
            "--speed",
            type=_register_value,
            default=DEFAULT_SPEED_REGISTER,
            help=f"Raw Hand-E speed register, capped at {MAX_SPEED_REGISTER}",
        )
        motion.add_argument(
            "--force",
            type=_register_value,
            default=DEFAULT_FORCE_REGISTER,
            help=f"Raw Hand-E force register, capped at {MAX_FORCE_REGISTER}",
        )
        motion.add_argument(
            "--timeout-s",
            type=_positive_finite,
            default=DEFAULT_TIMEOUT_S,
        )
        motion.add_argument(
            "--activate",
            action="store_true",
            help=(
                "Explicitly allow activation if the Hand-E is inactive; activation can move "
                "the fingers. The default is to fail without activating."
            ),
        )
        if name == "move":
            motion.add_argument(
                "--opening-mm",
                type=float,
                required=True,
                help=f"Absolute requested finger opening in millimetres (0..{MAX_WIDTH_MM:g})",
            )
    return parser


def validate_port(port: str) -> str:
    path = Path(port).expanduser()
    if not path.is_absolute():
        raise HandECliError("--port must be an explicit absolute serial-device path")
    return str(path)


def validate_motion_limits(speed: int, force: int) -> None:
    if not 0 <= speed <= MAX_SPEED_REGISTER:
        raise HandECliError(
            f"speed register {speed} exceeds the conservative range 0..{MAX_SPEED_REGISTER}"
        )
    if not 0 <= force <= MAX_FORCE_REGISTER:
        raise HandECliError(
            f"force register {force} exceeds the conservative range 0..{MAX_FORCE_REGISTER}"
        )


def _load_hande_types(lerobot_src: Path) -> tuple[type[Any], type[Any]]:
    source = lerobot_src.expanduser().resolve()
    expected_driver = (
        source / "lerobot" / "robots" / "franka_fr3" / "grippers" / "robotiq_hande.py"
    )
    if not expected_driver.is_file():
        raise HandECliError(f"corrected Hand-E driver not found at {expected_driver}")
    sys.path.insert(0, str(source))
    try:
        driver_module = importlib.import_module(
            "lerobot.robots.franka_fr3.grippers.robotiq_hande"
        )
        sdk_module = importlib.import_module("lerobot.robots.franka_fr3.grippers.robotiq_sdk")
    except ModuleNotFoundError as exc:
        if exc.name in {"modbus_tk", "serial"}:
            raise HandECliError(
                f"missing Hand-E serial dependency {exc.name!r}; install modbus-tk and pyserial "
                "in the Python environment running this command"
            ) from exc
        raise HandECliError(f"could not import the corrected Hand-E driver: {exc}") from exc

    loaded_driver = Path(driver_module.__file__).resolve()
    if loaded_driver != expected_driver.resolve():
        raise HandECliError(
            f"loaded Hand-E driver {loaded_driver}, expected the corrected driver "
            f"{expected_driver.resolve()}"
        )
    return driver_module.RobotiqHandE, sdk_module.HandEForRtu


def _status_tuple(feedback: Any) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(value) for value in feedback.status)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HandECliError("Hand-E returned malformed status feedback") from exc
    if len(values) != 4:
        raise HandECliError(f"Hand-E returned {len(values)} status fields; expected 4")
    return values  # type: ignore[return-value]


def _object_interpretation(
    object_status: int,
    closure_mm: float,
    *,
    command: str | None,
    endpoint_tolerance_mm: float = DEFAULT_ENDPOINT_TOLERANCE_MM,
    closed_endpoint_tolerance_mm: float = DEFAULT_CLOSED_ENDPOINT_TOLERANCE_MM,
) -> dict[str, Any]:
    grasped = object_status == 2
    fully_closed_without_object = (
        object_status == 3
        and closure_mm >= MAX_WIDTH_MM - closed_endpoint_tolerance_mm
    )
    fully_open = object_status == 3 and closure_mm <= endpoint_tolerance_mm
    meaning = OBJECT_STATUS_NAMES.get(object_status, "unknown")
    if command == "close" and fully_closed_without_object:
        meaning = "fully_closed_no_object"
    elif command == "open" and fully_open:
        meaning = "fully_open_no_object"
    return {
        "code": object_status,
        "meaning": meaning,
        "grasped": grasped,
        "fully_closed_without_object": fully_closed_without_object,
        "fully_open": fully_open,
    }


def feedback_payload(feedback: Any, *, command: str | None = None) -> dict[str, Any]:
    activation, go_to, activation_status, object_status = _status_tuple(feedback)
    closure_mm = float(feedback.position_mm)
    requested_closure_mm = float(feedback.requested_position_mm)
    current = float(feedback.current)
    for label, value in (
        ("position_mm", closure_mm),
        ("requested_position_mm", requested_closure_mm),
        ("current", current),
    ):
        if not math.isfinite(value):
            raise HandECliError(f"Hand-E returned non-finite {label}")
    return {
        "activated": activation == 1 and activation_status == 3,
        "activation_bit": activation,
        "go_to_bit": go_to,
        "activation_status": activation_status,
        "closure_mm": closure_mm,
        "opening_width_mm": MAX_WIDTH_MM - closure_mm,
        "requested_closure_mm": requested_closure_mm,
        "current_register": current,
        "fault_code": int(feedback.fault_code),
        "object_status": _object_interpretation(
            object_status,
            closure_mm,
            command=command,
        ),
    }


def driver_state_payload(state: Any, *, command: str) -> dict[str, Any]:
    closure_mm = float(state.position_mm)
    current = float(state.current)
    timestamp = float(state.timestamp)
    for label, value in (
        ("position_mm", closure_mm),
        ("current", current),
        ("timestamp", timestamp),
    ):
        if not math.isfinite(value):
            raise HandECliError(f"Hand-E returned non-finite {label}")
    return {
        "connected": bool(state.connected),
        "normalized_closure": float(state.position),
        "closure_mm": closure_mm,
        "opening_width_mm": MAX_WIDTH_MM - closure_mm,
        "requested_normalized_closure": float(state.requested_position),
        "current_register": current,
        "fault_code": int(state.fault_code),
        "timestamp_unix_s": timestamp,
        "object_status": _object_interpretation(
            int(state.object_status),
            closure_mm,
            command=command,
        ),
    }


def read_status(
    *,
    port: str,
    sdk_factory: Callable[..., Any],
    modbus_timeout_s: float,
) -> dict[str, Any]:
    """Read feedback without issuing activation, motion, or stop writes."""

    hand = sdk_factory(port=port, autoInit=False, timeout=modbus_timeout_s)
    try:
        feedback = hand.read_feedback()
        return feedback_payload(feedback)
    finally:
        # HandEForRtu.close only closes the serial transport.  In particular,
        # do not call stop(): that writes REQUEST_ACTION and would make status
        # cease to be a read-only operation.
        hand.close()


def wait_for_motion(
    driver: Any,
    *,
    command: str,
    timeout_s: float,
    target_closure_mm: float | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    deadline = clock() + timeout_s
    last_state = None
    while clock() < deadline:
        state = driver.get_state()
        last_state = state
        if not state.connected:
            raise HandECliError("Hand-E feedback connection was lost during motion")
        if int(state.fault_code) != 0:
            raise HandECliError(f"Hand-E reported fault code {int(state.fault_code)}")

        object_status = int(state.object_status)
        closure_mm = float(state.position_mm)
        if command == "close":
            # gOBJ=2 is an object detected while closing: this is a grasp.
            if object_status == 2:
                return state
            # gOBJ=3 means the requested position was reached without an
            # object. It counts as completion only when the fingers actually
            # reached the closed endpoint; it must never be called a grasp.
            if (
                object_status == 3
                and closure_mm >= MAX_WIDTH_MM - DEFAULT_CLOSED_ENDPOINT_TOLERANCE_MM
            ):
                return state
        elif command == "open":
            if object_status == 1:
                raise HandECliError("Hand-E detected an obstruction while opening")
            if object_status == 3 and closure_mm <= DEFAULT_ENDPOINT_TOLERANCE_MM:
                return state
        elif command == "move":
            if target_closure_mm is None:
                raise ValueError("move requires target_closure_mm")
            # gOBJ=2 is useful probe evidence: the fingers stopped on contact
            # before reaching the requested absolute opening.
            if object_status == 2:
                return state
            if object_status == 1:
                raise HandECliError("Hand-E detected an obstruction while opening")
            if (
                object_status == 3
                and abs(closure_mm - target_closure_mm)
                <= DEFAULT_ENDPOINT_TOLERANCE_MM
            ):
                return state
        else:  # pragma: no cover - guarded by the CLI and execute_motion
            raise ValueError(f"unknown Hand-E command {command!r}")
        sleep(poll_interval_s)

    detail = "no feedback" if last_state is None else (
        f"last object_status={int(last_state.object_status)}, "
        f"closure_mm={float(last_state.position_mm):.3f}"
    )
    raise HandECliError(f"Hand-E {command} timed out after {timeout_s:.2f}s ({detail})")


def execute_motion(
    *,
    command: str,
    port: str,
    driver_factory: Callable[..., Any],
    speed: int,
    force: int,
    timeout_s: float,
    modbus_timeout_s: float,
    activate: bool,
    opening_mm: float | None = None,
) -> dict[str, Any]:
    if command not in {"open", "close", "move"}:
        raise ValueError(f"unsupported Hand-E motion {command!r}")
    if command == "move":
        if opening_mm is None or not math.isfinite(float(opening_mm)):
            raise HandECliError("move requires a finite --opening-mm")
        if not 0.0 <= float(opening_mm) <= MAX_WIDTH_MM:
            raise HandECliError(f"--opening-mm must lie in 0..{MAX_WIDTH_MM:g}")
    elif opening_mm is not None:
        raise ValueError("opening_mm is only valid for move")
    validate_motion_limits(speed, force)
    driver = driver_factory(
        port=port,
        max_width_mm=MAX_WIDTH_MM,
        speed=speed,
        force=force,
        state_frequency=20.0,
        startup_timeout_s=timeout_s,
        modbus_timeout_s=modbus_timeout_s,
        auto_activate=activate,
        binary_latch=False,
        command_deadband=0.0,
    )
    try:
        if not driver.connect():
            detail = getattr(driver, "_connect_error", None)
            suffix = f": {detail}" if detail is not None else ""
            raise HandECliError(f"could not connect to an activated Hand-E{suffix}")
        if command == "open":
            target_closure_mm = 0.0
        elif command == "close":
            target_closure_mm = MAX_WIDTH_MM
        else:
            assert opening_mm is not None
            target_closure_mm = MAX_WIDTH_MM - float(opening_mm)
        driver.command(target_closure_mm / MAX_WIDTH_MM)
        state = wait_for_motion(
            driver,
            command=command,
            timeout_s=timeout_s,
            target_closure_mm=target_closure_mm,
        )
        payload = driver_state_payload(state, command=command)
        if command == "move":
            payload["requested_opening_width_mm"] = float(opening_mm)
            payload["requested_closure_mm"] = target_closure_mm
        return payload
    finally:
        driver.disconnect()


def run(
    args: argparse.Namespace,
    *,
    type_loader: Callable[[Path], tuple[type[Any], type[Any]]],
) -> dict:
    port = validate_port(args.port)
    driver_factory, sdk_factory = type_loader(args.lerobot_src)
    base = {
        "schema": "demo2gripper.hande_cli.v1",
        "ok": True,
        "command": args.command,
        "port": port,
        "fr3_fci_opened": False,
        "driver_source": str(args.lerobot_src.expanduser().resolve()),
    }
    if args.command == "status":
        base["read_only"] = True
        base["state"] = read_status(
            port=port,
            sdk_factory=sdk_factory,
            modbus_timeout_s=args.modbus_timeout_s,
        )
        return base

    base.update(
        {
            "read_only": False,
            "activation_requested": bool(args.activate),
            "speed_register": args.speed,
            "force_register": args.force,
            "safety_caps": {
                "max_speed_register": MAX_SPEED_REGISTER,
                "max_force_register": MAX_FORCE_REGISTER,
            },
        }
    )
    base["state"] = execute_motion(
        command=args.command,
        port=port,
        driver_factory=driver_factory,
        speed=args.speed,
        force=args.force,
        timeout_s=args.timeout_s,
        modbus_timeout_s=args.modbus_timeout_s,
        activate=args.activate,
        opening_mm=(args.opening_mm if args.command == "move" else None),
    )
    return base


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # The bundled Modbus retry helper prints diagnostics. Keep stdout as
        # exactly one machine-readable JSON object and route such diagnostics
        # to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args, type_loader=_load_hande_types)
    except (HandECliError, OSError, RuntimeError) as exc:
        result = {
            "schema": "demo2gripper.hande_cli.v1",
            "ok": False,
            "command": getattr(args, "command", None),
            "port": getattr(args, "port", None),
            "fr3_fci_opened": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
