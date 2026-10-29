"""Small dependency-light Robotiq Hand-E CLI over Modbus RTU.

This module deliberately avoids importing LeRobot.  ``status`` performs one
read-only Modbus transaction; motion commands leave the Hand-E's go-to command
latched when the serial transport closes.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLAVE_ID = 9
BAUDRATE = 115200
REQUEST_ACTION = 0x03E8
GRIPPER_STATUS = 0x07D0
ACTIVE_STOP_ACTION = 0x0100
ACTIVE_GO_TO_ACTION = 0x0900
MAX_WIDTH_MM = 50.0
POS_MM_PER_COUNT = 0.1953125
DEFAULT_SPEED_REGISTER = 50
DEFAULT_FORCE_REGISTER = 40
MAX_SPEED_REGISTER = 100
MAX_FORCE_REGISTER = 70
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MODBUS_TIMEOUT_S = 0.25
POLL_INTERVAL_S = 0.025


class HandEModbusError(RuntimeError):
    """Invalid configuration, Modbus frame, feedback, or motion result."""


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(payload: bytes) -> bytes:
    crc = crc16_modbus(payload)
    return payload + bytes((crc & 0xFF, crc >> 8))


def _validate_crc(frame: bytes) -> bytes:
    if len(frame) < 4:
        raise HandEModbusError("short Modbus RTU frame")
    expected = frame[-2] | (frame[-1] << 8)
    actual = crc16_modbus(frame[:-2])
    if actual != expected:
        raise HandEModbusError(
            f"Modbus CRC mismatch: received 0x{expected:04x}, computed 0x{actual:04x}"
        )
    return frame[:-2]


def build_read_holding_request(address: int, count: int) -> bytes:
    return with_crc(struct.pack(">BBHH", SLAVE_ID, 0x03, address, count))


def parse_read_holding_response(frame: bytes, *, count: int) -> tuple[int, ...]:
    payload = _validate_crc(frame)
    if len(payload) >= 2 and payload[1] == 0x83:
        code = payload[2] if len(payload) > 2 else -1
        raise HandEModbusError(f"Modbus read exception {code}")
    expected_length = 3 + count * 2
    if len(payload) != expected_length or payload[:3] != bytes((SLAVE_ID, 0x03, count * 2)):
        raise HandEModbusError("malformed Modbus read response")
    return struct.unpack(f">{count}H", payload[3:])


def build_write_multiple_request(address: int, registers: tuple[int, ...]) -> bytes:
    if not registers:
        raise ValueError("at least one register is required")
    data = struct.pack(f">{len(registers)}H", *registers)
    payload = struct.pack(">BBHHB", SLAVE_ID, 0x10, address, len(registers), len(data)) + data
    return with_crc(payload)


def parse_write_multiple_response(
    frame: bytes, *, address: int, count: int
) -> None:
    payload = _validate_crc(frame)
    if len(payload) >= 2 and payload[1] == 0x90:
        code = payload[2] if len(payload) > 2 else -1
        raise HandEModbusError(f"Modbus write exception {code}")
    expected = struct.pack(">BBHH", SLAVE_ID, 0x10, address, count)
    if payload != expected:
        raise HandEModbusError("malformed Modbus write response")


def build_write_single_request(address: int, value: int) -> bytes:
    return with_crc(struct.pack(">BBHH", SLAVE_ID, 0x06, address, value))


def parse_write_single_response(frame: bytes, *, address: int, value: int) -> None:
    payload = _validate_crc(frame)
    expected = struct.pack(">BBHH", SLAVE_ID, 0x06, address, value)
    if payload != expected:
        raise HandEModbusError("malformed Modbus single-register response")


@dataclass(frozen=True)
class Feedback:
    activation_bit: int
    go_to_bit: int
    activation_status: int
    object_status: int
    fault_code: int
    requested_count: int
    position_count: int
    current: int


def parse_feedback_registers(registers: tuple[int, ...]) -> Feedback:
    if len(registers) != 3:
        raise HandEModbusError("Hand-E feedback must contain three registers")
    status = registers[0] >> 8
    return Feedback(
        activation_bit=status & 0x01,
        go_to_bit=(status >> 3) & 0x01,
        activation_status=(status >> 4) & 0x03,
        object_status=(status >> 6) & 0x03,
        fault_code=registers[1] >> 8,
        requested_count=registers[1] & 0xFF,
        position_count=registers[2] >> 8,
        current=registers[2] & 0xFF,
    )


class SerialModbusRTU:
    def __init__(self, port: str, timeout_s: float) -> None:
        try:
            import serial
        except ModuleNotFoundError as exc:  # pragma: no cover - host dependency
            raise HandEModbusError("pyserial is required for Hand-E control") from exc
        self.serial = serial.Serial(
            port=port,
            baudrate=BAUDRATE,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=timeout_s,
            write_timeout=timeout_s,
        )

    def close(self) -> None:
        self.serial.close()

    def _exchange(self, request: bytes, response_length: int) -> bytes:
        self.serial.reset_input_buffer()
        self.serial.write(request)
        self.serial.flush()
        response = self.serial.read(response_length)
        if len(response) != response_length:
            raise HandEModbusError(
                f"short Modbus response: got {len(response)}, expected {response_length}"
            )
        return response

    def read_feedback(self) -> Feedback:
        frame = self._exchange(build_read_holding_request(GRIPPER_STATUS, 3), 11)
        return parse_feedback_registers(parse_read_holding_response(frame, count=3))

    def write_motion(self, target_count: int, speed: int, force: int) -> None:
        registers = (ACTIVE_GO_TO_ACTION, target_count, (speed << 8) | force)
        response = self._exchange(
            build_write_multiple_request(REQUEST_ACTION, registers), 8
        )
        parse_write_multiple_response(response, address=REQUEST_ACTION, count=3)

    def write_action(self, value: int) -> None:
        response = self._exchange(build_write_single_request(REQUEST_ACTION, value), 8)
        parse_write_single_response(response, address=REQUEST_ACTION, value=value)

    def stop_motion(self) -> None:
        """Stop finger motion while leaving the Hand-E activated.

        The Hand-E action byte places ``rACT`` at bit 0 and ``rGTO`` at bit 3.
        In the first Modbus register this is ``0x0100``: activation remains set
        while go-to is cleared.  This is a stop/hold command, not deactivation
        and not an open command.
        """

        self.write_action(ACTIVE_STOP_ACTION)


def _object_payload(feedback: Feedback, command: str | None) -> dict[str, Any]:
    closure_mm = feedback.position_count * POS_MM_PER_COUNT
    code = feedback.object_status
    fully_closed = code == 3 and closure_mm >= MAX_WIDTH_MM - 2.5
    fully_open = code == 3 and closure_mm <= 2.0
    meanings = {
        0: "moving_no_object",
        1: "object_detected_opening",
        2: "grasped_object_detected_closing",
        3: "motion_complete_no_object",
    }
    meaning = meanings.get(code, "unknown")
    if command == "close" and fully_closed:
        meaning = "fully_closed_no_object"
    elif command == "open" and fully_open:
        meaning = "fully_open_no_object"
    return {
        "code": code,
        "meaning": meaning,
        "grasped": code == 2,
        "fully_closed_without_object": fully_closed,
        "fully_open": fully_open,
    }


def feedback_payload(feedback: Feedback, *, command: str | None = None) -> dict[str, Any]:
    closure_mm = feedback.position_count * POS_MM_PER_COUNT
    return {
        "connected": True,
        "activated": feedback.activation_bit == 1 and feedback.activation_status == 3,
        "activation_bit": feedback.activation_bit,
        "go_to_bit": feedback.go_to_bit,
        "activation_status": feedback.activation_status,
        "normalized_closure": closure_mm / MAX_WIDTH_MM,
        "closure_mm": closure_mm,
        "opening_width_mm": MAX_WIDTH_MM - closure_mm,
        "requested_normalized_closure": feedback.requested_count / 255.0,
        "requested_closure_mm": feedback.requested_count * POS_MM_PER_COUNT,
        "current_register": feedback.current,
        "fault_code": feedback.fault_code,
        "timestamp_unix_s": time.time(),
        "object_status": _object_payload(feedback, command),
    }


def _target_count(command: str, opening_mm: float | None) -> tuple[int, float]:
    if command == "open":
        target_closure_mm = 0.0
    elif command == "close":
        target_closure_mm = MAX_WIDTH_MM
    else:
        if opening_mm is None or not math.isfinite(opening_mm) or not 0 <= opening_mm <= 50:
            raise HandEModbusError("--opening-mm must be finite and lie in 0..50")
        target_closure_mm = MAX_WIDTH_MM - opening_mm
    return min(255, max(0, int(target_closure_mm / POS_MM_PER_COUNT))), target_closure_mm


def _wait_motion(
    transport: SerialModbusRTU, *, command: str, target_count: int, timeout_s: float
) -> Feedback:
    deadline = time.monotonic() + timeout_s
    last: Feedback | None = None
    while time.monotonic() < deadline:
        last = transport.read_feedback()
        if last.fault_code:
            raise HandEModbusError(f"Hand-E reported fault code {last.fault_code}")
        if command != "open" and last.object_status == 2:
            return last
        if last.object_status == 1:
            raise HandEModbusError("Hand-E detected an obstruction while opening")
        tolerance_counts = math.ceil(2.0 / POS_MM_PER_COUNT)
        if last.object_status == 3 and abs(last.position_count - target_count) <= tolerance_counts:
            return last
        time.sleep(POLL_INTERVAL_S)
    detail = "no feedback" if last is None else (
        f"gOBJ={last.object_status}, position_count={last.position_count}"
    )
    raise HandEModbusError(f"Hand-E {command} timed out after {timeout_s:.2f}s ({detail})")


def _activate(transport: SerialModbusRTU, timeout_s: float) -> None:
    transport.write_action(0x0000)
    time.sleep(0.1)
    transport.write_action(ACTIVE_STOP_ACTION)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        feedback = transport.read_feedback()
        if feedback.activation_bit == 1 and feedback.activation_status == 3:
            return
        time.sleep(POLL_INTERVAL_S)
    raise HandEModbusError("Hand-E activation timed out")


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct pyserial Hand-E Modbus RTU CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "open", "close", "move"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--port", required=True)
        sub.add_argument("--modbus-timeout-s", type=_positive_float, default=0.25)
        sub.add_argument("--lerobot-src", type=Path, help=argparse.SUPPRESS)
        if name != "status":
            sub.add_argument("--speed", type=int, default=DEFAULT_SPEED_REGISTER)
            sub.add_argument("--force", type=int, default=DEFAULT_FORCE_REGISTER)
            sub.add_argument("--timeout-s", type=_positive_float, default=DEFAULT_TIMEOUT_S)
            sub.add_argument("--activate", action="store_true")
        if name == "move":
            sub.add_argument("--opening-mm", type=float, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    port = Path(args.port).expanduser()
    if not port.is_absolute():
        raise HandEModbusError("--port must be an absolute serial-device path")
    transport = SerialModbusRTU(str(port), args.modbus_timeout_s)
    try:
        if args.command == "status":
            state = feedback_payload(transport.read_feedback())
        else:
            if not 0 <= args.speed <= MAX_SPEED_REGISTER:
                raise HandEModbusError(f"--speed must lie in 0..{MAX_SPEED_REGISTER}")
            if not 0 <= args.force <= MAX_FORCE_REGISTER:
                raise HandEModbusError(f"--force must lie in 0..{MAX_FORCE_REGISTER}")
            initial = transport.read_feedback()
            if not (initial.activation_bit == 1 and initial.activation_status == 3):
                if not args.activate:
                    raise HandEModbusError("Hand-E is not activated")
                _activate(transport, args.timeout_s)
            opening = args.opening_mm if args.command == "move" else None
            target, target_closure_mm = _target_count(args.command, opening)
            transport.write_motion(target, args.speed, args.force)
            state = feedback_payload(
                _wait_motion(
                    transport,
                    command=args.command,
                    target_count=target,
                    timeout_s=args.timeout_s,
                ),
                command=args.command,
            )
            state["requested_closure_mm"] = target_closure_mm
            if args.command == "move":
                state["requested_opening_width_mm"] = opening
        return {
            "schema": "demo2gripper.hande_cli.v1",
            "ok": True,
            "command": args.command,
            "port": str(port),
            "fr3_fci_opened": False,
            "driver_source": "direct_pyserial_modbus_rtu",
            "read_only": args.command == "status",
            "state": state,
        }
    finally:
        transport.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (HandEModbusError, OSError, ValueError) as exc:
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
