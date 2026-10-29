#!/usr/bin/env python3
"""Lightweight Robotiq Hand-E Modbus RTU command line client.

This entry point deliberately imports only pyserial.  It is wire-compatible
with the Hand-E path used by ``hande_cli.py`` but avoids importing LeRobot and
PyTorch on the real-time robot host.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import serial

SLAVE = 0x09
BAUD = 115200
MAX_OPENING_MM = 50.0
MM_PER_REGISTER = 0.1953125
STATUS_REGISTER = 0x07D0
ACTION_REGISTER = 0x03E8


class HandEError(RuntimeError):
    pass


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _frame(payload: bytes) -> bytes:
    return payload + struct.pack("<H", _crc16(payload))


def _read_exact(port: serial.Serial, count: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < count and time.monotonic() < deadline:
        data.extend(port.read(count - len(data)))
    return bytes(data)


def _transact(
    port: serial.Serial,
    payload: bytes,
    *,
    timeout_s: float,
    retries: int = 3,
) -> bytes:
    last_error: BaseException | None = None
    for _ in range(retries + 1):
        try:
            port.reset_input_buffer()
            port.write(_frame(payload))
            port.flush()
            deadline = time.monotonic() + timeout_s
            head = _read_exact(port, 3, deadline)
            if len(head) != 3:
                raise HandEError(f"short Modbus header: {head.hex()}")
            if head[0] != SLAVE:
                raise HandEError(f"unexpected Modbus slave {head[0]}")
            if head[1] == (payload[1] | 0x80):
                tail = _read_exact(port, 2, deadline)
                response = head + tail
                if len(response) != 5:
                    raise HandEError(f"short Modbus exception: {response.hex()}")
                raise HandEError(f"Modbus exception code {head[2]}")
            if head[1] != payload[1]:
                raise HandEError(f"unexpected Modbus function {head[1]}")
            remaining = head[2] + 2 if payload[1] == 0x03 else 5
            response = head + _read_exact(port, remaining, deadline)
            if len(response) != 3 + remaining:
                raise HandEError(f"short Modbus response: {response.hex()}")
            if _crc16(response[:-2]) != int.from_bytes(response[-2:], "little"):
                raise HandEError("Modbus response CRC mismatch")
            return response
        except (HandEError, serial.SerialException) as exc:
            last_error = exc
            time.sleep(0.05)
    raise HandEError(f"Modbus transaction failed: {last_error}") from last_error


def _read_state(port: serial.Serial, *, timeout_s: float) -> dict[str, object]:
    request = struct.pack(">BBHH", SLAVE, 0x03, STATUS_REGISTER, 3)
    response = _transact(port, request, timeout_s=timeout_s)
    registers = [
        int.from_bytes(response[offset : offset + 2], "big")
        for offset in (3, 5, 7)
    ]
    status = (registers[0] >> 8) & 0xFF
    activation = status & 0x01
    go_to = (status >> 3) & 0x01
    activation_status = (status >> 4) & 0x03
    object_code = (status >> 6) & 0x03
    fault = (registers[1] >> 8) & 0xFF
    requested_closure = (registers[1] & 0xFF) * MM_PER_REGISTER
    closure = ((registers[2] >> 8) & 0xFF) * MM_PER_REGISTER
    opening = MAX_OPENING_MM - closure
    meanings = {
        0: "moving_no_object",
        1: "object_detected_opening",
        2: "grasped_object_detected_closing",
        3: "motion_complete_no_object",
    }
    return {
        "activated": activation == 1 and activation_status == 3,
        "activation_bit": activation,
        "go_to_bit": go_to,
        "activation_status": activation_status,
        "closure_mm": closure,
        "opening_width_mm": opening,
        "requested_closure_mm": requested_closure,
        "current_register": registers[2] & 0xFF,
        "fault_code": fault,
        "object_status": {
            "code": object_code,
            "meaning": meanings.get(object_code, "unknown"),
            "grasped": object_code == 2,
            "fully_closed_without_object": object_code == 3 and closure >= 47.5,
            "fully_open": object_code == 3 and closure <= 2.0,
        },
    }


def _write_target(
    port: serial.Serial,
    *,
    opening_mm: float,
    speed: int,
    force: int,
    timeout_s: float,
) -> None:
    closure_mm = MAX_OPENING_MM - opening_mm
    target = max(0, min(255, int(closure_mm / MM_PER_REGISTER)))
    registers = struct.pack(">HHH", 0x0900, target, (speed << 8) | force)
    request = struct.pack(">BBHHB", SLAVE, 0x10, ACTION_REGISTER, 3, 6) + registers
    _transact(port, request, timeout_s=timeout_s)


def _move(
    port: serial.Serial,
    *,
    opening_mm: float,
    speed: int,
    force: int,
    timeout_s: float,
) -> dict[str, object]:
    initial = _read_state(port, timeout_s=min(0.5, timeout_s))
    if initial["activated"] is not True:
        raise HandEError("Hand-E is not activated")
    initial_closure = float(initial["closure_mm"])
    target_closure = MAX_OPENING_MM - opening_mm
    opening_motion = target_closure < initial_closure
    _write_target(
        port,
        opening_mm=opening_mm,
        speed=speed,
        force=force,
        timeout_s=min(0.5, timeout_s),
    )
    deadline = time.monotonic() + timeout_s
    last = initial
    while time.monotonic() < deadline:
        last = _read_state(port, timeout_s=min(0.5, timeout_s))
        if int(last["fault_code"]) != 0:
            raise HandEError(f"Hand-E reported fault code {last['fault_code']}")
        code = int(last["object_status"]["code"])  # type: ignore[index]
        closure = float(last["closure_mm"])
        if not opening_motion and code == 2:
            return last
        if code == 3 and abs(closure - target_closure) <= 2.0:
            return last
        if opening_motion and code == 1:
            raise HandEError("Hand-E detected an obstruction while opening")
        time.sleep(0.025)
    raise HandEError(
        f"Hand-E move timed out after {timeout_s:.2f}s "
        f"(opening={float(last['opening_width_mm']):.3f} mm)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "open", "close", "move"):
        item = sub.add_parser(name)
        item.add_argument("--port", required=True)
        item.add_argument("--modbus-timeout-s", type=float, default=0.25)
        if name != "status":
            item.add_argument("--speed", type=int, default=50)
            item.add_argument("--force", type=int, default=40)
            item.add_argument("--timeout-s", type=float, default=8.0)
        if name == "move":
            item.add_argument("--opening-mm", type=float, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        port_path = Path(args.port).expanduser()
        if not port_path.is_absolute():
            raise HandEError("--port must be absolute")
        modbus_timeout = float(args.modbus_timeout_s)
        if not math.isfinite(modbus_timeout) or modbus_timeout <= 0:
            raise HandEError("--modbus-timeout-s must be positive")
        with serial.Serial(
            str(port_path),
            BAUD,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.05,
            write_timeout=0.5,
        ) as port:
            if args.command == "status":
                state = _read_state(port, timeout_s=modbus_timeout)
            else:
                speed = int(args.speed)
                force = int(args.force)
                if not 0 <= speed <= 100 or not 0 <= force <= 70:
                    raise HandEError("speed must be 0..100 and force must be 0..70")
                opening = (
                    MAX_OPENING_MM
                    if args.command == "open"
                    else 0.0
                    if args.command == "close"
                    else float(args.opening_mm)
                )
                if not math.isfinite(opening) or not 0 <= opening <= MAX_OPENING_MM:
                    raise HandEError("opening must be within 0..50 mm")
                state = _move(
                    port,
                    opening_mm=opening,
                    speed=speed,
                    force=force,
                    timeout_s=float(args.timeout_s),
                )
        print(json.dumps({"ok": True, "command": args.command, "state": state}))
        return 0
    except BaseException as exc:
        print(json.dumps({"ok": False, "command": args.command, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
