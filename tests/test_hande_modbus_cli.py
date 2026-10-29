from __future__ import annotations

import pytest

from demo2gripper.hande_modbus_cli import (
    ACTIVE_GO_TO_ACTION,
    ACTIVE_STOP_ACTION,
    HandEModbusError,
    build_read_holding_request,
    build_write_multiple_request,
    crc16_modbus,
    parse_feedback_registers,
    parse_read_holding_response,
    parse_write_multiple_response,
)


def test_crc16_matches_modbus_reference_vector() -> None:
    # Published Modbus request: CRC is transmitted low byte first as C5 CD.
    assert crc16_modbus(bytes.fromhex("01 03 00 00 00 0A")) == 0xCDC5


def test_build_hand_e_read_status_frame() -> None:
    assert build_read_holding_request(0x07D0, 3) == bytes.fromhex(
        "09 03 07 D0 00 03 04 0E"
    )


def test_parse_hand_e_status_frame_and_byte_layout() -> None:
    frame = bytes.fromhex("09 03 06 F9 00 00 2C 80 21 32 CD")
    registers = parse_read_holding_response(frame, count=3)
    feedback = parse_feedback_registers(registers)

    assert registers == (0xF900, 0x002C, 0x8021)
    assert feedback.activation_bit == 1
    assert feedback.go_to_bit == 1
    assert feedback.activation_status == 3
    assert feedback.object_status == 3
    assert feedback.fault_code == 0
    assert feedback.requested_count == 0x2C
    assert feedback.position_count == 0x80
    assert feedback.current == 0x21


def test_build_hand_e_write_multiple_frame_uses_existing_sdk_register_layout() -> None:
    assert ACTIVE_GO_TO_ACTION == 0x0900
    assert ACTIVE_STOP_ACTION == 0x0100
    assert build_write_multiple_request(
        0x03E8, (0x0900, 0x00A3, 0x0F0A)
    ) == bytes.fromhex("09 10 03 E8 00 03 06 09 00 00 A3 0F 0A 06 7C")


def test_parse_write_ack_and_reject_corrupt_crc() -> None:
    parse_write_multiple_response(
        bytes.fromhex("09 10 03 E8 00 03 01 30"), address=0x03E8, count=3
    )

    corrupt = bytes.fromhex("09 03 06 F9 00 00 2C 80 21 32 CC")
    with pytest.raises(HandEModbusError, match="CRC mismatch"):
        parse_read_holding_response(corrupt, count=3)
