"""
canopen_bridge.py  —  CAN Bus / CANopen bridge (python-can + canopen).

Supports:
  - Raw CAN frame publish/subscribe
  - CANopen NMT state machine (start, stop, pre-op, reset)
  - CANopen SDO read/write (object dictionary access)
  - CANopen PDO receive/transmit (real-time process data)
  - Standard motor controller objects (0x6040 control word, 0x607A target pos)

Typical use cases
-----------------
  Motor controllers, battery BMS, AGV/AMR wheel drives,
  EV powertrain components (EVSE CAN), servo drives.

Install
-------
  pip install python-can canopen
  # Also install a CAN interface driver, e.g.:
  # socketcan (Linux built-in), pcan, kvaser, vector
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Callable, Optional

try:
    import can
    _CAN_OK = True
except ImportError:
    _CAN_OK = False

try:
    import canopen
    _CANOPEN_OK = True
except ImportError:
    _CANOPEN_OK = False


# ── CANopen standard object indices ──────────────────────────────────────────

OD = {
    "control_word":     (0x6040, 0x00),  # CIA 402 drive state machine
    "status_word":      (0x6041, 0x00),
    "target_position":  (0x607A, 0x00),  # position mode target
    "target_velocity":  (0x60FF, 0x00),  # velocity mode target (rpm)
    "actual_position":  (0x6064, 0x00),
    "actual_velocity":  (0x606C, 0x00),
    "actual_torque":    (0x6077, 0x00),
    "mode_of_operation":(0x6060, 0x00),  # 1=profile pos, 3=profile vel, 4=torque
    "max_motor_speed":  (0x6080, 0x00),
    "error_register":   (0x1001, 0x00),
    "heartbeat_time":   (0x1017, 0x00),
}

# CIA 402 control word commands
CW_SHUTDOWN        = 0x0006
CW_SWITCH_ON       = 0x0007
CW_ENABLE_OP       = 0x000F
CW_FAULT_RESET     = 0x0080
CW_QUICK_STOP      = 0x000B


class CANBus:
    """
    Raw CAN bus interface — send/receive frames.

    Parameters
    ----------
    interface : "socketcan", "pcan", "kvaser", "vector", "virtual"
    channel   : "can0", "PCAN_USBBUS1", etc.
    bitrate   : 250000, 500000, 1000000
    """

    def __init__(self,
                 interface: str = "socketcan",
                 channel:   str = "can0",
                 bitrate:   int = 500_000,
                 on_frame:  Optional[Callable] = None) -> None:
        self._interface = interface
        self._channel   = channel
        self._bitrate   = bitrate
        self._on_frame  = on_frame
        self._bus:      Optional["can.Bus"] = None
        self._thread:   Optional[threading.Thread] = None
        self._running   = False

    @property
    def available(self) -> bool:
        return _CAN_OK

    def connect(self) -> bool:
        if not _CAN_OK:
            return False
        try:
            self._bus = can.Bus(
                interface=self._interface,
                channel=self._channel,
                bitrate=self._bitrate,
            )
            self._running = True
            if self._on_frame:
                self._thread = threading.Thread(
                    target=self._rx_loop, daemon=True)
                self._thread.start()
            return True
        except Exception as e:
            print(f"[CAN] Connect failed: {e}")
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._bus:
            self._bus.shutdown()
            self._bus = None

    def send(self, arb_id: int, data: bytes, is_extended: bool = False) -> bool:
        if not self._bus:
            return False
        try:
            msg = can.Message(
                arbitration_id=arb_id,
                data=data,
                is_extended_id=is_extended,
            )
            self._bus.send(msg)
            return True
        except Exception:
            return False

    def _rx_loop(self) -> None:
        while self._running and self._bus:
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg and self._on_frame:
                    self._on_frame(msg.arbitration_id, bytes(msg.data),
                                   msg.timestamp)
            except Exception:
                time.sleep(0.01)

    def get_status(self) -> dict:
        return {
            "available":  _CAN_OK,
            "connected":  self._bus is not None,
            "interface":  self._interface,
            "channel":    self._channel,
            "bitrate":    self._bitrate,
        }


class CANopenBridge:
    """
    CANopen higher-level bridge over a CAN bus.

    Manages a CANopen network, exposing motor SDO/PDO access and
    NMT control in a ROS2-friendly API.
    """

    def __init__(self,
                 interface: str = "socketcan",
                 channel:   str = "can0",
                 bitrate:   int = 500_000) -> None:
        self._iface   = interface
        self._chan    = channel
        self._brate   = bitrate
        self._network: Optional["canopen.Network"] = None
        self._nodes:   dict[int, "canopen.RemoteNode"] = {}

    @property
    def available(self) -> bool:
        return _CANOPEN_OK

    def connect(self) -> bool:
        if not _CANOPEN_OK:
            return False
        try:
            self._network = canopen.Network()
            self._network.connect(
                interface=self._iface,
                channel=self._chan,
                bitrate=self._brate,
            )
            return True
        except Exception as e:
            print(f"[CANopen] Connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._network:
            self._network.disconnect()

    def add_node(self, node_id: int, eds_path: str = "") -> bool:
        """Add a remote CANopen node (optionally with EDS object dictionary)."""
        if not self._network:
            return False
        try:
            if eds_path:
                node = self._network.add_node(node_id, eds_path)
            else:
                node = canopen.RemoteNode(node_id, canopen.ObjectDictionary())
                self._network.add_node(node)
            self._nodes[node_id] = node
            return True
        except Exception as e:
            print(f"[CANopen] Add node {node_id} failed: {e}")
            return False

    # ── NMT ──────────────────────────────────────────────────────────────────

    def nmt_start(self, node_id: int = 0) -> None:
        """Send NMT Start (operational). node_id=0 = broadcast."""
        if self._network:
            self._network.send_message(0x000, bytes([0x01, node_id]))

    def nmt_stop(self, node_id: int = 0) -> None:
        if self._network:
            self._network.send_message(0x000, bytes([0x02, node_id]))

    def nmt_preop(self, node_id: int = 0) -> None:
        if self._network:
            self._network.send_message(0x000, bytes([0x80, node_id]))

    def nmt_reset(self, node_id: int = 0) -> None:
        if self._network:
            self._network.send_message(0x000, bytes([0x81, node_id]))

    # ── SDO ──────────────────────────────────────────────────────────────────

    def sdo_read(self, node_id: int, index: int, subindex: int = 0):
        node = self._nodes.get(node_id)
        if not node:
            return None
        try:
            return node.sdo[index][subindex].raw
        except Exception:
            return None

    def sdo_write(self, node_id: int, index: int, subindex: int, value) -> bool:
        node = self._nodes.get(node_id)
        if not node:
            return False
        try:
            node.sdo[index][subindex].raw = value
            return True
        except Exception:
            return False

    # ── Motor control shortcuts ───────────────────────────────────────────────

    def enable_drive(self, node_id: int) -> None:
        """CIA 402 state machine: shutdown → switch on → enable operation."""
        self.sdo_write(node_id, *OD["control_word"], CW_SHUTDOWN)
        time.sleep(0.05)
        self.sdo_write(node_id, *OD["control_word"], CW_SWITCH_ON)
        time.sleep(0.05)
        self.sdo_write(node_id, *OD["control_word"], CW_ENABLE_OP)

    def set_velocity(self, node_id: int, rpm: int) -> None:
        self.sdo_write(node_id, *OD["mode_of_operation"], 3)   # velocity mode
        self.sdo_write(node_id, *OD["target_velocity"], rpm)

    def set_position(self, node_id: int, counts: int) -> None:
        self.sdo_write(node_id, *OD["mode_of_operation"], 1)   # profile pos
        self.sdo_write(node_id, *OD["target_position"], counts)
        # Trigger new setpoint
        self.sdo_write(node_id, *OD["control_word"], CW_ENABLE_OP | 0x0010)
        time.sleep(0.01)
        self.sdo_write(node_id, *OD["control_word"], CW_ENABLE_OP)

    def fault_reset(self, node_id: int) -> None:
        self.sdo_write(node_id, *OD["control_word"], CW_FAULT_RESET)

    def get_actual_velocity(self, node_id: int) -> Optional[int]:
        return self.sdo_read(node_id, *OD["actual_velocity"])

    def get_actual_position(self, node_id: int) -> Optional[int]:
        return self.sdo_read(node_id, *OD["actual_position"])

    def get_status(self) -> dict:
        return {
            "available":  _CANOPEN_OK,
            "connected":  self._network is not None,
            "nodes":      list(self._nodes.keys()),
            "interface":  self._iface,
            "channel":    self._chan,
        }
