"""
modbus_bridge.py  —  Modbus RTU + TCP bridge (pymodbus).

Acts as a Modbus TCP server (slave) exposing robot registers, and as a
Modbus client to read/write external PLCs, VFDs, and sensor devices.

Register Map (server)
---------------------
  Holding Registers (4xxxx):
    40001–40008   Joint positions × 1000 (int16, rad × 1000)
    40009–40016   Joint velocities × 1000
    40017         AMR position X × 100 (cm)
    40018         AMR position Y × 100 (cm)
    40019         AMR heading × 100 (deg × 100)
    40020         Detection count
    40021         CmdVel linear × 100  (writeable)
    40022         CmdVel angular × 100 (writeable)
    40023         Status code (0=idle,1=running,2=error)

  Coils (0xxxx):
    00001         E-Stop (write 1 to trigger)
    00002         Reset
    00003         Enable RL policy

Install
-------
  pip install pymodbus
"""
from __future__ import annotations

import asyncio
import struct
import threading
import time
from typing import Callable, Optional

try:
    from pymodbus.server import StartAsyncTcpServer
    from pymodbus.datastore import (
        ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext,
    )
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient
    from pymodbus.framer import FramerType
    _MODBUS_OK = True
except ImportError:
    _MODBUS_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _f2reg(val: float, scale: int = 100) -> int:
    """Float → int16 register value with scale factor."""
    return max(-32768, min(32767, int(val * scale)))

def _reg2f(reg: int, scale: int = 100) -> float:
    """int16 register → float."""
    v = reg if reg < 32768 else reg - 65536
    return v / scale


# ── Modbus server (slave) ─────────────────────────────────────────────────────

class ModbusBridge:
    """
    Modbus TCP server exposing robot state + Modbus client for external devices.

    Parameters
    ----------
    host        : bind address for TCP server
    port        : TCP port (default 502; use 5020 if not root)
    on_estop    : callback() when E-Stop coil is written
    on_cmd_vel  : callback(linear, angular) when CmdVel registers are written
    """

    def __init__(self,
                 host:       str = "0.0.0.0",
                 port:       int = 5020,
                 on_estop:   Optional[Callable] = None,
                 on_cmd_vel: Optional[Callable] = None) -> None:
        self._host      = host
        self._port      = port
        self._on_estop  = on_estop
        self._on_cmd_vel = on_cmd_vel
        self._context:  Optional["ModbusServerContext"] = None
        self._store:    Optional["ModbusSlaveContext"]  = None
        self._running   = False
        self._thread:   Optional[threading.Thread]     = None
        self._loop:     Optional[asyncio.AbstractEventLoop] = None

    @property
    def available(self) -> bool:
        return _MODBUS_OK

    # ── server ────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not _MODBUS_OK:
            return False
        try:
            # Registers: 40 holding, 10 coils, all zero-init
            self._store = ModbusSlaveContext(
                hr=ModbusSequentialDataBlock(0, [0] * 40),
                co=ModbusSequentialDataBlock(0, [0] * 10),
                di=ModbusSequentialDataBlock(0, [0] * 10),
                ir=ModbusSequentialDataBlock(0, [0] * 10),
            )
            self._context = ModbusServerContext(
                slaves=self._store, single=True)
            self._running = True
            self._thread  = threading.Thread(
                target=self._run_server, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[Modbus] Server start failed: {e}")
            return False

    def _run_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(
                StartAsyncTcpServer(
                    context=self._context,
                    address=(self._host, self._port),
                )
            )
        except Exception as e:
            print(f"[Modbus] Server error: {e}")
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── register writes (from ROS2 data) ──────────────────────────────────────

    def update_joint_states(self,
                             positions: list[float],
                             velocities: list[float]) -> None:
        if not self._store:
            return
        for i, p in enumerate(positions[:8]):
            self._store.setValues(3, i, [_f2reg(p, 1000)])          # HR 0-7
        for i, v in enumerate(velocities[:8]):
            self._store.setValues(3, 8 + i, [_f2reg(v, 1000)])      # HR 8-15

    def update_odom(self, x: float, y: float, heading_deg: float) -> None:
        if not self._store:
            return
        self._store.setValues(3, 16, [_f2reg(x, 100)])              # HR 16
        self._store.setValues(3, 17, [_f2reg(y, 100)])              # HR 17
        self._store.setValues(3, 18, [_f2reg(heading_deg, 100)])    # HR 18

    def update_detections(self, count: int, status_code: int = 1) -> None:
        if not self._store:
            return
        self._store.setValues(3, 19, [count])                        # HR 19
        self._store.setValues(3, 22, [status_code])                  # HR 22

    def poll_writes(self) -> None:
        """
        Poll for external writes to CmdVel registers and E-Stop coil.
        Call periodically from a timer.
        """
        if not self._store:
            return
        # E-Stop coil
        estop = self._store.getValues(1, 0, 1)[0]
        if estop and self._on_estop:
            self._on_estop()
            self._store.setValues(1, 0, [0])   # auto-clear
        # CmdVel holding registers
        lin_raw = self._store.getValues(3, 20, 1)[0]
        ang_raw = self._store.getValues(3, 21, 1)[0]
        lin = _reg2f(lin_raw, 100)
        ang = _reg2f(ang_raw, 100)
        if self._on_cmd_vel and (lin != 0.0 or ang != 0.0):
            self._on_cmd_vel(lin, ang)

    # ── client (read/write external Modbus devices) ───────────────────────────

    def read_holding_registers(self,
                                host: str,
                                port: int = 502,
                                start: int = 0,
                                count: int = 10) -> Optional[list[int]]:
        """Read holding registers from an external Modbus TCP device."""
        if not _MODBUS_OK:
            return None
        try:
            with ModbusTcpClient(host=host, port=port) as cl:
                result = cl.read_holding_registers(start, count)
                if result.isError():
                    return None
                return list(result.registers)
        except Exception:
            return None

    def write_holding_registers(self,
                                 host: str,
                                 port: int = 502,
                                 start: int = 0,
                                 values: Optional[list[int]] = None) -> bool:
        if not _MODBUS_OK or not values:
            return False
        try:
            with ModbusTcpClient(host=host, port=port) as cl:
                result = cl.write_registers(start, values)
                return not result.isError()
        except Exception:
            return False

    def read_rtu(self,
                 port: str,
                 unit: int,
                 start: int = 0,
                 count: int = 10,
                 baudrate: int = 9600) -> Optional[list[int]]:
        """Read holding registers from a Modbus RTU (serial) device."""
        if not _MODBUS_OK:
            return None
        try:
            with ModbusSerialClient(
                port=port, baudrate=baudrate,
                bytesize=8, parity="N", stopbits=1,
            ) as cl:
                result = cl.read_holding_registers(start, count, slave=unit)
                if result.isError():
                    return None
                return list(result.registers)
        except Exception:
            return None

    def get_status(self) -> dict:
        return {
            "available": _MODBUS_OK,
            "running":   self._running,
            "host":      self._host,
            "port":      self._port,
        }
