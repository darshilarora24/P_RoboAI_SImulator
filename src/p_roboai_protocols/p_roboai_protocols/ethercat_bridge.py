"""
ethercat_bridge.py  —  EtherCAT bridge (pysoem).

EtherCAT is a deterministic fieldbus used for high-performance servo systems,
industrial robot arms, and real-time motion control.

Provides:
  - Slave enumeration and configuration
  - Cyclic PDO exchange (process data objects) in a dedicated RT thread
  - SDO read/write for slave configuration
  - Slave state machine control (INIT → PREOP → SAFEOP → OP)

Use cases
---------
  Servo drives (Beckhoff EL7201, AMC, Elmo)
  Robot arm joints (EtherCAT servo drives per joint)
  I/O modules (Beckhoff EL1008, EL2008)
  Force/torque sensors

Install
-------
  pip install pysoem
  # Requires raw socket access (run as root or with CAP_NET_RAW):
  sudo setcap cap_net_raw+ep $(which python3)
"""
from __future__ import annotations

import ctypes
import struct
import threading
import time
from typing import Callable, Optional

try:
    import pysoem
    _SOEM_OK = True
except ImportError:
    _SOEM_OK = False


# ── slave states ──────────────────────────────────────────────────────────────

SLAVE_STATE = {
    "NONE":    0x00,
    "INIT":    0x01,
    "PREOP":   0x02,
    "BOOT":    0x03,
    "SAFEOP":  0x04,
    "OP":      0x08,
    "ACK":     0x10,
    "ERROR":   0x40,
}


class EtherCATSlave:
    """Represents one EtherCAT slave with its PDO layout."""

    def __init__(self, position: int, name: str,
                 rx_pdo_fmt: str = "",   # struct format for output PDO
                 tx_pdo_fmt: str = "") -> None:  # struct format for input PDO
        self.position   = position
        self.name       = name
        self.rx_fmt     = rx_pdo_fmt    # master → slave (commands)
        self.tx_fmt     = tx_pdo_fmt    # slave → master (feedback)
        self.rx_size    = struct.calcsize(rx_pdo_fmt) if rx_pdo_fmt else 0
        self.tx_size    = struct.calcsize(tx_pdo_fmt) if tx_pdo_fmt else 0
        self.state      = "NONE"
        self.error_flag = False

    def pack_rx(self, *values) -> bytes:
        if self.rx_fmt:
            return struct.pack(self.rx_fmt, *values)
        return b""

    def unpack_tx(self, data: bytes):
        if self.tx_fmt and len(data) >= self.tx_size:
            return struct.unpack(self.tx_fmt, data[:self.tx_size])
        return ()


class EtherCATBridge:
    """
    EtherCAT master using SOEM (Simple Open EtherCAT Master).

    Parameters
    ----------
    interface   : network interface name, e.g. "eth0", "enp3s0"
    cycle_us    : PDO cycle time in microseconds (default 1000 = 1 ms)
    on_feedback : callback(slave_pos, feedback_tuple) called each PDO cycle
    """

    def __init__(self,
                 interface:   str = "eth0",
                 cycle_us:    int = 1000,
                 on_feedback: Optional[Callable] = None) -> None:
        self._iface      = interface
        self._cycle_us   = cycle_us
        self._on_fb      = on_feedback
        self._master:    Optional["pysoem.CdefMaster"] = None
        self._slaves:    list[EtherCATSlave] = []
        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._lock       = threading.Lock()
        self._commands:  dict[int, bytes] = {}   # slave_pos → PDO bytes

    @property
    def available(self) -> bool:
        return _SOEM_OK

    # ── initialization ────────────────────────────────────────────────────────

    def init(self) -> bool:
        if not _SOEM_OK:
            return False
        try:
            self._master = pysoem.find_adapters()   # enumerate adapters
            self._master = pysoem.CdefMaster()
            if self._master.open(self._iface) == 0:
                print(f"[EtherCAT] Cannot open {self._iface}")
                return False
            n = self._master.config_init()
            if n == 0:
                print("[EtherCAT] No slaves found")
                return False
            print(f"[EtherCAT] Found {n} slaves")
            self._master.config_map()
            self._master.config_dc()
            self._transition_to_op()
            return True
        except Exception as e:
            print(f"[EtherCAT] Init failed: {e}")
            return False

    def _transition_to_op(self) -> None:
        self._master.state = pysoem.SAFEOP_STATE
        self._master.write_state()
        self._master.state = pysoem.OP_STATE
        self._master.write_state()
        chk = 40
        while chk > 0:
            self._master.state_check(pysoem.OP_STATE, 50_000)
            chk -= 1
            if self._master.state == pysoem.OP_STATE:
                break

    def register_slave(self, slave: EtherCATSlave) -> None:
        self._slaves.append(slave)
        self._commands[slave.position] = bytes(slave.rx_size)

    # ── PDO cyclic exchange ───────────────────────────────────────────────────

    def start_cyclic(self) -> bool:
        if not self._master:
            return False
        self._running = True
        self._thread  = threading.Thread(
            target=self._cyclic_loop, daemon=True)
        self._thread.start()
        return True

    def stop_cyclic(self) -> None:
        self._running = False

    def _cyclic_loop(self) -> None:
        dt = self._cycle_us / 1_000_000
        while self._running:
            t0 = time.perf_counter()
            try:
                # Write outputs to slaves
                with self._lock:
                    for slave in self._slaves:
                        pos  = slave.position
                        data = self._commands.get(pos, b"")
                        if data and pos < len(self._master.slaves):
                            self._master.slaves[pos].output = data

                self._master.send_processdata()
                self._master.receive_processdata(2000)

                # Read inputs from slaves
                for slave in self._slaves:
                    pos = slave.position
                    if pos < len(self._master.slaves):
                        raw = bytes(self._master.slaves[pos].input)
                        fb  = slave.unpack_tx(raw)
                        if fb and self._on_fb:
                            self._on_fb(pos, fb)
            except Exception:
                pass

            elapsed = time.perf_counter() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def close(self) -> None:
        self.stop_cyclic()
        if self._master:
            self._master.state = pysoem.INIT_STATE
            self._master.write_state()
            self._master.close()

    # ── commands ──────────────────────────────────────────────────────────────

    def set_pdo_output(self, slave_pos: int, data: bytes) -> None:
        with self._lock:
            self._commands[slave_pos] = data

    def set_joint_command(self, slave_pos: int, slave: EtherCATSlave,
                           *values) -> None:
        """Pack and queue a PDO output for a servo slave."""
        self.set_pdo_output(slave_pos, slave.pack_rx(*values))

    # ── SDO ──────────────────────────────────────────────────────────────────

    def sdo_read(self, slave_pos: int, index: int, subindex: int,
                  fmt: str = "H") -> Optional[int]:
        if not self._master or slave_pos >= len(self._master.slaves):
            return None
        try:
            raw  = self._master.slaves[slave_pos].sdo_read(index, subindex)
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, raw[:size])[0]
        except Exception:
            return None

    def sdo_write(self, slave_pos: int, index: int, subindex: int,
                   value: int, fmt: str = "H") -> bool:
        if not self._master or slave_pos >= len(self._master.slaves):
            return False
        try:
            data = struct.pack(fmt, value)
            self._master.slaves[slave_pos].sdo_write(index, subindex, data)
            return True
        except Exception:
            return False

    def get_status(self) -> dict:
        slaves_info = []
        if self._master:
            for i, s in enumerate(self._master.slaves):
                slaves_info.append({
                    "pos":   i,
                    "name":  s.name,
                    "state": s.state,
                })
        return {
            "available": _SOEM_OK,
            "interface": self._iface,
            "running":   self._running,
            "cycle_us":  self._cycle_us,
            "slaves":    slaves_info,
        }


# ── Pre-built slave profiles for common hardware ──────────────────────────────

def beckhoff_el7201_slave(pos: int) -> EtherCATSlave:
    """Beckhoff EL7201 servo terminal — velocity mode."""
    return EtherCATSlave(
        position=pos, name="EL7201",
        rx_pdo_fmt="<HI",    # control_word(u16), target_velocity(i32)
        tx_pdo_fmt="<HiI",   # status_word(u16), actual_pos(i32), actual_vel(u32)
    )

def generic_servo_slave(pos: int) -> EtherCATSlave:
    """Generic CIA 402 servo drive."""
    return EtherCATSlave(
        position=pos, name="GenericServo",
        rx_pdo_fmt="<Hi",    # control_word(u16), target_position(i32)
        tx_pdo_fmt="<Hii",   # status_word(u16), actual_pos(i32), actual_vel(i32)
    )
