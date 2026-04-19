"""
profinet_bridge.py  —  PROFINET IO bridge.

PROFINET is Siemens' industrial Ethernet protocol used in manufacturing plants.

This module provides:
  1. PROFINET IO Device (slave) — expose robot I/O to a Siemens PLC controller
  2. GSD/GSDML generation — generate device description file for TIA Portal
  3. S7 communication bridge via Snap7 — direct Siemens S7-300/400/1200/1500 DB access

Architecture
------------
  True PROFINET RT requires kernel modules (profinet-rt, rt-preempt kernel).
  For Python-accessible integration, this bridge uses two approaches:

  A. snap7 library — connects to S7 PLCs over standard Ethernet (S7 protocol,
     not PROFINET RT, but reads/writes the same DB variables that PROFINET IO
     would exchange). Works with S7-1200/1500 with PUT/GET enabled.

  B. Raw socket PROFINET DCP — device discovery and basic DCP frames.
     Full RT IO requires profinet-rt kernel module (Linux only).

Install
-------
  pip install python-snap7          # S7 PLC communication
  # For full PROFINET RT: use profinet-rt Linux kernel module
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import snap7
    from snap7 import util as s7util
    _SNAP7_OK = True
except ImportError:
    _SNAP7_OK = False


# ── PROFINET DCP constants ────────────────────────────────────────────────────

PROFINET_ETHERTYPE = 0x8892
DCP_MULTICAST_MAC  = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x00])

DCP_IDENTIFY_REQUEST   = 0xFEFF
DCP_SERVICE_ID_IDENTIFY = 0x05
DCP_SERVICE_TYPE_REQ    = 0x00


class PROFINETDiscovery:
    """
    Passive PROFINET DCP device discovery via raw socket.
    Listens for DCP Identify requests/responses on the network.

    Requires CAP_NET_RAW or root to open raw sockets.
    """

    def __init__(self,
                 interface: str = "eth0",
                 on_device: Optional[Callable] = None) -> None:
        self._iface    = interface
        self._on_dev   = on_device
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._devices: list[dict] = []

    def start(self) -> bool:
        try:
            self._sock = socket.socket(
                socket.AF_PACKET, socket.SOCK_RAW,
                socket.htons(PROFINET_ETHERTYPE))
            self._sock.bind((self._iface, 0))
            self._running = True
            self._thread  = threading.Thread(
                target=self._listen, daemon=True)
            self._thread.start()
            return True
        except PermissionError:
            print("[PROFINET] Raw socket needs CAP_NET_RAW or root")
            return False
        except Exception as e:
            print(f"[PROFINET] Discovery start failed: {e}")
            return False

    def _listen(self) -> None:
        self._sock.settimeout(0.5)
        while self._running:
            try:
                data, addr = self._sock.recvfrom(1514)
                self._parse_dcp(data, addr)
            except socket.timeout:
                pass
            except Exception:
                break

    def _parse_dcp(self, data: bytes, addr) -> None:
        if len(data) < 26:
            return
        # Skip Ethernet header (14 bytes), check EtherType
        frame_id = struct.unpack_from(">H", data, 14)[0]
        if frame_id not in (0xFEFE, 0xFEFF):
            return
        src_mac = ":".join(f"{b:02x}" for b in data[6:12])
        device  = {
            "mac":       src_mac,
            "frame_id":  hex(frame_id),
            "timestamp": time.time(),
        }
        # Try to parse station name from DCP blocks
        try:
            offset = 26
            while offset + 4 <= len(data):
                blk_type, blk_len = struct.unpack_from(">HH", data, offset)
                blk_data = data[offset + 4: offset + 4 + blk_len]
                if blk_type == 0x0202 and blk_len > 2:  # station name
                    device["station_name"] = blk_data[2:].decode("ascii", errors="ignore")
                offset += 4 + blk_len + (blk_len % 2)
        except Exception:
            pass

        self._devices.append(device)
        if self._on_dev:
            self._on_dev(device)

    def send_identify_request(self) -> None:
        """Broadcast DCP Identify All request."""
        if not hasattr(self, "_sock"):
            return
        # Ethernet frame: DST=multicast, SRC=zeros, EtherType=0x8892
        eth_hdr = DCP_MULTICAST_MAC + bytes(6) + struct.pack(">H", PROFINET_ETHERTYPE)
        # PROFINET real-time header: FrameID + DCP header
        pn_hdr  = struct.pack(">HBBHH",
                               DCP_IDENTIFY_REQUEST,        # FrameID
                               DCP_SERVICE_ID_IDENTIFY,     # ServiceID
                               DCP_SERVICE_TYPE_REQ,        # ServiceType
                               1,                           # XID
                               0x0000,                      # ResponseDelay
                               ) + struct.pack(">H", 4)     # DCPDataLength
        # Block: All/All (0xFFFF), length 0
        block  = struct.pack(">HH", 0xFFFF, 0x0000)
        frame  = eth_hdr + pn_hdr + block
        try:
            self._sock.send(frame)
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_sock"):
            self._sock.close()

    @property
    def devices(self) -> list[dict]:
        return list(self._devices)


# ── S7 PLC bridge (Snap7) ─────────────────────────────────────────────────────

class S7Bridge:
    """
    Siemens S7 PLC communication via snap7.

    Reads/writes Data Blocks (DB) on S7-300/400/1200/1500 PLCs.
    The PLC must have PUT/GET access enabled (in TIA Portal: device properties
    → Protection → Allow access with PUT/GET).

    Parameters
    ----------
    ip      : PLC IP address
    rack    : rack number (usually 0)
    slot    : CPU slot (1 for S7-300, 0 for S7-1200/1500)
    """

    def __init__(self, ip: str, rack: int = 0, slot: int = 1) -> None:
        self._ip   = ip
        self._rack = rack
        self._slot = slot
        self._plc: Optional["snap7.client.Client"] = None

    @property
    def available(self) -> bool:
        return _SNAP7_OK

    def connect(self) -> bool:
        if not _SNAP7_OK:
            return False
        try:
            self._plc = snap7.client.Client()
            self._plc.connect(self._ip, self._rack, self._slot)
            return self._plc.get_connected()
        except Exception as e:
            print(f"[S7] Connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._plc:
            self._plc.disconnect()

    # ── DB read/write ─────────────────────────────────────────────────────────

    def read_real(self, db: int, offset: int) -> Optional[float]:
        """Read a REAL (float32) from a Data Block."""
        if not self._plc:
            return None
        try:
            raw = self._plc.db_read(db, offset, 4)
            return s7util.get_real(raw, 0)
        except Exception:
            return None

    def write_real(self, db: int, offset: int, value: float) -> bool:
        if not self._plc:
            return False
        try:
            buf = bytearray(4)
            s7util.set_real(buf, 0, value)
            self._plc.db_write(db, offset, buf)
            return True
        except Exception:
            return False

    def read_int(self, db: int, offset: int) -> Optional[int]:
        if not self._plc:
            return None
        try:
            raw = self._plc.db_read(db, offset, 2)
            return s7util.get_int(raw, 0)
        except Exception:
            return None

    def write_int(self, db: int, offset: int, value: int) -> bool:
        if not self._plc:
            return False
        try:
            buf = bytearray(2)
            s7util.set_int(buf, 0, value)
            self._plc.db_write(db, offset, buf)
            return True
        except Exception:
            return False

    def read_bool(self, db: int, offset: int, bit: int) -> Optional[bool]:
        if not self._plc:
            return None
        try:
            raw = self._plc.db_read(db, offset, 1)
            return s7util.get_bool(raw, 0, bit)
        except Exception:
            return None

    def write_bool(self, db: int, offset: int, bit: int, value: bool) -> bool:
        if not self._plc:
            return False
        try:
            raw = self._plc.db_read(db, offset, 1)
            buf = bytearray(raw)
            s7util.set_bool(buf, 0, bit, value)
            self._plc.db_write(db, offset, buf)
            return True
        except Exception:
            return False

    def push_robot_state(self, db: int,
                          joint_pos: list[float],
                          x: float, y: float, heading: float) -> None:
        """
        Write robot state into a PLC Data Block.
        Layout: REAL[8] joint_pos (0..31), REAL x(32), REAL y(36), REAL h(40)
        """
        for i, p in enumerate(joint_pos[:8]):
            self.write_real(db, i * 4, p)
        self.write_real(db, 32, x)
        self.write_real(db, 36, y)
        self.write_real(db, 40, heading)

    def read_cmd_vel(self, db: int) -> tuple[float, float]:
        """Read CmdVel from PLC DB. Layout: REAL linear(44), REAL angular(48)."""
        lin = self.read_real(db, 44) or 0.0
        ang = self.read_real(db, 48) or 0.0
        return lin, ang

    def get_status(self) -> dict:
        connected = False
        if self._plc:
            try:
                connected = self._plc.get_connected()
            except Exception:
                pass
        return {
            "available":  _SNAP7_OK,
            "connected":  connected,
            "ip":         self._ip,
            "rack":       self._rack,
            "slot":       self._slot,
        }


# ── GSDML generator ───────────────────────────────────────────────────────────

def generate_gsdml(device_name: str = "P-RoboAI",
                   vendor: str     = "P RoboAI",
                   output_path: str = "/tmp/P_RoboAI.xml") -> str:
    """
    Generate a minimal GSDML file for importing P_RoboAI as a PROFINET
    IO device in TIA Portal or Step 7.
    """
    gsdml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ISO15745Profile>
  <ProfileHeader>
    <ProfileIdentification>PROFINET Device Profile</ProfileIdentification>
    <ProfileRevision>1.0</ProfileRevision>
    <ProfileName>Device Profile for {device_name}</ProfileName>
    <ProfileSource>IEC 61784-2</ProfileSource>
    <ProfileClassID>Device</ProfileClassID>
    <ISO15745Reference>
      <ISO15745Part>4</ISO15745Part>
      <ISO15745Edition>1</ISO15745Edition>
      <ProfileTechnology>PROFINET IO</ProfileTechnology>
    </ISO15745Reference>
  </ProfileHeader>
  <ProfileBody>
    <DeviceIdentity>
      <VendorID>0x0001</VendorID>
      <DeviceID>0x0001</DeviceID>
      <VendorName Value="{vendor}"/>
      <InfoText TextId="DeviceDescription">
        <ExternalTextRef TextId="DeviceDescription"/>
      </InfoText>
    </DeviceIdentity>
    <DeviceFunction>
      <Family MainFamily="Drives" ProductFamily="{device_name}"/>
    </DeviceFunction>
    <ApplicationProcess>
      <DeviceAccessPointList>
        <DeviceAccessPointItem ID="DAP1" PhysicalSlots="0..1"
                               MinDeviceInterval="32"
                               DNS_CompatibleName="{device_name.lower().replace(' ','-')}"
                               LLDP_NoD="false">
          <!-- Input module: robot state (joint positions, odom) -->
          <!-- Output module: cmd_vel, joint commands -->
        </DeviceAccessPointItem>
      </DeviceAccessPointList>
    </ApplicationProcess>
  </ProfileBody>
</ISO15745Profile>
"""
    Path(output_path).write_text(gsdml)
    return output_path
