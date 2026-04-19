"""
dds_config.py  —  DDS middleware configuration and QoS management.

Supports Fast DDS, Cyclone DDS, and RTI Connext DDS via environment
variable and XML profile management.  Also provides helper functions
to set per-topic QoS policies via rclpy.

Supported RMW implementations
------------------------------
  rmw_fastrtps_cpp       — Fast DDS (default in ROS2 Humble)
  rmw_cyclonedds_cpp     — Cyclone DDS
  rmw_connextdds         — RTI Connext DDS (requires RTI license)

Usage
-----
  from dds_config import DDSConfig
  cfg = DDSConfig()
  cfg.set_rmw("cyclone")          # switch to Cyclone DDS
  cfg.write_fastdds_profile(...)  # generate Fast DDS XML profile
  qos = cfg.realtime_qos()        # rclpy QoSProfile for real-time topics
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    from rclpy.qos import (
        QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
        QoSHistoryPolicy, QoSLivelinessPolicy,
    )
    from rclpy.duration import Duration
    _RCLPY_OK = True
except ImportError:
    _RCLPY_OK = False


# ── RMW implementations ───────────────────────────────────────────────────────

RMW_MAP = {
    "fastdds":   "rmw_fastrtps_cpp",
    "fast":      "rmw_fastrtps_cpp",
    "cyclone":   "rmw_cyclonedds_cpp",
    "cyclonedds":"rmw_cyclonedds_cpp",
    "connext":   "rmw_connextdds",
    "rti":       "rmw_connextdds",
}


class DDSConfig:
    """Manage DDS RMW selection and QoS profiles."""

    def __init__(self, profile_dir: str = "/tmp/p_roboai_dds") -> None:
        self._profile_dir = Path(profile_dir)
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._current_rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    # ── RMW selection ─────────────────────────────────────────────────────────

    def set_rmw(self, name: str) -> str:
        """Set RMW_IMPLEMENTATION. Returns the full RMW identifier string."""
        rmw = RMW_MAP.get(name.lower(), name)
        os.environ["RMW_IMPLEMENTATION"] = rmw
        self._current_rmw = rmw
        return rmw

    @property
    def current_rmw(self) -> str:
        return self._current_rmw

    @staticmethod
    def available_rmw() -> list[str]:
        """Return list of installed RMW packages."""
        result = []
        for rmw in RMW_MAP.values():
            try:
                out = subprocess.run(
                    ["ros2", "pkg", "list"], capture_output=True, text=True)
                if rmw.replace("rmw_", "").split("_")[0] in out.stdout:
                    result.append(rmw)
            except Exception:
                pass
        return list(set(result)) or ["rmw_fastrtps_cpp"]

    # ── Fast DDS XML profiles ─────────────────────────────────────────────────

    def write_fastdds_profile(self,
                               transport: str = "udp",
                               max_part: int = 10,
                               discovery: str = "SIMPLE") -> str:
        """
        Generate a Fast DDS XML profile and set FASTRTPS_DEFAULT_PROFILES_FILE.
        Returns the profile file path.
        """
        xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">

  <participant profile_name="p_roboai_participant" is_default_profile="true">
    <rtps>
      <builtin>
        <discovery_config>
          <discoveryProtocol>{discovery}</discoveryProtocol>
          <leaseDuration><sec>20</sec><nanosec>0</nanosec></leaseDuration>
        </discovery_config>
        <metatrafficUnicastLocatorList>
          <locator><kind>{transport.upper()}</kind></locator>
        </metatrafficUnicastLocatorList>
      </builtin>
      <participantID>-1</participantID>
    </rtps>
  </participant>

  <!-- Real-time sensor topics: BEST_EFFORT, KEEP_LAST(1) -->
  <data_writer profile_name="rt_sensor_writer">
    <qos>
      <reliability><kind>BEST_EFFORT</kind></reliability>
      <durability><kind>VOLATILE</kind></durability>
      <history><kind>KEEP_LAST</kind><depth>1</depth></history>
    </qos>
  </data_writer>

  <!-- Command topics: RELIABLE, TRANSIENT_LOCAL -->
  <data_writer profile_name="command_writer">
    <qos>
      <reliability><kind>RELIABLE</kind></reliability>
      <durability><kind>TRANSIENT_LOCAL</kind></durability>
      <history><kind>KEEP_LAST</kind><depth>10</depth></history>
    </qos>
  </data_writer>

  <!-- Map / large data: RELIABLE, TRANSIENT_LOCAL, KEEP_ALL -->
  <data_writer profile_name="map_writer">
    <qos>
      <reliability><kind>RELIABLE</kind></reliability>
      <durability><kind>TRANSIENT_LOCAL</kind></durability>
      <history><kind>KEEP_ALL</kind></history>
    </qos>
  </data_writer>

</profiles>
"""
        path = self._profile_dir / "fastdds_profile.xml"
        path.write_text(xml)
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(path)
        return str(path)

    def write_cyclone_config(self, interface: str = "lo") -> str:
        """
        Generate a Cyclone DDS XML config and set CYCLONEDDS_URI.
        Returns the config file path.
        """
        xml = f"""<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>{interface}</NetworkInterfaceAddress>
      <AllowMulticast>true</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Internal>
      <Watermarks>
        <WhcHigh>500kB</WhcHigh>
      </Watermarks>
    </Internal>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>9</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
"""
        path = self._profile_dir / "cyclone_config.xml"
        path.write_text(xml)
        os.environ["CYCLONEDDS_URI"] = f"file://{path}"
        return str(path)

    # ── QoS profiles ─────────────────────────────────────────────────────────

    def realtime_qos(self) -> "QoSProfile":
        """Best-effort, volatile, depth=1 — for sensors/odom at high rate."""
        if not _RCLPY_OK:
            return None
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

    def command_qos(self) -> "QoSProfile":
        """Reliable, transient-local — for commands that must be delivered."""
        if not _RCLPY_OK:
            return None
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

    def map_qos(self) -> "QoSProfile":
        """Reliable, transient-local — for maps published once."""
        if not _RCLPY_OK:
            return None
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

    def get_status(self) -> dict:
        return {
            "rmw":         self._current_rmw,
            "profile_dir": str(self._profile_dir),
            "fastdds_xml": os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "not set"),
            "cyclone_uri": os.environ.get("CYCLONEDDS_URI", "not set"),
        }
