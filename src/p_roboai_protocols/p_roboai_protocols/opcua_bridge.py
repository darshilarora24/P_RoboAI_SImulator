"""
opcua_bridge.py  —  OPC UA bridge (asyncua).

Runs an OPC UA server exposing robot state as OPC UA nodes, and optionally
connects as a client to an external PLC/SCADA OPC UA server.

Node structure (server mode)
-----------------------------
  Objects/
    Robot/
      JointPositions[n]   — Float  (from /joint_states)
      JointVelocities[n]  — Float
      Position_X          — Float  (from /odom)
      Position_Y          — Float
      Heading             — Float
      CmdVel_Linear       — Float  (writeable → /cmd_vel)
      CmdVel_Angular      — Float  (writeable → /cmd_vel)
      DetectionCount      — Int    (from /yolo/detections)
      Status              — String

Install
-------
  pip install asyncua
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Callable, Optional

try:
    from asyncua import Server as _OPCServer, Client as _OPCClient, ua
    from asyncua.common.methods import uamethod
    _OPCUA_OK = True
except ImportError:
    _OPCUA_OK = False


class OPCUABridge:
    """
    OPC UA server + optional client bridge.

    Parameters
    ----------
    endpoint    : OPC UA server URL to host, e.g. "opc.tcp://0.0.0.0:4840/p_roboai"
    namespace   : OPC UA namespace URI
    client_url  : connect as client to this URL (PLC/SCADA) if set
    on_cmd_vel  : callback(linear, angular) called when CmdVel nodes are written
    """

    NS_URI = "urn:p_roboai:robotics"

    def __init__(self,
                 endpoint:   str = "opc.tcp://0.0.0.0:4840/p_roboai",
                 client_url: str = "",
                 on_cmd_vel: Optional[Callable] = None) -> None:
        self._endpoint   = endpoint
        self._client_url = client_url
        self._on_cmd_vel = on_cmd_vel
        self._server: Optional["_OPCServer"] = None
        self._client: Optional["_OPCClient"] = None
        self._nodes: dict  = {}
        self._running = False
        self._loop:   Optional[asyncio.AbstractEventLoop] = None

    @property
    def available(self) -> bool:
        return _OPCUA_OK

    # ── server lifecycle ──────────────────────────────────────────────────────

    async def start_server(self) -> None:
        if not _OPCUA_OK:
            return
        self._server = _OPCServer()
        await self._server.init()
        self._server.set_endpoint(self._endpoint)
        self._server.set_server_name("P_RoboAI OPC UA Server")
        idx = await self._server.register_namespace(self.NS_URI)

        # Build node tree
        objects = self._server.nodes.objects
        robot   = await objects.add_object(idx, "Robot")

        async def _var(parent, name, val, writable=False):
            node = await parent.add_variable(idx, name, val)
            if writable:
                await node.set_writable()
            return node

        self._nodes["joint_pos"]     = [await _var(robot, f"JointPos_{i}", 0.0)     for i in range(8)]
        self._nodes["joint_vel"]     = [await _var(robot, f"JointVel_{i}", 0.0)     for i in range(8)]
        self._nodes["pos_x"]         = await _var(robot, "Position_X", 0.0)
        self._nodes["pos_y"]         = await _var(robot, "Position_Y", 0.0)
        self._nodes["heading"]       = await _var(robot, "Heading", 0.0)
        self._nodes["cmd_linear"]    = await _var(robot, "CmdVel_Linear",  0.0, writable=True)
        self._nodes["cmd_angular"]   = await _var(robot, "CmdVel_Angular", 0.0, writable=True)
        self._nodes["detect_count"]  = await _var(robot, "DetectionCount", 0)
        self._nodes["status"]        = await _var(robot, "Status", "idle")
        self._nodes["timestamp"]     = await _var(robot, "Timestamp", 0.0)

        # Subscribe to CmdVel writes
        if self._on_cmd_vel:
            handler = _CmdVelHandler(
                self._nodes["cmd_linear"],
                self._nodes["cmd_angular"],
                self._on_cmd_vel,
            )
            sub = await self._server.create_subscription(100, handler)
            await sub.subscribe_data_change([
                self._nodes["cmd_linear"],
                self._nodes["cmd_angular"],
            ])

        await self._server.start()
        self._running = True

    async def stop_server(self) -> None:
        if self._server:
            await self._server.stop()
            self._running = False

    # ── data update (called from ROS2 callbacks) ──────────────────────────────

    async def update_joint_states(self,
                                   positions: list[float],
                                   velocities: list[float]) -> None:
        if not self._running:
            return
        for i, (p, v) in enumerate(zip(positions[:8], velocities[:8])):
            await self._nodes["joint_pos"][i].write_value(float(p))
            await self._nodes["joint_vel"][i].write_value(float(v))

    async def update_odom(self, x: float, y: float, heading: float) -> None:
        if not self._running:
            return
        await self._nodes["pos_x"].write_value(x)
        await self._nodes["pos_y"].write_value(y)
        await self._nodes["heading"].write_value(heading)
        await self._nodes["timestamp"].write_value(time.time())

    async def update_detections(self, count: int, status: str = "") -> None:
        if not self._running:
            return
        await self._nodes["detect_count"].write_value(count)
        if status:
            await self._nodes["status"].write_value(status)

    # ── client mode (connect to external PLC OPC UA) ─────────────────────────

    async def connect_client(self) -> bool:
        if not _OPCUA_OK or not self._client_url:
            return False
        try:
            self._client = _OPCClient(self._client_url)
            await self._client.connect()
            return True
        except Exception as e:
            print(f"[OPC UA] Client connect failed: {e}")
            return False

    async def read_client_node(self, node_id: str):
        """Read a value from the connected PLC OPC UA server."""
        if not self._client:
            return None
        try:
            node = self._client.get_node(node_id)
            return await node.read_value()
        except Exception:
            return None

    async def write_client_node(self, node_id: str, value) -> bool:
        if not self._client:
            return False
        try:
            node = self._client.get_node(node_id)
            dv   = ua.DataValue(ua.Variant(value))
            await node.write_value(dv)
            return True
        except Exception:
            return False

    def get_status(self) -> dict:
        return {
            "available":  _OPCUA_OK,
            "running":    self._running,
            "endpoint":   self._endpoint,
            "client_url": self._client_url,
            "client_connected": self._client is not None,
        }


class _CmdVelHandler:
    def __init__(self, lin_node, ang_node, callback):
        self._lin  = lin_node
        self._ang  = ang_node
        self._cb   = callback
        self._lin_val = 0.0
        self._ang_val = 0.0

    async def datachange_notification(self, node, val, data):
        if node == self._lin:
            self._lin_val = float(val)
        elif node == self._ang:
            self._ang_val = float(val)
        if self._cb:
            self._cb(self._lin_val, self._ang_val)
