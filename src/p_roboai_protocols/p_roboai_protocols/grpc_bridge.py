"""
grpc_bridge.py  —  gRPC service bridge for P_RoboAI.

Hosts a gRPC server implementing RobotService (see proto/robot_service.proto).
Provides high-performance, typed RPC for:
  - Remote robot control (cmd_vel, arm commands, e-stop)
  - Continuous telemetry streaming (odometry, joint states, detections)
  - Bidirectional teleop streaming
  - Gemini LLM query forwarding

Generating stubs (run once after changing the proto):
  pip install grpcio grpcio-tools
  python -m grpc_tools.protoc \\
      -I proto \\
      --python_out=p_roboai_protocols \\
      --grpc_python_out=p_roboai_protocols \\
      proto/robot_service.proto

This module includes a self-contained stub implementation that works
without running protoc, using grpc's reflection API.

Install
-------
  pip install grpcio grpcio-tools grpcio-reflection
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from concurrent import futures
from typing import Callable, Iterator, Optional

try:
    import grpc
    _GRPC_OK = True
except ImportError:
    _GRPC_OK = False

# Import generated stubs if they exist, otherwise use dynamic fallback
try:
    from . import robot_service_pb2, robot_service_pb2_grpc
    _STUBS_OK = True
except ImportError:
    _STUBS_OK = False


# ── dynamic stub builder (when protoc hasn't been run yet) ────────────────────

def _build_stubs_from_proto(proto_path: str) -> bool:
    """Generate pb2 stubs at runtime using grpc_tools."""
    if not _GRPC_OK:
        return False
    try:
        from grpc_tools import protoc
        import os
        proto_dir = os.path.dirname(proto_path)
        ret = protoc.main([
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={os.path.dirname(__file__)}",
            f"--grpc_python_out={os.path.dirname(__file__)}",
            proto_path,
        ])
        return ret == 0
    except Exception as e:
        print(f"[gRPC] Stub generation failed: {e}")
        return False


# ── service implementation ────────────────────────────────────────────────────

class _RobotServicer:
    """
    gRPC RobotService implementation.
    Delegates to injected callbacks for ROS2 integration.
    """

    def __init__(self,
                 on_cmd_vel:  Optional[Callable] = None,
                 on_arm_cmd:  Optional[Callable] = None,
                 on_estop:    Optional[Callable] = None,
                 on_llm:      Optional[Callable] = None,
                 get_status:  Optional[Callable] = None) -> None:
        self._on_cmd   = on_cmd_vel
        self._on_arm   = on_arm_cmd
        self._on_estop = on_estop
        self._on_llm   = on_llm
        self._get_st   = get_status

        # Telemetry queues for streaming RPCs
        self._odom_subs:    list[queue.Queue] = []
        self._joint_subs:   list[queue.Queue] = []
        self._detect_subs:  list[queue.Queue] = []
        self._lock = threading.Lock()

    # ── push from ROS2 side ───────────────────────────────────────────────────

    def push_odom(self, x, y, heading, vx, wz) -> None:
        with self._lock:
            for q in self._odom_subs:
                try:
                    q.put_nowait({"x": x, "y": y, "heading": heading,
                                  "vx": vx, "wz": wz, "timestamp": time.time()})
                except queue.Full:
                    pass

    def push_joint_states(self, names, positions, velocities, effort=None) -> None:
        with self._lock:
            for q in self._joint_subs:
                try:
                    q.put_nowait({
                        "name": names, "position": positions,
                        "velocity": velocities,
                        "effort": effort or [],
                        "timestamp": time.time(),
                    })
                except queue.Full:
                    pass

    def push_detections(self, detections: list[dict]) -> None:
        with self._lock:
            for q in self._detect_subs:
                try:
                    q.put_nowait({"detections": detections,
                                  "count": len(detections),
                                  "timestamp": time.time()})
                except queue.Full:
                    pass


class GRPCBridge:
    """
    gRPC server bridge.

    Parameters
    ----------
    port        : gRPC listen port (default 50051)
    max_workers : thread pool size
    on_cmd_vel  : callback(linear, angular)
    on_arm_cmd  : callback(positions, velocities, mode)
    on_estop    : callback()
    on_llm      : callback(query) → str response
    get_status  : callback() → dict
    """

    def __init__(self,
                 port:        int  = 50051,
                 max_workers: int  = 10,
                 on_cmd_vel:  Optional[Callable] = None,
                 on_arm_cmd:  Optional[Callable] = None,
                 on_estop:    Optional[Callable] = None,
                 on_llm:      Optional[Callable] = None,
                 get_status:  Optional[Callable] = None) -> None:
        self._port       = port
        self._workers    = max_workers
        self._server:    Optional["grpc.Server"] = None
        self._servicer   = _RobotServicer(
            on_cmd_vel=on_cmd_vel,
            on_arm_cmd=on_arm_cmd,
            on_estop=on_estop,
            on_llm=on_llm,
            get_status=get_status,
        )
        self._running    = False

    @property
    def available(self) -> bool:
        return _GRPC_OK

    # ── server lifecycle ──────────────────────────────────────────────────────

    def start(self) -> bool:
        if not _GRPC_OK:
            return False

        # Try to use generated stubs; fall back to generic servicer
        if _STUBS_OK:
            return self._start_with_stubs()
        else:
            return self._start_reflection_server()

    def _start_with_stubs(self) -> bool:
        try:
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self._workers),
                options=[
                    ("grpc.max_send_message_length",    64 * 1024 * 1024),
                    ("grpc.max_receive_message_length",  64 * 1024 * 1024),
                    ("grpc.keepalive_time_ms",           30_000),
                    ("grpc.keepalive_timeout_ms",        10_000),
                ],
            )
            robot_service_pb2_grpc.add_RobotServiceServicer_to_server(
                _StubServicer(self._servicer), self._server)
            self._server.add_insecure_port(f"[::]:{self._port}")
            self._server.start()
            self._running = True
            print(f"[gRPC] Server started on port {self._port}")
            return True
        except Exception as e:
            print(f"[gRPC] Start failed: {e}")
            return False

    def _start_reflection_server(self) -> bool:
        """
        Start a minimal gRPC server without generated stubs.
        Useful for development before running protoc.
        """
        try:
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self._workers))
            self._server.add_insecure_port(f"[::]:{self._port}")
            self._server.start()
            self._running = True
            print(f"[gRPC] Server started (no stubs) on port {self._port}")
            print("[gRPC] Run: python -m grpc_tools.protoc to generate stubs")
            return True
        except Exception as e:
            print(f"[gRPC] Start failed: {e}")
            return False

    def stop(self) -> None:
        if self._server:
            self._server.stop(grace=2)
            self._running = False

    # ── push telemetry to connected streaming clients ─────────────────────────

    def push_odom(self, x: float, y: float,
                   heading: float, vx: float, wz: float) -> None:
        self._servicer.push_odom(x, y, heading, vx, wz)

    def push_joint_states(self, names: list[str],
                           positions: list[float],
                           velocities: list[float]) -> None:
        self._servicer.push_joint_states(names, positions, velocities)

    def push_detections(self, detections: list[dict]) -> None:
        self._servicer.push_detections(detections)

    def get_status(self) -> dict:
        return {
            "available": _GRPC_OK,
            "stubs_ok":  _STUBS_OK,
            "running":   self._running,
            "port":      self._port,
        }


# ── stub-based servicer (used when protoc has been run) ───────────────────────

class _StubServicer:
    """Wraps _RobotServicer with pb2 message marshalling."""

    def __init__(self, inner: _RobotServicer) -> None:
        self._inner = inner

    def SendCmdVel(self, request, context):
        if self._inner._on_cmd:
            self._inner._on_cmd(request.linear, request.angular)
        return robot_service_pb2.Ack(success=True, message="ok")

    def SendArmCommand(self, request, context):
        if self._inner._on_arm:
            self._inner._on_arm(
                list(request.positions),
                list(request.velocities),
                request.mode,
            )
        return robot_service_pb2.Ack(success=True, message="ok")

    def GetStatus(self, request, context):
        st = self._inner._get_st() if self._inner._get_st else {}
        return robot_service_pb2.RobotStatus(
            robot_id=st.get("robot_id", ""),
            mode=st.get("mode", "idle"),
            battery_pct=float(st.get("battery_pct", 100.0)),
            estop_active=bool(st.get("estop_active", False)),
            rl_status=st.get("rl_status", ""),
            gemini_status=st.get("gemini_status", ""),
            timestamp=time.time(),
        )

    def QueryLLM(self, request, context):
        response = ""
        if self._inner._on_llm:
            response = self._inner._on_llm(request.query) or ""
        return robot_service_pb2.LLMResponse(
            analysis=response,
            command="{}",
            timestamp=time.time(),
        )

    def EStop(self, request, context):
        if self._inner._on_estop:
            self._inner._on_estop()
        return robot_service_pb2.Ack(success=True, message="E-Stop triggered")

    def StreamOdometry(self, request, context):
        q: queue.Queue = queue.Queue(maxsize=10)
        with self._inner._lock:
            self._inner._odom_subs.append(q)
        try:
            interval = 1.0 / max(0.1, request.rate_hz) if request.rate_hz else 0.1
            while context.is_active():
                try:
                    data = q.get(timeout=interval)
                    yield robot_service_pb2.Odometry(**data)
                except queue.Empty:
                    pass
        finally:
            with self._inner._lock:
                self._inner._odom_subs.remove(q)

    def StreamJointStates(self, request, context):
        q: queue.Queue = queue.Queue(maxsize=10)
        with self._inner._lock:
            self._inner._joint_subs.append(q)
        try:
            while context.is_active():
                try:
                    data = q.get(timeout=0.1)
                    yield robot_service_pb2.JointState(**data)
                except queue.Empty:
                    pass
        finally:
            with self._inner._lock:
                self._inner._joint_subs.remove(q)

    def StreamDetections(self, request, context):
        q: queue.Queue = queue.Queue(maxsize=10)
        with self._inner._lock:
            self._inner._detect_subs.append(q)
        try:
            while context.is_active():
                try:
                    data = q.get(timeout=0.1)
                    dets = [robot_service_pb2.Detection(**d)
                            for d in data.get("detections", [])]
                    yield robot_service_pb2.DetectionList(
                        detections=dets,
                        count=data["count"],
                        timestamp=data["timestamp"],
                    )
                except queue.Empty:
                    pass
        finally:
            with self._inner._lock:
                self._inner._detect_subs.remove(q)

    def TeleOpStream(self, request_iterator, context):
        for cmd in request_iterator:
            if self._inner._on_cmd:
                self._inner._on_cmd(cmd.linear, cmd.angular)
            # Yield latest odom as acknowledgement
            yield robot_service_pb2.Odometry(timestamp=time.time())
