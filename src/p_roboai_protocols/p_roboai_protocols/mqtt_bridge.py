"""
mqtt_bridge.py  —  MQTT bridge for cloud robotics and IoT.

Connects to an MQTT broker and bridges ROS2 topics to MQTT topics
for fleet management, remote monitoring, and cloud dashboards.

Topic mapping (default)
-----------------------
  ROS2 → MQTT (publish):
    /odom               → p_roboai/{robot_id}/odom          (JSON)
    /joint_states       → p_roboai/{robot_id}/joint_states  (JSON)
    /yolo/detections    → p_roboai/{robot_id}/detections     (JSON)
    /amr_rl/status      → p_roboai/{robot_id}/rl_status      (JSON)
    /gemini/response    → p_roboai/{robot_id}/llm_response   (string)

  MQTT → ROS2 (subscribe):
    p_roboai/{robot_id}/cmd_vel       → /cmd_vel
    p_roboai/{robot_id}/arm_command   → /arm/position_commands
    p_roboai/{robot_id}/goal          → /move_base_simple/goal

Install
-------
  pip install paho-mqtt
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

try:
    import paho.mqtt.client as mqtt
    _MQTT_OK = True
except ImportError:
    _MQTT_OK = False


class MQTTBridge:
    """
    MQTT ↔ ROS2 bridge.

    Parameters
    ----------
    broker      : MQTT broker hostname or IP
    port        : broker port (1883 plain, 8883 TLS, 9001 WebSocket)
    robot_id    : unique robot identifier used in topic paths
    username    : broker username (optional)
    password    : broker password (optional)
    use_tls     : enable TLS (requires CA cert)
    ca_cert     : path to CA certificate for TLS
    qos         : MQTT QoS level (0, 1, 2)
    on_cmd_vel  : callback(linear, angular)
    on_goal     : callback(x, y, theta)
    on_arm_cmd  : callback(list[float])
    """

    BASE_TOPIC = "p_roboai"

    def __init__(self,
                 broker:     str = "localhost",
                 port:       int = 1883,
                 robot_id:   str = "robot_01",
                 username:   str = "",
                 password:   str = "",
                 use_tls:    bool = False,
                 ca_cert:    str = "",
                 qos:        int = 1,
                 on_cmd_vel: Optional[Callable] = None,
                 on_goal:    Optional[Callable] = None,
                 on_arm_cmd: Optional[Callable] = None) -> None:
        self._broker    = broker
        self._port      = port
        self._robot_id  = robot_id
        self._qos       = qos
        self._on_cmd    = on_cmd_vel
        self._on_goal   = on_goal
        self._on_arm    = on_arm_cmd
        self._client:   Optional["mqtt.Client"] = None
        self._connected = False
        self._lock      = threading.Lock()
        self._stats     = {"published": 0, "received": 0, "errors": 0}

        if not _MQTT_OK:
            return

        self._client = mqtt.Client(
            client_id=f"p_roboai_{robot_id}",
            protocol=mqtt.MQTTv5,
        )
        if username:
            self._client.username_pw_set(username, password)
        if use_tls and ca_cert:
            self._client.tls_set(ca_cert)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    @property
    def available(self) -> bool:
        return _MQTT_OK

    def _topic(self, suffix: str) -> str:
        return f"{self.BASE_TOPIC}/{self._robot_id}/{suffix}"

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, keepalive: int = 60) -> bool:
        if not _MQTT_OK or not self._client:
            return False
        try:
            self._client.connect_async(self._broker, self._port, keepalive)
            self._client.loop_start()
            timeout = 5.0
            while not self._connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            return self._connected
        except Exception as e:
            print(f"[MQTT] Connect failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc == 0:
            self._connected = True
            # Subscribe to inbound command topics
            subs = [
                (self._topic("cmd_vel"),     self._qos),
                (self._topic("arm_command"), self._qos),
                (self._topic("goal"),        self._qos),
                (self._topic("estop"),       0),
            ]
            client.subscribe(subs)
        else:
            print(f"[MQTT] Connect refused: rc={rc}")

    def _on_disconnect(self, client, userdata, rc, props=None) -> None:
        self._connected = False

    def _on_message(self, client, userdata, msg) -> None:
        with self._lock:
            self._stats["received"] += 1
        try:
            payload = json.loads(msg.payload.decode())
            topic   = msg.topic

            if topic == self._topic("cmd_vel") and self._on_cmd:
                self._on_cmd(
                    float(payload.get("linear",  0.0)),
                    float(payload.get("angular", 0.0)),
                )
            elif topic == self._topic("goal") and self._on_goal:
                self._on_goal(
                    float(payload.get("x",     0.0)),
                    float(payload.get("y",     0.0)),
                    float(payload.get("theta", 0.0)),
                )
            elif topic == self._topic("arm_command") and self._on_arm:
                self._on_arm(
                    [float(v) for v in payload.get("positions", [])])
            elif topic == self._topic("estop"):
                print("[MQTT] E-Stop received!")
        except Exception as e:
            with self._lock:
                self._stats["errors"] += 1

    # ── publish helpers ───────────────────────────────────────────────────────

    def publish(self, suffix: str, payload: dict, retain: bool = False) -> bool:
        if not self._connected:
            return False
        try:
            self._client.publish(
                self._topic(suffix),
                json.dumps(payload),
                qos=self._qos,
                retain=retain,
            )
            with self._lock:
                self._stats["published"] += 1
            return True
        except Exception:
            with self._lock:
                self._stats["errors"] += 1
            return False

    def publish_odom(self, x: float, y: float,
                      heading: float, vx: float, wz: float) -> None:
        self.publish("odom", {
            "x": round(x, 3), "y": round(y, 3),
            "heading": round(heading, 4),
            "vx": round(vx, 3), "wz": round(wz, 3),
            "ts": time.time(),
        })

    def publish_joint_states(self,
                              names: list[str],
                              positions: list[float],
                              velocities: list[float]) -> None:
        self.publish("joint_states", {
            "names":     names,
            "positions": [round(p, 4) for p in positions],
            "velocities":[round(v, 4) for v in velocities],
            "ts":        time.time(),
        })

    def publish_detections(self, detections: list[dict]) -> None:
        self.publish("detections", {
            "count":      len(detections),
            "detections": detections,
            "ts":         time.time(),
        })

    def publish_rl_status(self, episode: int, reward: float,
                           step: int, mode: str) -> None:
        self.publish("rl_status", {
            "episode": episode,
            "reward":  round(reward, 3),
            "step":    step,
            "mode":    mode,
            "ts":      time.time(),
        })

    def publish_llm_response(self, query: str, response: str) -> None:
        self.publish("llm_response", {
            "query":    query[:500],
            "response": response[:2000],
            "ts":       time.time(),
        })

    def publish_status(self, status: dict) -> None:
        self.publish("status", {**status, "ts": time.time()}, retain=True)

    def publish_alarm(self, code: int, message: str, severity: str = "WARNING") -> None:
        self.publish("alarm", {
            "code":     code,
            "message":  message,
            "severity": severity,
            "ts":       time.time(),
        })

    def get_status(self) -> dict:
        return {
            "available":  _MQTT_OK,
            "connected":  self._connected,
            "broker":     self._broker,
            "port":       self._port,
            "robot_id":   self._robot_id,
            "stats":      dict(self._stats),
        }
