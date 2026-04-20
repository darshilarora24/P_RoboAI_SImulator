"""
protocol_manager_node.py  —  P_RoboAI Protocol Manager ROS2 Node.

Single node that manages all protocol bridges simultaneously:
  DDS  | OPC UA | Modbus | CAN/CANopen | EtherCAT | PROFINET | MQTT | gRPC | WebSocket

Subscribes (ROS2 → external protocols)
---------------------------------------
  /joint_states       → OPC UA, Modbus, MQTT, gRPC, WebSocket
  /odom               → OPC UA, Modbus, MQTT, gRPC, WebSocket
  /yolo/detections    → MQTT, gRPC, WebSocket
  /amr_rl/status      → MQTT, WebSocket
  /gemini/response    → MQTT, WebSocket

Publishes (external protocols → ROS2)
--------------------------------------
  /cmd_vel                  ← OPC UA, Modbus, MQTT, gRPC, WebSocket
  /protocol/status          ← periodic status JSON for all protocols
  /arm/position_commands    ← gRPC ArmCommand

Parameters
----------
  enable_opcua      : bool   (default false)
  opcua_endpoint    : str    (default "opc.tcp://0.0.0.0:4840/p_roboai")
  opcua_client_url  : str

  enable_modbus     : bool   (default false)
  modbus_host       : str    (default "0.0.0.0")
  modbus_port       : int    (default 5020)

  enable_can        : bool   (default false)
  can_interface     : str    (default "socketcan")
  can_channel       : str    (default "can0")
  can_bitrate       : int    (default 500000)

  enable_ethercat   : bool   (default false)
  ethercat_iface    : str    (default "eth0")

  enable_profinet   : bool   (default false)
  profinet_iface    : str    (default "eth0")
  plc_ip            : str

  enable_mqtt       : bool   (default false)
  mqtt_broker       : str    (default "localhost")
  mqtt_port         : int    (default 1883)
  mqtt_robot_id     : str    (default "robot_01")
  mqtt_username     : str
  mqtt_password     : str

  enable_grpc       : bool   (default false)
  grpc_port         : int    (default 50051)

  enable_websocket  : bool   (default true)
  ws_port           : int    (default 8765)
  http_port         : int    (default 8766)

  dds_rmw           : str    (default "") — switch RMW implementation
  dds_profile       : str    (default "fastdds") — write profile XML
"""
from __future__ import annotations

import asyncio
import json
import math
import site
import sys
import threading
import time
from pathlib import Path


def _bootstrap_venv() -> None:
    for root in Path(__file__).resolve().parents:
        venv = root / ".venv"
        if venv.exists():
            for sp in venv.glob("lib/python*/site-packages"):
                if str(sp) not in sys.path:
                    site.addsitedir(str(sp))
                    sys.path.insert(0, str(sp))
            break

_bootstrap_venv()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist

try:
    from sensor_msgs.msg import JointState
    from nav_msgs.msg import Odometry
    _MSGS_OK = True
except ImportError:
    _MSGS_OK = False

from .dds_config      import DDSConfig
from .opcua_bridge    import OPCUABridge
from .modbus_bridge   import ModbusBridge
from .canopen_bridge  import CANopenBridge, CANBus
from .ethercat_bridge import EtherCATBridge
from .profinet_bridge import PROFINETDiscovery, S7Bridge
from .mqtt_bridge     import MQTTBridge
from .grpc_bridge     import GRPCBridge
from .websocket_bridge import WebSocketBridge


class ProtocolManagerNode(Node):

    def __init__(self) -> None:
        super().__init__("protocol_manager")
        self._declare_params()
        self._load_params()

        # Publishers
        self._pub_cmd    = self.create_publisher(Twist,             "/cmd_vel",               10)
        self._pub_arm    = self.create_publisher(Float64MultiArray, "/arm/position_commands",  10)
        self._pub_status = self.create_publisher(String,            "/protocol/status",        10)

        # Subscribers
        if _MSGS_OK:
            self.create_subscription(JointState, "/joint_states",    self._on_joints,    10)
            self.create_subscription(Odometry,   "/odom",            self._on_odom,      10)
        self.create_subscription(String, "/yolo/detections", self._on_detections, 10)
        self.create_subscription(String, "/amr_rl/status",   self._on_rl_status,  10)
        self.create_subscription(String, "/gemini/response", self._on_llm_resp,   10)

        # Init DDS config
        self._dds = DDSConfig()
        if self._p["dds_rmw"]:
            self._dds.set_rmw(self._p["dds_rmw"])
        if self._p["dds_profile"] == "fastdds":
            self._dds.write_fastdds_profile()
        elif self._p["dds_profile"] == "cyclone":
            self._dds.write_cyclone_config()

        # Init bridges
        self._opcua   = self._init_opcua()
        self._modbus  = self._init_modbus()
        self._can     = self._init_can()
        self._ecat    = self._init_ethercat()
        self._profinet = self._init_profinet()
        self._mqtt    = self._init_mqtt()
        self._grpc    = self._init_grpc()
        self._ws      = self._init_websocket()

        # Modbus poll timer (checks for external writes at 10 Hz)
        if self._modbus and self._modbus.available:
            self.create_timer(0.1, self._poll_modbus)

        # Status publish timer (1 Hz)
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info("Protocol Manager started")
        self._log_enabled()

    # ── parameter declaration ─────────────────────────────────────────────────

    def _declare_params(self) -> None:
        defaults = {
            "enable_opcua":    False,
            "opcua_endpoint":  "opc.tcp://0.0.0.0:4840/p_roboai",
            "opcua_client_url": "",
            "enable_modbus":   False,
            "modbus_host":     "0.0.0.0",
            "modbus_port":     5020,
            "enable_can":      False,
            "can_interface":   "socketcan",
            "can_channel":     "can0",
            "can_bitrate":     500000,
            "enable_ethercat": False,
            "ethercat_iface":  "eth0",
            "enable_profinet": False,
            "profinet_iface":  "eth0",
            "plc_ip":          "",
            "enable_mqtt":     False,
            "mqtt_broker":     "localhost",
            "mqtt_port":       1883,
            "mqtt_robot_id":   "robot_01",
            "mqtt_username":   "",
            "mqtt_password":   "",
            "enable_grpc":     False,
            "grpc_port":       50051,
            "enable_websocket": True,
            "ws_port":         8765,
            "http_port":       8766,
            "dds_rmw":         "",
            "dds_profile":     "fastdds",
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)

    def _load_params(self) -> None:
        self._p = {k: self.get_parameter(k).value
                   for k in [
                       "enable_opcua", "opcua_endpoint", "opcua_client_url",
                       "enable_modbus", "modbus_host", "modbus_port",
                       "enable_can", "can_interface", "can_channel", "can_bitrate",
                       "enable_ethercat", "ethercat_iface",
                       "enable_profinet", "profinet_iface", "plc_ip",
                       "enable_mqtt", "mqtt_broker", "mqtt_port",
                       "mqtt_robot_id", "mqtt_username", "mqtt_password",
                       "enable_grpc", "grpc_port",
                       "enable_websocket", "ws_port", "http_port",
                       "dds_rmw", "dds_profile",
                   ]}

    # ── bridge initialisation ─────────────────────────────────────────────────

    def _init_opcua(self) -> OPCUABridge:
        b = OPCUABridge(
            endpoint=self._p["opcua_endpoint"],
            client_url=self._p["opcua_client_url"],
            on_cmd_vel=self._on_ext_cmd_vel,
        )
        if self._p["enable_opcua"] and b.available:
            threading.Thread(
                target=lambda: asyncio.run(b.start_server()),
                daemon=True).start()
            self.get_logger().info(
                f"OPC UA server: {self._p['opcua_endpoint']}")
        return b

    def _init_modbus(self) -> ModbusBridge:
        b = ModbusBridge(
            host=self._p["modbus_host"],
            port=int(self._p["modbus_port"]),
            on_estop=self._on_ext_estop,
            on_cmd_vel=self._on_ext_cmd_vel,
        )
        if self._p["enable_modbus"]:
            ok = b.start()
            if ok:
                self.get_logger().info(
                    f"Modbus TCP server: {self._p['modbus_host']}:{self._p['modbus_port']}")
        return b

    def _init_can(self) -> CANopenBridge:
        b = CANopenBridge(
            interface=self._p["can_interface"],
            channel=self._p["can_channel"],
            bitrate=int(self._p["can_bitrate"]),
        )
        if self._p["enable_can"] and b.available:
            ok = b.connect()
            if ok:
                self.get_logger().info(
                    f"CANopen: {self._p['can_interface']} ch={self._p['can_channel']}")
        return b

    def _init_ethercat(self) -> EtherCATBridge:
        def _on_ecat_fb(slave_pos, feedback):
            pass  # forward to ROS2 joint_states publisher if needed
        b = EtherCATBridge(
            interface=self._p["ethercat_iface"],
            on_feedback=_on_ecat_fb,
        )
        if self._p["enable_ethercat"] and b.available:
            ok = b.init()
            if ok:
                b.start_cyclic()
                self.get_logger().info(
                    f"EtherCAT master: {self._p['ethercat_iface']}")
        return b

    def _init_profinet(self):
        disc = PROFINETDiscovery(
            interface=self._p["profinet_iface"],
            on_device=lambda d: self.get_logger().info(
                f"[PROFINET] Device found: {d}"),
        )
        if self._p["enable_profinet"]:
            disc.start()
            self.get_logger().info(
                f"PROFINET discovery: {self._p['profinet_iface']}")
        s7 = None
        if self._p["plc_ip"]:
            s7 = S7Bridge(ip=self._p["plc_ip"])
            ok = s7.connect()
            if ok:
                self.get_logger().info(
                    f"S7 PLC connected: {self._p['plc_ip']}")
        return {"discovery": disc, "s7": s7}

    def _init_mqtt(self) -> MQTTBridge:
        b = MQTTBridge(
            broker=self._p["mqtt_broker"],
            port=int(self._p["mqtt_port"]),
            robot_id=self._p["mqtt_robot_id"],
            username=self._p["mqtt_username"],
            password=self._p["mqtt_password"],
            on_cmd_vel=self._on_ext_cmd_vel,
            on_goal=self._on_ext_goal,
            on_arm_cmd=self._on_ext_arm_cmd,
        )
        if self._p["enable_mqtt"] and b.available:
            ok = b.connect()
            if ok:
                self.get_logger().info(
                    f"MQTT connected: {self._p['mqtt_broker']}:{self._p['mqtt_port']}")
        return b

    def _init_grpc(self) -> GRPCBridge:
        b = GRPCBridge(
            port=int(self._p["grpc_port"]),
            on_cmd_vel=self._on_ext_cmd_vel,
            on_arm_cmd=self._on_ext_arm_cmd,
            on_estop=self._on_ext_estop,
            on_llm=None,   # wire to gemini node if needed
            get_status=self._build_status,
        )
        if self._p["enable_grpc"] and b.available:
            ok = b.start()
            if ok:
                self.get_logger().info(
                    f"gRPC server: port {self._p['grpc_port']}")
        return b

    def _init_websocket(self) -> WebSocketBridge:
        b = WebSocketBridge(
            ws_port=int(self._p["ws_port"]),
            http_port=int(self._p["http_port"]),
            on_cmd_vel=self._on_ext_cmd_vel,
            on_estop=self._on_ext_estop,
            on_llm=None,
        )
        if self._p["enable_websocket"] and b.available:
            ok = b.start()
            if ok:
                self.get_logger().info(
                    f"WebSocket: ws://0.0.0.0:{self._p['ws_port']}  "
                    f"Dashboard: http://0.0.0.0:{self._p['http_port']}/")
        return b

    # ── external command callbacks (protocol → ROS2) ──────────────────────────

    def _on_ext_cmd_vel(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self._pub_cmd.publish(msg)

    def _on_ext_estop(self) -> None:
        self.get_logger().warn("E-STOP received from external protocol!")
        self._on_ext_cmd_vel(0.0, 0.0)

    def _on_ext_goal(self, x: float, y: float, theta: float) -> None:
        try:
            from geometry_msgs.msg import PoseStamped
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.header.stamp    = self.get_clock().now().to_msg()
            ps.pose.position.x = x
            ps.pose.position.y = y
            import math
            ps.pose.orientation.z = math.sin(theta / 2)
            ps.pose.orientation.w = math.cos(theta / 2)
            if not hasattr(self, "_pub_goal"):
                from geometry_msgs.msg import PoseStamped
                self._pub_goal = self.create_publisher(
                    PoseStamped, "/move_base_simple/goal", 10)
            self._pub_goal.publish(ps)
        except Exception:
            pass

    def _on_ext_arm_cmd(self, positions: list[float],
                         velocities: list[float] = None,
                         mode: str = "position") -> None:
        msg = Float64MultiArray()
        msg.data = [float(p) for p in positions]
        self._pub_arm.publish(msg)

    # ── ROS2 subscriber callbacks (ROS2 → external protocols) ─────────────────

    def _on_joints(self, msg: "JointState") -> None:
        names = list(msg.name)
        pos   = list(msg.position)
        vel   = list(msg.velocity) if msg.velocity else [0.0] * len(pos)
        eff   = list(msg.effort)   if msg.effort   else [0.0] * len(pos)

        if self._p["enable_opcua"]:
            asyncio.run_coroutine_threadsafe(
                self._opcua.update_joint_states(pos, vel),
                asyncio.get_event_loop() if asyncio.get_event_loop().is_running()
                else asyncio.new_event_loop(),
            )
        if self._p["enable_modbus"]:
            self._modbus.update_joint_states(pos, vel)
        if self._p["enable_mqtt"]:
            self._mqtt.publish_joint_states(names, pos, vel)
        if self._p["enable_grpc"]:
            self._grpc.push_joint_states(names, pos, vel)
        if self._p["enable_websocket"]:
            self._ws.broadcast_joint_states(names, pos, vel)

    def _on_odom(self, msg: "Odometry") -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist
        # quaternion → heading (yaw)
        heading = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

        if self._p["enable_opcua"]:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        self._opcua.update_odom(p.x, p.y, heading))
            except Exception:
                pass
        if self._p["enable_modbus"]:
            self._modbus.update_odom(p.x, p.y, math.degrees(heading))
        if self._p["enable_mqtt"]:
            self._mqtt.publish_odom(p.x, p.y, heading, t.linear.x, t.angular.z)
        if self._p["enable_grpc"]:
            self._grpc.push_odom(p.x, p.y, heading, t.linear.x, t.angular.z)
        if self._p["enable_websocket"]:
            self._ws.broadcast_odom(p.x, p.y, heading, t.linear.x, t.angular.z)

        # S7 PLC push
        s7 = self._profinet.get("s7")
        if s7 and self._p["enable_profinet"]:
            s7.push_robot_state(10, [], p.x, p.y, math.degrees(heading))

    def _on_detections(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            dets = data.get("detections", [])
            if self._p["enable_mqtt"]:
                self._mqtt.publish_detections(dets)
            if self._p["enable_grpc"]:
                self._grpc.push_detections(dets)
            if self._p["enable_websocket"]:
                self._ws.broadcast_detections(dets)
            if self._p["enable_opcua"]:
                try:
                    asyncio.ensure_future(
                        self._opcua.update_detections(len(dets)))
                except Exception:
                    pass
        except Exception:
            pass

    def _on_rl_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            ep   = data.get("episode", 0)
            rew  = data.get("reward",  0.0)
            step = data.get("step",    0)
            mode = data.get("mode",    "train")
            if self._p["enable_mqtt"]:
                self._mqtt.publish_rl_status(ep, rew, step, mode)
            if self._p["enable_websocket"]:
                self._ws.broadcast_rl_status(ep, rew, step, mode)
        except Exception:
            pass

    def _on_llm_resp(self, msg: String) -> None:
        if self._p["enable_mqtt"]:
            self._mqtt.publish_llm_response("", msg.data)
        if self._p["enable_websocket"]:
            self._ws.broadcast_llm_response(msg.data)

    # ── Modbus poll ───────────────────────────────────────────────────────────

    def _poll_modbus(self) -> None:
        if self._p["enable_modbus"]:
            self._modbus.poll_writes()

    # ── status ────────────────────────────────────────────────────────────────

    def _build_status(self) -> dict:
        s7 = self._profinet.get("s7")
        return {
            "dds":       self._dds.get_status(),
            "opcua":     self._opcua.get_status(),
            "modbus":    self._modbus.get_status(),
            "canopen":   self._can.get_status(),
            "ethercat":  self._ecat.get_status(),
            "profinet":  {"discovery": True, "s7": s7.get_status() if s7 else {}},
            "mqtt":      self._mqtt.get_status(),
            "grpc":      self._grpc.get_status(),
            "websocket": self._ws.get_status(),
            "timestamp": time.time(),
        }

    def _publish_status(self) -> None:
        self._pub_status.publish(
            String(data=json.dumps(self._build_status())))

    def _log_enabled(self) -> None:
        enabled = [k.replace("enable_","").upper()
                   for k, v in self._p.items()
                   if k.startswith("enable_") and v]
        self.get_logger().info(
            f"Active protocols: {', '.join(enabled) if enabled else 'none (all disabled)'}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ProtocolManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
