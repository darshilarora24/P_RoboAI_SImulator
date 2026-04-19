"""
gemini_robot_node.py  —  Gemini Robotics LLM ROS2 Node.

Integrates Google Gemini 2.0 Flash (multimodal) with:
  - RAG over robot knowledge base (URDF specs, task docs, sensor history)
  - Sim2Real adapter (ONNX policy loading, calibration, domain gap tracking)
  - Real2Sim adapter (online system identification from real sensors)

Subscribes
----------
  /joint_states       sensor_msgs/JointState
  /odom               nav_msgs/Odometry
  /scan               sensor_msgs/LaserScan
  /amr/camera/image   sensor_msgs/Image
  /gemini/query       std_msgs/String          — natural language task query

Publishes
---------
  /gemini/response        std_msgs/String   — Gemini NL response
  /gemini/status          std_msgs/String   — node status JSON
  /gemini/robot_command   std_msgs/String   — structured robot command JSON
  /gemini/domain_gap      std_msgs/String   — Sim2Real domain gap JSON
  /gemini/real2sim_params std_msgs/String   — identified sim params JSON

Parameters
----------
  api_key       : str   — Google API key (or set GOOGLE_API_KEY env var)
  model_name    : str   — Gemini model ID (default: gemini-2.0-flash)
  policy_zip    : str   — SB3 policy ZIP for Sim2Real
  onnx_path     : str   — Exported ONNX policy path
  calib_path    : str   — Calibration JSON path
  kb_persist    : str   — Knowledge base persistence JSON path
  robot_type    : str   — "arm" or "amr"
  n_joints      : int   — Number of DOF (for Real2Sim)
  enable_real2sim : bool
  enable_sim2real : bool
"""
from __future__ import annotations

import json
import os
import sys
import site
import time
import threading
from pathlib import Path
from typing import Optional


# ── venv bootstrap ────────────────────────────────────────────────────────────

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
from std_msgs.msg import String

try:
    from sensor_msgs.msg import JointState, Image, LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    _ROS_MSGS_OK = True
except ImportError:
    _ROS_MSGS_OK = False

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

from .rag_engine     import RobotKnowledgeBase
from .sim2real_bridge import Sim2RealAdapter
from .real2sim_bridge import Real2SimAdapter


# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """You are a robotics AI assistant integrated into a robot control system.
You have access to the robot's current sensor state and a knowledge base of robot specifications.

Your role:
1. Interpret natural language task commands for the robot
2. Provide situational awareness based on sensor data
3. Generate structured robot commands when asked
4. Explain robot behavior and state in natural language
5. Advise on sim-to-real transfer quality and system identification

Always respond in two parts:
ANALYSIS: <brief situational analysis>
COMMAND: <JSON command or "none" if informational only>

Valid command types: "navigate_to", "pick", "place", "arm_move", "stop", "query"
"""

_SYSTEM_PROMPT_ER = """You are Gemini Robotics ER (Embodied Reasoning), an advanced robotics AI
integrated into a real-time robot control system with MuJoCo simulation, RL policies, and sensors.

Your embodied reasoning capabilities:
1. Spatial understanding: interpret 3D poses, distances, collision geometries from sensor data
2. Manipulation planning: generate precise joint-space and task-space arm motion plans
3. Navigation reasoning: path planning, obstacle avoidance, goal-directed AMR control
4. Sim2Real awareness: flag when sim-policy transfer may fail; suggest calibration steps
5. Real2Sim grounding: interpret identified dynamic parameters and their physical meaning
6. Multimodal perception: reason jointly over camera frames, LiDAR scans, joint states

Always respond in two parts:
ANALYSIS: <embodied spatial/dynamic analysis with numerical reasoning>
COMMAND: <JSON command or "none" if informational only>

Valid command types: "navigate_to", "pick", "place", "arm_move", "stop", "query",
                    "set_joint", "cartesian_move", "grasp", "release"
"""


# ── Gemini client wrapper ─────────────────────────────────────────────────────

class _GeminiClient:
    DEFAULT_MODEL = "gemini-robotics-er-1.6"

    def __init__(self, api_key: str, model_name: str) -> None:
        self._ok    = False
        self._model = None
        if not _GENAI_OK:
            return
        model_id = model_name or self.DEFAULT_MODEL
        is_er    = "robotics-er" in model_id
        sys_prompt = _SYSTEM_PROMPT_ER if is_er else _SYSTEM_PROMPT_BASE
        try:
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""))
            self._model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=sys_prompt,
            )
            self._ok = True
        except Exception as e:
            print(f"[Gemini] Init failed: {e}")

    @property
    def ready(self) -> bool:
        return self._ok

    def query(self, prompt: str, image_bytes: Optional[bytes] = None) -> str:
        if not self._ok:
            return '{"error": "Gemini not available — set GOOGLE_API_KEY"}'
        try:
            parts: list = [prompt]
            if image_bytes and _GENAI_OK:
                parts = [{"mime_type": "image/jpeg", "data": image_bytes}, prompt]
            response = self._model.generate_content(parts)
            return response.text
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class GeminiRobotNode(Node):

    def __init__(self) -> None:
        super().__init__("gemini_robot_node")

        # Parameters
        self.declare_parameter("api_key",         "")
        self.declare_parameter("model_name",      "gemini-robotics-er-1.6")
        self.declare_parameter("policy_zip",      "")
        self.declare_parameter("onnx_path",       "")
        self.declare_parameter("calib_path",      "")
        self.declare_parameter("kb_persist",      "")
        self.declare_parameter("robot_type",      "amr")
        self.declare_parameter("n_joints",        6)
        self.declare_parameter("enable_real2sim", True)
        self.declare_parameter("enable_sim2real", False)

        api_key      = self.get_parameter("api_key").value
        model_name   = self.get_parameter("model_name").value
        policy_zip   = self.get_parameter("policy_zip").value
        onnx_path    = self.get_parameter("onnx_path").value
        calib_path   = self.get_parameter("calib_path").value
        kb_persist   = self.get_parameter("kb_persist").value
        self._robot  = self.get_parameter("robot_type").value
        n_joints     = int(self.get_parameter("n_joints").value)
        en_r2s       = self.get_parameter("enable_real2sim").value
        en_s2r       = self.get_parameter("enable_sim2real").value

        # Publishers
        self._pub_resp   = self.create_publisher(String, "/gemini/response",        10)
        self._pub_status = self.create_publisher(String, "/gemini/status",           10)
        self._pub_cmd    = self.create_publisher(String, "/gemini/robot_command",    10)
        self._pub_gap    = self.create_publisher(String, "/gemini/domain_gap",       10)
        self._pub_r2s    = self.create_publisher(String, "/gemini/real2sim_params",  10)

        # Gemini client
        self._gemini = _GeminiClient(api_key, model_name)

        # RAG knowledge base
        self._kb = RobotKnowledgeBase(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
            persist_path=kb_persist,
        )
        self._init_default_knowledge()

        # Sim2Real
        self._s2r: Optional[Sim2RealAdapter] = None
        if en_s2r:
            self._s2r = Sim2RealAdapter()
            if onnx_path and Path(onnx_path).exists():
                self._s2r.load_onnx(onnx_path)
            elif policy_zip and Path(policy_zip).exists():
                out = str(Path(policy_zip).with_suffix(".onnx"))
                if Sim2RealAdapter.export_onnx(policy_zip, out):
                    self._s2r.load_onnx(out)
            if calib_path and Path(calib_path).exists():
                self._s2r.load_calibration(calib_path)

        # Real2Sim
        self._r2s: Optional[Real2SimAdapter] = None
        if en_r2s:
            self._r2s = Real2SimAdapter(n_joints=n_joints)

        # Sensor state
        self._joint_state: Optional[JointState] = None
        self._odom:        Optional[Odometry]   = None
        self._last_image:  Optional[bytes]      = None
        self._query_lock   = threading.Lock()
        self._pending_query: Optional[str]      = None

        # Subscribers
        if _ROS_MSGS_OK:
            self.create_subscription(String,     "/gemini/query",       self._on_query,  10)
            self.create_subscription(JointState, "/joint_states",       self._on_joints, 10)
            self.create_subscription(Odometry,   "/odom",               self._on_odom,   10)
            self.create_subscription(Image,      "/amr/camera/image",   self._on_image,  10)
            self.create_subscription(LaserScan,  "/scan",               self._on_scan,   10)

        # Timers
        self.create_timer(0.5,  self._process_query)   # process pending queries
        self.create_timer(2.0,  self._publish_status)
        self.create_timer(5.0,  self._publish_r2s)

        self._pub_status.publish(String(data=json.dumps({
            "gemini_ready":  self._gemini.ready,
            "sim2real":      en_s2r,
            "real2sim":      en_r2s,
            "robot_type":    self._robot,
            "kb_chunks":     0,
        })))
        self.get_logger().info(
            f"Gemini Robot Node started  model={model_name}  "
            f"gemini={'OK' if self._gemini.ready else 'NO KEY'}  "
            f"real2sim={'on' if en_r2s else 'off'}  "
            f"sim2real={'on' if en_s2r else 'off'}"
        )

    # ── sensor callbacks ──────────────────────────────────────────────────────

    def _on_query(self, msg: String) -> None:
        with self._query_lock:
            self._pending_query = msg.data

    def _on_joints(self, msg: "JointState") -> None:
        self._joint_state = msg
        if self._r2s and _NP_OK:
            n = len(msg.name)
            pos = list(msg.position[:n]) if msg.position else [0.0] * n
            vel = list(msg.velocity[:n]) if msg.velocity else [0.0] * n
            eff = list(msg.effort[:n])   if msg.effort   else [0.0] * n
            self._r2s.update(
                actions=pos,    # use position as proxy for command
                joint_vel=vel,
                joint_torque=eff,
            )

    def _on_odom(self, msg: "Odometry") -> None:
        self._odom = msg
        if self._r2s:
            twist = msg.twist.twist
            self._r2s.update_from_odom(
                cmd_vel_linear=0.0,   # would need stored cmd_vel
                cmd_vel_angular=0.0,
                odom_linear=twist.linear.x,
                odom_angular=twist.angular.z,
            )

    def _on_image(self, msg: "Image") -> None:
        # Store raw bytes for multimodal queries
        try:
            if _NP_OK:
                import numpy as np
                frame = np.frombuffer(msg.data, np.uint8).reshape(
                    msg.height, msg.width, -1)
                # Simple JPEG encode via numpy (no cv2)
                # Store as raw RGB bytes — convert to JPEG only if PIL available
                try:
                    from io import BytesIO
                    from PIL import Image as PILImage
                    pil = PILImage.fromarray(frame[:, :, :3])
                    buf = BytesIO()
                    pil.save(buf, format="JPEG", quality=70)
                    self._last_image = buf.getvalue()
                except ImportError:
                    self._last_image = None
        except Exception:
            pass

    def _on_scan(self, msg: "LaserScan") -> None:
        # Store summary in KB every 30s
        pass

    # ── query processing ──────────────────────────────────────────────────────

    def _process_query(self) -> None:
        with self._query_lock:
            query = self._pending_query
            self._pending_query = None
        if not query:
            return
        threading.Thread(target=self._run_query, args=(query,), daemon=True).start()

    def _run_query(self, query: str) -> None:
        # Build context
        sensor_ctx = self._build_sensor_context()
        rag_results = self._kb.retrieve(query, top_k=4)
        rag_ctx     = self._kb.format_context(rag_results)

        full_prompt = (
            f"=== Robot Sensor State ===\n{sensor_ctx}\n\n"
            f"{rag_ctx}\n\n"
            f"=== User Query ===\n{query}"
        )

        response = self._gemini.query(full_prompt, self._last_image)

        self._pub_resp.publish(String(data=response))

        # Extract and publish structured command
        cmd = self._extract_command(response)
        if cmd:
            self._pub_cmd.publish(String(data=json.dumps(cmd)))

        # Log query as sensor snapshot in KB
        self._kb.add_sensor_snapshot({
            "query": query[:200],
            "timestamp": time.time(),
        })

    def _build_sensor_context(self) -> str:
        lines = []
        if self._joint_state:
            js = self._joint_state
            joints = dict(zip(js.name, js.position))
            lines.append(f"Joint positions: {json.dumps({k: round(v, 3) for k, v in joints.items()})}")
        if self._odom:
            p = self._odom.pose.pose.position
            t = self._odom.twist.twist
            lines.append(f"Position: ({p.x:.2f}, {p.y:.2f}), "
                         f"Velocity: linear={t.linear.x:.2f} angular={t.angular.z:.2f}")
        if not lines:
            lines.append("No sensor data available yet.")
        return "\n".join(lines)

    @staticmethod
    def _extract_command(response: str) -> Optional[dict]:
        import re
        # Look for COMMAND: {...} in response
        m = re.search(r"COMMAND:\s*(\{.*?\})", response, re.DOTALL)
        if not m:
            return None
        try:
            cmd = json.loads(m.group(1))
            if cmd.get("type") == "none" or "none" in str(cmd).lower():
                return None
            return cmd
        except Exception:
            return None

    # ── periodic status ───────────────────────────────────────────────────────

    def _publish_status(self) -> None:
        status = {
            "gemini_ready": self._gemini.ready,
            "kb_chunks":    len(self._kb._chunks),
            "robot_type":   self._robot,
            "timestamp":    time.time(),
        }
        if self._s2r:
            status["domain_gap"]  = round(self._s2r.domain_gap, 4)
            status["gap_trend"]   = self._s2r.domain_gap_trend
            status["dr_suggest"]  = self._s2r.suggest_domain_randomization()
            self._pub_gap.publish(String(data=json.dumps({
                "gap":   status["domain_gap"],
                "trend": status["gap_trend"],
                "dr":    status["dr_suggest"],
            })))
        self._pub_status.publish(String(data=json.dumps(status)))

    def _publish_r2s(self) -> None:
        if self._r2s:
            self._pub_r2s.publish(String(data=json.dumps(self._r2s.get_status())))

    # ── default knowledge ─────────────────────────────────────────────────────

    def _init_default_knowledge(self) -> None:
        self._kb.add_task_doc("navigation",
            "The AMR navigates autonomously using a PPO/SAC policy trained in MuJoCo. "
            "Observations: relative goal position (dx_r, dy_r), distance, heading error "
            "(sin/cos), wheel velocities. Actions: left/right wheel velocity commands. "
            "Reward: -0.1*distance + 3.0 on reach + -0.5 on collision.")
        self._kb.add_task_doc("arm_control",
            "The robot arm is controlled by a PPO policy for end-effector reaching. "
            "Observations: joint positions, joint velocities, EE position, target position. "
            "Actions: joint torques/velocities. Reward: -distance to target + 2.0 if within 5cm.")
        self._kb.add_task_doc("sim2real",
            "Sim-to-Real transfer uses ONNX-exported policies with a calibration layer. "
            "The calibration layer compensates for actuator delays, friction differences, "
            "and sensor noise. Domain gap is tracked via RMS residual between sim predictions "
            "and real observations.")
        self._kb.add_task_doc("real2sim",
            "Real-to-Sim system identification uses recursive least squares (RLS) to estimate "
            "joint damping, Coulomb friction, and actuator gains from real sensor data. "
            "These parameters are written into the live MuJoCo model to improve sim fidelity.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GeminiRobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
