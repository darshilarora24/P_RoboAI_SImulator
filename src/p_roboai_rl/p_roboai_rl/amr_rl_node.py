"""
amr_rl_node.py  —  P_RoboAI RL

ROS2 node that provides reinforcement-learning-based navigation for the AMR.

Modes (ROS2 parameter  mode:  'train' | 'run')
-----------------------------------------------
train  — Trains a PPO policy against a local MuJoCo copy of the AMR model.
         Progress is published on /amr_rl/status (std_msgs/String).
         Policy is saved to ~/p_roboai_rl/amr_policy.zip on completion.

run    — Loads the saved policy and publishes /amr/cmd_vel (Twist) at 20 Hz
         based on current /amr/odom pose and the active /amr/goal_pose target.

Topics (run mode)
-----------------
Subscribes:   /amr/odom          nav_msgs/Odometry
              /amr/goal_pose     geometry_msgs/PoseStamped
Publishes:    /amr/cmd_vel       geometry_msgs/Twist
              /amr_rl/status     std_msgs/String

Topics (train mode)
-------------------
Publishes:    /amr_rl/status     std_msgs/String   (progress updates)
              /amr_rl/reward     std_msgs/Float32  (per-episode reward)
"""
from __future__ import annotations

import math
import os
import site
import sys
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32


# ── venv bootstrap (same pattern as existing nodes) ──────────────────────────

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

try:
    import mujoco
    import numpy as np
    _MUJOCO_OK = True
except ImportError:
    _MUJOCO_OK = False
    np = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym
    from gymnasium import spaces
    _SB3_OK = True
except ImportError:
    _SB3_OK = False


# ── Minimal MuJoCo AMR gym environment ───────────────────────────────────────

class _AMREnv:
    """Lightweight diff-drive navigation environment backed by MuJoCo."""

    _MAX_SPEED = 3.0
    _WHEEL_R   = 0.10
    _TRACK     = 0.25

    def __init__(self, xml_path: str, max_steps: int = 500) -> None:
        self._model     = mujoco.MjModel.from_xml_path(xml_path)
        self._data      = mujoco.MjData(self._model)
        self._max_steps = max_steps
        self._step_n    = 0
        self._goal      = np.zeros(2, np.float32)

        # Find wheel actuators
        self._lw_act = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_vel")
        self._rw_act = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_vel")
        if self._lw_act < 0:
            # Fallback: use first two actuators
            self._lw_act, self._rw_act = 0, 1

        self._root_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "root")

        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def _pose(self):
        qadr = self._model.jnt_qposadr[self._root_jid]
        q    = self._data.qpos
        x, y = float(q[qadr]), float(q[qadr + 1])
        qw   = float(q[qadr + 3])
        qz   = float(q[qadr + 6])
        th   = 2.0 * math.atan2(qz, qw)
        return x, y, th

    def _obs(self) -> np.ndarray:
        x, y, th = self._pose()
        dx, dy   = self._goal[0] - x, self._goal[1] - y
        dist     = math.hypot(dx, dy)
        c, s     = math.cos(-th), math.sin(-th)
        th_err   = math.atan2(dy, dx) - th
        vl = float(self._data.ctrl[self._lw_act]) / self._MAX_SPEED
        vr = float(self._data.ctrl[self._rw_act]) / self._MAX_SPEED
        return np.array([c * dx - s * dy, s * dx + c * dy, dist,
                         math.sin(th_err), math.cos(th_err),
                         vl, vr], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        angle        = np.random.uniform(0, 2 * math.pi)
        r            = np.random.uniform(1.0, 3.0)
        self._goal   = np.array([r * math.cos(angle), r * math.sin(angle)], np.float32)
        self._step_n = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        vl = float(np.clip(action[0], -1, 1)) * self._MAX_SPEED
        vr = float(np.clip(action[1], -1, 1)) * self._MAX_SPEED
        self._data.ctrl[self._lw_act] = vl
        self._data.ctrl[self._rw_act] = vr
        for _ in range(10):
            mujoco.mj_step(self._model, self._data)
        self._step_n += 1
        x, y, _ = self._pose()
        dist     = math.hypot(self._goal[0] - x, self._goal[1] - y)
        reached  = dist < 0.20
        reward   = -0.1 * dist + (3.0 if reached else 0.0)
        return self._obs(), float(reward), reached, self._step_n >= self._max_steps, {"dist": dist}

    def close(self): pass


# ── SB3 callback that publishes to ROS ───────────────────────────────────────

class _ROSCallback(BaseCallback if _SB3_OK else object):
    def __init__(self, node: "AMRRLNode", total_steps: int, verbose=0) -> None:
        if _SB3_OK:
            super().__init__(verbose)
        self._node        = node
        self._total       = total_steps
        self._ep_rew: list[float] = []
        self._episodes    = 0

    def _on_step(self) -> bool:
        if not _SB3_OK:
            return True
        r     = float(self.locals["rewards"][0])
        done  = bool(self.locals["dones"][0])
        dist  = float(self.locals["infos"][0].get("dist", 0))
        self._ep_rew.append(r)
        if done:
            ep_r = sum(self._ep_rew)
            self._episodes += 1
            self._ep_rew = []
            pct = int(self.num_timesteps * 100 / max(1, self._total))
            self._node._pub_status.publish(String(data=(
                f"[TRAIN] ep={self._episodes}  reward={ep_r:.2f}"
                f"  dist={dist:.2f}m  {pct}%")))
            self._node._pub_reward.publish(Float32(data=float(ep_r)))
        return not self._node._stop_training


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class AMRRLNode(Node):

    def __init__(self) -> None:
        super().__init__("amr_rl_node")

        self.declare_parameter("mode",        "train")   # 'train' | 'run'
        self.declare_parameter("policy_path", str(Path.home() / "p_roboai_rl" / "amr_policy"))
        self.declare_parameter("total_steps", 200_000)
        self.declare_parameter("xml_path",    "")        # MuJoCo XML (for training)

        self._mode        = self.get_parameter("mode").value
        self._policy_path = Path(self.get_parameter("policy_path").value)
        self._total_steps = int(self.get_parameter("total_steps").value)
        self._xml_path    = self.get_parameter("xml_path").value
        self._stop_training = False

        # Publishers
        self._pub_cmd    = self.create_publisher(Twist,  "/amr/cmd_vel",   10)
        self._pub_status = self.create_publisher(String, "/amr_rl/status", 10)
        self._pub_reward = self.create_publisher(Float32,"/amr_rl/reward", 10)

        # RL state
        self._policy     = None
        self._robot_x    = 0.0
        self._robot_y    = 0.0
        self._robot_th   = 0.0
        self._goal_x     = 0.0
        self._goal_y     = 0.0
        self._has_goal   = False

        if not _MUJOCO_OK:
            self._pub_status.publish(String(data="ERROR: mujoco not available"))
            return
        if not _SB3_OK:
            self._pub_status.publish(String(data="ERROR: stable-baselines3 not available — pip install stable-baselines3 gymnasium"))
            return

        if self._mode == "train":
            self._start_training()
        else:
            self._load_and_run()

    def _find_xml(self) -> str:
        if self._xml_path:
            return self._xml_path
        # Try installed share directory
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("robot_amr_mujoco_sim")
            return os.path.join(share, "models", "amr_warehouse.xml")
        except Exception:
            pass
        # Fallback: search workspace
        for p in Path("/home").rglob("amr_warehouse.xml"):
            return str(p)
        raise FileNotFoundError("amr_warehouse.xml not found — set xml_path parameter")

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self) -> None:
        self._pub_status.publish(String(data="[TRAIN] Starting AMR RL training…"))

        def _train():
            try:
                xml = self._find_xml()
                env = _AMREnv(xml, max_steps=500)
                model = PPO("MlpPolicy", env, learning_rate=3e-4,
                            n_steps=2048, batch_size=64, verbose=0)
                cb = _ROSCallback(self, self._total_steps)
                model.learn(total_timesteps=self._total_steps, callback=cb,
                            reset_num_timesteps=True)
                self._policy_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(self._policy_path))
                self._policy = model
                self._pub_status.publish(String(data=(
                    f"[TRAIN] Complete — policy saved to {self._policy_path}")))
            except Exception as exc:
                import traceback
                self._pub_status.publish(String(data=f"[TRAIN] ERROR: {exc}\n{traceback.format_exc()}"))

        threading.Thread(target=_train, daemon=True).start()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _load_and_run(self) -> None:
        policy_zip = Path(str(self._policy_path) + ".zip")
        if not policy_zip.exists():
            self._pub_status.publish(String(data=(
                f"[RUN] Policy not found at {policy_zip} — train first")))
            return

        try:
            self._policy = PPO.load(str(self._policy_path))
            self._pub_status.publish(String(data=f"[RUN] Policy loaded from {self._policy_path}"))
        except Exception as exc:
            self._pub_status.publish(String(data=f"[RUN] Load error: {exc}"))
            return

        self.create_subscription(Odometry,    "/amr/odom",      self._on_odom,  10)
        self.create_subscription(PoseStamped, "/amr/goal_pose", self._on_goal,  10)
        self.create_timer(0.05, self._inference_tick)   # 20 Hz
        self._pub_status.publish(String(data="[RUN] Running — waiting for /amr/goal_pose"))

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_x  = msg.pose.pose.position.x
        self._robot_y  = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_th = math.atan2(siny, cosy)

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_x  = msg.pose.position.x
        self._goal_y  = msg.pose.position.y
        self._has_goal = True
        self._pub_status.publish(String(data=(
            f"[RUN] Goal set → ({self._goal_x:.2f}, {self._goal_y:.2f})")))

    def _inference_tick(self) -> None:
        if self._policy is None or not self._has_goal:
            return

        dx, dy  = self._goal_x - self._robot_x, self._goal_y - self._robot_y
        dist    = math.hypot(dx, dy)

        if dist < 0.25:
            self._pub_cmd.publish(Twist())   # stop at goal
            self._pub_status.publish(String(data="[RUN] Goal reached!"))
            self._has_goal = False
            return

        th   = self._robot_th
        c, s = math.cos(-th), math.sin(-th)
        th_err = math.atan2(dy, dx) - th

        obs = np.array([c * dx - s * dy, s * dx + c * dy, dist,
                        math.sin(th_err), math.cos(th_err),
                        0.0, 0.0], np.float32)

        action, _ = self._policy.predict(obs, deterministic=True)
        vl = float(np.clip(action[0], -1, 1))
        vr = float(np.clip(action[1], -1, 1))

        # Convert wheel velocities to cmd_vel
        wheel_r = 0.10
        track   = 0.25
        v_ms  = (vl + vr) * 0.5 * wheel_r * 3.0   # scale back from normalised
        w_rads = (vr - vl) * wheel_r * 3.0 / track

        cmd = Twist()
        cmd.linear.x  = float(np.clip(v_ms,  -1.2, 1.2))
        cmd.angular.z = float(np.clip(w_rads, -2.5, 2.5))
        self._pub_cmd.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AMRRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop_training = True
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
