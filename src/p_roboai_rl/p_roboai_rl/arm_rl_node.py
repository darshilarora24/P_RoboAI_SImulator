"""
arm_rl_node.py  —  P_RoboAI RL

ROS2 node that provides reinforcement-learning-based control for the robot arm.

Modes (ROS2 parameter  mode:  'train' | 'run')
-----------------------------------------------
train  — Trains a PPO policy for end-effector reaching using the arm MJCF.
         Progress published on /arm_rl/status.
         Policy saved to ~/p_roboai_rl/arm_policy.zip.

run    — Loads policy, subscribes to /joint_states for arm state,
         publishes Float64MultiArray to /arm/position_commands at 20 Hz.
         Target set via /arm_rl/target (geometry_msgs/PointStamped).

Topics
------
Subscribes:   /joint_states         sensor_msgs/JointState
              /arm_rl/target        geometry_msgs/PointStamped   (run mode)
Publishes:    /arm/position_commands std_msgs/Float64MultiArray
              /arm_rl/status         std_msgs/String
              /arm_rl/reward         std_msgs/Float32
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
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32, Float64MultiArray


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


# ── Joint spec (must match arm URDF / MJCF) ──────────────────────────────────

JOINT_NAMES = ["shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"]
N_JOINTS    = len(JOINT_NAMES)


# ── Minimal arm reaching gym environment ──────────────────────────────────────

class _ArmEnv:
    """
    End-effector reaching with the 4-DOF arm MJCF.
    Observation: joint_pos(4) + joint_vel(4) + ee_pos(3) + target(3) = 14
    Action:      normalised joint position targets [-1, 1]
    """

    def __init__(self, mjcf_xml: str, max_steps: int = 200) -> None:
        self._model     = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._data      = mujoco.MjData(self._model)
        self._max_steps = max_steps
        self._step_n    = 0
        self._target    = np.zeros(3, np.float32)

        # Joint ids and qpos addresses
        self._jids  = []
        self._qadr  = []
        self._dadr  = []
        self._ctrl  = []
        self._lo    = []
        self._hi    = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._jids.append(jid)
                self._qadr.append(int(self._model.jnt_qposadr[jid]))
                self._dadr.append(int(self._model.jnt_dofadr[jid]))
                lo = float(self._model.jnt_range[jid, 0])
                hi = float(self._model.jnt_range[jid, 1])
                self._lo.append(lo if lo != 0 else -3.14159)
                self._hi.append(hi if hi != 0 else  3.14159)
        n = len(self._jids)
        # Actuator ctrl indices (assume same order as joints)
        self._ctrl = list(range(min(n, self._model.nu)))

        # End-effector = last body
        has_ch: set[int] = set()
        for i in range(self._model.nbody):
            p = self._model.body_parentid[i]
            if p != i:
                has_ch.add(p)
        self._ee_id = max((i for i in range(self._model.nbody)
                           if i not in has_ch), default=self._model.nbody - 1)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (2 * n + 6,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n,), np.float32)
        self._n = n

    def _obs(self) -> np.ndarray:
        pos = np.array([float(self._data.qpos[a]) for a in self._qadr], np.float32)
        vel = np.array([float(self._data.qvel[a]) for a in self._dadr], np.float32)
        ee  = self._data.xpos[self._ee_id].astype(np.float32)
        return np.concatenate([pos, vel, ee, self._target])

    def _sample_target(self) -> np.ndarray:
        r     = np.random.uniform(0.20, 0.45)
        theta = np.random.uniform(0, 2 * math.pi)
        z     = np.random.uniform(0.20, 0.65)
        return np.array([r * math.cos(theta), r * math.sin(theta), z], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self._model, self._data)
        for i, c in enumerate(self._ctrl):
            lo, hi = self._lo[i] * 0.3, self._hi[i] * 0.3
            self._data.ctrl[c] = np.random.uniform(lo, hi)
        mujoco.mj_forward(self._model, self._data)
        self._target  = self._sample_target()
        self._step_n  = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        for i, c in enumerate(self._ctrl):
            lo, hi = self._lo[i], self._hi[i]
            t = lo + (float(action[i]) + 1.0) * 0.5 * (hi - lo)
            self._data.ctrl[c] = float(np.clip(t, lo, hi))
        for _ in range(5):
            mujoco.mj_step(self._model, self._data)
        self._step_n += 1
        ee   = self._data.xpos[self._ee_id].astype(np.float32)
        dist = float(np.linalg.norm(ee - self._target))
        reached   = dist < 0.05
        reward    = -dist + (2.0 if reached else 0.0)
        return self._obs(), float(reward), reached, self._step_n >= self._max_steps, {"dist": dist}

    def close(self): pass

    @property
    def n_joints(self) -> int:
        return self._n

    def ctrl_for_action(self, action: np.ndarray) -> list[float]:
        cmds = []
        for i in range(self._n):
            lo, hi = self._lo[i], self._hi[i]
            t = lo + (float(action[i]) + 1.0) * 0.5 * (hi - lo)
            cmds.append(float(np.clip(t, lo, hi)))
        return cmds


# ── SB3 callback ─────────────────────────────────────────────────────────────

class _ROSCallback(BaseCallback if _SB3_OK else object):
    def __init__(self, node: "ArmRLNode", total_steps: int, verbose=0) -> None:
        if _SB3_OK:
            super().__init__(verbose)
        self._node     = node
        self._total    = total_steps
        self._ep_rew: list[float] = []
        self._episodes = 0

    def _on_step(self) -> bool:
        if not _SB3_OK:
            return True
        r    = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        dist = float(self.locals["infos"][0].get("dist", 0))
        self._ep_rew.append(r)
        if done:
            ep_r = sum(self._ep_rew)
            self._episodes += 1
            self._ep_rew = []
            pct = int(self.num_timesteps * 100 / max(1, self._total))
            self._node._pub_status.publish(String(data=(
                f"[TRAIN] ep={self._episodes}  reward={ep_r:.2f}"
                f"  dist={dist:.3f}m  {pct}%")))
            self._node._pub_reward.publish(Float32(data=float(ep_r)))
        return not self._node._stop_training


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class ArmRLNode(Node):

    def __init__(self) -> None:
        super().__init__("arm_rl_node")

        self.declare_parameter("mode",        "train")
        self.declare_parameter("policy_path", str(Path.home() / "p_roboai_rl" / "arm_policy"))
        self.declare_parameter("total_steps", 150_000)
        self.declare_parameter("urdf_path",   "")

        self._mode        = self.get_parameter("mode").value
        self._policy_path = Path(self.get_parameter("policy_path").value)
        self._total_steps = int(self.get_parameter("total_steps").value)
        self._urdf_path   = self.get_parameter("urdf_path").value
        self._stop_training = False

        self._pub_cmd    = self.create_publisher(Float64MultiArray, "/arm/position_commands", 10)
        self._pub_status = self.create_publisher(String,  "/arm_rl/status", 10)
        self._pub_reward = self.create_publisher(Float32, "/arm_rl/reward", 10)

        self._policy      = None
        self._env: _ArmEnv | None = None
        self._joint_pos   = np.zeros(N_JOINTS, np.float32)
        self._joint_vel   = np.zeros(N_JOINTS, np.float32)
        self._target      = np.array([0.3, 0.0, 0.4], np.float32)
        self._has_target  = False

        if not _MUJOCO_OK:
            self._pub_status.publish(String(data="ERROR: mujoco not available"))
            return
        if not _SB3_OK:
            self._pub_status.publish(String(data="ERROR: stable-baselines3/gymnasium not installed"))
            return

        if self._mode == "train":
            self._start_training()
        else:
            self._load_and_run()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_mjcf(self) -> str:
        """Convert URDF → MJCF or load from XML directly."""
        if self._urdf_path:
            p = Path(self._urdf_path)
        else:
            try:
                from ament_index_python.packages import get_package_share_directory
                share = get_package_share_directory("robot_arm_mujoco_sim")
                # Use the existing MJCF model directly
                xml_path = os.path.join(share, "models", "robot_arm.xml")
                if os.path.exists(xml_path):
                    return open(xml_path).read()
            except Exception:
                pass
            # Fallback: find URDF and convert
            for p in Path("/home").rglob("robot_arm.urdf"):
                break
            else:
                raise FileNotFoundError(
                    "robot_arm.urdf not found — set urdf_path parameter")

        if str(p).endswith(".xml"):
            return p.read_text()

        # URDF → MJCF via p_roboai_studio urdf_loader
        studio = None
        for root in Path(__file__).resolve().parents:
            candidate = root / "p_roboai_studio"
            if candidate.exists():
                studio = candidate
                break
        if studio:
            sys.path.insert(0, str(studio))
        import urdf_loader as ul
        result = ul.load(p)
        return result.mjcf_xml

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self) -> None:
        self._pub_status.publish(String(data="[TRAIN] Starting arm RL training…"))

        def _train():
            try:
                mjcf = self._get_mjcf()
                env  = _ArmEnv(mjcf, max_steps=200)
                self._env = env
                model = PPO("MlpPolicy", env, learning_rate=3e-4,
                            n_steps=1024, batch_size=64, verbose=0)
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
            mjcf = self._get_mjcf()
            self._env = _ArmEnv(mjcf)
            self._env.reset()
            self._pub_status.publish(String(data=f"[RUN] Policy loaded"))
        except Exception as exc:
            self._pub_status.publish(String(data=f"[RUN] Load error: {exc}"))
            return

        self.create_subscription(JointState,    "/joint_states",  self._on_joints, 10)
        self.create_subscription(PointStamped,  "/arm_rl/target", self._on_target, 10)
        self.create_timer(0.05, self._inference_tick)
        self._pub_status.publish(String(data=(
            "[RUN] Running — set target via /arm_rl/target (PointStamped)")))

    def _on_joints(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        for i, name in enumerate(JOINT_NAMES):
            idx = name_to_idx.get(name)
            if idx is not None:
                if idx < len(msg.position):
                    self._joint_pos[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self._joint_vel[i] = msg.velocity[idx]

    def _on_target(self, msg: PointStamped) -> None:
        self._target    = np.array([msg.point.x, msg.point.y, msg.point.z], np.float32)
        self._has_target = True
        self._pub_status.publish(String(data=(
            f"[RUN] Target → ({self._target[0]:.3f}, {self._target[1]:.3f},"
            f" {self._target[2]:.3f})")))

    def _inference_tick(self) -> None:
        if self._policy is None or self._env is None:
            return
        env = self._env

        # Build observation from live joint state + current target
        obs = np.concatenate([
            self._joint_pos[:env.n_joints],
            self._joint_vel[:env.n_joints],
            env._data.xpos[env._ee_id].astype(np.float32),
            self._target,
        ]).astype(np.float32)

        action, _ = self._policy.predict(obs, deterministic=True)
        cmds      = env.ctrl_for_action(action)

        msg = Float64MultiArray()
        msg.data = cmds
        self._pub_cmd.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._stop_training = True
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
