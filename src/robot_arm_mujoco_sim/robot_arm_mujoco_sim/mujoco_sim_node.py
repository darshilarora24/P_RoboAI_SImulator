from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import site
import sys

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory

def _add_workspace_venv_site_packages() -> list[Path]:
    """Look for a workspace-local virtualenv and prepend its site-packages."""
    discovered_paths: list[Path] = []
    search_roots = []

    current_file = Path(__file__).resolve()
    search_roots.extend(current_file.parents)
    search_roots.append(Path.cwd().resolve())

    checked_roots = []
    for root in search_roots:
        if root in checked_roots:
            continue
        checked_roots.append(root)

        venv_directory = root / ".venv"
        if not venv_directory.exists():
            continue

        for site_packages_path in venv_directory.glob("lib/python*/site-packages"):
            resolved_path = site_packages_path.resolve()
            if str(resolved_path) not in sys.path:
                site.addsitedir(str(resolved_path))
                sys.path.insert(0, str(resolved_path))
            discovered_paths.append(resolved_path)

    return discovered_paths


WORKSPACE_VENV_SITE_PACKAGES = _add_workspace_venv_site_packages()


try:
    import mujoco
    import mujoco.viewer
    import numpy as np
except ImportError as import_error:
    mujoco = None
    np = None
    mujoco_import_error = import_error
else:
    mujoco_import_error = None


@dataclass(frozen=True)
class JointSpec:
    joint_name: str
    actuator_name: str


JOINT_SPECS = (
    JointSpec("shoulder_yaw", "shoulder_yaw_servo"),
    JointSpec("shoulder_pitch", "shoulder_pitch_servo"),
    JointSpec("elbow_pitch", "elbow_pitch_servo"),
    JointSpec("wrist_pitch", "wrist_pitch_servo"),
)


class MujocoArmSimulator(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_arm_simulator")

        if mujoco is None:
            venv_hint = ""
            if WORKSPACE_VENV_SITE_PACKAGES:
                venv_hint = (
                    " A workspace virtualenv was detected, but 'mujoco' is not installed in "
                    "it yet."
                )
            raise RuntimeError(
                "The 'mujoco' Python package is required. Install it with "
                "'python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt'."
                + venv_hint
            ) from mujoco_import_error

        package_share = Path(get_package_share_directory("robot_arm_mujoco_sim"))
        default_model_path = package_share / "models" / "robot_arm.xml"

        self.declare_parameter("model_path", str(default_model_path))
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("sim_rate_hz", 500.0)
        self.declare_parameter("state_pub_rate_hz", 50.0)
        self.declare_parameter("publish_clock", True)
        self.declare_parameter("position_command_topic", "joint_position_cmd")
        self.declare_parameter("trajectory_command_topic", "joint_trajectory")
        self.declare_parameter("home_keyframe", "home")

        model_path = Path(self.get_parameter("model_path").value).expanduser()
        use_viewer = bool(self.get_parameter("use_viewer").value)
        self.publish_clock = bool(self.get_parameter("publish_clock").value)
        sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        state_pub_rate_hz = float(self.get_parameter("state_pub_rate_hz").value)
        self.home_keyframe = str(self.get_parameter("home_keyframe").value)

        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = [spec.joint_name for spec in JOINT_SPECS]
        self.joint_index_by_name = {
            joint_name: index for index, joint_name in enumerate(self.joint_names)
        }

        self.joint_ids = []
        self.actuator_ids = []
        self.qpos_indices = []
        self.qvel_indices = []
        self.joint_limits = []

        for spec in JOINT_SPECS:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name
            )
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.actuator_name
            )
            if joint_id == -1:
                raise RuntimeError(f"Joint '{spec.joint_name}' not found in MuJoCo model.")
            if actuator_id == -1:
                raise RuntimeError(
                    f"Actuator '{spec.actuator_name}' not found in MuJoCo model."
                )

            self.joint_ids.append(joint_id)
            self.actuator_ids.append(actuator_id)
            self.qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self.qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

            limit_lower = float(self.model.jnt_range[joint_id][0])
            limit_upper = float(self.model.jnt_range[joint_id][1])
            self.joint_limits.append((limit_lower, limit_upper))

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        if self.ee_site_id == -1:
            raise RuntimeError("Site 'ee_site' not found in MuJoCo model.")

        self.viewer = None
        self._reset_to_home()

        self.joint_state_pub = self.create_publisher(JointState, "joint_states", 10)
        self.ee_pose_pub = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self.clock_pub = (
            self.create_publisher(Clock, "/clock", 10) if self.publish_clock else None
        )

        position_topic = str(self.get_parameter("position_command_topic").value)
        trajectory_topic = str(self.get_parameter("trajectory_command_topic").value)

        self.position_cmd_sub = self.create_subscription(
            Float64MultiArray, position_topic, self._handle_position_command, 10
        )
        self.position_cmd_alias_sub = self.create_subscription(
            Float64MultiArray,
            "/joint_group_position_controller/commands",
            self._handle_position_command,
            10,
        )
        self.trajectory_sub = self.create_subscription(
            JointTrajectory, trajectory_topic, self._handle_trajectory_command, 10
        )
        self.reset_srv = self.create_service(
            Empty, "reset_simulation", self._handle_reset_request
        )

        self.sim_timer = self.create_timer(1.0 / sim_rate_hz, self._step_simulation)
        self.state_timer = self.create_timer(
            1.0 / state_pub_rate_hz, self._publish_ros_outputs
        )

        if use_viewer:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.get_logger().info("Started MuJoCo viewer.")
            except Exception as exc:
                self.get_logger().warning(
                    f"MuJoCo viewer could not be started, continuing headless: {exc}"
                )

        self.get_logger().info(f"Loaded MuJoCo arm model from {model_path}")
        self.get_logger().info(
            "Command topics: "
            f"'{position_topic}' (Float64MultiArray) and "
            f"'{trajectory_topic}' (JointTrajectory)"
        )

    def _reset_to_home(self) -> None:
        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.home_keyframe
        )
        if keyframe_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self.target_positions = [
            float(self.data.qpos[qpos_index]) for qpos_index in self.qpos_indices
        ]
        for actuator_id, target in zip(self.actuator_ids, self.target_positions):
            self.data.ctrl[actuator_id] = target

    def _handle_reset_request(self, request: Empty.Request, response: Empty.Response):
        del request
        self._reset_to_home()
        self.get_logger().info("Simulation reset to the home keyframe.")
        return response

    def _handle_position_command(self, message: Float64MultiArray) -> None:
        if len(message.data) != len(self.joint_names):
            self.get_logger().warning(
                "Received direct position command with "
                f"{len(message.data)} values, expected {len(self.joint_names)}."
            )
            return

        self.target_positions = [
            self._clip_to_joint_limit(index, float(position))
            for index, position in enumerate(message.data)
        ]

    def _handle_trajectory_command(self, message: JointTrajectory) -> None:
        if not message.points:
            self.get_logger().warning("Received empty JointTrajectory command.")
            return

        point = message.points[-1]
        command_joint_names = list(message.joint_names) or self.joint_names

        if len(point.positions) != len(command_joint_names):
            self.get_logger().warning(
                "Received JointTrajectory with mismatched joint_names and positions."
            )
            return

        updated_targets = list(self.target_positions)
        for joint_name, position in zip(command_joint_names, point.positions):
            joint_index = self.joint_index_by_name.get(joint_name)
            if joint_index is None:
                self.get_logger().warning(
                    f"Ignoring command for unknown joint '{joint_name}'."
                )
                continue

            updated_targets[joint_index] = self._clip_to_joint_limit(
                joint_index, float(position)
            )

        self.target_positions = updated_targets

    def _clip_to_joint_limit(self, joint_index: int, position: float) -> float:
        lower_limit, upper_limit = self.joint_limits[joint_index]
        return max(lower_limit, min(upper_limit, position))

    def _step_simulation(self) -> None:
        for actuator_id, target in zip(self.actuator_ids, self.target_positions):
            self.data.ctrl[actuator_id] = target

        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            try:
                self.viewer.sync()
            except Exception as exc:
                self.get_logger().warning(
                    f"Viewer sync failed, switching to headless mode: {exc}"
                )
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None

    def _publish_ros_outputs(self) -> None:
        sim_time = self._sim_time_to_msg()

        joint_state = JointState()
        joint_state.header.stamp = sim_time
        joint_state.name = list(self.joint_names)
        joint_state.position = [
            float(self.data.qpos[qpos_index]) for qpos_index in self.qpos_indices
        ]
        joint_state.velocity = [
            float(self.data.qvel[qvel_index]) for qvel_index in self.qvel_indices
        ]
        joint_state.effort = [
            float(self.data.qfrc_actuator[qvel_index])
            for qvel_index in self.qvel_indices
        ]
        self.joint_state_pub.publish(joint_state)

        ee_pose = PoseStamped()
        ee_pose.header.stamp = sim_time
        ee_pose.header.frame_id = "world"
        ee_pose.pose.position.x = float(self.data.site_xpos[self.ee_site_id][0])
        ee_pose.pose.position.y = float(self.data.site_xpos[self.ee_site_id][1])
        ee_pose.pose.position.z = float(self.data.site_xpos[self.ee_site_id][2])
        quat_x, quat_y, quat_z, quat_w = self._site_quaternion_xyzw()
        ee_pose.pose.orientation.x = quat_x
        ee_pose.pose.orientation.y = quat_y
        ee_pose.pose.orientation.z = quat_z
        ee_pose.pose.orientation.w = quat_w
        self.ee_pose_pub.publish(ee_pose)

        if self.clock_pub is not None:
            clock_message = Clock()
            clock_message.clock = sim_time
            self.clock_pub.publish(clock_message)

    def _site_quaternion_xyzw(self) -> tuple[float, float, float, float]:
        quat_wxyz = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat_wxyz, self.data.site_xmat[self.ee_site_id])
        return (
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        )

    def _sim_time_to_msg(self) -> Time:
        seconds = int(math.floor(self.data.time))
        nanoseconds = int((self.data.time - seconds) * 1e9)
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        return Time(sec=seconds, nanosec=nanoseconds)

    def destroy_node(self) -> bool:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = MujocoArmSimulator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
