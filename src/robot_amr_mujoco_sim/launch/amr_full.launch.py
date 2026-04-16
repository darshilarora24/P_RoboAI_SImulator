"""
amr_full.launch.py  —  P_RoboAI AMR full stack

Nodes launched
--------------
  1. amr_mujoco_node   — 3-D MuJoCo physics sim (lidar, odom, imu)
  2. robot_state_pub   — URDF → TF static frames (amr_base_link, lidar_link …)
  3. slam_node         — P_RoboAI_SLAM (builds /map, publishes map→odom TF)
  4. nav_node          — P_RoboAI_Nav2 (costmap + A* + DWA → /amr/cmd_vel)

Usage
-----
  ros2 launch robot_amr_mujoco_sim amr_full.launch.py

Optional teleop (separate terminal):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/amr/cmd_vel

Visualise in RViz2 (separate terminal):
  rviz2
  Add: Map (/p_roboai_slam/map), LaserScan (/amr/scan),
       Odometry (/amr/odom), Path (/p_roboai_nav2/path), TF, RobotModel
"""
import glob
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _mujoco_pythonpath() -> str:
    """
    Locate the workspace .venv site-packages that contains MuJoCo and return
    a PYTHONPATH string that prepends it to any existing PYTHONPATH.

    Strategy: the package is installed to <workspace>/install/robot_amr_mujoco_sim,
    so the workspace root is two directories above the ament install prefix.
    """
    try:
        install_prefix = get_package_prefix("robot_amr_mujoco_sim")
        workspace_root = os.path.dirname(os.path.dirname(install_prefix))
    except Exception:
        workspace_root = os.path.expanduser("~/P_RoboAI_SImulator")

    candidates = glob.glob(
        os.path.join(workspace_root, ".venv", "lib", "python*", "site-packages"))
    if not candidates:
        raise RuntimeError(
            f"Could not find MuJoCo venv under {workspace_root}/.venv — "
            "run 'pip install -r requirements.txt' inside the venv first.")

    venv_sp = candidates[0]
    existing = os.environ.get("PYTHONPATH", "")
    return f"{venv_sp}:{existing}" if existing else venv_sp


def generate_launch_description() -> LaunchDescription:
    pkg_sim  = get_package_share_directory("robot_amr_mujoco_sim")
    urdf_path = os.path.join(pkg_sim, "urdf", "amr_robot.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    mujoco_pythonpath = _mujoco_pythonpath()

    return LaunchDescription([
        # ── Launch arguments ─────────────────────────────────────────────────
        DeclareLaunchArgument("start_x",     default_value="1.0"),
        DeclareLaunchArgument("start_y",     default_value="1.0"),
        DeclareLaunchArgument("start_theta", default_value="0.0"),
        DeclareLaunchArgument("use_slam_matching", default_value="true"),

        # ── 1. MuJoCo AMR simulator ──────────────────────────────────────────
        # additional_env injects the venv site-packages so `import mujoco` works
        # inside the system-Python ROS2 node process.
        Node(
            package="robot_amr_mujoco_sim",
            executable="amr_mujoco_node",
            name="amr_mujoco_sim",
            output="screen",
            additional_env={"PYTHONPATH": mujoco_pythonpath},
            parameters=[{
                "start_x":     LaunchConfiguration("start_x"),
                "start_y":     LaunchConfiguration("start_y"),
                "start_theta": LaunchConfiguration("start_theta"),
                "sim_rate_hz": 200.0,
            }],
        ),

        # ── 2. Robot State Publisher (URDF → TF static frames) ───────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        # ── 3. P_RoboAI_SLAM ─────────────────────────────────────────────────
        Node(
            package="p_roboai_slam",
            executable="slam_node",
            name="p_roboai_slam",
            output="screen",
            parameters=[{
                "map_width_m":  10.0,
                "map_height_m": 10.0,
                "resolution":   0.05,
                "origin_x":     0.0,
                "origin_y":     0.0,
                "use_scan_matching": LaunchConfiguration("use_slam_matching"),
            }],
        ),

        # ── 4. P_RoboAI_Nav2 ─────────────────────────────────────────────────
        Node(
            package="p_roboai_nav2",
            executable="nav_node",
            name="p_roboai_nav2",
            output="screen",
        ),
    ])
