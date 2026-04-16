"""
viz.launch.py  —  P_RoboAI visualizer

Launches the Qt RViz-like visualizer alongside the full AMR stack.

Usage (visualizer only, assumes AMR stack already running):
  ros2 launch p_roboai_viz viz.launch.py

Usage (full stack + visualizer):
  ros2 launch p_roboai_viz viz.launch.py launch_stack:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "launch_stack", default_value="false",
            description="Also launch the full AMR stack (sim + SLAM + Nav2)"),

        # ── Optional: bring up the full AMR stack ─────────────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("robot_amr_mujoco_sim"),
                    "launch", "amr_full.launch.py")),
            condition=IfCondition(LaunchConfiguration("launch_stack")),
        ),

        # ── P_RoboAI Visualizer ───────────────────────────────────────────────
        Node(
            package="p_roboai_viz",
            executable="p_roboai_viz",
            name="p_roboai_viz",
            output="screen",
        ),
    ])
