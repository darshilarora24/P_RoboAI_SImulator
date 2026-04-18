"""
arm_rl.launch.py  —  Launch arm RL training or inference.

Usage
-----
  ros2 launch p_roboai_rl arm_rl.launch.py
  ros2 launch p_roboai_rl arm_rl.launch.py mode:=run
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("mode",        default_value="train"),
        DeclareLaunchArgument("policy_path", default_value="~/p_roboai_rl/arm_policy"),
        DeclareLaunchArgument("total_steps", default_value="150000"),
        DeclareLaunchArgument("urdf_path",   default_value="",
                              description="URDF or MJCF XML path (auto-detected if empty)"),

        Node(
            package    = "p_roboai_rl",
            executable = "arm_rl_node",
            name       = "arm_rl_node",
            output     = "screen",
            parameters = [{
                "mode":        LaunchConfiguration("mode"),
                "policy_path": LaunchConfiguration("policy_path"),
                "total_steps": LaunchConfiguration("total_steps"),
                "urdf_path":   LaunchConfiguration("urdf_path"),
            }],
        ),
    ])
