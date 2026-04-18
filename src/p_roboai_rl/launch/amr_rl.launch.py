"""
amr_rl.launch.py  —  Launch AMR RL training or inference.

Usage
-----
  # Train (default):
  ros2 launch p_roboai_rl amr_rl.launch.py

  # Run trained policy:
  ros2 launch p_roboai_rl amr_rl.launch.py mode:=run

  # Train with custom steps:
  ros2 launch p_roboai_rl amr_rl.launch.py total_steps:=500000
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("mode",        default_value="train",
                              description="'train' or 'run'"),
        DeclareLaunchArgument("policy_path", default_value="~/p_roboai_rl/amr_policy",
                              description="Path to save/load policy ZIP"),
        DeclareLaunchArgument("total_steps", default_value="200000",
                              description="Training timesteps"),
        DeclareLaunchArgument("xml_path",    default_value="",
                              description="MuJoCo XML path (auto-detected if empty)"),

        Node(
            package    = "p_roboai_rl",
            executable = "amr_rl_node",
            name       = "amr_rl_node",
            output     = "screen",
            parameters = [{
                "mode":        LaunchConfiguration("mode"),
                "policy_path": LaunchConfiguration("policy_path"),
                "total_steps": LaunchConfiguration("total_steps"),
                "xml_path":    LaunchConfiguration("xml_path"),
            }],
        ),
    ])
