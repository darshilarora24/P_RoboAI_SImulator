"""
AMR Studio launch file.

Starts:
  1. amr_sim_node        — 2-D kinematic simulator (Python)
  2. amr_navigation_node — A* planner + pure-pursuit controller (Python)
  3. amr_studio          — Qt 2-D map visualisation (C++)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("start_x",     default_value="1.0"),
        DeclareLaunchArgument("start_y",     default_value="1.0"),
        DeclareLaunchArgument("start_theta", default_value="0.0"),

        Node(
            package="robot_amr_sim",
            executable="amr_sim_node",
            name="amr_sim",
            output="screen",
            parameters=[{
                "start_x":     LaunchConfiguration("start_x"),
                "start_y":     LaunchConfiguration("start_y"),
                "start_theta": LaunchConfiguration("start_theta"),
                "sim_rate_hz": 50.0,
            }],
        ),

        Node(
            package="robot_amr_sim",
            executable="amr_navigation_node",
            name="amr_navigation",
            output="screen",
        ),

        Node(
            package="robot_arm_qt_ui",
            executable="amr_studio",
            name="amr_studio",
            output="screen",
        ),
    ])
