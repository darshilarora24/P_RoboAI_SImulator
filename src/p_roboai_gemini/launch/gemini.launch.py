"""
gemini.launch.py  —  Launch Gemini Robotics LLM node.

Usage
-----
  # Default: Gemini Robotics ER 1.6 (requires GOOGLE_API_KEY + allowlist access):
  ros2 launch p_roboai_gemini gemini.launch.py

  # Fallback to gemini-2.0-flash if not on allowlist:
  ros2 launch p_roboai_gemini gemini.launch.py model_name:=gemini-2.0-flash

  # With API key and sim2real:
  ros2 launch p_roboai_gemini gemini.launch.py \\
      api_key:=AIza... \\
      robot_type:=amr \\
      enable_sim2real:=true \\
      policy_zip:=~/p_roboai_rl/amr_policy.zip

  # Real2Sim for arm (6-DOF):
  ros2 launch p_roboai_gemini gemini.launch.py \\
      robot_type:=arm  n_joints:=6  enable_real2sim:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("api_key",          default_value="",
                              description="Google API key (or set GOOGLE_API_KEY env var)"),
        DeclareLaunchArgument("model_name",       default_value="gemini-robotics-er-1.6",
                              description="Gemini model ID (e.g. gemini-robotics-er-1.6, gemini-2.0-flash)"),
        DeclareLaunchArgument("robot_type",       default_value="amr",
                              description="'arm' or 'amr'"),
        DeclareLaunchArgument("n_joints",         default_value="6",
                              description="Number of robot DOF for Real2Sim"),
        DeclareLaunchArgument("policy_zip",       default_value="",
                              description="SB3 policy ZIP for Sim2Real export"),
        DeclareLaunchArgument("onnx_path",        default_value="",
                              description="Pre-exported ONNX policy path"),
        DeclareLaunchArgument("calib_path",       default_value="",
                              description="Calibration JSON path"),
        DeclareLaunchArgument("kb_persist",       default_value="",
                              description="Knowledge base persistence JSON path"),
        DeclareLaunchArgument("enable_real2sim",  default_value="true",
                              description="Enable Real2Sim system identification"),
        DeclareLaunchArgument("enable_sim2real",  default_value="false",
                              description="Enable Sim2Real policy transfer"),

        Node(
            package    = "p_roboai_gemini",
            executable = "gemini_robot_node",
            name       = "gemini_robot_node",
            output     = "screen",
            parameters = [{
                "api_key":         LaunchConfiguration("api_key"),
                "model_name":      LaunchConfiguration("model_name"),
                "robot_type":      LaunchConfiguration("robot_type"),
                "n_joints":        LaunchConfiguration("n_joints"),
                "policy_zip":      LaunchConfiguration("policy_zip"),
                "onnx_path":       LaunchConfiguration("onnx_path"),
                "calib_path":      LaunchConfiguration("calib_path"),
                "kb_persist":      LaunchConfiguration("kb_persist"),
                "enable_real2sim": LaunchConfiguration("enable_real2sim"),
                "enable_sim2real": LaunchConfiguration("enable_sim2real"),
            }],
        ),
    ])
