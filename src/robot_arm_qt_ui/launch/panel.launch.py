from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    simulator_share = Path(get_package_share_directory("robot_arm_mujoco_sim"))
    model_path = simulator_share / "models" / "robot_arm.xml"
    state_pub_rate_hz = LaunchConfiguration("state_pub_rate_hz")
    publish_clock = LaunchConfiguration("publish_clock")

    return LaunchDescription(
        [
            DeclareLaunchArgument("state_pub_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("publish_clock", default_value="true"),
            Node(
                package="robot_arm_qt_ui",
                executable="robot_arm_qt_panel",
                name="robot_arm_qt_panel",
                output="screen",
                parameters=[
                    {
                        "model_path": str(model_path),
                        "state_pub_rate_hz": ParameterValue(
                            state_pub_rate_hz, value_type=float
                        ),
                        "publish_clock": ParameterValue(
                            publish_clock, value_type=bool
                        ),
                    }
                ],
            )
        ]
    )
