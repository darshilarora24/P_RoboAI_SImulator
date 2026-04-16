from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    simulator_share = Path(get_package_share_directory("robot_arm_mujoco_sim"))
    model_path = simulator_share / "models" / "robot_arm.xml"
    urdf_path = simulator_share / "urdf" / "robot_arm.urdf"
    robot_description = urdf_path.read_text()

    start_demo_commander = LaunchConfiguration("start_demo_commander")
    start_robot_state_publisher = LaunchConfiguration("start_robot_state_publisher")
    state_pub_rate_hz = LaunchConfiguration("state_pub_rate_hz")
    publish_clock = LaunchConfiguration("publish_clock")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_demo_commander", default_value="false"),
            DeclareLaunchArgument(
                "start_robot_state_publisher", default_value="true"
            ),
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
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                condition=IfCondition(start_robot_state_publisher),
                parameters=[
                    {
                        "robot_description": robot_description,
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="robot_arm_mujoco_sim",
                executable="sine_commander_node",
                name="sine_commander",
                output="screen",
                condition=IfCondition(start_demo_commander),
                parameters=[{"use_sim_time": False}],
            ),
        ]
    )
