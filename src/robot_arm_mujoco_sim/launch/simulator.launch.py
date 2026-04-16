from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("robot_arm_mujoco_sim"))
    model_path = package_share / "models" / "robot_arm.xml"
    urdf_path = package_share / "urdf" / "robot_arm.urdf"
    robot_description = urdf_path.read_text()

    use_viewer = LaunchConfiguration("use_viewer")
    start_demo_commander = LaunchConfiguration("start_demo_commander")
    start_robot_state_publisher = LaunchConfiguration("start_robot_state_publisher")
    sim_rate_hz = LaunchConfiguration("sim_rate_hz")
    state_pub_rate_hz = LaunchConfiguration("state_pub_rate_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_viewer", default_value="true"),
            DeclareLaunchArgument("start_demo_commander", default_value="false"),
            DeclareLaunchArgument(
                "start_robot_state_publisher", default_value="true"
            ),
            DeclareLaunchArgument("sim_rate_hz", default_value="500.0"),
            DeclareLaunchArgument("state_pub_rate_hz", default_value="50.0"),
            Node(
                package="robot_arm_mujoco_sim",
                executable="mujoco_sim_node",
                name="mujoco_arm_simulator",
                output="screen",
                parameters=[
                    {
                        "model_path": str(model_path),
                        "use_viewer": ParameterValue(use_viewer, value_type=bool),
                        "sim_rate_hz": ParameterValue(sim_rate_hz, value_type=float),
                        "state_pub_rate_hz": ParameterValue(
                            state_pub_rate_hz, value_type=float
                        ),
                        "publish_clock": True,
                        "position_command_topic": "joint_position_cmd",
                        "trajectory_command_topic": "joint_trajectory",
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
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
