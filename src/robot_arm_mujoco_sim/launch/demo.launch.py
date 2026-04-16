from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("robot_arm_mujoco_sim"))
    simulator_launch = package_share / "launch" / "simulator.launch.py"

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(simulator_launch)),
                launch_arguments={"start_demo_commander": "true"}.items(),
            )
        ]
    )
