from glob import glob

from setuptools import setup


package_name = "robot_arm_mujoco_sim"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/models", glob("models/*.xml")),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="codex@example.com",
    description="Minimal ROS 2 + MuJoCo 3D simulator for a 4-DOF robot arm.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = robot_arm_mujoco_sim.mujoco_sim_node:main",
            "sine_commander_node = robot_arm_mujoco_sim.sine_commander_node:main",
        ],
    },
)
