from glob import glob
from setuptools import find_packages, setup

package_name = "robot_amr_mujoco_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/models",  glob("models/*")),
        (f"share/{package_name}/urdf",    glob("urdf/*")),
        (f"share/{package_name}/launch",  glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="P RoboAI",
    maintainer_email="user@example.com",
    description="3-D MuJoCo AMR simulator",
    license="MIT",
    entry_points={
        "console_scripts": [
            "amr_mujoco_node = robot_amr_mujoco_sim.amr_mujoco_node:main",
        ],
    },
)
