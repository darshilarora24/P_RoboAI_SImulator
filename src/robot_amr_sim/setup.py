from setuptools import find_packages, setup

package_name = "robot_amr_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/amr_sim.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="P RoboAI",
    maintainer_email="user@example.com",
    description="2-D kinematic AMR simulator with A* navigation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "amr_sim_node = robot_amr_sim.amr_sim_node:main",
            "amr_navigation_node = robot_amr_sim.amr_navigation_node:main",
        ],
    },
)
