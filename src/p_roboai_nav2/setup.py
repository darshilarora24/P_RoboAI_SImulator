from glob import glob
from setuptools import find_packages, setup

package_name = "p_roboai_nav2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="P RoboAI",
    maintainer_email="user@example.com",
    description="P_RoboAI custom navigation stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "nav_node = p_roboai_nav2.nav_node:main",
        ],
    },
)
