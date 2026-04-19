from glob import glob
from setuptools import find_packages, setup

package_name = "p_roboai_protocols"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/proto",  glob("proto/*.proto")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="P RoboAI",
    maintainer_email="user@example.com",
    description="Multi-protocol bridge: DDS, OPC UA, Modbus, CAN, EtherCAT, PROFINET, MQTT, gRPC, WebSocket",
    license="MIT",
    entry_points={
        "console_scripts": [
            "protocol_manager = p_roboai_protocols.protocol_manager_node:main",
        ],
    },
)
