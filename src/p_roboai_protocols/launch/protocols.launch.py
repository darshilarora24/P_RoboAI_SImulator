"""
protocols.launch.py  —  Launch P_RoboAI Protocol Manager.

Usage
-----
  # WebSocket dashboard only (default):
  ros2 launch p_roboai_protocols protocols.launch.py

  # Full industrial stack:
  ros2 launch p_roboai_protocols protocols.launch.py \\
      enable_opcua:=true  opcua_endpoint:=opc.tcp://0.0.0.0:4840/p_roboai \\
      enable_modbus:=true modbus_port:=5020 \\
      enable_mqtt:=true   mqtt_broker:=192.168.1.100 mqtt_robot_id:=amr_01 \\
      enable_grpc:=true   grpc_port:=50051 \\
      enable_websocket:=true ws_port:=8765 http_port:=8766

  # With Siemens PLC (PROFINET/S7):
  ros2 launch p_roboai_protocols protocols.launch.py \\
      enable_profinet:=true plc_ip:=192.168.1.10

  # CAN Bus for motor controllers:
  ros2 launch p_roboai_protocols protocols.launch.py \\
      enable_can:=true can_channel:=can0 can_bitrate:=500000

  # Switch to Cyclone DDS:
  ros2 launch p_roboai_protocols protocols.launch.py dds_rmw:=cyclone
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        # DDS
        DeclareLaunchArgument("dds_rmw",          default_value="",         description="RMW: fastdds/cyclone/connext"),
        DeclareLaunchArgument("dds_profile",      default_value="fastdds",  description="DDS profile to write: fastdds/cyclone"),

        # OPC UA
        DeclareLaunchArgument("enable_opcua",     default_value="false"),
        DeclareLaunchArgument("opcua_endpoint",   default_value="opc.tcp://0.0.0.0:4840/p_roboai"),
        DeclareLaunchArgument("opcua_client_url", default_value=""),

        # Modbus
        DeclareLaunchArgument("enable_modbus",    default_value="false"),
        DeclareLaunchArgument("modbus_host",      default_value="0.0.0.0"),
        DeclareLaunchArgument("modbus_port",      default_value="5020"),

        # CAN / CANopen
        DeclareLaunchArgument("enable_can",       default_value="false"),
        DeclareLaunchArgument("can_interface",    default_value="socketcan"),
        DeclareLaunchArgument("can_channel",      default_value="can0"),
        DeclareLaunchArgument("can_bitrate",      default_value="500000"),

        # EtherCAT
        DeclareLaunchArgument("enable_ethercat",  default_value="false"),
        DeclareLaunchArgument("ethercat_iface",   default_value="eth0"),

        # PROFINET / S7
        DeclareLaunchArgument("enable_profinet",  default_value="false"),
        DeclareLaunchArgument("profinet_iface",   default_value="eth0"),
        DeclareLaunchArgument("plc_ip",           default_value=""),

        # MQTT
        DeclareLaunchArgument("enable_mqtt",      default_value="false"),
        DeclareLaunchArgument("mqtt_broker",      default_value="localhost"),
        DeclareLaunchArgument("mqtt_port",        default_value="1883"),
        DeclareLaunchArgument("mqtt_robot_id",    default_value="robot_01"),
        DeclareLaunchArgument("mqtt_username",    default_value=""),
        DeclareLaunchArgument("mqtt_password",    default_value=""),

        # gRPC
        DeclareLaunchArgument("enable_grpc",      default_value="false"),
        DeclareLaunchArgument("grpc_port",        default_value="50051"),

        # WebSocket
        DeclareLaunchArgument("enable_websocket", default_value="true"),
        DeclareLaunchArgument("ws_port",          default_value="8765"),
        DeclareLaunchArgument("http_port",        default_value="8766"),
    ]

    node = Node(
        package    = "p_roboai_protocols",
        executable = "protocol_manager",
        name       = "protocol_manager",
        output     = "screen",
        parameters = [{
            "dds_rmw":          LaunchConfiguration("dds_rmw"),
            "dds_profile":      LaunchConfiguration("dds_profile"),
            "enable_opcua":     LaunchConfiguration("enable_opcua"),
            "opcua_endpoint":   LaunchConfiguration("opcua_endpoint"),
            "opcua_client_url": LaunchConfiguration("opcua_client_url"),
            "enable_modbus":    LaunchConfiguration("enable_modbus"),
            "modbus_host":      LaunchConfiguration("modbus_host"),
            "modbus_port":      LaunchConfiguration("modbus_port"),
            "enable_can":       LaunchConfiguration("enable_can"),
            "can_interface":    LaunchConfiguration("can_interface"),
            "can_channel":      LaunchConfiguration("can_channel"),
            "can_bitrate":      LaunchConfiguration("can_bitrate"),
            "enable_ethercat":  LaunchConfiguration("enable_ethercat"),
            "ethercat_iface":   LaunchConfiguration("ethercat_iface"),
            "enable_profinet":  LaunchConfiguration("enable_profinet"),
            "profinet_iface":   LaunchConfiguration("profinet_iface"),
            "plc_ip":           LaunchConfiguration("plc_ip"),
            "enable_mqtt":      LaunchConfiguration("enable_mqtt"),
            "mqtt_broker":      LaunchConfiguration("mqtt_broker"),
            "mqtt_port":        LaunchConfiguration("mqtt_port"),
            "mqtt_robot_id":    LaunchConfiguration("mqtt_robot_id"),
            "mqtt_username":    LaunchConfiguration("mqtt_username"),
            "mqtt_password":    LaunchConfiguration("mqtt_password"),
            "enable_grpc":      LaunchConfiguration("enable_grpc"),
            "grpc_port":        LaunchConfiguration("grpc_port"),
            "enable_websocket": LaunchConfiguration("enable_websocket"),
            "ws_port":          LaunchConfiguration("ws_port"),
            "http_port":        LaunchConfiguration("http_port"),
        }],
    )

    return LaunchDescription(args + [node])
