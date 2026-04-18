"""
yolo.launch.py  —  Launch YOLO detection node.

Usage
-----
  # Default: subscribe to /camera/image_raw (falls back to MuJoCo if no image)
  ros2 launch p_roboai_yolo yolo.launch.py

  # Use AMR camera:
  ros2 launch p_roboai_yolo yolo.launch.py image_topic:=/amr/camera/image

  # Custom model and confidence:
  ros2 launch p_roboai_yolo yolo.launch.py model_name:=yolov8s.pt conf_thresh:=0.4
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/amr/camera/image",
                              description="Camera image topic to subscribe"),
        DeclareLaunchArgument("model_name",  default_value="yolo11n.pt",
                              description="YOLO model name or path"),
        DeclareLaunchArgument("conf_thresh", default_value="0.35",
                              description="Detection confidence threshold"),
        DeclareLaunchArgument("detect_hz",   default_value="10.0",
                              description="Max detection rate (Hz)"),

        Node(
            package    = "p_roboai_yolo",
            executable = "yolo_node",
            name       = "yolo_node",
            output     = "screen",
            parameters = [{
                "image_topic": LaunchConfiguration("image_topic"),
                "model_name":  LaunchConfiguration("model_name"),
                "conf_thresh": LaunchConfiguration("conf_thresh"),
                "detect_hz":   LaunchConfiguration("detect_hz"),
                "publish_annotated": True,
            }],
        ),
    ])
