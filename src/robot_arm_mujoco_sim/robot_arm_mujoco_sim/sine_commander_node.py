from __future__ import annotations

import math

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_JOINT_NAMES = [
    "shoulder_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
]


class SineCommander(Node):
    def __init__(self) -> None:
        super().__init__("sine_commander")

        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("amplitudes", [0.75, 0.35, 0.55, 0.25])
        self.declare_parameter("offsets", [0.0, 0.45, -0.95, 0.55])
        self.declare_parameter("frequencies", [0.05, 0.08, 0.07, 0.12])
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("trajectory_topic", "joint_trajectory")

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.amplitudes = self._expand_values(
            list(self.get_parameter("amplitudes").value), "amplitudes"
        )
        self.offsets = self._expand_values(
            list(self.get_parameter("offsets").value), "offsets"
        )
        self.frequencies = self._expand_values(
            list(self.get_parameter("frequencies").value), "frequencies"
        )
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        trajectory_topic = str(self.get_parameter("trajectory_topic").value)

        self.publisher = self.create_publisher(JointTrajectory, trajectory_topic, 10)
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._publish_command)

        self.get_logger().info(
            f"Publishing demo trajectory commands to '{trajectory_topic}'."
        )

    def _expand_values(self, values: list[float], label: str) -> list[float]:
        if len(values) == len(self.joint_names):
            return [float(value) for value in values]
        if len(values) == 1:
            return [float(values[0]) for _ in self.joint_names]

        self.get_logger().warning(
            f"Parameter '{label}' has {len(values)} values, expected "
            f"{len(self.joint_names)}. Repeating the final value."
        )
        if not values:
            return [0.0 for _ in self.joint_names]

        expanded = [float(value) for value in values]
        while len(expanded) < len(self.joint_names):
            expanded.append(expanded[-1])
        return expanded[: len(self.joint_names)]

    def _publish_command(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        point = JointTrajectoryPoint()
        point.positions = [
            offset
            + amplitude * math.sin((2.0 * math.pi * frequency * elapsed) + (index * 0.8))
            for index, (amplitude, offset, frequency) in enumerate(
                zip(self.amplitudes, self.offsets, self.frequencies)
            )
        ]
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)

        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(self.joint_names)
        message.points = [point]
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = SineCommander()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
