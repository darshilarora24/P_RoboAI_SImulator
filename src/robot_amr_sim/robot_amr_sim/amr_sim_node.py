"""
AMR kinematic simulation node.

Models a differential-drive robot on a 10 m × 10 m floor.
Publishes odometry and a simulated 2-D laser scan.

Topics
------
Subscribes:  /amr/cmd_vel   (geometry_msgs/Twist)
Publishes:   /amr/odom      (nav_msgs/Odometry)
             /amr/scan      (sensor_msgs/LaserScan)
"""
from __future__ import annotations

import math

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

# ---------------------------------------------------------------------------
# Shared map definition  (identical to the one in amr_navigation_node.py)
# Each entry is (x_min, x_max, y_min, y_max) in metres.
# ---------------------------------------------------------------------------
OBSTACLES: list[tuple[float, float, float, float]] = [
    (0.0, 10.0,  0.0,  0.3),   # south wall
    (0.0, 10.0,  9.7, 10.0),   # north wall
    (0.0,  0.3,  0.0, 10.0),   # west wall
    (9.7, 10.0,  0.0, 10.0),   # east wall
    (3.0,  3.3,  1.5,  6.0),   # interior wall A
    (5.0,  8.2,  4.0,  4.3),   # interior wall B
    (1.5,  2.5,  6.5,  7.5),   # box 1
    (7.0,  8.0,  1.0,  2.0),   # box 2
    (6.0,  7.0,  6.5,  7.5),   # box 3
    (4.5,  5.5,  1.5,  2.5),   # box 4
]

ROBOT_RADIUS = 0.28   # collision radius (m)
MAX_LINEAR   = 1.0    # m/s  clamp
MAX_ANGULAR  = 2.5    # rad/s clamp
LIDAR_RAYS   = 90     # number of scan rays
LIDAR_RANGE  = 6.0    # max range (m)
LIDAR_STEP   = 0.04   # ray-march step (m)


def _circle_rect_collision(cx: float, cy: float, r: float,
                            x1: float, x2: float,
                            y1: float, y2: float) -> bool:
    """True if a circle (cx, cy, r) overlaps rectangle (x1..x2, y1..y2)."""
    nearest_x = max(x1, min(cx, x2))
    nearest_y = max(y1, min(cy, y2))
    dx = cx - nearest_x
    dy = cy - nearest_y
    return (dx * dx + dy * dy) < (r * r)


class AMRSimNode(Node):
    def __init__(self) -> None:
        super().__init__("amr_sim")

        self.declare_parameter("start_x", 1.0)
        self.declare_parameter("start_y", 1.0)
        self.declare_parameter("start_theta", 0.0)
        self.declare_parameter("sim_rate_hz", 50.0)

        self._x     = float(self.get_parameter("start_x").value)
        self._y     = float(self.get_parameter("start_y").value)
        self._theta = float(self.get_parameter("start_theta").value)
        self._vx    = 0.0
        self._wz    = 0.0

        sim_rate = float(self.get_parameter("sim_rate_hz").value)
        self._dt = 1.0 / sim_rate

        self._cmd_sub = self.create_subscription(
            Twist, "/amr/cmd_vel", self._cmd_cb, 10)
        self._odom_pub  = self.create_publisher(Odometry,   "/amr/odom", 10)
        self._scan_pub  = self.create_publisher(LaserScan,  "/amr/scan", 10)
        self._tf_broad  = TransformBroadcaster(self)

        self._timer = self.create_timer(self._dt, self._step)
        self.get_logger().info(
            f"AMR simulator ready at ({self._x:.2f}, {self._y:.2f}, "
            f"θ={math.degrees(self._theta):.1f}°)")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist) -> None:
        self._vx = max(-MAX_LINEAR,  min(MAX_LINEAR,  msg.linear.x))
        self._wz = max(-MAX_ANGULAR, min(MAX_ANGULAR, msg.angular.z))

    def _step(self) -> None:
        # Integrate unicycle model
        new_x     = self._x + self._vx * math.cos(self._theta) * self._dt
        new_y     = self._y + self._vx * math.sin(self._theta) * self._dt
        new_theta = self._theta + self._wz * self._dt

        # Normalise heading to (−π, π]
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

        # Collision detection: block translation into obstacles
        if self._collides(new_x, new_y):
            new_x = self._x
            new_y = self._y
            self._vx = 0.0

        self._x     = new_x
        self._y     = new_y
        self._theta = new_theta

        self._publish_odom()
        self._publish_scan()

    # ── Publishing ───────────────────────────────────────────────────────────

    def _publish_odom(self) -> None:
        now = self.get_clock().now().to_msg()
        qw  = math.cos(self._theta / 2.0)
        qz  = math.sin(self._theta / 2.0)

        # TF: odom → amr_base_link
        tf = TransformStamped()
        tf.header.stamp        = now
        tf.header.frame_id     = "odom"
        tf.child_frame_id      = "amr_base_link"
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.rotation.w    = qw
        tf.transform.rotation.z    = qz
        self._tf_broad.sendTransform(tf)

        # Odometry message
        odom = Odometry()
        odom.header.stamp              = now
        odom.header.frame_id           = "odom"
        odom.child_frame_id            = "amr_base_link"
        odom.pose.pose.position.x      = self._x
        odom.pose.pose.position.y      = self._y
        odom.pose.pose.orientation.w   = qw
        odom.pose.pose.orientation.z   = qz
        odom.twist.twist.linear.x      = self._vx
        odom.twist.twist.angular.z     = self._wz
        self._odom_pub.publish(odom)

    def _publish_scan(self) -> None:
        angle_inc = 2.0 * math.pi / LIDAR_RAYS
        ranges: list[float] = []

        for i in range(LIDAR_RAYS):
            angle = self._theta + (-math.pi + i * angle_inc)
            ranges.append(self._cast_ray(angle))

        scan = LaserScan()
        scan.header.stamp       = self.get_clock().now().to_msg()
        scan.header.frame_id    = "amr_base_link"
        scan.angle_min          = -math.pi
        scan.angle_max          = math.pi
        scan.angle_increment    = angle_inc
        scan.range_min          = 0.1
        scan.range_max          = LIDAR_RANGE
        scan.ranges             = ranges
        self._scan_pub.publish(scan)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _collides(self, x: float, y: float) -> bool:
        for obs in OBSTACLES:
            if _circle_rect_collision(x, y, ROBOT_RADIUS, *obs):
                return True
        return False

    def _cast_ray(self, angle: float) -> float:
        dx = math.cos(angle) * LIDAR_STEP
        dy = math.sin(angle) * LIDAR_STEP
        rx, ry = self._x, self._y
        for step in range(int(LIDAR_RANGE / LIDAR_STEP)):
            rx += dx
            ry += dy
            for (x1, x2, y1, y2) in OBSTACLES:
                if x1 <= rx <= x2 and y1 <= ry <= y2:
                    return step * LIDAR_STEP
        return LIDAR_RANGE


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: AMRSimNode | None = None
    try:
        node = AMRSimNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
