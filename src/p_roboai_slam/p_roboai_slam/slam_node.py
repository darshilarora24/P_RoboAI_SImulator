"""
slam_node.py  —  P_RoboAI_SLAM

Online occupancy-grid SLAM node.

Pipeline per scan
-----------------
1. Motion model: propagate pose by odometry delta.
2. Scan matching: correlative search around predicted pose.
3. Map update:   log-odds Bresenham ray updates.
4. Publish:      /p_roboai_slam/map  (OccupancyGrid)
                 TF: map → odom

Topics
------
Subscribes:   /amr/scan    (sensor_msgs/LaserScan)
              /amr/odom    (nav_msgs/Odometry)
Publishes:    /p_roboai_slam/map   (nav_msgs/OccupancyGrid)
              /p_roboai_slam/pose  (geometry_msgs/PoseStamped)
TF:           map → odom
"""
from __future__ import annotations

import math

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid as ROSOccGrid, Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from .occupancy_grid import OccupancyGrid
from .scan_matcher   import match as scan_match


# How many scans to skip between map updates (controls CPU load)
_MAP_UPDATE_EVERY = 3   # update map every N scans
# Minimum distance / angle moved before accepting a new scan into the map
_MIN_DIST   = 0.15   # m
_MIN_ANGLE  = 0.08   # rad


class SlamNode(Node):
    def __init__(self) -> None:
        super().__init__("p_roboai_slam")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("map_width_m",  10.0)
        self.declare_parameter("map_height_m", 10.0)
        self.declare_parameter("resolution",   0.05)
        self.declare_parameter("origin_x",     0.0)
        self.declare_parameter("origin_y",     0.0)
        self.declare_parameter("use_scan_matching", True)

        w   = float(self.get_parameter("map_width_m").value)
        h   = float(self.get_parameter("map_height_m").value)
        res = float(self.get_parameter("resolution").value)
        ox  = float(self.get_parameter("origin_x").value)
        oy  = float(self.get_parameter("origin_y").value)
        self._use_matching = bool(self.get_parameter("use_scan_matching").value)

        # ── Occupancy grid ────────────────────────────────────────────────────
        self._grid = OccupancyGrid(w, h, res, ox, oy)

        # ── SLAM pose (map frame) ─────────────────────────────────────────────
        self._map_x:     float = 1.0
        self._map_y:     float = 1.0
        self._map_theta: float = 0.0

        # ── Odometry tracking ─────────────────────────────────────────────────
        self._odom_x:     float | None = None
        self._odom_y:     float | None = None
        self._odom_theta: float | None = None

        # Last pose at which we inserted a scan into the map
        self._last_insert_x:     float = 1.0
        self._last_insert_y:     float = 1.0
        self._last_insert_theta: float = 0.0

        # ── Scan-update counter ───────────────────────────────────────────────
        self._scan_count = 0

        # ── ROS2 interfaces ───────────────────────────────────────────────────
        self._scan_sub = self.create_subscription(
            LaserScan, "/amr/scan", self._scan_cb, 5)
        self._odom_sub = self.create_subscription(
            Odometry, "/amr/odom", self._odom_cb, 10)

        self._map_pub  = self.create_publisher(ROSOccGrid,   "/p_roboai_slam/map",  1)
        self._pose_pub = self.create_publisher(PoseStamped,  "/p_roboai_slam/pose", 10)
        self._tf_broad = TransformBroadcaster(self)

        # Publish map at fixed rate regardless of scan rate
        self._map_timer = self.create_timer(2.0, self._publish_map)

        self.get_logger().info(
            f"P_RoboAI_SLAM ready — {int(w/res)}×{int(h/res)} cells "
            f"@ {res*100:.0f} cm,  scan_matching={'ON' if self._use_matching else 'OFF'}")

    # ── Odometry callback ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_theta = math.atan2(siny, cosy)

    # ── Scan callback (main SLAM loop) ────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan) -> None:
        if self._odom_x is None:
            return   # wait for first odometry

        # ── Motion model: propagate pose via odometry ─────────────────────────
        ox, oy, ot = self._odom_x, self._odom_y, self._odom_theta

        # Delta in odom frame since last scan
        dx_odom = ox - (self._odom_x if self._odom_x is not None else ox)
        dy_odom = oy - (self._odom_y if self._odom_y is not None else oy)
        dth     = ot - (self._odom_theta if self._odom_theta is not None else ot)

        # Actually use absolute odom → compute where the robot is in map frame
        # by tracking the cumulative map→odom transform.
        # map_pose = map_T_odom ∘ odom_pose
        # We maintain map_T_odom implicitly: difference between map_pose and odom_pose.
        # Simple approach: predicted map pose = last map pose + odom delta
        # Compute odom delta since the last scan callback
        if not hasattr(self, "_prev_odom_x"):
            self._prev_odom_x     = ox
            self._prev_odom_y     = oy
            self._prev_odom_theta = ot

        delta_x   = ox - self._prev_odom_x
        delta_y   = oy - self._prev_odom_y
        delta_th  = ot - self._prev_odom_theta
        # Normalise
        delta_th  = (delta_th + math.pi) % (2 * math.pi) - math.pi

        # Apply delta in map frame (rotate by current map heading)
        c, s = math.cos(self._map_theta), math.sin(self._map_theta)
        self._map_x     += c * delta_x - s * delta_y
        self._map_y     += s * delta_x + c * delta_y
        self._map_theta  = (self._map_theta + delta_th + math.pi) % (2 * math.pi) - math.pi

        self._prev_odom_x     = ox
        self._prev_odom_y     = oy
        self._prev_odom_theta = ot

        # ── Scan matching: refine predicted pose ──────────────────────────────
        if self._use_matching:
            cx, cy, ct, score = scan_match(
                self._grid,
                self._map_x, self._map_y, self._map_theta,
                msg.angle_min, msg.angle_increment,
                list(msg.ranges), msg.range_max,
            )
            self._map_x     = cx
            self._map_y     = cy
            self._map_theta = ct

        # ── Decide whether to update the map ─────────────────────────────────
        self._scan_count += 1
        dist  = math.hypot(self._map_x - self._last_insert_x,
                           self._map_y - self._last_insert_y)
        angle = abs((self._map_theta - self._last_insert_theta + math.pi)
                    % (2 * math.pi) - math.pi)
        should_update = (
            self._scan_count % _MAP_UPDATE_EVERY == 0
            and (dist >= _MIN_DIST or angle >= _MIN_ANGLE
                 or self._scan_count <= 10)  # always update first 10 scans
        )

        if should_update:
            self._grid.update_scan(
                self._map_x, self._map_y, self._map_theta,
                msg.angle_min, msg.angle_increment,
                list(msg.ranges), msg.range_max,
            )
            self._last_insert_x     = self._map_x
            self._last_insert_y     = self._map_y
            self._last_insert_theta = self._map_theta

        # ── Publish TF map → odom ─────────────────────────────────────────────
        self._broadcast_map_odom_tf(msg.header.stamp)

        # ── Publish pose ──────────────────────────────────────────────────────
        ps = PoseStamped()
        ps.header.stamp    = msg.header.stamp
        ps.header.frame_id = "map"
        ps.pose.position.x = self._map_x
        ps.pose.position.y = self._map_y
        ht = self._map_theta / 2.0
        ps.pose.orientation.w = math.cos(ht)
        ps.pose.orientation.z = math.sin(ht)
        self._pose_pub.publish(ps)

    # ── TF broadcaster ────────────────────────────────────────────────────────

    def _broadcast_map_odom_tf(self, stamp) -> None:
        """
        Compute and broadcast TF map → odom.

        SLAM says robot is at (map_x, map_y, map_θ) in map frame.
        Odometry says robot is at (odom_x, odom_y, odom_θ) in odom frame.

        T_map_odom = T_map_robot × inv(T_odom_robot)
        """
        if self._odom_x is None:
            return

        mx, my, mt = self._map_x, self._map_y, self._map_theta
        ox, oy, ot = self._odom_x, self._odom_y, self._odom_theta  # type: ignore

        # inv(T_odom_robot)
        inv_x =  -(ox * math.cos(ot) + oy * math.sin(ot))
        inv_y =  -(- ox * math.sin(ot) + oy * math.cos(ot))
        inv_t = -ot

        # T_map_odom = compose(T_map_robot, inv_T_odom_robot)
        tmo_x = mx + inv_x * math.cos(mt) - inv_y * math.sin(mt)
        tmo_y = my + inv_x * math.sin(mt) + inv_y * math.cos(mt)
        tmo_t = mt + inv_t

        qw = math.cos(tmo_t / 2.0)
        qz = math.sin(tmo_t / 2.0)

        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = "map"
        tf.child_frame_id  = "odom"
        tf.transform.translation.x = tmo_x
        tf.transform.translation.y = tmo_y
        tf.transform.rotation.w    = qw
        tf.transform.rotation.z    = qz
        self._tf_broad.sendTransform(tf)

    # ── Periodic map publish ──────────────────────────────────────────────────

    def _publish_map(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._map_pub.publish(self._grid.to_ros_msg(stamp))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node: SlamNode | None = None
    try:
        node = SlamNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
