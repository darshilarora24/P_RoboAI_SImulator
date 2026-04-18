"""
nav_node.py  —  P_RoboAI_Nav2

Navigation stack:  SLAM costmap → A* global path → DWA local control.

Topics
------
Subscribes:   /p_roboai_slam/map  (nav_msgs/OccupancyGrid)
              /amr/odom            (nav_msgs/Odometry)
              /amr/goal_pose       (geometry_msgs/PoseStamped)

Publishes:    /amr/cmd_vel                (geometry_msgs/Twist)
              /p_roboai_nav2/path         (nav_msgs/Path)
              /p_roboai_nav2/costmap      (nav_msgs/OccupancyGrid)
              /p_roboai_nav2/status       (std_msgs/String)
"""
from __future__ import annotations

import math

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid as ROSOccGrid, Odometry, Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .costmap       import Costmap
from .global_planner import plan as global_plan
from .local_planner  import DWAConfig, dwa_step, _angle_diff


GOAL_TOL      = 0.25    # m — goal reached tolerance
LOOKAHEAD     = 0.70    # m — pure-pursuit look-ahead (shorter = tighter tracking)
REPLAN_DIST   = 0.25    # m — replan if goal moves farther than this
STUCK_TIME    = 4.0     # s — seconds without progress → recovery
STUCK_DIST    = 0.04    # m — "no progress" if moved less than this
CTRL_HZ       = 20.0    # Hz — control loop rate


class NavNode(Node):
    def __init__(self) -> None:
        super().__init__("p_roboai_nav2")

        # ── State ─────────────────────────────────────────────────────────────
        self._costmap    = Costmap(inflation_radius=0.35)
        self._dwa_cfg    = DWAConfig()
        self._path:      list[tuple[float, float]] = []
        self._goal:      tuple[float, float] | None = None

        self._robot_x     = 1.0
        self._robot_y     = 1.0
        self._robot_theta = 0.0
        self._robot_v     = 0.0
        self._robot_w     = 0.0

        self._reached  = True
        self._map_ready = False

        # Stuck detection
        self._last_progress_x     = 1.0
        self._last_progress_y     = 1.0
        self._last_progress_stamp = self.get_clock().now().nanoseconds * 1e-9
        self._recovering           = False
        self._recovery_end         = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(ROSOccGrid,   "/p_roboai_slam/map", self._map_cb,  1)
        self.create_subscription(Odometry,     "/amr/odom",          self._odom_cb, 10)
        self.create_subscription(PoseStamped,  "/amr/goal_pose",     self._goal_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_pub     = self.create_publisher(Twist,      "/amr/cmd_vel",            10)
        self._path_pub    = self.create_publisher(Path,       "/p_roboai_nav2/path",     1)
        self._cost_pub    = self.create_publisher(ROSOccGrid, "/p_roboai_nav2/costmap",  1)
        self._status_pub  = self.create_publisher(String,     "/p_roboai_nav2/status",   10)

        # ── Control loop timer ────────────────────────────────────────────────
        self._ctrl_dt = 1.0 / CTRL_HZ
        self._timer   = self.create_timer(self._ctrl_dt, self._control_loop)

        self.get_logger().info("P_RoboAI_Nav2 ready — waiting for map and goal.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _map_cb(self, msg: ROSOccGrid) -> None:
        self._costmap.update_from_slam(msg)
        self._map_ready = True
        # Republish inflated costmap for visualisation
        self._cost_pub.publish(
            self._costmap.to_ros_msg(self.get_clock().now().to_msg()))
        # Replan if we have a pending goal
        if self._goal and not self._reached:
            self._do_replan()

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_theta = math.atan2(siny, cosy)
        self._robot_v = msg.twist.twist.linear.x
        self._robot_w = msg.twist.twist.angular.z

    def _goal_cb(self, msg: PoseStamped) -> None:
        new_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"New goal: ({new_goal[0]:.2f}, {new_goal[1]:.2f})")

        if self._goal is not None:
            prev = self._goal
            if math.hypot(new_goal[0]-prev[0], new_goal[1]-prev[1]) < 0.05:
                return   # same goal — ignore

        self._goal    = new_goal
        self._reached = False
        self._recovering = False

        if self._map_ready:
            self._do_replan()
        else:
            self.get_logger().warn("Map not ready yet — will plan when map arrives.")

    # ── Global replan ─────────────────────────────────────────────────────────

    def _do_replan(self) -> None:
        if self._goal is None:
            return
        path = global_plan(
            self._costmap,
            self._robot_x, self._robot_y,
            self._goal[0], self._goal[1],
        )
        if not path:
            self.get_logger().warn("Global planner: no path found.")
            self._publish_status("NO_PATH")
            return
        self._path = path
        self.get_logger().info(f"New path: {len(path)} waypoints.")
        self._publish_path(path)

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # ── Recovery mode ─────────────────────────────────────────────────────
        if self._recovering:
            if now_s < self._recovery_end:
                # Rotate in place
                cmd = Twist()
                cmd.angular.z = 0.8
                self._cmd_pub.publish(cmd)
                return
            else:
                self._recovering = False
                self._do_replan()

        # ── Idle ──────────────────────────────────────────────────────────────
        if self._reached or not self._path or self._goal is None:
            self._publish_stop()
            return

        # ── Goal reached? ─────────────────────────────────────────────────────
        gx, gy = self._goal
        if math.hypot(gx - self._robot_x, gy - self._robot_y) < GOAL_TOL:
            self._reached = True
            self._path    = []
            self._publish_stop()
            self._publish_status("GOAL_REACHED")
            self.get_logger().info("Goal reached.")
            return

        # ── Prune passed waypoints ─────────────────────────────────────────────
        # Remove waypoints the robot has already passed (closer than 30% of lookahead)
        while len(self._path) > 1:
            wx, wy = self._path[0]
            if math.hypot(wx - self._robot_x, wy - self._robot_y) < LOOKAHEAD * 0.30:
                self._path.pop(0)
            else:
                break

        # ── Find lookahead point ───────────────────────────────────────────────
        # Default to last waypoint (goal); prefer the first waypoint >= LOOKAHEAD away
        lx, ly = self._path[-1]
        for wx, wy in self._path:
            d = math.hypot(wx - self._robot_x, wy - self._robot_y)
            if d >= LOOKAHEAD * 0.5:   # use 50% of lookahead to avoid ignoring close waypoints
                lx, ly = wx, wy
                break

        # ── DWA local command ─────────────────────────────────────────────────
        v, w = dwa_step(
            self._costmap,
            self._robot_x, self._robot_y, self._robot_theta,
            self._robot_v, self._robot_w,
            lx, ly,
            self._dwa_cfg,
            self._ctrl_dt,
        )

        cmd = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(w)
        self._cmd_pub.publish(cmd)
        self._publish_status("NAVIGATING")

        # ── Stuck detection ───────────────────────────────────────────────────
        moved = math.hypot(self._robot_x - self._last_progress_x,
                           self._robot_y - self._last_progress_y)
        if moved > STUCK_DIST:
            self._last_progress_x = self._robot_x
            self._last_progress_y = self._robot_y
            self._last_progress_stamp = now_s

        if now_s - self._last_progress_stamp > STUCK_TIME:
            self.get_logger().warn("Robot stuck — entering recovery.")
            self._recovering      = True
            self._recovery_end    = now_s + 2.5   # rotate for 2.5 s
            self._last_progress_stamp = now_s

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _publish_status(self, s: str) -> None:
        msg = String()
        msg.data = s
        self._status_pub.publish(msg)

    def _publish_path(self, path: list[tuple[float, float]]) -> None:
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for wx, wy in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node: NavNode | None = None
    try:
        node = NavNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
