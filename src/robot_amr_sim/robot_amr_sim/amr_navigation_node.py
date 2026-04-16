"""
AMR navigation node.

Builds an inflated occupancy grid from the static obstacle map, plans an A*
path to a received goal pose, and executes the path with a pure-pursuit
controller published as /amr/cmd_vel.

Topics
------
Subscribes:  /amr/odom       (nav_msgs/Odometry)
             /amr/goal_pose  (geometry_msgs/PoseStamped)
Publishes:   /amr/cmd_vel    (geometry_msgs/Twist)
             /amr/path       (nav_msgs/Path)
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node

# ---------------------------------------------------------------------------
# Shared map definition  (identical to amr_sim_node.py)
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

GRID_SIZE  = 100     # cells per axis (10 m / 0.1 m)
CELL_SIZE  = 0.1     # metres per cell
INFLATE_R  = 4       # cells of obstacle inflation (≥ robot_radius / cell_size = 2.8)

LOOKAHEAD  = 0.8     # pure-pursuit lookahead distance (m)
MAX_VEL    = 0.5     # linear speed cap (m/s)
MAX_OMG    = 2.0     # angular speed cap (rad/s)
GOAL_TOL   = 0.15    # stop when within this distance of goal (m)


# ---------------------------------------------------------------------------
# Occupancy grid builder
# ---------------------------------------------------------------------------

def _build_grid() -> list[list[bool]]:
    """Return GRID_SIZE × GRID_SIZE boolean grid; True = occupied / inflated."""
    grid: list[list[bool]] = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    for (x1, x2, y1, y2) in OBSTACLES:
        gx1 = max(0,            int(x1 / CELL_SIZE) - INFLATE_R)
        gx2 = min(GRID_SIZE - 1, int(math.ceil(x2 / CELL_SIZE)) + INFLATE_R)
        gy1 = max(0,            int(y1 / CELL_SIZE) - INFLATE_R)
        gy2 = min(GRID_SIZE - 1, int(math.ceil(y2 / CELL_SIZE)) + INFLATE_R)
        for gx in range(gx1, gx2 + 1):
            for gy in range(gy1, gy2 + 1):
                grid[gy][gx] = True
    return grid


# ---------------------------------------------------------------------------
# A* path planner
# ---------------------------------------------------------------------------

_DIRS = [(-1, -1), (0, -1), (1, -1),
         (-1,  0),           (1,  0),
         (-1,  1), (0,  1), (1,  1)]
_COSTS = [math.sqrt(2), 1.0, math.sqrt(2),
          1.0,               1.0,
          math.sqrt(2), 1.0, math.sqrt(2)]


def _astar(
    grid: list[list[bool]],
    start: tuple[int, int],
    goal:  tuple[int, int],
) -> list[tuple[int, int]]:
    """Return grid-cell path from *start* to *goal*, or [] if unreachable."""

    def h(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start, goal), 0.0, start))

    came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    g_score:   dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path: list[tuple[int, int]] = []
            node: Optional[tuple[int, int]] = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path
        if g > g_score.get(cur, float("inf")) + 1e-9:
            continue
        for (dx, dy), cost in zip(_DIRS, _COSTS):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if grid[ny][nx]:
                continue
            ng = g + cost
            neighbour = (nx, ny)
            if ng < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = ng
                came_from[neighbour] = cur
                heapq.heappush(open_heap,
                               (ng + h(neighbour, goal), ng, neighbour))
    return []


# ---------------------------------------------------------------------------
# Navigation node
# ---------------------------------------------------------------------------

class AMRNavigationNode(Node):
    def __init__(self) -> None:
        super().__init__("amr_navigation")

        self._grid = _build_grid()
        self.get_logger().info("Occupancy grid built.")

        # Robot state
        self._x:     float = 1.0
        self._y:     float = 1.0
        self._theta: float = 0.0

        # Path tracking
        self._path:         list[tuple[float, float]] = []
        self._goal:         Optional[tuple[float, float]] = None
        self._reached_goal: bool = True

        # ROS2 I/O
        self._odom_sub = self.create_subscription(
            Odometry, "/amr/odom", self._odom_cb, 10)
        self._goal_sub = self.create_subscription(
            PoseStamped, "/amr/goal_pose", self._goal_cb, 10)

        self._cmd_pub  = self.create_publisher(Twist, "/amr/cmd_vel", 10)
        self._path_pub = self.create_publisher(Path,  "/amr/path",    10)

        self._timer = self.create_timer(0.05, self._control_loop)  # 20 Hz
        self.get_logger().info("AMR navigation node ready.")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._theta = math.atan2(siny, cosy)

    def _goal_cb(self, msg: PoseStamped) -> None:
        gx_w = msg.pose.position.x
        gy_w = msg.pose.position.y
        self.get_logger().info(f"New goal: ({gx_w:.2f}, {gy_w:.2f})")

        start_cell = self._world_to_cell(self._x, self._y)
        goal_cell  = self._world_to_cell(gx_w, gy_w)

        cells = _astar(self._grid, start_cell, goal_cell)
        if not cells:
            self.get_logger().warn("A*: no path found — goal may be inside an obstacle.")
            return

        self._path = [self._cell_to_world(c[0], c[1]) for c in cells]
        self._goal = (gx_w, gy_w)
        self._reached_goal = False
        self._publish_path()
        self.get_logger().info(f"Path found: {len(self._path)} waypoints.")

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if self._reached_goal or not self._path:
            self._publish_stop()
            return

        # Check if goal reached
        assert self._goal is not None
        dist_to_goal = math.hypot(self._goal[0] - self._x, self._goal[1] - self._y)
        if dist_to_goal < GOAL_TOL:
            self._reached_goal = True
            self._path = []
            self._publish_stop()
            self.get_logger().info("Goal reached.")
            return

        # Prune path points that the robot has already passed
        while len(self._path) > 1:
            wx, wy = self._path[0]
            if math.hypot(wx - self._x, wy - self._y) < LOOKAHEAD * 0.4:
                self._path.pop(0)
            else:
                break

        # Pure pursuit: find lookahead point
        lx, ly = self._find_lookahead()

        # Compute steering
        target_angle = math.atan2(ly - self._y, lx - self._x)
        angle_err    = self._angle_diff(target_angle, self._theta)

        v = MAX_VEL * max(0.0, math.cos(angle_err))
        w = MAX_OMG * math.tanh(angle_err * 1.5)

        # Decelerate near goal
        if dist_to_goal < 0.5:
            v *= dist_to_goal / 0.5

        cmd = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(w)
        self._cmd_pub.publish(cmd)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_lookahead(self) -> tuple[float, float]:
        """Return first path point at least LOOKAHEAD ahead, else last point."""
        for wx, wy in self._path:
            if math.hypot(wx - self._x, wy - self._y) >= LOOKAHEAD:
                return (wx, wy)
        return self._path[-1]

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _publish_path(self) -> None:
        msg = Path()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        for wx, wy in self._path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    @staticmethod
    def _world_to_cell(x: float, y: float) -> tuple[int, int]:
        gx = max(0, min(GRID_SIZE - 1, int(x / CELL_SIZE)))
        gy = max(0, min(GRID_SIZE - 1, int(y / CELL_SIZE)))
        return (gx, gy)

    @staticmethod
    def _cell_to_world(gx: int, gy: int) -> tuple[float, float]:
        return (gx * CELL_SIZE + CELL_SIZE * 0.5,
                gy * CELL_SIZE + CELL_SIZE * 0.5)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        while d >  math.pi: d -= 2.0 * math.pi
        while d < -math.pi: d += 2.0 * math.pi
        return d


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: AMRNavigationNode | None = None
    try:
        node = AMRNavigationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
