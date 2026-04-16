"""
occupancy_grid.py  —  P_RoboAI_SLAM

Log-odds occupancy grid with Bresenham ray updates.

Coordinate convention
---------------------
  World : (x, y) in metres, origin at grid corner (0, 0)
  Grid  : (col, row) = (gx, gy) — gx increases with x, gy with y
  numpy : grid[gy, gx]  (row-major, y = row)
"""
from __future__ import annotations

import math
from typing import Generator

import numpy as np
from nav_msgs.msg import MapMetaData, OccupancyGrid as ROSOccGrid
from geometry_msgs.msg import Pose


class OccupancyGrid:
    """
    Parameters
    ----------
    width_m, height_m : float   map extent in metres
    resolution        : float   metres per cell (e.g. 0.05)
    origin_x, origin_y: float  world coordinates of cell (0, 0) corner
    """

    # Log-odds update values
    L_OCC  =  0.40   # occupied hit
    L_FREE = -0.20   # free pass-through
    L_MIN  = -5.0    # clamp
    L_MAX  =  5.0

    def __init__(
        self,
        width_m: float  = 10.0,
        height_m: float = 10.0,
        resolution: float = 0.05,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> None:
        self.resolution = resolution
        self.origin_x   = origin_x
        self.origin_y   = origin_y
        self.cols       = int(math.ceil(width_m  / resolution))
        self.rows       = int(math.ceil(height_m / resolution))
        # Log-odds grid (float32)
        self._log       = np.zeros((self.rows, self.cols), dtype=np.float32)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def cell_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        cx = self.origin_x + (gx + 0.5) * self.resolution
        cy = self.origin_y + (gy + 0.5) * self.resolution
        return cx, cy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    # ── Map update ────────────────────────────────────────────────────────────

    def update_scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        angle_min: float,
        angle_inc: float,
        ranges: list[float],
        max_range: float,
    ) -> None:
        """Update log-odds with one LaserScan reading."""
        rx, ry = self.world_to_cell(robot_x, robot_y)

        for i, r in enumerate(ranges):
            if r <= 0.01:
                continue
            angle     = robot_theta + angle_min + i * angle_inc
            hit_x     = robot_x + r * math.cos(angle)
            hit_y     = robot_y + r * math.sin(angle)
            hx, hy    = self.world_to_cell(hit_x, hit_y)

            hit = r < max_range * 0.99   # True = obstacle endpoint

            # Walk cells along the ray, marking free
            for gx, gy in _bresenham(rx, ry, hx, hy):
                if not self.in_bounds(gx, gy):
                    break   # ray left map
                self._log[gy, gx] = np.clip(
                    self._log[gy, gx] + self.L_FREE, self.L_MIN, self.L_MAX)

            # Mark hit cell occupied
            if hit and self.in_bounds(hx, hy):
                self._log[hy, hx] = np.clip(
                    self._log[hy, hx] + self.L_OCC, self.L_MIN, self.L_MAX)

    # ── Score a candidate pose against the current map ────────────────────────

    def score_scan(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        angle_min: float,
        angle_inc: float,
        ranges: list[float],
        max_range: float,
        subsample: int = 3,
    ) -> float:
        """
        Return sum of log-odds values at scan-endpoint cells (higher = better match).
        Only considers non-max-range rays; subsamples for speed.
        """
        score = 0.0
        n     = 0
        for i in range(0, len(ranges), subsample):
            r = ranges[i]
            if r >= max_range * 0.99 or r <= 0.01:
                continue
            angle = robot_theta + angle_min + i * angle_inc
            hx, hy = self.world_to_cell(
                robot_x + r * math.cos(angle),
                robot_y + r * math.sin(angle))
            if self.in_bounds(hx, hy):
                score += float(self._log[hy, hx])
                n += 1
        return score / max(n, 1)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def is_occupied(self, gx: int, gy: int, threshold: float = 0.3) -> bool:
        if not self.in_bounds(gx, gy):
            return True   # treat out-of-bounds as occupied (wall)
        return bool(self._log[gy, gx] > threshold)

    def is_free(self, gx: int, gy: int, threshold: float = -0.1) -> bool:
        if not self.in_bounds(gx, gy):
            return False
        return bool(self._log[gy, gx] < threshold)

    # ── ROS message serialisation ─────────────────────────────────────────────

    def to_ros_msg(self, stamp, frame_id: str = "map") -> ROSOccGrid:
        """Convert to nav_msgs/OccupancyGrid (-1=unknown, 0=free, 100=occupied)."""
        msg = ROSOccGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = frame_id

        info = MapMetaData()
        info.resolution = self.resolution
        info.width      = self.cols
        info.height     = self.rows
        pose = Pose()
        pose.position.x    = self.origin_x
        pose.position.y    = self.origin_y
        pose.orientation.w = 1.0
        info.origin = pose
        msg.info = info

        # Convert log-odds → int8 occupancy
        occ = np.full((self.rows, self.cols), -1, dtype=np.int8)
        occ[self._log >  0.3] = 100    # occupied
        occ[self._log < -0.1] = 0      # free
        msg.data = occ.flatten().tolist()
        return msg

    # ── Copy ──────────────────────────────────────────────────────────────────

    def clone(self) -> "OccupancyGrid":
        g = OccupancyGrid(
            self.cols * self.resolution,
            self.rows * self.resolution,
            self.resolution,
            self.origin_x,
            self.origin_y,
        )
        g._log = self._log.copy()
        return g


# ── Bresenham integer ray-march ───────────────────────────────────────────────

def _bresenham(
    x0: int, y0: int, x1: int, y1: int
) -> Generator[tuple[int, int], None, None]:
    """Yield (gx, gy) cells along the straight line from (x0,y0) to (x1,y1)."""
    dx = abs(x1 - x0);  dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    max_steps = dx + dy + 2
    for _ in range(max_steps):
        yield x, y
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy
