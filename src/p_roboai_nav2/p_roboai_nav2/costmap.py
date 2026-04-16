"""
costmap.py  —  P_RoboAI_Nav2

Inflated costmap derived from a SLAM occupancy grid.

Layers
------
  Static layer : copy of SLAM map (100 = lethal, 0 = free, -1 = unknown)
  Inflation    : Euclidean-distance inflate around every lethal cell.
                 Cost = 100 at lethal, linearly decays to 0 at inflation_radius.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from nav_msgs.msg import OccupancyGrid as ROSOccGrid, MapMetaData
from geometry_msgs.msg import Pose


class Costmap:
    """
    Parameters
    ----------
    inflation_radius : float  metres around obstacles that receive a cost > 0
    """

    LETHAL    = 100
    FREE      = 0
    UNKNOWN   = -1

    def __init__(self, inflation_radius: float = 0.35) -> None:
        self.inflation_radius = inflation_radius
        self._raw:  Optional[np.ndarray] = None   # int8  (rows, cols)
        self._cost: Optional[np.ndarray] = None   # int16 (rows, cols)
        self.resolution = 0.05
        self.origin_x   = 0.0
        self.origin_y   = 0.0
        self.rows = 0
        self.cols = 0

    # ── Build from SLAM map ───────────────────────────────────────────────────

    def update_from_slam(self, msg: ROSOccGrid) -> None:
        """Rebuild static layer + inflation from an incoming OccupancyGrid msg."""
        self.resolution = msg.info.resolution
        self.origin_x   = msg.info.origin.position.x
        self.origin_y   = msg.info.origin.position.y
        self.rows       = msg.info.height
        self.cols       = msg.info.width

        raw = np.array(msg.data, dtype=np.int8).reshape(self.rows, self.cols)
        self._raw  = raw
        self._cost = self._inflate(raw)

    def _inflate(self, raw: np.ndarray) -> np.ndarray:
        """Return inflated cost grid (int16)."""
        inflate_cells = int(math.ceil(self.inflation_radius / self.resolution))
        cost = np.where(raw == 100, 100, 0).astype(np.int16)

        # Fast distance-based inflation using a square kernel
        lethal_mask = (raw == 100)
        if not lethal_mask.any():
            return cost

        # Build distance transform approximation: iterate outward
        # Use a simple BFS-style approach with NumPy erosion
        from scipy.ndimage import distance_transform_edt  # type: ignore
        dist_m = distance_transform_edt(~lethal_mask) * self.resolution
        # Cells within inflation_radius get a cost proportional to distance
        mask = (dist_m < self.inflation_radius) & (dist_m > 0)
        cost[mask] = np.clip(
            (100 * (1.0 - dist_m[mask] / self.inflation_radius)).astype(np.int16),
            1, 99)
        cost[lethal_mask] = 100
        return cost

    # ── Query helpers ─────────────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def cell_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return (self.origin_x + (gx + 0.5) * self.resolution,
                self.origin_y + (gy + 0.5) * self.resolution)

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cols and 0 <= gy < self.rows

    def cost_at(self, gx: int, gy: int) -> int:
        if self._cost is None or not self.in_bounds(gx, gy):
            return self.LETHAL
        return int(self._cost[gy, gx])

    def is_lethal(self, gx: int, gy: int, threshold: int = 90) -> bool:
        return self.cost_at(gx, gy) >= threshold

    def is_ready(self) -> bool:
        return self._cost is not None

    # ── ROS msg ───────────────────────────────────────────────────────────────

    def to_ros_msg(self, stamp, frame_id: str = "map") -> ROSOccGrid:
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
        if self._cost is not None:
            msg.data = self._cost.flatten().astype(np.int8).tolist()
        return msg
