"""
global_planner.py  —  P_RoboAI_Nav2

A* global path planner on the inflated costmap.

Returns a list of (world_x, world_y) waypoints from current pose to goal.
Supports 8-connected grid with diagonal movement costs.
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

from .costmap import Costmap


# Maximum cost a cell may have and still be traversable
_MAX_TRAVERSABLE = 89   # cells with cost ≥ 90 are blocked


def plan(
    costmap: Costmap,
    start_x: float, start_y: float,
    goal_x:  float, goal_y:  float,
) -> list[tuple[float, float]]:
    """
    Return list of (x, y) world waypoints from start to goal, or [] if
    no path exists or costmap is not ready.
    """
    if not costmap.is_ready():
        return []

    sx, sy = costmap.world_to_cell(start_x, start_y)
    gx, gy = costmap.world_to_cell(goal_x,  goal_y)

    # Clamp goal into map
    gx = max(0, min(costmap.cols - 1, gx))
    gy = max(0, min(costmap.rows - 1, gy))

    if costmap.is_lethal(gx, gy, 90):
        # Try to find nearest free goal cell
        gx, gy = _nearest_free(costmap, gx, gy)
        if gx is None:
            return []

    cells = _astar(costmap, (sx, sy), (gx, gy))
    if not cells:
        return []

    # Convert to world coords
    waypoints = [costmap.cell_to_world(c[0], c[1]) for c in cells]
    return _smooth_path(waypoints)


# ── A* ────────────────────────────────────────────────────────────────────────

_DIRS   = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
_DCOSTS = [math.sqrt(2),1.0,math.sqrt(2),1.0,1.0,math.sqrt(2),1.0,math.sqrt(2)]


def _astar(
    costmap: Costmap,
    start: tuple[int, int],
    goal:  tuple[int, int],
) -> list[tuple[int, int]]:
    def h(a: tuple[int, int]) -> float:
        return math.hypot(a[0] - goal[0], a[1] - goal[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start), 0.0, start))
    came_from: dict[tuple[int,int], Optional[tuple[int,int]]] = {start: None}
    g_score:   dict[tuple[int,int], float]                    = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            return _reconstruct(came_from, goal)
        if g > g_score.get(cur, 1e18) + 1e-9:
            continue
        for (dx, dy), dc in zip(_DIRS, _DCOSTS):
            nx, ny = cur[0] + dx, cur[1] + dy
            if not costmap.in_bounds(nx, ny):
                continue
            cell_cost = costmap.cost_at(nx, ny)
            if cell_cost >= _MAX_TRAVERSABLE:
                continue
            # Penalise high-cost cells proportionally
            ng = g + dc * (1.0 + cell_cost / 50.0)
            nb = (nx, ny)
            if ng < g_score.get(nb, 1e18):
                g_score[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
    return []


def _reconstruct(
    came_from: dict[tuple[int,int], Optional[tuple[int,int]]],
    goal: tuple[int,int],
) -> list[tuple[int,int]]:
    path: list[tuple[int,int]] = []
    node: Optional[tuple[int,int]] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def _nearest_free(
    costmap: Costmap, gx: int, gy: int, radius: int = 10
) -> tuple[Optional[int], Optional[int]]:
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                nx, ny = gx + dx, gy + dy
                if costmap.in_bounds(nx, ny) and not costmap.is_lethal(nx, ny):
                    return nx, ny
    return None, None


# ── Path smoothing (Ramer-Douglas-Peucker) ────────────────────────────────────

def _smooth_path(
    path: list[tuple[float, float]], epsilon: float = 0.12
) -> list[tuple[float, float]]:
    """Reduce waypoint count with RDP."""
    if len(path) <= 2:
        return path
    return _rdp(path, epsilon)


def _rdp(
    pts: list[tuple[float, float]], eps: float
) -> list[tuple[float, float]]:
    if len(pts) <= 2:
        return pts
    start, end = pts[0], pts[-1]
    max_dist = 0.0
    max_idx  = 0
    for i in range(1, len(pts) - 1):
        d = _perp_dist(pts[i], start, end)
        if d > max_dist:
            max_dist = d
            max_idx  = i
    if max_dist > eps:
        left  = _rdp(pts[:max_idx + 1], eps)
        right = _rdp(pts[max_idx:],     eps)
        return left[:-1] + right
    return [start, end]


def _perp_dist(
    p: tuple[float,float],
    a: tuple[float,float],
    b: tuple[float,float],
) -> float:
    dx, dy = b[0]-a[0], b[1]-a[1]
    if dx == 0 and dy == 0:
        return math.hypot(p[0]-a[0], p[1]-a[1])
    t = ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    nx = a[0] + t*dx - p[0]
    ny = a[1] + t*dy - p[1]
    return math.hypot(nx, ny)
