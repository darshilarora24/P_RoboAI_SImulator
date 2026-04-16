"""
scan_matcher.py  —  P_RoboAI_SLAM

Correlative scan matching: searches a 3-D window (dx, dy, dθ) around the
predicted pose and returns the offset that maximises map-score.

Algorithm
---------
1.  Coarse search over (dx, dy, dθ) on the full grid.
2.  Fine search around the coarse winner at 1/4 step sizes.
3.  Returns (best_dx, best_dy, best_dθ, score).

Only uses the log-odds grid's score_scan() — no extra data structures.
"""
from __future__ import annotations

import math

from .occupancy_grid import OccupancyGrid


# ── Default search windows ────────────────────────────────────────────────────
# Coarse: ±0.30 m, ±0.15 rad  (radius × number of steps each side)
_COARSE_DXY_R   = 0.30   # m
_COARSE_DXY_S   = 0.05   # m  step
_COARSE_DTH_R   = 0.15   # rad
_COARSE_DTH_S   = 0.03   # rad step

# Fine: ±1 coarse step in xy, ±1 coarse step in θ, at 1/4 size
_FINE_RATIO     = 0.25


def match(
    grid: OccupancyGrid,
    pred_x: float,
    pred_y: float,
    pred_theta: float,
    angle_min: float,
    angle_inc: float,
    ranges: list[float],
    max_range: float,
    min_map_cells: int = 50,
) -> tuple[float, float, float, float]:
    """
    Returns (corrected_x, corrected_y, corrected_theta, score).

    If the map has fewer than min_map_cells occupied cells the predicted
    pose is returned unchanged (map not yet reliable).
    """
    # Guard: skip matching on an empty map
    occupied_count = int((grid._log > 0.3).sum())
    if occupied_count < min_map_cells:
        base = grid.score_scan(pred_x, pred_y, pred_theta,
                               angle_min, angle_inc, ranges, max_range)
        return pred_x, pred_y, pred_theta, base

    # ── Coarse search ─────────────────────────────────────────────────────────
    best_score = -1e9
    best_dx = best_dy = best_dth = 0.0

    dxy_steps = max(1, round(_COARSE_DXY_R / _COARSE_DXY_S))
    dth_steps = max(1, round(_COARSE_DTH_R / _COARSE_DTH_S))

    for ix in range(-dxy_steps, dxy_steps + 1):
        dx = ix * _COARSE_DXY_S
        for iy in range(-dxy_steps, dxy_steps + 1):
            dy = iy * _COARSE_DXY_S
            for it in range(-dth_steps, dth_steps + 1):
                dth = it * _COARSE_DTH_S
                sc = grid.score_scan(
                    pred_x + dx, pred_y + dy, pred_theta + dth,
                    angle_min, angle_inc, ranges, max_range,
                    subsample=4,    # faster coarse scan
                )
                if sc > best_score:
                    best_score = sc
                    best_dx, best_dy, best_dth = dx, dy, dth

    # ── Fine search around coarse winner ──────────────────────────────────────
    fine_dxy = _COARSE_DXY_S * _FINE_RATIO
    fine_dth = _COARSE_DTH_S * _FINE_RATIO
    fine_steps = 4   # ±4 fine steps = ±1 coarse step

    for ix in range(-fine_steps, fine_steps + 1):
        dx = best_dx + ix * fine_dxy
        for iy in range(-fine_steps, fine_steps + 1):
            dy = best_dy + iy * fine_dxy
            for it in range(-fine_steps, fine_steps + 1):
                dth = best_dth + it * fine_dth
                sc = grid.score_scan(
                    pred_x + dx, pred_y + dy, pred_theta + dth,
                    angle_min, angle_inc, ranges, max_range,
                    subsample=2,
                )
                if sc > best_score:
                    best_score = sc
                    best_dx, best_dy, best_dth = dx, dy, dth

    cx = pred_x + best_dx
    cy = pred_y + best_dy
    ct = (pred_theta + best_dth + math.pi) % (2 * math.pi) - math.pi

    return cx, cy, ct, best_score
