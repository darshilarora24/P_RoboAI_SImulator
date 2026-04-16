"""
local_planner.py  —  P_RoboAI_Nav2

Dynamic Window Approach (DWA) local planner.

At each timestep:
  1. Build feasible velocity window [v_min..v_max] × [w_min..w_max].
  2. Sample N×M (v, ω) pairs.
  3. Simulate each for horizon seconds.
  4. Score: heading_gain + clearance_gain + velocity_gain.
  5. Return best (v, ω).

If no trajectory is admissible, emit a rotate-in-place command.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .costmap import Costmap


@dataclass
class DWAConfig:
    # Robot kinematics
    max_v:     float = 0.8    # m/s
    min_v:     float = 0.0    # m/s  (no reverse in forward navigation)
    max_w:     float = 2.2    # rad/s
    max_acc_v: float = 0.8    # m/s²
    max_acc_w: float = 2.5    # rad/s²

    # Sampling
    v_samples: int   = 12
    w_samples: int   = 24

    # Simulation
    sim_time:  float = 1.8    # seconds
    sim_steps: int   = 15     # trajectory steps

    # Scoring weights
    w_heading:   float = 1.8
    w_clearance: float = 1.5
    w_velocity:  float = 0.5

    # Safety
    min_clearance: float = 0.22   # metres — abort if less than this


def dwa_step(
    costmap: Costmap,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    robot_v: float,
    robot_w: float,
    lookahead_x: float,
    lookahead_y: float,
    cfg: DWAConfig = DWAConfig(),
    dt: float = 0.05,          # control period
) -> tuple[float, float]:
    """
    Return (v, ω) best command toward (lookahead_x, lookahead_y).
    Falls back to (0, w_turn) if no safe trajectory exists.
    """
    # Dynamic window
    v_lo = max(cfg.min_v, robot_v - cfg.max_acc_v * dt)
    v_hi = min(cfg.max_v, robot_v + cfg.max_acc_v * dt)
    w_lo = max(-cfg.max_w, robot_w - cfg.max_acc_w * dt)
    w_hi = min( cfg.max_w, robot_w + cfg.max_acc_w * dt)

    best_score = -1e9
    best_v = 0.0
    best_w = 0.0
    found  = False

    sim_dt = cfg.sim_time / cfg.sim_steps

    for vi in range(cfg.v_samples):
        v = v_lo + (v_hi - v_lo) * vi / max(cfg.v_samples - 1, 1)
        for wi in range(cfg.w_samples):
            w = w_lo + (w_hi - w_lo) * wi / max(cfg.w_samples - 1, 1)

            # Simulate trajectory
            sx, sy, sth = robot_x, robot_y, robot_theta
            min_dist = 1e9
            ok = True

            for _ in range(cfg.sim_steps):
                sth += w * sim_dt
                sx  += v * math.cos(sth) * sim_dt
                sy  += v * math.sin(sth) * sim_dt

                gx, gy = costmap.world_to_cell(sx, sy)
                cell_cost = costmap.cost_at(gx, gy)
                if cell_cost >= 90:
                    ok = False
                    break
                # Approximate clearance from cost (cost ∝ 1/dist)
                dist_approx = (1.0 - cell_cost / 100.0) * cfg.min_clearance * 3
                min_dist = min(min_dist, dist_approx)

            if not ok:
                continue
            if min_dist < cfg.min_clearance:
                continue

            # ── Score ──────────────────────────────────────────────────────
            # Heading toward lookahead point
            target_angle   = math.atan2(lookahead_y - sx, lookahead_x - sx)
            heading_err    = abs(_angle_diff(target_angle, sth))
            heading_score  = (math.pi - heading_err) / math.pi

            # Clearance (higher = better)
            clearance_score = min_dist / (cfg.min_clearance * 3)

            # Forward velocity reward
            vel_score = v / cfg.max_v

            score = (cfg.w_heading    * heading_score
                   + cfg.w_clearance  * clearance_score
                   + cfg.w_velocity   * vel_score)

            if score > best_score:
                best_score = score
                best_v, best_w = v, w
                found = True

    if not found:
        # Recovery: rotate in place toward goal
        goal_angle = math.atan2(lookahead_y - robot_y, lookahead_x - robot_x)
        err = _angle_diff(goal_angle, robot_theta)
        best_v = 0.0
        best_w = math.copysign(min(cfg.max_w * 0.5, abs(err) * 2.0), err)

    return best_v, best_w


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d
