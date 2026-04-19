"""
real2sim_bridge.py  —  Real-to-Sim online system identification.

Subscribes to real robot sensor streams and continuously updates MuJoCo
model parameters (damping, friction, mass, actuator gear) to minimize the
gap between simulated and real robot dynamics.

Uses recursive least squares (RLS) for online parameter estimation.

Workflow
--------
  1. real_robot_topics → Real2SimAdapter.update(real_obs, real_action)
  2. Adapter identifies: damping, friction per joint via RLS
  3. Adapter.apply(mj_model) writes estimated params into MuJoCo model
  4. Sim now better reflects real robot dynamics
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False


# ── identified parameter set ──────────────────────────────────────────────────

@dataclass
class IdentifiedParams:
    """Estimated real-robot dynamic parameters."""
    joint_damping:   list[float] = field(default_factory=list)
    joint_friction:  list[float] = field(default_factory=list)
    joint_armature:  list[float] = field(default_factory=list)
    geom_friction:   list[float] = field(default_factory=list)   # [slide, spin, roll]
    actuator_gain:   list[float] = field(default_factory=list)
    timestamp:       float       = field(default_factory=time.time)
    confidence:      float       = 0.0   # 0–1, increases with more data

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "IdentifiedParams":
        d.pop("timestamp", None)
        d.pop("confidence", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── recursive least squares (per DOF) ─────────────────────────────────────────

class _RLSEstimator:
    """
    Single-DOF recursive least-squares estimator.

    Estimates [damping, coulomb_friction, actuator_gain] from:
      tau = gain * u - damping * dq - friction * sign(dq)
    """

    def __init__(self, forgetting: float = 0.98) -> None:
        self._lam = forgetting
        self._theta = None   # [gain, damping, friction]
        self._P     = None

    def _init(self, n: int = 3) -> None:
        if not _NP_OK:
            return
        self._theta = np.zeros(n, np.float64)
        self._P     = np.eye(n, dtype=np.float64) * 1e4

    def update(self, u: float, dq: float, tau: float) -> Optional["np.ndarray"]:
        """Update with one sample. Returns current param estimate."""
        if not _NP_OK:
            return None
        if self._theta is None:
            self._init()
        phi   = np.array([u, -dq, -float(np.sign(dq))], np.float64)
        y_hat = float(self._theta @ phi)
        err   = tau - y_hat
        Pphi  = self._P @ phi
        denom = self._lam + float(phi @ Pphi)
        K     = Pphi / denom
        self._theta = self._theta + K * err
        self._P     = (self._P - np.outer(K, phi @ self._P)) / self._lam
        return self._theta.copy()


# ── main adapter ──────────────────────────────────────────────────────────────

class Real2SimAdapter:
    """
    Online system identification from real robot data.

    Each call to `update()` advances the RLS estimators with one timestep of
    real-robot data.  Call `apply(mj_model)` to write estimated parameters
    into the MuJoCo model in place.
    """

    def __init__(self,
                 n_joints: int = 6,
                 forgetting: float = 0.98,
                 update_interval: int = 20) -> None:
        self._n     = n_joints
        self._rls   = [_RLSEstimator(forgetting) for _ in range(n_joints)]
        self._count = 0
        self._interval = update_interval
        self._params = IdentifiedParams(
            joint_damping  = [0.5] * n_joints,
            joint_friction = [0.1] * n_joints,
            joint_armature = [0.01] * n_joints,
            geom_friction  = [1.0, 0.005, 0.0001],
            actuator_gain  = [1.0] * n_joints,
        )
        # History for confidence estimation
        self._theta_hist: list[list[float]] = []

    def update(self,
               actions: list[float],
               joint_vel: list[float],
               joint_torque: list[float]) -> IdentifiedParams:
        """
        Feed one timestep of real-robot data.

        Parameters
        ----------
        actions      : commanded actuator inputs (same length as n_joints)
        joint_vel    : measured joint velocities
        joint_torque : measured / estimated joint torques
        """
        if not _NP_OK:
            return self._params

        n = min(self._n, len(actions), len(joint_vel), len(joint_torque))
        for i in range(n):
            theta = self._rls[i].update(
                u=actions[i], dq=joint_vel[i], tau=joint_torque[i])
            if theta is not None and theta[0] != 0:
                self._params.actuator_gain[i]  = max(0.01, float(theta[0]))
                self._params.joint_damping[i]  = max(0.0,  float(theta[1]))
                self._params.joint_friction[i] = max(0.0,  float(theta[2]))

        self._count += 1
        self._theta_hist.append(list(self._params.joint_damping))
        if len(self._theta_hist) > 200:
            self._theta_hist.pop(0)

        # Confidence: how stable is the estimate over the last 50 steps
        if len(self._theta_hist) >= 50:
            recent = np.array(self._theta_hist[-50:], np.float64)
            cv     = float(np.mean(np.std(recent, axis=0) /
                                   (np.mean(np.abs(recent), axis=0) + 1e-6)))
            self._params.confidence = max(0.0, min(1.0, 1.0 - cv))

        self._params.timestamp = time.time()
        return self._params

    def apply(self, mj_model: Any) -> None:
        """Write identified parameters into a live MuJoCo MjModel."""
        if not _NP_OK:
            return
        n = min(self._n, mj_model.nv)
        try:
            mj_model.dof_damping[:n]  = np.array(
                self._params.joint_damping[:n], np.float64)
            mj_model.dof_frictionloss[:n] = np.array(
                self._params.joint_friction[:n], np.float64)
            mj_model.dof_armature[:n] = np.array(
                self._params.joint_armature[:n], np.float64)
            # Update floor friction (geom 0 is typically the floor)
            if mj_model.ngeom > 0 and len(self._params.geom_friction) >= 3:
                mj_model.geom_friction[0] = np.array(
                    self._params.geom_friction[:3], np.float64)
            # Actuator gains (kp in position actuators)
            na = min(len(self._params.actuator_gain), mj_model.nu)
            for i in range(na):
                # gainprm[0] = position gain
                mj_model.actuator_gainprm[i, 0] = self._params.actuator_gain[i]
        except Exception as exc:
            pass  # model layout mismatch — skip silently

    def update_from_odom(self,
                         cmd_vel_linear: float,
                         cmd_vel_angular: float,
                         odom_linear: float,
                         odom_angular: float) -> None:
        """
        AMR-specific: identify wheel slip and effective radius from
        commanded vs measured velocities.
        """
        if not _NP_OK:
            return
        err_lin = abs(odom_linear - cmd_vel_linear)
        err_ang = abs(odom_angular - cmd_vel_angular)
        # Simple EMA update of floor friction from velocity tracking error
        slip = (err_lin + 0.5 * err_ang) / (abs(cmd_vel_linear) + 0.1)
        cur  = self._params.geom_friction[0]
        self._params.geom_friction[0] = 0.95 * cur + 0.05 * max(0.3, 2.0 - slip * 5)

    def get_status(self) -> dict:
        return {
            "samples":       self._count,
            "confidence":    round(self._params.confidence, 3),
            "joint_damping": [round(v, 4) for v in self._params.joint_damping],
            "joint_friction":[round(v, 4) for v in self._params.joint_friction],
            "actuator_gain": [round(v, 4) for v in self._params.actuator_gain],
            "geom_friction": [round(v, 4) for v in self._params.geom_friction],
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self._params.to_dict()))

    def load(self, path: str) -> bool:
        try:
            d = json.loads(Path(path).read_text())
            self._params = IdentifiedParams.from_dict(d)
            # Re-initialize RLS with loaded values
            for i, rls in enumerate(self._rls):
                rls._init()
                if i < len(self._params.actuator_gain):
                    rls._theta = np.array([
                        self._params.actuator_gain[i],
                        self._params.joint_damping[i],
                        self._params.joint_friction[i],
                    ], np.float64)
            return True
        except Exception:
            return False
