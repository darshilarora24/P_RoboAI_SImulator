"""
sim2real_bridge.py  —  Sim-to-Real policy transfer utilities.

Exports a trained SB3 policy to ONNX, applies a calibration layer to
compensate for domain gap (actuator delays, friction differences, sensor
noise), and estimates the current domain gap magnitude.

Workflow
--------
  1. Train policy in MuJoCo sim with RLTrainer / amr_rl_node
  2. Call Sim2RealAdapter.export_onnx(policy_zip, out_path)
  3. Load ONNX on real robot: adapter.load_onnx(path)
  4. Each control cycle: real_action = adapter.predict(real_obs)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False


# ── calibration parameters ────────────────────────────────────────────────────

@dataclass
class CalibrationParams:
    """Per-joint calibration offsets and scale factors."""
    action_scale:  list[float] = field(default_factory=list)   # multiply sim action
    action_offset: list[float] = field(default_factory=list)   # add after scale
    obs_scale:     list[float] = field(default_factory=list)
    obs_offset:    list[float] = field(default_factory=list)
    delay_steps:   int = 1                                       # action delay compensation

    def to_dict(self) -> dict:
        return {
            "action_scale":  self.action_scale,
            "action_offset": self.action_offset,
            "obs_scale":     self.obs_scale,
            "obs_offset":    self.obs_offset,
            "delay_steps":   self.delay_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationParams":
        return cls(**d)

    @classmethod
    def identity(cls, n_action: int, n_obs: int) -> "CalibrationParams":
        return cls(
            action_scale=[1.0] * n_action,
            action_offset=[0.0] * n_action,
            obs_scale=[1.0] * n_obs,
            obs_offset=[0.0] * n_obs,
            delay_steps=1,
        )


# ── domain gap estimator ──────────────────────────────────────────────────────

class DomainGapEstimator:
    """
    Maintains a rolling estimate of the sim→real domain gap by comparing
    predicted next observations from the sim model against actual real
    observations.  Reports gap as RMS residual.
    """

    def __init__(self, window: int = 50) -> None:
        self._residuals: list[float] = []
        self._window = window

    def update(self, sim_pred_obs: "np.ndarray", real_obs: "np.ndarray") -> float:
        if not _NP_OK:
            return 0.0
        rms = float(np.sqrt(np.mean((sim_pred_obs - real_obs) ** 2)))
        self._residuals.append(rms)
        if len(self._residuals) > self._window:
            self._residuals.pop(0)
        return rms

    @property
    def gap(self) -> float:
        if not self._residuals:
            return 0.0
        return sum(self._residuals) / len(self._residuals)

    @property
    def trend(self) -> str:
        if len(self._residuals) < 10:
            return "unknown"
        early = sum(self._residuals[:5]) / 5
        late  = sum(self._residuals[-5:]) / 5
        if late < early * 0.9:
            return "improving"
        if late > early * 1.1:
            return "worsening"
        return "stable"


# ── main adapter ──────────────────────────────────────────────────────────────

class Sim2RealAdapter:
    """
    Bridges a simulated RL policy to real robot execution.

    Supports:
      - ONNX export from Stable Baselines 3 ZIP
      - Calibration layer (per-joint scale/offset + delay buffer)
      - Domain gap tracking
      - Domain randomization parameter suggestions
    """

    def __init__(self, calib: Optional[CalibrationParams] = None) -> None:
        self._session: Optional["ort.InferenceSession"] = None
        self._calib   = calib
        self._gap_est = DomainGapEstimator()
        self._action_buf: list["np.ndarray"] = []
        self._n_action = 0
        self._n_obs    = 0

    # ── ONNX export ───────────────────────────────────────────────────────────

    @staticmethod
    def export_onnx(policy_zip_path: str, out_onnx_path: str) -> bool:
        """Export an SB3 policy ZIP to ONNX. Returns True on success."""
        try:
            from stable_baselines3 import PPO, SAC, TD3
            import torch

            # Try each algorithm
            model = None
            for Cls in (PPO, SAC, TD3):
                try:
                    model = Cls.load(policy_zip_path)
                    break
                except Exception:
                    continue
            if model is None:
                return False

            policy = model.policy
            policy.eval()
            obs_dim = model.observation_space.shape[0]
            dummy   = torch.zeros(1, obs_dim)

            torch.onnx.export(
                policy,
                dummy,
                out_onnx_path,
                input_names=["obs"],
                output_names=["action"],
                opset_version=17,
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            )
            return True
        except Exception as e:
            print(f"[Sim2Real] ONNX export failed: {e}")
            return False

    def load_onnx(self, onnx_path: str) -> bool:
        if not _ORT_OK:
            return False
        try:
            self._session = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            meta = self._session.get_inputs()
            self._n_obs = meta[0].shape[-1] if meta else 0
            return True
        except Exception as e:
            print(f"[Sim2Real] ONNX load failed: {e}")
            return False

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, real_obs: "np.ndarray") -> Optional["np.ndarray"]:
        """Apply calibration → ONNX inference → delay buffer → return action."""
        if not _NP_OK or self._session is None:
            return None

        obs = real_obs.astype(np.float32)
        if self._calib and self._calib.obs_scale:
            sc = np.array(self._calib.obs_scale, np.float32)
            of = np.array(self._calib.obs_offset, np.float32)
            if len(sc) == len(obs):
                obs = obs * sc + of

        inp_name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {inp_name: obs[None]})[0][0]
        action = raw.astype(np.float32)

        if self._calib and self._calib.action_scale:
            sc = np.array(self._calib.action_scale, np.float32)
            of = np.array(self._calib.action_offset, np.float32)
            if len(sc) == len(action):
                action = action * sc + of

        # Delay buffer
        delay = self._calib.delay_steps if self._calib else 1
        self._action_buf.append(action)
        if len(self._action_buf) > max(1, delay):
            self._action_buf.pop(0)
        return self._action_buf[0]

    # ── calibration helpers ───────────────────────────────────────────────────

    def calibrate_from_rollout(self,
                               sim_obs_traj: list["np.ndarray"],
                               real_obs_traj: list["np.ndarray"]) -> CalibrationParams:
        """
        Compute per-dimension scale/offset to minimize ||sim_obs - real_obs||
        using least-squares per dimension.
        """
        if not _NP_OK or not sim_obs_traj or not real_obs_traj:
            return CalibrationParams()
        n = min(len(sim_obs_traj), len(real_obs_traj))
        S = np.array(sim_obs_traj[:n], np.float32)
        R = np.array(real_obs_traj[:n], np.float32)
        scales  = []
        offsets = []
        for d in range(S.shape[1]):
            s_col = S[:, d]
            r_col = R[:, d]
            s_std = float(np.std(s_col)) or 1.0
            r_std = float(np.std(r_col)) or 1.0
            scale  = r_std / s_std
            offset = float(np.mean(r_col)) - scale * float(np.mean(s_col))
            scales.append(scale)
            offsets.append(offset)
        self._n_obs = S.shape[1]
        calib = CalibrationParams(
            obs_scale=scales,
            obs_offset=offsets,
            action_scale=[1.0] * self._n_action,
            action_offset=[0.0] * self._n_action,
        )
        self._calib = calib
        return calib

    def update_gap(self, sim_pred: "np.ndarray", real_obs: "np.ndarray") -> float:
        return self._gap_est.update(sim_pred, real_obs)

    @property
    def domain_gap(self) -> float:
        return self._gap_est.gap

    @property
    def domain_gap_trend(self) -> str:
        return self._gap_est.trend

    def save_calibration(self, path: str) -> None:
        if self._calib:
            Path(path).write_text(json.dumps(self._calib.to_dict()))

    def load_calibration(self, path: str) -> bool:
        try:
            d = json.loads(Path(path).read_text())
            self._calib = CalibrationParams.from_dict(d)
            return True
        except Exception:
            return False

    def suggest_domain_randomization(self) -> dict:
        """Return suggested DR ranges based on current gap estimate."""
        gap = self.domain_gap
        base = 0.05 + gap * 0.5
        return {
            "friction_range":   [max(0.1, 1.0 - base), 1.0 + base],
            "damping_range":    [max(0.01, 1.0 - base * 0.5), 1.0 + base * 0.5],
            "mass_range":       [max(0.5, 1.0 - base * 0.3), 1.0 + base * 0.3],
            "delay_steps_max":  max(1, int(gap * 10)),
            "noise_std":        gap * 0.1,
            "gap_magnitude":    gap,
            "gap_trend":        self.domain_gap_trend,
        }
