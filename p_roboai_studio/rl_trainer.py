"""
rl_trainer.py  —  P_RoboAI Studio

QThread wrapper around Stable Baselines3 that trains an RL policy on a
MuJoCo gym environment and emits progress signals back to the Qt UI.

Signals emitted on the main thread:
  episode_done(episode: int, mean_reward: float, dist: float)
  step_done(total_steps: int)
  training_finished(msg: str)
  error(msg: str)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

# SB3 is imported lazily so the module loads even if SB3 is not installed.
_SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    pass


# ── Callback that feeds progress back through Qt signals ──────────────────────

class _QtCallback:
    """
    Pseudo-callback (works without SB3 too) that the training loop calls
    after every episode to emit Qt signals and honour stop requests.
    """

    def __init__(self, trainer: "RLTrainer") -> None:
        self._trainer   = trainer
        self._ep_rew    : list[float] = []
        self._ep_dist   : list[float] = []
        self._ep_count  = 0
        self._total_steps = 0
        self._window    = 20   # moving-average window

    def on_step(self, reward: float, done: bool, info: dict) -> bool:
        self._total_steps += 1
        self._ep_rew.append(reward)
        if done:
            ep_total = float(np.sum(self._ep_rew))
            mean_r   = float(np.mean(self._ep_rew[-self._window:]))
            dist     = float(info.get("dist", 0.0))
            self._ep_count += 1
            self._ep_rew = []
            self._trainer.episode_done.emit(self._ep_count, mean_r, dist)
        self._trainer.step_done.emit(self._total_steps)
        return not self._trainer._stop_requested


class _SB3QtCallback(BaseCallback if _SB3_AVAILABLE else object):
    """SB3 BaseCallback that bridges into _QtCallback."""

    def __init__(self, qt_cb: _QtCallback, verbose: int = 0) -> None:
        if _SB3_AVAILABLE:
            super().__init__(verbose)
        self._qt = qt_cb

    def _on_step(self) -> bool:
        if not _SB3_AVAILABLE:
            return True
        rews  = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        r     = float(rews[0])  if hasattr(rews,  "__len__") else float(rews)
        d     = bool(dones[0])  if hasattr(dones, "__len__") else bool(dones)
        info  = infos[0]        if hasattr(infos, "__len__") else {}
        return self._qt.on_step(r, d, info)


# ── Main trainer thread ───────────────────────────────────────────────────────

class RLTrainer(QThread):
    """
    Parameters
    ----------
    env          : MuJoCoArmEnv | MuJoCoAMREnv
    algorithm    : 'PPO' | 'SAC' | 'TD3'
    total_steps  : int   — total environment steps to train
    lr           : float — learning rate
    save_dir     : Path  — where to write the policy after training
    """

    episode_done      = pyqtSignal(int, float, float)   # episode, mean_r, dist
    step_done         = pyqtSignal(int)                  # total steps
    training_finished = pyqtSignal(str)                  # message
    error             = pyqtSignal(str)

    def __init__(self, env, algorithm: str = "PPO",
                 total_steps: int = 100_000,
                 lr: float = 3e-4,
                 save_dir: Path | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._env          = env
        self._algorithm    = algorithm
        self._total_steps  = total_steps
        self._lr           = lr
        self._save_dir     = save_dir or Path.home() / ".p_roboai_rl"
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._stop_requested = False
        self._model: Any     = None     # SB3 model

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_requested = True

    def save_policy(self, path: Path | None = None) -> Path:
        p = path or (self._save_dir / f"{self._algorithm}_policy")
        if self._model:
            self._model.save(str(p))
        return p

    def load_policy(self, path: Path) -> None:
        if not _SB3_AVAILABLE:
            return
        cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}.get(self._algorithm, PPO)
        self._model = cls.load(str(path), env=self._env)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run a single inference step (called from main thread)."""
        if self._model is None:
            return self._env.action_space.sample()
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        if not _SB3_AVAILABLE:
            self.error.emit(
                "stable-baselines3 not installed.\n"
                "Run: pip install stable-baselines3")
            return

        try:
            self._stop_requested = False
            qt_cb = _QtCallback(self)
            sb3_cb = _SB3QtCallback(qt_cb)

            cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}.get(self._algorithm, PPO)

            policy_type = "MlpPolicy"

            if self._model is None:
                self._model = cls(
                    policy_type,
                    self._env,
                    learning_rate=self._lr,
                    verbose=0,
                )
            else:
                # Continue training — update env
                self._model.set_env(self._env)

            self._model.learn(
                total_timesteps=self._total_steps,
                callback=sb3_cb,
                reset_num_timesteps=False,
            )

            save_path = self.save_policy()
            self.training_finished.emit(
                f"Training complete — {self._total_steps:,} steps.\n"
                f"Policy saved to {save_path}")

        except Exception as exc:
            import traceback
            self.error.emit(f"Training error:\n{exc}\n\n{traceback.format_exc()}")


# ── Policy evaluator (runs in main thread via QTimer) ─────────────────────────

class PolicyPlayer:
    """
    Applies the trained policy to the LIVE simulation data at a fixed Hz.
    Call tick() from a QTimer to write actions into data.ctrl.
    """

    def __init__(self, trainer: RLTrainer, env) -> None:
        self._trainer = trainer
        self._env     = env
        self._obs, _  = env.reset()
        self._active  = False

    def start(self) -> None:
        self._obs, _ = self._env.reset()
        self._active = True

    def stop(self) -> None:
        self._active = False

    def tick(self, live_data) -> None:
        """Called by QTimer; live_data is the shared mujoco.MjData."""
        if not self._active:
            return
        action = self._trainer.predict(self._obs)
        obs, _, terminated, truncated, _ = self._env.step(action)
        self._obs = obs
        # Mirror ctrl into live simulation
        live_data.ctrl[:] = self._env._data.ctrl[:]
        if terminated or truncated:
            self._obs, _ = self._env.reset()
