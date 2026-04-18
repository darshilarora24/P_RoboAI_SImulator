"""
rl_panel.py  —  P_RoboAI Studio

Qt control panel for Reinforcement Learning.

Tabs
----
  Train    — configure algorithm / hyperparams, start/stop training
  Evaluate — deploy trained policy into live simulation, watch it run
  Chart    — live reward plot (custom QPainter, no matplotlib)
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional

from PyQt6.QtCore    import Qt, QTimer, pyqtSignal
from PyQt6.QtGui     import QColor, QPainter, QPen, QFont, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, QTabWidget,
    QFileDialog, QProgressBar, QFrame, QSizePolicy,
)

from rl_trainer import RLTrainer, PolicyPlayer, _SB3_AVAILABLE

try:
    from rl_env import MuJoCoArmEnv, MuJoCoAMREnv, GYM_AVAILABLE
except ImportError:
    GYM_AVAILABLE = False


# ── Reward chart widget ───────────────────────────────────────────────────────

class _RewardChart(QWidget):
    """Scrolling line chart of per-episode rewards, drawn with QPainter."""

    _BG   = QColor(18,  18,  18)
    _GRID = QColor(45,  45,  45)
    _LINE = QColor(0,   200, 100)
    _MEAN = QColor(255, 165, 0)
    _TEXT = QColor(150, 150, 150)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._rewards: deque[float] = deque(maxlen=200)
        self._mean20:  deque[float] = deque(maxlen=200)

    def push(self, reward: float) -> None:
        self._rewards.append(reward)
        window = list(self._rewards)[-20:]
        self._mean20.append(sum(window) / len(window))
        self.update()

    def clear(self) -> None:
        self._rewards.clear()
        self._mean20.clear()
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, self._BG)

        if len(self._rewards) < 2:
            p.setPen(self._TEXT)
            p.drawText(0, 0, W, H, Qt.AlignmentFlag.AlignCenter,
                       "No data yet — start training")
            return

        rews = list(self._rewards)
        mn, mx = min(rews), max(rews)
        span = mx - mn or 1.0
        pad  = 24

        # Grid lines
        p.setPen(QPen(self._GRID, 1))
        for i in range(5):
            y = pad + (H - 2 * pad) * i // 4
            p.drawLine(pad, y, W - 4, y)

        def to_px(v: float, i: int) -> tuple[int, int]:
            x = pad + int((W - pad - 4) * i / (len(rews) - 1))
            y = pad + int((H - 2 * pad) * (1 - (v - mn) / span))
            return x, y

        # Episode rewards
        p.setPen(QPen(self._LINE, 1))
        for i in range(1, len(rews)):
            p.drawLine(*to_px(rews[i - 1], i - 1), *to_px(rews[i], i))

        # 20-ep moving average
        mean = list(self._mean20)
        p.setPen(QPen(self._MEAN, 2))
        for i in range(1, len(mean)):
            p.drawLine(*to_px(mean[i - 1], i - 1), *to_px(mean[i], i))

        # Labels
        f = QFont("Monospace", 7)
        p.setFont(f)
        p.setPen(self._TEXT)
        p.drawText(2, pad,     50, 14, Qt.AlignmentFlag.AlignLeft, f"{mx:.2f}")
        p.drawText(2, H - pad, 50, 14, Qt.AlignmentFlag.AlignLeft, f"{mn:.2f}")
        p.setPen(self._MEAN)
        p.drawText(W - 60, 4, 56, 14,  Qt.AlignmentFlag.AlignRight,
                   f"avg {mean[-1]:.2f}" if mean else "")


# ── RL Panel ──────────────────────────────────────────────────────────────────

class RLPanel(QWidget):
    """
    Parameters
    ----------
    model_getter   : callable → (mjcf_xml, joint_infos, kind)
    live_data_getter : callable → mujoco.MjData  (the shared sim data)
    """

    request_pause_physics = pyqtSignal(bool)   # True = pause

    _GRP = ("QGroupBox{color:#aaa;font-size:11px;border:1px solid #444;"
            "border-radius:4px;margin-top:8px;padding:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")

    def __init__(self, model_getter, live_data_getter, parent=None) -> None:
        super().__init__(parent)
        self._get_model     = model_getter
        self._get_live_data = live_data_getter
        self._trainer:  Optional[RLTrainer]  = None
        self._player:   Optional[PolicyPlayer] = None
        self._eval_timer = QTimer(self)
        self._eval_timer.setInterval(50)   # 20 Hz policy inference
        self._eval_timer.timeout.connect(self._eval_tick)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("  Reinforcement Learning")
        title.setFixedHeight(32)
        title.setStyleSheet(
            "background:#1e1e1e;color:#ddd;font-size:13px;font-weight:bold;"
            "border-bottom:1px solid #444;")
        outer.addWidget(title)

        if not _SB3_AVAILABLE or not GYM_AVAILABLE:
            warn = QLabel(
                "<p style='color:#f90;padding:10px;'>"
                "⚠ Missing dependencies:<br>"
                "<tt>pip install stable-baselines3 gymnasium</tt></p>")
            warn.setWordWrap(True)
            outer.addWidget(warn)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane{border:0;background:#1e1e1e;}"
            "QTabBar::tab{background:#2a2a2a;color:#aaa;padding:5px 10px;"
            "border-radius:3px 3px 0 0;}"
            "QTabBar::tab:selected{background:#1e1e1e;color:#eee;}")
        tabs.addTab(self._build_train_tab(),  "Train")
        tabs.addTab(self._build_eval_tab(),   "Run Policy")
        tabs.addTab(self._build_chart_tab(),  "Reward")
        outer.addWidget(tabs, 1)

    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#1e1e1e;")
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Algorithm
        alg_grp = QGroupBox("Algorithm")
        alg_grp.setStyleSheet(self._GRP)
        ag = QHBoxLayout(alg_grp)
        self._alg_combo = QComboBox()
        self._alg_combo.addItems(["PPO", "SAC", "TD3"])
        self._alg_combo.setStyleSheet(
            "QComboBox{background:#2a2a2a;color:#ccc;border:1px solid #555;"
            "border-radius:3px;padding:3px;}")
        ag.addWidget(QLabel("Algorithm:"))
        ag.addWidget(self._alg_combo)
        v.addWidget(alg_grp)

        # Hyperparameters
        hp_grp = QGroupBox("Hyperparameters")
        hp_grp.setStyleSheet(self._GRP)
        hg = QVBoxLayout(hp_grp)

        def _spin_row(label, lo, hi, val, dec=5, step=0.0001):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#aaa;font-size:11px;")
            lbl.setFixedWidth(110)
            sb  = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setValue(val)
            sb.setDecimals(dec); sb.setSingleStep(step)
            sb.setStyleSheet(
                "QDoubleSpinBox{background:#2a2a2a;color:#eee;"
                "border:1px solid #555;border-radius:3px;}")
            row.addWidget(lbl); row.addWidget(sb)
            hg.addLayout(row)
            return sb

        self._lr_spin     = _spin_row("Learning rate:", 1e-6, 1e-2, 3e-4, 6, 1e-5)
        self._steps_spin  = QSpinBox()
        self._steps_spin.setRange(1_000, 10_000_000)
        self._steps_spin.setValue(100_000)
        self._steps_spin.setSingleStep(10_000)
        self._steps_spin.setStyleSheet(
            "QSpinBox{background:#2a2a2a;color:#eee;border:1px solid #555;"
            "border-radius:3px;}")
        row2 = QHBoxLayout()
        lbl2 = QLabel("Total steps:")
        lbl2.setStyleSheet("color:#aaa;font-size:11px;"); lbl2.setFixedWidth(110)
        row2.addWidget(lbl2); row2.addWidget(self._steps_spin)
        hg.addLayout(row2)
        v.addWidget(hp_grp)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            "QProgressBar{background:#2a2a2a;border:1px solid #555;"
            "border-radius:3px;text-align:center;color:#ccc;}"
            "QProgressBar::chunk{background:#1a6b3a;border-radius:2px;}")
        v.addWidget(self._progress)

        # Status
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            "color:#aaa;font-size:11px;background:#1a1a1a;padding:4px;"
            "border:1px solid #333;border-radius:3px;")
        self._status_lbl.setWordWrap(True)
        v.addWidget(self._status_lbl)

        # Buttons
        btn_row = QHBoxLayout()
        self._train_btn = self._mk_btn("▶  Train", "#1a6b3a")
        self._stop_btn  = self._mk_btn("⏹  Stop",  "#6b1a1a")
        self._save_btn  = self._mk_btn("💾  Save",  "#1a3a6b")
        self._load_btn  = self._mk_btn("📂  Load",  "#3a3a1a")
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train)
        self._stop_btn.clicked.connect(self._on_stop)
        self._save_btn.clicked.connect(self._on_save)
        self._load_btn.clicked.connect(self._on_load)
        for b in (self._train_btn, self._stop_btn,
                  self._save_btn, self._load_btn):
            btn_row.addWidget(b)
        v.addLayout(btn_row)
        v.addStretch()
        return w

    def _build_eval_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#1e1e1e;")
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        info = QLabel(
            "<p style='color:#888;font-size:11px;'>"
            "Runs the trained policy inside the live physics simulation.<br>"
            "Train a policy first, then press <b>Run Policy</b>.</p>")
        info.setWordWrap(True)
        v.addWidget(info)

        self._eval_status = QLabel("Policy: none")
        self._eval_status.setStyleSheet(
            "color:#4a9;font-family:monospace;font-size:11px;"
            "background:#1a1a1a;padding:4px;border:1px solid #333;"
            "border-radius:3px;")
        v.addWidget(self._eval_status)

        self._ep_label = QLabel("Episode: —   Dist: —")
        self._ep_label.setStyleSheet(
            "color:#00e5ff;font-family:monospace;font-size:11px;"
            "background:#1a1a1a;padding:4px;border:1px solid #333;border-radius:3px;")
        v.addWidget(self._ep_label)

        row = QHBoxLayout()
        self._run_btn  = self._mk_btn("▶  Run Policy", "#1a6b3a")
        self._kill_btn = self._mk_btn("⏹  Stop",       "#6b1a1a")
        self._run_btn.setEnabled(False)
        self._kill_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_policy)
        self._kill_btn.clicked.connect(self._on_stop_policy)
        row.addWidget(self._run_btn); row.addWidget(self._kill_btn)
        v.addLayout(row)
        v.addStretch()
        return w

    def _build_chart_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#1e1e1e;")
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        self._chart = _RewardChart()
        v.addWidget(self._chart, 1)

        legend = QLabel(
            "<span style='color:#00c864;'>▬</span> Episode reward  "
            "<span style='color:#ffa500;'>▬</span> 20-ep avg")
        legend.setStyleSheet("color:#888;font-size:10px;")
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(legend)

        clr = QPushButton("Clear")
        clr.setStyleSheet(
            "QPushButton{background:#333;color:#aaa;border-radius:3px;padding:3px;}"
            "QPushButton:hover{background:#444;}")
        clr.clicked.connect(self._chart.clear)
        v.addWidget(clr)
        return w

    @staticmethod
    def _mk_btn(label: str, color: str) -> QPushButton:
        b = QPushButton(label)
        b.setStyleSheet(
            f"QPushButton{{background:{color};color:#fff;border-radius:4px;"
            f"padding:5px;font-size:11px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{color}cc;}}"
            f"QPushButton:disabled{{background:#333;color:#666;}}")
        return b

    # ── Train slots ───────────────────────────────────────────────────────────

    def _on_train(self) -> None:
        result = self._get_model()
        if result is None:
            self._status_lbl.setText("Load a URDF first.")
            return
        mjcf_xml, joint_infos, kind = result

        try:
            from rl_env import MuJoCoArmEnv, MuJoCoAMREnv, RobotKind
            from urdf_loader import RobotKind as RK
            env = (MuJoCoArmEnv(mjcf_xml, joint_infos)
                   if kind == RK.ARM
                   else MuJoCoAMREnv(mjcf_xml, joint_infos))
        except Exception as exc:
            self._status_lbl.setText(f"Env error: {exc}")
            return

        self._trainer = RLTrainer(
            env,
            algorithm   = self._alg_combo.currentText(),
            total_steps = self._steps_spin.value(),
            lr          = self._lr_spin.value(),
        )
        self._trainer.episode_done.connect(self._on_episode)
        self._trainer.step_done.connect(self._on_step)
        self._trainer.training_finished.connect(self._on_finished)
        self._trainer.error.connect(self._on_error)
        self._trainer.start()

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("Training …")
        self._chart.clear()
        self._progress.setValue(0)

    def _on_stop(self) -> None:
        if self._trainer:
            self._trainer.stop()
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Policy", "",
                                              "ZIP files (*.zip)")
        if path and self._trainer:
            self._trainer.save_policy(Path(path))

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Policy", "",
                                              "ZIP files (*.zip)")
        if not path:
            return
        result = self._get_model()
        if result is None:
            return
        mjcf_xml, joint_infos, kind = result
        try:
            from rl_env import MuJoCoArmEnv, MuJoCoAMREnv
            from urdf_loader import RobotKind as RK
            env = (MuJoCoArmEnv(mjcf_xml, joint_infos)
                   if kind == RK.ARM
                   else MuJoCoAMREnv(mjcf_xml, joint_infos))
            if self._trainer is None:
                self._trainer = RLTrainer(
                    env, algorithm=self._alg_combo.currentText())
            self._trainer.load_policy(Path(path))
            self._save_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
            self._eval_status.setText(f"Policy: {Path(path).name}")
        except Exception as exc:
            self._status_lbl.setText(f"Load error: {exc}")

    # ── Training signal handlers ──────────────────────────────────────────────

    def _on_episode(self, ep: int, mean_r: float, dist: float) -> None:
        self._chart.push(mean_r)
        self._status_lbl.setText(
            f"Ep {ep}  |  mean reward {mean_r:+.3f}  |  dist {dist:.3f} m")
        pct = min(100, int(ep * 100 / max(1, self._steps_spin.value() // 200)))
        self._progress.setValue(pct)

    def _on_step(self, total: int) -> None:
        pct = min(100, int(total * 100 / max(1, self._steps_spin.value())))
        self._progress.setValue(pct)

    def _on_finished(self, msg: str) -> None:
        self._status_lbl.setText(msg)
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._progress.setValue(100)

    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText(msg)
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Eval slots ────────────────────────────────────────────────────────────

    def _on_run_policy(self) -> None:
        if not self._trainer:
            return
        result = self._get_model()
        if result is None:
            return
        mjcf_xml, joint_infos, kind = result
        try:
            from rl_env import MuJoCoArmEnv, MuJoCoAMREnv
            from urdf_loader import RobotKind as RK
            env = (MuJoCoArmEnv(mjcf_xml, joint_infos)
                   if kind == RK.ARM
                   else MuJoCoAMREnv(mjcf_xml, joint_infos))
        except Exception as exc:
            self._eval_status.setText(f"Error: {exc}")
            return

        self._player = PolicyPlayer(self._trainer, env)
        self._player.start()
        self._eval_timer.start()
        self._run_btn.setEnabled(False)
        self._kill_btn.setEnabled(True)
        self.request_pause_physics.emit(True)   # pause main physics

    def _on_stop_policy(self) -> None:
        if self._player:
            self._player.stop()
        self._eval_timer.stop()
        self._run_btn.setEnabled(True)
        self._kill_btn.setEnabled(False)
        self.request_pause_physics.emit(False)

    def _eval_tick(self) -> None:
        live_data = self._get_live_data()
        if self._player and live_data is not None:
            self._player.tick(live_data)
