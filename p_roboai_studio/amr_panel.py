"""
amr_panel.py  —  P_RoboAI Studio

Control panel for an AMR (differential-drive robot) loaded from URDF.

Keyboard controls (focus must be on the main window or viewport):
  W / Up    — drive forward
  S / Down  — drive backward
  A / Left  — turn left
  D / Right — turn right
  Space     — emergency stop

The panel also shows live pose (x, y, θ) read from the freejoint qpos,
and publishes wheel velocity targets to data.ctrl[].
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from PyQt6.QtCore    import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QGroupBox, QPushButton, QFrame,
)
from PyQt6.QtGui import QFont, QKeyEvent

from urdf_loader import JointInfo


class AMRPanel(QWidget):
    """
    Parameters
    ----------
    joints       : list[JointInfo]  — only wheel joints used (is_wheel==True)
    data         : mujoco.MjData
    model        : mujoco.MjModel
    """

    def __init__(self, joints: list[JointInfo],
                 data:  mujoco.MjData,
                 model: mujoco.MjModel,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data        = data
        self._model       = model
        self._wheel_joints = [j for j in joints if j.is_wheel]
        # Fallback: use all continuous joints if none flagged as wheel
        if not self._wheel_joints:
            self._wheel_joints = [j for j in joints if j.joint_type == "continuous"]

        self._max_speed   = 2.0    # rad/s wheel speed at full throttle
        self._lin_speed   = 0.0    # current linear command  [-1 .. 1]
        self._ang_speed   = 0.0    # current angular command [-1 .. 1]
        self._keys_held: set[int] = set()

        # Key repeat timer
        self._key_timer = QTimer(self)
        self._key_timer.setInterval(50)   # 20 Hz command update
        self._key_timer.timeout.connect(self._apply_keys)
        self._key_timer.start()

        # Pose refresh timer
        self._pose_timer = QTimer(self)
        self._pose_timer.setInterval(100)
        self._pose_timer.timeout.connect(self._refresh_pose)
        self._pose_timer.start()

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("  AMR Drive Control")
        title.setFixedHeight(32)
        title.setStyleSheet(
            "background:#1e1e1e;color:#ddd;font-size:13px;font-weight:bold;"
            "border-bottom:1px solid #444;")
        outer.addWidget(title)

        inner = QWidget()
        inner.setStyleSheet("background:#262626;")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(10)

        # ── WASD graphic ──────────────────────────────────────────────────
        wasd = QLabel(
            "<div style='text-align:center;font-family:monospace;font-size:14px;"
            "color:#aaa;line-height:1.6;'>"
            "<b style='color:#4a9'>W</b> — Forward<br>"
            "<b style='color:#4a9'>S</b> — Backward<br>"
            "<b style='color:#4a9'>A</b> — Turn Left<br>"
            "<b style='color:#4a9'>D</b> — Turn Right<br>"
            "<b style='color:#f90'>Space</b> — Stop<br>"
            "</div>")
        wasd.setWordWrap(True)
        vbox.addWidget(wasd)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        vbox.addWidget(sep)

        # ── Speed slider ──────────────────────────────────────────────────
        spd_grp = self._make_group("Speed")
        spd_vbox = QVBoxLayout(spd_grp)

        spd_row = QHBoxLayout()
        spd_lbl = QLabel("Max speed:")
        spd_lbl.setStyleSheet("color:#999;font-size:11px;")
        self._spd_val = QLabel(f"{self._max_speed:.1f} rad/s")
        self._spd_val.setStyleSheet(
            "color:#00e5ff;font-family:monospace;font-size:12px;"
            "background:#1a1a1a;border:1px solid #333;padding:1px 4px;border-radius:3px;")
        spd_row.addWidget(spd_lbl)
        spd_row.addStretch()
        spd_row.addWidget(self._spd_val)

        self._spd_slider = QSlider(Qt.Orientation.Horizontal)
        self._spd_slider.setRange(1, 40)   # 0.1 .. 4.0 rad/s in 0.1 steps
        self._spd_slider.setValue(int(self._max_speed * 10))
        self._spd_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:5px;background:#444;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;margin:-4px 0;"
            "background:#f90;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#f90;border-radius:2px;}")
        self._spd_slider.valueChanged.connect(self._on_speed_changed)

        spd_vbox.addLayout(spd_row)
        spd_vbox.addWidget(self._spd_slider)
        vbox.addWidget(spd_grp)

        # ── Velocity display ──────────────────────────────────────────────
        vel_grp = self._make_group("Current Command")
        vel_vbox = QVBoxLayout(vel_grp)
        self._cmd_label = self._val_label("v=0.00 m/s   ω=0.00 rad/s")
        vel_vbox.addWidget(self._cmd_label)
        vbox.addWidget(vel_grp)

        # ── Pose display ──────────────────────────────────────────────────
        pose_grp = self._make_group("Robot Pose")
        pose_vbox = QVBoxLayout(pose_grp)
        self._pose_label = self._val_label("x=—  y=—  θ=—")
        pose_vbox.addWidget(self._pose_label)
        vbox.addWidget(pose_grp)

        # ── E-Stop ────────────────────────────────────────────────────────
        btn_stop = QPushButton("⏹  STOP  (Space)")
        btn_stop.setStyleSheet(
            "QPushButton{background:#8b0000;color:#fff;font-size:14px;font-weight:bold;"
            "border-radius:6px;padding:8px;border:2px solid #ff4444;}"
            "QPushButton:hover{background:#c00000;}"
            "QPushButton:pressed{background:#600000;}")
        btn_stop.clicked.connect(self.stop)
        vbox.addWidget(btn_stop)

        vbox.addStretch()
        outer.addWidget(inner, 1)

    @staticmethod
    def _make_group(title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            "QGroupBox{color:#aaa;font-size:11px;border:1px solid #444;"
            "border-radius:4px;margin-top:8px;padding:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
        return g

    @staticmethod
    def _val_label(text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(
            "color:#00e5ff;font-family:monospace;font-size:12px;"
            "background:#1a1a1a;border:1px solid #333;padding:4px;"
            "border-radius:3px;")
        l.setWordWrap(True)
        return l

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_speed_changed(self, v: int) -> None:
        self._max_speed = v / 10.0
        self._spd_val.setText(f"{self._max_speed:.1f} rad/s")

    def stop(self) -> None:
        self._lin_speed = 0.0
        self._ang_speed = 0.0
        self._set_wheels(0.0, 0.0)

    # ── Keyboard handling ─────────────────────────────────────────────────────

    def handle_key_press(self, ev: QKeyEvent) -> bool:
        """Returns True if key consumed."""
        k = ev.key()
        if k in (Qt.Key.Key_W, Qt.Key.Key_Up,
                 Qt.Key.Key_S, Qt.Key.Key_Down,
                 Qt.Key.Key_A, Qt.Key.Key_Left,
                 Qt.Key.Key_D, Qt.Key.Key_Right,
                 Qt.Key.Key_Space):
            self._keys_held.add(k)
            if k == Qt.Key.Key_Space:
                self.stop()
            return True
        return False

    def handle_key_release(self, ev: QKeyEvent) -> bool:
        k = ev.key()
        if k in self._keys_held:
            self._keys_held.discard(k)
            # If no drive key held, stop gradually
            if not self._drive_keys_held():
                self.stop()
            return True
        return False

    def _drive_keys_held(self) -> bool:
        return bool(self._keys_held - {Qt.Key.Key_Space})

    def _apply_keys(self) -> None:
        if not self._drive_keys_held():
            return
        fwd = (Qt.Key.Key_W in self._keys_held or Qt.Key.Key_Up    in self._keys_held)
        bwd = (Qt.Key.Key_S in self._keys_held or Qt.Key.Key_Down  in self._keys_held)
        lft = (Qt.Key.Key_A in self._keys_held or Qt.Key.Key_Left  in self._keys_held)
        rgt = (Qt.Key.Key_D in self._keys_held or Qt.Key.Key_Right in self._keys_held)

        lin = (1.0 if fwd else 0.0) - (1.0 if bwd else 0.0)
        ang = (1.0 if lft else 0.0) - (1.0 if rgt else 0.0)

        # Differential drive: v_l = lin - ang,  v_r = lin + ang
        v_l = (lin - ang) * self._max_speed
        v_r = (lin + ang) * self._max_speed
        self._set_wheels(v_l, v_r)

        # Update command label
        wheel_r = 0.10   # assume 10 cm wheel radius; adjust from model if known
        v_lin = (v_l + v_r) / 2.0 * wheel_r
        v_ang = (v_r - v_l) / (2.0 * wheel_r)   # approx track = 2*wheel_r
        self._cmd_label.setText(
            f"v={v_lin:+.2f} m/s   ω={v_ang:+.2f} rad/s")

    def _set_wheels(self, v_left: float, v_right: float) -> None:
        """Write wheel velocities to data.ctrl."""
        if len(self._wheel_joints) >= 2:
            lji = self._wheel_joints[0]
            rji = self._wheel_joints[1]
            if lji.ctrl_index < len(self._data.ctrl):
                self._data.ctrl[lji.ctrl_index] = v_left
            if rji.ctrl_index < len(self._data.ctrl):
                self._data.ctrl[rji.ctrl_index] = v_right
        elif len(self._wheel_joints) == 1:
            ji = self._wheel_joints[0]
            if ji.ctrl_index < len(self._data.ctrl):
                self._data.ctrl[ji.ctrl_index] = (v_left + v_right) / 2.0

    # ── Pose readout ──────────────────────────────────────────────────────────

    def _refresh_pose(self) -> None:
        # Try to read freejoint qpos
        try:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "root")
            if jid >= 0:
                qadr = self._model.jnt_qposadr[jid]
                qp = self._data.qpos
                x, y = float(qp[qadr]), float(qp[qadr + 1])
                qw = float(qp[qadr + 3])
                qz = float(qp[qadr + 6])
                theta = 2.0 * math.atan2(qz, qw) * 180.0 / math.pi
                self._pose_label.setText(
                    f"x={x:+.3f} m\ny={y:+.3f} m\nθ={theta:+.1f}°")
                return
        except Exception:
            pass
        self._pose_label.setText("Pose unavailable\n(no freejoint named 'root')")
