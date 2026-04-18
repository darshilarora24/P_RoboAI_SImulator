"""
arm_panel.py  —  P_RoboAI Studio

Auto-generated control panel for a robot arm loaded from URDF.
Creates one slider + value readout per controllable joint, derived
from JointInfo extracted by urdf_loader.

Signals
-------
  joint_changed(name: str, value: float) — emitted when any slider moves
"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QSlider, QGroupBox, QPushButton,
    QDoubleSpinBox, QSizePolicy,
)
from PyQt6.QtGui import QFont

from urdf_loader import JointInfo


class _JointRow(QWidget):
    """Single row: label | slider | value spinbox."""

    changed = pyqtSignal(float)   # emits new radian/metre value

    def __init__(self, info: JointInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info  = info
        self._scale = 1000   # slider integer = value * scale

        lo, hi = info.lower, info.upper
        if math.isinf(lo) or lo < -100:  lo = -3.14159
        if math.isinf(hi) or hi >  100:  hi =  3.14159

        self._lo = lo
        self._hi = hi

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Joint name label
        lbl = QLabel(info.name.replace("_", " "))
        lbl.setFixedWidth(160)
        lbl.setStyleSheet("color:#bbb; font-size:11px;")
        lbl.setWordWrap(True)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(lo * self._scale), int(hi * self._scale))
        self._slider.setValue(0)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal{height:5px;background:#444;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;margin:-4px 0;"
            "background:#4a9;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#4a9;border-radius:2px;}")

        # Numeric readout / input
        unit = "m" if info.joint_type == "prismatic" else "rad"
        self._spin = QDoubleSpinBox()
        self._spin.setRange(lo, hi)
        self._spin.setDecimals(3)
        self._spin.setSingleStep(0.01)
        self._spin.setSuffix(f" {unit}")
        self._spin.setFixedWidth(90)
        self._spin.setStyleSheet(
            "QDoubleSpinBox{background:#1a1a1a;color:#eee;border:1px solid #555;"
            "border-radius:3px;padding:2px;font-family:monospace;font-size:11px;}"
            "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button"
            "{background:#333;width:14px;border-radius:2px;}")

        layout.addWidget(lbl)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

        # Wiring
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, v: int) -> None:
        val = v / self._scale
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        self.changed.emit(val)

    def _on_spin(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(val * self._scale))
        self._slider.blockSignals(False)
        self.changed.emit(val)

    def set_value(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setValue(int(val * self._scale))
        self._spin.setValue(val)
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)

    def value(self) -> float:
        return self._spin.value()


class ArmPanel(QWidget):
    """
    Right-side panel for arm control.

    Parameters
    ----------
    joints : list[JointInfo]  — from urdf_loader
    data   : mujoco.MjData   — shared with viewport; ctrl[] written here
    """

    joint_changed = pyqtSignal(str, float)

    def __init__(self, joints: list[JointInfo], data: mujoco.MjData,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data   = data
        self._joints = joints
        self._rows:  dict[str, _JointRow] = {}
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title
        title = QLabel("  Joint Control")
        title.setFixedHeight(32)
        title.setStyleSheet(
            "background:#1e1e1e;color:#ddd;font-size:13px;font-weight:bold;"
            "border-bottom:1px solid #444;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:#262626;}"
                              "QScrollBar:vertical{background:#2a2a2a;width:8px;}"
                              "QScrollBar::handle:vertical{background:#555;border-radius:4px;}")

        container = QWidget()
        container.setStyleSheet("background:#262626;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        if not self._joints:
            lbl = QLabel("No controllable joints found.\nCheck the URDF.")
            lbl.setStyleSheet("color:#888;font-size:12px;padding:20px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl)
        else:
            for ji in self._joints:
                row = _JointRow(ji)
                row.changed.connect(lambda val, j=ji: self._on_joint(j, val))
                self._rows[ji.name] = row
                vbox.addWidget(row)

        vbox.addStretch()

        # ── Buttons ────────────────────────────────────────────────────────
        btn_grp = QGroupBox("Actions")
        btn_grp.setStyleSheet(
            "QGroupBox{color:#aaa;font-size:11px;border:1px solid #444;"
            "border-radius:4px;margin-top:8px;padding:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")
        btn_lay = QVBoxLayout(btn_grp)
        btn_lay.setSpacing(4)

        btn_zero = self._make_btn("Zero all joints", "#1a4a6b")
        btn_zero.clicked.connect(self.zero_all)
        btn_lay.addWidget(btn_zero)

        vbox.addWidget(btn_grp)

        # ── Joint positions readout ─────────────────────────────────────
        self._pos_label = QLabel("Position: —")
        self._pos_label.setStyleSheet(
            "color:#888;font-size:10px;font-family:monospace;"
            "background:#1a1a1a;padding:6px;border-radius:3px;")
        self._pos_label.setWordWrap(True)
        vbox.addWidget(self._pos_label)

        scroll.setWidget(container)
        outer.addWidget(scroll, 1)

    @staticmethod
    def _make_btn(label: str, color: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(
            f"QPushButton{{background:{color};color:#fff;border-radius:4px;"
            f"padding:5px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{color}cc;}}"
            f"QPushButton:pressed{{background:{color}88;}}")
        return btn

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_joint(self, ji: JointInfo, val: float) -> None:
        if ji.ctrl_index < len(self._data.ctrl):
            self._data.ctrl[ji.ctrl_index] = val
        self.joint_changed.emit(ji.name, val)
        self._update_pos_label()

    def zero_all(self) -> None:
        for ji in self._joints:
            if ji.ctrl_index < len(self._data.ctrl):
                self._data.ctrl[ji.ctrl_index] = 0.0
            row = self._rows.get(ji.name)
            if row:
                row.set_value(0.0)
        self._update_pos_label()

    def _update_pos_label(self) -> None:
        lines = []
        for ji in self._joints:
            if ji.ctrl_index < len(self._data.ctrl):
                v = self._data.ctrl[ji.ctrl_index]
                unit = "m" if ji.joint_type == "prismatic" else "rad"
                lines.append(f"{ji.name[:18]}: {v:+.3f} {unit}")
        self._pos_label.setText("\n".join(lines) if lines else "—")

    # ── Live update from physics ───────────────────────────────────────────────

    def sync_from_data(self) -> None:
        """Read actual joint positions from data.qpos and update slider readouts."""
        for ji in self._joints:
            row = self._rows.get(ji.name)
            if row is None:
                continue
            # Find qpos address for this joint
            try:
                jid = mujoco.mj_name2id(
                    self._data.model if hasattr(self._data, 'model') else self._data,
                    mujoco.mjtObj.mjOBJ_JOINT, ji.name)
                # Note: MjData doesn't hold model ref in Python; qpos tracking is via ctrl
            except Exception:
                pass
