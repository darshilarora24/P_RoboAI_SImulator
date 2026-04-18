"""
main_window.py  —  P_RoboAI Studio

QMainWindow that ties together:
  • URDF file picker
  • MuJoCo viewport (3D rendering)
  • ArmPanel   — joint sliders for manipulators
  • AMRPanel   — WASD keyboard drive for mobile robots
  • RLPanel    — reinforcement learning train / evaluate
  • YOLOPanel  — live object detection on rendered frames
  • Physics simulation loop (200 Hz, QTimer)
  • Mode auto-detection (ARM / AMR) with manual override
"""
from __future__ import annotations

import traceback
from pathlib import Path

import mujoco
import numpy as np

from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import QKeyEvent, QAction, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QDockWidget,
    QLabel, QPushButton, QComboBox, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QToolBar,
    QStatusBar,
)

from mujoco_viewport import MuJoCoViewport
from arm_panel       import ArmPanel
from amr_panel       import AMRPanel
from rl_panel        import RLPanel
from yolo_panel      import YOLOPanel
import urdf_loader
from urdf_loader import RobotKind


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("P_RoboAI Studio")
        self.setMinimumSize(1050, 680)
        self.resize(1280, 800)

        # ── State ─────────────────────────────────────────────────────────
        self._model:   mujoco.MjModel | None = None
        self._data:    mujoco.MjData  | None = None
        self._kind:    RobotKind | None       = None
        self._joints:  list[urdf_loader.JointInfo] = []

        self._viewport:   MuJoCoViewport | None = None
        self._arm_panel:  ArmPanel        | None = None
        self._amr_panel:  AMRPanel        | None = None
        self._ctrl_dock:  QDockWidget     | None = None
        self._rl_dock:    QDockWidget     | None = None
        self._yolo_dock:  QDockWidget     | None = None
        self._rl_panel:   RLPanel         | None = None
        self._yolo_panel: YOLOPanel       | None = None
        self._mjcf_xml:   str | None             = None

        # Physics simulation timer (200 Hz)
        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(5)   # 5 ms = 200 Hz
        self._sim_timer.timeout.connect(self._physics_step)

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._build_rl_dock()
        self._build_yolo_dock()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Placeholder central widget shown before a URDF is loaded
        placeholder = QWidget()
        placeholder.setStyleSheet("background:#111;")
        pbox = QVBoxLayout(placeholder)
        pbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel("🤖")
        icon_lbl.setStyleSheet("font-size:64px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        msg = QLabel(
            "<h2 style='color:#ddd;'>P_RoboAI Studio</h2>"
            "<p style='color:#888;'>Open a URDF file to load your robot.<br>"
            "Supports robot arms and differential-drive AMRs.</p>")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_btn = QPushButton("Open URDF…")
        open_btn.setFixedWidth(160)
        open_btn.setStyleSheet(
            "QPushButton{background:#1a6b3a;color:#fff;border-radius:6px;"
            "padding:8px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#22914e;}")
        open_btn.clicked.connect(self._open_urdf_dialog)

        pbox.addStretch()
        pbox.addWidget(icon_lbl)
        pbox.addWidget(msg)
        pbox.addSpacing(20)
        pbox.addWidget(open_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        pbox.addStretch()

        self.setCentralWidget(placeholder)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        act_open = QAction("&Open URDF…\tCtrl+O", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_urdf_dialog)
        fm.addAction(act_open)

        act_reload = QAction("&Reload URDF\tCtrl+R", self)
        act_reload.setShortcut("Ctrl+R")
        act_reload.triggered.connect(self._reload_urdf)
        fm.addAction(act_reload)
        self._act_reload = act_reload

        fm.addSeparator()
        act_quit = QAction("&Quit\tCtrl+Q", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        fm.addAction(act_quit)

        # View
        vm = mb.addMenu("&View")
        act_reset_cam = QAction("Reset Camera\tDbl-click 3D", self)
        act_reset_cam.triggered.connect(self._reset_camera)
        vm.addAction(act_reset_cam)

        act_physics = QAction("Pause / Resume Physics\tP", self)
        act_physics.setShortcut("P")
        act_physics.triggered.connect(self._toggle_physics)
        vm.addAction(act_physics)

        vm.addSeparator()
        act_rl = QAction("Reinforcement Learning Panel", self)
        act_rl.triggered.connect(
            lambda: self._rl_dock and self._rl_dock.setVisible(
                not self._rl_dock.isVisible()))
        vm.addAction(act_rl)

        act_yolo = QAction("YOLO Vision Panel", self)
        act_yolo.triggered.connect(
            lambda: self._yolo_dock and self._yolo_dock.setVisible(
                not self._yolo_dock.isVisible()))
        vm.addAction(act_yolo)

        # Help
        hm = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        hm.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar{background:#1e1e1e;border-bottom:1px solid #333;spacing:4px;}"
            "QToolButton{color:#ccc;padding:4px 10px;border-radius:4px;font-size:12px;}"
            "QToolButton:hover{background:#333;}"
            "QToolButton:checked{background:#1a5c3a;color:#7fff9a;}")
        self.addToolBar(tb)

        # Open button
        open_btn = QPushButton("📂  Open URDF")
        open_btn.setStyleSheet(
            "QPushButton{background:#2a4a6a;color:#fff;border-radius:4px;"
            "padding:5px 10px;font-size:12px;}"
            "QPushButton:hover{background:#3a6a9a;}")
        open_btn.clicked.connect(self._open_urdf_dialog)
        tb.addWidget(open_btn)

        tb.addSeparator()

        # Mode selector (shown only after load)
        mode_lbl = QLabel("  Mode: ")
        mode_lbl.setStyleSheet("color:#888;font-size:12px;")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Auto-detect", "Arm", "AMR"])
        self._mode_combo.setStyleSheet(
            "QComboBox{background:#2a2a2a;color:#ccc;border:1px solid #444;"
            "border-radius:3px;padding:3px;}"
            "QComboBox::drop-down{background:#333;border-radius:2px;}")
        self._mode_combo.setFixedWidth(120)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(mode_lbl)
        tb.addWidget(self._mode_combo)

        tb.addSeparator()

        # Physics toggle
        self._physics_btn = QPushButton("⏸  Pause")
        self._physics_btn.setStyleSheet(
            "QPushButton{background:#444;color:#ccc;border-radius:4px;"
            "padding:5px 10px;font-size:12px;}"
            "QPushButton:hover{background:#555;}")
        self._physics_btn.clicked.connect(self._toggle_physics)
        self._physics_btn.setEnabled(False)
        tb.addWidget(self._physics_btn)

        # Reset physics
        reset_btn = QPushButton("↺  Reset")
        reset_btn.setStyleSheet(
            "QPushButton{background:#444;color:#ccc;border-radius:4px;"
            "padding:5px 10px;font-size:12px;}"
            "QPushButton:hover{background:#555;}")
        reset_btn.clicked.connect(self._reset_simulation)
        tb.addWidget(reset_btn)

        tb.addSeparator()

        # Help hint
        hint = QLabel("  Drag 3D: orbit  |  Right-drag: zoom  |  Mid-drag: pan  |  Dbl-click: reset")
        hint.setStyleSheet("color:#555;font-size:11px;")
        tb.addWidget(hint)

        self._last_urdf: Path | None = None

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setStyleSheet(
            "QStatusBar{background:#1a1a1a;border-top:1px solid #333;color:#777;font-size:11px;}")
        self._sb_model  = QLabel("No model loaded")
        self._sb_sim_hz = QLabel("")
        self._sb_kind   = QLabel("")
        sb.addWidget(self._sb_model)
        sb.addPermanentWidget(self._sb_kind)
        sb.addPermanentWidget(self._sb_sim_hz)

        # Sim-rate timer
        self._step_count = 0
        self._hz_timer = QTimer(self)
        self._hz_timer.setInterval(1000)
        self._hz_timer.timeout.connect(self._update_hz)
        self._hz_timer.start()

    # ── RL dock ───────────────────────────────────────────────────────────────

    def _build_rl_dock(self) -> None:
        self._rl_panel = RLPanel(
            model_getter     = self._rl_model_getter,
            live_data_getter = lambda: self._data,
        )
        self._rl_panel.request_pause_physics.connect(self._on_rl_pause)
        self._rl_panel.setMinimumWidth(270)
        self._rl_panel.setMaximumWidth(360)

        dock = QDockWidget("Reinforcement Learning", self)
        dock.setObjectName("rl_dock")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)
        dock.setWidget(self._rl_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(360, 520)
        self._rl_dock = dock

    def _build_yolo_dock(self) -> None:
        self._yolo_panel = YOLOPanel(frame_getter=self._yolo_frame_getter)
        self._yolo_panel.setMinimumWidth(270)
        self._yolo_panel.setMaximumWidth(380)

        dock = QDockWidget("YOLO Vision", self)
        dock.setObjectName("yolo_dock")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)
        dock.setWidget(self._yolo_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setFloating(True)
        dock.resize(380, 560)
        self._yolo_dock = dock

    # ── Getters used by sub-panels ────────────────────────────────────────────

    def _rl_model_getter(self):
        """Returns (mjcf_xml, joint_infos, kind) or None."""
        if self._mjcf_xml and self._joints and self._kind:
            return self._mjcf_xml, self._joints, self._kind
        return None

    def _yolo_frame_getter(self):
        """Returns latest rendered RGB frame (H,W,3) uint8 or None."""
        if self._viewport is None or self._viewport._renderer is None:
            return None
        try:
            import numpy as np
            pixels = self._viewport._renderer.render()
            return np.ascontiguousarray(np.flipud(pixels))
        except Exception:
            return None

    def _on_rl_pause(self, pause: bool) -> None:
        self._physics_paused = pause
        self._physics_btn.setText("▶  Resume" if pause else "⏸  Pause")

    # ── URDF loading ──────────────────────────────────────────────────────────

    def _open_urdf_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open URDF", str(Path.home()),
            "URDF Files (*.urdf *.xacro);;All Files (*)")
        if path:
            self._load_urdf(Path(path))

    def _reload_urdf(self) -> None:
        if self._last_urdf:
            self._load_urdf(self._last_urdf)

    def _load_urdf(self, path: Path) -> None:
        self._sb_model.setText(f"Loading {path.name}…")
        try:
            result = urdf_loader.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "URDF Load Error",
                                 f"Failed to parse URDF:\n\n{exc}\n\n"
                                 f"{traceback.format_exc()}")
            self._sb_model.setText("Load failed")
            return

        try:
            model = mujoco.MjModel.from_xml_string(result.mjcf_xml)
        except Exception as exc:
            QMessageBox.critical(self, "MuJoCo Error",
                                 f"MuJoCo rejected the generated MJCF:\n\n{exc}\n\n"
                                 "Check mesh paths or joint limits in the URDF.")
            self._sb_model.setText("MuJoCo load failed")
            return

        self._sim_timer.stop()
        self._model    = model
        self._data     = mujoco.MjData(model)
        self._joints   = result.joints
        self._kind     = result.kind
        self._mjcf_xml = result.mjcf_xml
        self._last_urdf = path

        # Apply mode override from combo
        idx = self._mode_combo.currentIndex()
        if idx == 1:
            self._kind = RobotKind.ARM
        elif idx == 2:
            self._kind = RobotKind.AMR

        self._install_panels()
        mujoco.mj_forward(model, self._data)
        self._sim_timer.start()
        self._physics_btn.setEnabled(True)
        self._physics_paused = False
        self._physics_btn.setText("⏸  Pause")

        n_ctrl = len(self._joints)
        kind_str = "Arm" if self._kind == RobotKind.ARM else "AMR"
        self._sb_model.setText(f"{result.model_name}  |  {path.name}")
        self._sb_kind.setText(f"  {kind_str}  |  {n_ctrl} joints  ")

        if self._viewport:
            self._viewport.show_overlay(
                f"✓  Loaded: {result.model_name}  ({kind_str})")

    # ── Panel installation ────────────────────────────────────────────────────

    def _install_panels(self) -> None:
        """(Re-)create viewport and control dock for the new model."""
        assert self._model is not None and self._data is not None

        self._sim_timer.stop()

        # ── Viewport ──────────────────────────────────────────────────────
        if self._viewport is None:
            self._viewport = MuJoCoViewport(self._model, self._data)
            # Splitter: viewport left, nothing right yet
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self._viewport)
            splitter.setStyleSheet("QSplitter::handle{background:#333;}")
            self.setCentralWidget(splitter)
        else:
            self._viewport.replace_model(self._model, self._data)

        # ── Remove old control dock ────────────────────────────────────────
        if self._ctrl_dock is not None:
            self.removeDockWidget(self._ctrl_dock)
            self._ctrl_dock.deleteLater()
            self._ctrl_dock = None
        self._arm_panel = None
        self._amr_panel = None

        # ── Create new control dock ────────────────────────────────────────
        if self._kind == RobotKind.ARM:
            panel = ArmPanel(self._joints, self._data)
            self._arm_panel = panel
            dock_title = "Joint Control"
        else:
            panel = AMRPanel(self._joints, self._data, self._model)
            self._amr_panel = panel
            dock_title = "AMR Drive"

        panel.setMinimumWidth(240)
        panel.setMaximumWidth(340)

        dock = QDockWidget(dock_title, self)
        dock.setObjectName("ctrl_dock")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._ctrl_dock = dock

        self._sim_timer.start()

    # ── Physics loop ──────────────────────────────────────────────────────────

    _physics_paused: bool = False

    def _physics_step(self) -> None:
        if self._model is None or self._data is None or self._physics_paused:
            return
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

    def _toggle_physics(self) -> None:
        self._physics_paused = not self._physics_paused
        self._physics_btn.setText(
            "▶  Resume" if self._physics_paused else "⏸  Pause")

    def _reset_simulation(self) -> None:
        if self._model is not None and self._data is not None:
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)
            if self._arm_panel:
                self._arm_panel.zero_all()
            if self._amr_panel:
                self._amr_panel.stop()

    def _reset_camera(self) -> None:
        if self._viewport:
            self._viewport._reset_camera()
            self._viewport.update()

    # ── Status bar updates ────────────────────────────────────────────────────

    def _update_hz(self) -> None:
        if self._model:
            self._sb_sim_hz.setText(f"  {self._step_count} steps/s  ")
        self._step_count = 0

    # ── Mode selector ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, _: int) -> None:
        # Only re-install panels if a model is already loaded
        if self._model is not None and self._last_urdf is not None:
            self._load_urdf(self._last_urdf)

    # ── Keyboard → AMR panel ──────────────────────────────────────────────────

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if self._amr_panel and self._amr_panel.handle_key_press(ev):
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev: QKeyEvent) -> None:
        if self._amr_panel and self._amr_panel.handle_key_release(ev):
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    # ── About dialog ──────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(self, "About P_RoboAI Studio",
            "<h3>P_RoboAI Studio</h3>"
            "<p>Standalone robot simulator and controller.<br>"
            "Load any URDF — arm or AMR — and interact in real time.</p>"
            "<p><b>Controls:</b><br>"
            "3D view: Left-drag orbit · Right-drag zoom · Mid-drag pan · Scroll zoom · Dbl-click reset<br>"
            "AMR: W/A/S/D or arrow keys · Space = stop</p>"
            "<p>Powered by <b>MuJoCo</b> · <b>PyQt6</b> · Python 3</p>")
