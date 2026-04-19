"""
gemini_panel.py  —  Gemini Robotics LLM panel for P_RoboAI Studio.

A Qt dock widget providing:
  - Chat interface with Gemini (natural language robot queries)
  - RAG context viewer (collapsed by default)
  - Sim2Real tab: ONNX export, calibration, domain gap gauge
  - Real2Sim tab: live system-ID display, apply-to-sim button
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore    import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui     import QColor, QPainter, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QTabWidget, QGroupBox, QComboBox,
    QSlider, QProgressBar, QFileDialog, QSplitter, QListWidget,
    QListWidgetItem, QCheckBox, QSpinBox, QDoubleSpinBox, QFrame,
)

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

try:
    from rag_engine      import RobotKnowledgeBase, RetrievalResult
    from sim2real_bridge import Sim2RealAdapter, CalibrationParams
    from real2sim_bridge import Real2SimAdapter
except ImportError:
    from .rag_engine      import RobotKnowledgeBase, RetrievalResult
    from .sim2real_bridge import Sim2RealAdapter, CalibrationParams
    from .real2sim_bridge import Real2SimAdapter


# ── domain gap gauge ──────────────────────────────────────────────────────────

class _GapGauge(QWidget):
    """Horizontal bar showing sim2real domain gap (green→red)."""

    def __init__(self) -> None:
        super().__init__()
        self._gap   = 0.0
        self._trend = "unknown"
        self.setFixedHeight(24)
        self.setMinimumWidth(100)

    def set_gap(self, gap: float, trend: str) -> None:
        self._gap   = min(1.0, max(0.0, gap))
        self._trend = trend
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        # Background
        p.fillRect(0, 0, W, H, QColor(40, 40, 40))
        # Bar colour: green (gap=0) → red (gap=1)
        r = int(self._gap * 255)
        g = int((1 - self._gap) * 180)
        fill_w = int(self._gap * W)
        p.fillRect(0, 0, fill_w, H, QColor(r, g, 30))
        # Label
        p.setPen(QPen(Qt.GlobalColor.white))
        p.setFont(QFont("monospace", 9))
        txt = f"Domain Gap: {self._gap:.3f}  ({self._trend})"
        p.drawText(4, H - 5, txt)


# ── signal bridge ─────────────────────────────────────────────────────────────

class _Emitter(QObject):
    response_ready = pyqtSignal(str, str)   # (query, response)
    error          = pyqtSignal(str)


# ── main panel ────────────────────────────────────────────────────────────────

class GeminiPanel(QWidget):
    """
    Gemini Robotics LLM panel.

    Parameters
    ----------
    get_mj_model : callable returning live mujoco.MjModel or None
    get_frame    : callable returning np.ndarray (H×W×3 RGB) or None
    """

    def __init__(self,
                 get_mj_model: Optional[Callable] = None,
                 get_frame:    Optional[Callable] = None) -> None:
        super().__init__()
        self._get_model = get_mj_model
        self._get_frame = get_frame
        self._emitter   = _Emitter()
        self._emitter.response_ready.connect(self._on_response)
        self._emitter.error.connect(self._on_error)

        # Components
        self._kb  = RobotKnowledgeBase(
            api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self._s2r = Sim2RealAdapter()
        self._r2s: Optional[Real2SimAdapter] = None

        self._gemini_model: Optional["genai.GenerativeModel"] = None
        self._chat_history: list[dict] = []

        self._build_ui()
        self._init_default_kb()

        # Status poll timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_status)
        self._timer.start(3000)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(4, 4, 4, 4)

        # API key row
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("AIza… (or set GOOGLE_API_KEY env var)")
        self._api_key_edit.setText(os.environ.get("GOOGLE_API_KEY", ""))
        key_row.addWidget(self._api_key_edit)
        model_combo = QComboBox()
        model_combo.addItems([
            "gemini-robotics-er-1.6",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ])
        model_combo.setFixedWidth(190)
        self._model_combo = model_combo
        key_row.addWidget(model_combo)
        connect_btn = QPushButton("Connect")
        connect_btn.setFixedWidth(72)
        connect_btn.clicked.connect(self._connect_gemini)
        key_row.addWidget(connect_btn)
        self._conn_label = QLabel("⬤")
        self._conn_label.setStyleSheet("color: #888;")
        key_row.addWidget(self._conn_label)
        root.addLayout(key_row)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(),    "Chat")
        tabs.addTab(self._build_rag_tab(),     "Knowledge")
        tabs.addTab(self._build_s2r_tab(),     "Sim2Real")
        tabs.addTab(self._build_r2s_tab(),     "Real2Sim")
        root.addWidget(tabs)

    def _build_chat_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(4)

        # Chat display
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setFont(QFont("monospace", 9))
        lay.addWidget(self._chat_display, stretch=1)

        # RAG context (collapsible)
        self._rag_group = QGroupBox("Retrieved Context (last query)")
        self._rag_group.setCheckable(True)
        self._rag_group.setChecked(False)
        rag_inner = QVBoxLayout(self._rag_group)
        self._rag_display = QTextEdit()
        self._rag_display.setReadOnly(True)
        self._rag_display.setMaximumHeight(120)
        self._rag_display.setFont(QFont("monospace", 8))
        rag_inner.addWidget(self._rag_display)
        lay.addWidget(self._rag_group)

        # Input row
        inp_row = QHBoxLayout()
        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText("Ask robot… e.g. 'navigate to the charging station'")
        self._query_edit.returnPressed.connect(self._send_query)
        inp_row.addWidget(self._query_edit)
        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(60)
        send_btn.clicked.connect(self._send_query)
        inp_row.addWidget(send_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self._chat_display.clear)
        inp_row.addWidget(clear_btn)
        lay.addLayout(inp_row)

        self._status_label = QLabel("Gemini: not connected")
        self._status_label.setStyleSheet("color: #888; font-size: 10px;")
        lay.addWidget(self._status_label)
        return w

    def _build_rag_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        lay.addWidget(QLabel("Knowledge Chunks in RAG DB:"))
        self._kb_list = QListWidget()
        lay.addWidget(self._kb_list, stretch=1)

        btn_row = QHBoxLayout()
        add_urdf_btn = QPushButton("Add URDF from file…")
        add_urdf_btn.clicked.connect(self._add_urdf_to_kb)
        btn_row.addWidget(add_urdf_btn)

        add_doc_btn = QPushButton("Add text doc…")
        add_doc_btn.clicked.connect(self._add_doc_to_kb)
        btn_row.addWidget(add_doc_btn)

        snapshot_btn = QPushButton("Snapshot sensor state")
        snapshot_btn.clicked.connect(self._snapshot_sensors)
        btn_row.addWidget(snapshot_btn)
        lay.addLayout(btn_row)

        return w

    def _build_s2r_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self._gap_gauge = _GapGauge()
        lay.addWidget(self._gap_gauge)

        grp = QGroupBox("Export Policy → ONNX")
        grp_lay = QVBoxLayout(grp)
        row1 = QHBoxLayout()
        self._policy_zip_edit = QLineEdit()
        self._policy_zip_edit.setPlaceholderText("SB3 policy ZIP path…")
        row1.addWidget(self._policy_zip_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_policy)
        row1.addWidget(browse_btn)
        grp_lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._onnx_out_edit = QLineEdit()
        self._onnx_out_edit.setPlaceholderText("ONNX output path…")
        row2.addWidget(self._onnx_out_edit)
        export_btn = QPushButton("Export")
        export_btn.setFixedWidth(60)
        export_btn.clicked.connect(self._export_onnx)
        row2.addWidget(export_btn)
        grp_lay.addLayout(row2)

        load_btn = QPushButton("Load ONNX for inference")
        load_btn.clicked.connect(self._load_onnx)
        grp_lay.addWidget(load_btn)
        lay.addWidget(grp)

        grp2 = QGroupBox("Calibration")
        grp2_lay = QHBoxLayout(grp2)
        save_calib = QPushButton("Save calibration…")
        save_calib.clicked.connect(self._save_calib)
        load_calib = QPushButton("Load calibration…")
        load_calib.clicked.connect(self._load_calib)
        grp2_lay.addWidget(save_calib)
        grp2_lay.addWidget(load_calib)
        lay.addWidget(grp2)

        self._s2r_status = QTextEdit()
        self._s2r_status.setReadOnly(True)
        self._s2r_status.setMaximumHeight(100)
        self._s2r_status.setFont(QFont("monospace", 8))
        lay.addWidget(self._s2r_status)

        lay.addStretch()
        return w

    def _build_r2s_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        n_joints_lbl = QLabel("Joints:")
        self._n_joints_spin = QSpinBox()
        self._n_joints_spin.setRange(1, 20)
        self._n_joints_spin.setValue(6)
        ctrl_row.addWidget(n_joints_lbl)
        ctrl_row.addWidget(self._n_joints_spin)
        start_btn = QPushButton("Start Real2Sim ID")
        start_btn.clicked.connect(self._start_r2s)
        ctrl_row.addWidget(start_btn)
        apply_btn = QPushButton("Apply to Sim")
        apply_btn.clicked.connect(self._apply_r2s)
        ctrl_row.addWidget(apply_btn)
        lay.addLayout(ctrl_row)

        self._r2s_display = QTextEdit()
        self._r2s_display.setReadOnly(True)
        self._r2s_display.setFont(QFont("monospace", 8))
        lay.addWidget(self._r2s_display, stretch=1)

        row2 = QHBoxLayout()
        save_r2s = QPushButton("Save params…")
        save_r2s.clicked.connect(self._save_r2s)
        load_r2s = QPushButton("Load params…")
        load_r2s.clicked.connect(self._load_r2s)
        row2.addWidget(save_r2s)
        row2.addWidget(load_r2s)
        lay.addLayout(row2)

        return w

    # ── Gemini connection ─────────────────────────────────────────────────────

    def _connect_gemini(self) -> None:
        if not _GENAI_OK:
            self._status_label.setText("Gemini: install google-generativeai")
            return
        key = self._api_key_edit.text().strip()
        if not key:
            self._status_label.setText("Gemini: no API key")
            return
        try:
            genai.configure(api_key=key)
            model_id = self._model_combo.currentText()
            is_er    = "robotics-er" in model_id
            sys_instr = (
                "You are Gemini Robotics ER, an embodied reasoning AI for robotic systems. "
                "You specialize in spatial reasoning, manipulation planning, navigation, "
                "and real-time robot control. Analyse sensor data precisely and generate "
                "structured robot commands. Always respond with ANALYSIS: and COMMAND: sections."
            ) if is_er else (
                "You are an expert robotics AI assistant. "
                "Help interpret robot sensor data and generate robot commands. "
                "Respond concisely with ANALYSIS: and COMMAND: sections."
            )
            self._gemini_model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=sys_instr,
            )
            self._kb = RobotKnowledgeBase(api_key=key)
            self._init_default_kb()
            self._conn_label.setStyleSheet("color: #4f4;")
            er_note = "  [Robotics ER — requires allowlist access]" if is_er else ""
            self._status_label.setText(
                f"Gemini: connected ({model_id}){er_note}")
        except Exception as e:
            self._conn_label.setStyleSheet("color: #f44;")
            self._status_label.setText(f"Gemini: error — {e}")

    # ── chat ──────────────────────────────────────────────────────────────────

    def _send_query(self) -> None:
        query = self._query_edit.text().strip()
        if not query:
            return
        self._query_edit.clear()
        self._append_chat("You", query)
        if not self._gemini_model:
            self._append_chat("System", "Gemini not connected — use RAG-only fallback")
            self._show_rag_only(query)
            return
        threading.Thread(target=self._run_query_thread, args=(query,), daemon=True).start()

    def _run_query_thread(self, query: str) -> None:
        # RAG retrieval
        results = self._kb.retrieve(query, top_k=4)
        rag_ctx = self._kb.format_context(results)

        # Build sensor context
        sensor_ctx = self._build_sensor_ctx()

        prompt = (
            f"=== Robot Sensor State ===\n{sensor_ctx}\n\n"
            f"{rag_ctx}\n\n"
            f"User: {query}"
        )

        try:
            response = self._gemini_model.generate_content(prompt)
            txt      = response.text
        except Exception as e:
            txt = f"Error: {e}"

        self._emitter.response_ready.emit(query, txt)
        # Update RAG display from thread
        rag_text = "\n".join(
            f"[{i+1}] {r.chunk.source}: {r.chunk.text[:120]}…  (score={r.score:.3f})"
            for i, r in enumerate(results)
        )
        # Can't update Qt directly from thread — store and update in _on_response
        self._last_rag_text = rag_text

    def _show_rag_only(self, query: str) -> None:
        results = self._kb.retrieve(query, top_k=4)
        if results:
            txt = "\n".join(
                f"[{i+1}] {r.chunk.source}: {r.chunk.text[:200]}"
                for i, r in enumerate(results)
            )
            self._append_chat("RAG", txt)

    def _on_response(self, query: str, response: str) -> None:
        self._append_chat("Gemini", response)
        if hasattr(self, "_last_rag_text"):
            self._rag_display.setPlainText(self._last_rag_text)

    def _on_error(self, msg: str) -> None:
        self._append_chat("Error", msg)

    def _append_chat(self, sender: str, text: str) -> None:
        colors = {"You": "#7af", "Gemini": "#af7", "RAG": "#fa7",
                  "System": "#aaa", "Error": "#f77"}
        col = colors.get(sender, "#ddd")
        html = (f'<span style="color:{col};font-weight:bold">{sender}:</span>'
                f'<span style="color:#ddd"> {text.replace(chr(10),"<br>")}</span><br><br>')
        self._chat_display.append(html)

    def _build_sensor_ctx(self) -> str:
        lines = []
        if self._get_model:
            try:
                model = self._get_model()
                if model is not None and _NP_OK:
                    lines.append(f"MuJoCo model: {model.nq} DOF, {model.nbody} bodies")
            except Exception:
                pass
        if self._get_frame:
            try:
                frame = self._get_frame()
                if frame is not None:
                    lines.append(f"Camera frame: {frame.shape[1]}×{frame.shape[0]} available")
            except Exception:
                pass
        return "\n".join(lines) if lines else "No live sensor data."

    # ── knowledge base ────────────────────────────────────────────────────────

    def _add_urdf_to_kb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open URDF/XML",
                                               "", "Robot files (*.urdf *.xml)")
        if not path:
            return
        try:
            content = Path(path).read_text()
            name    = Path(path).stem
            self._kb.add_urdf_spec(name, content)
            self._refresh_kb_list()
        except Exception as e:
            self._status_label.setText(f"KB: error loading — {e}")

    def _add_doc_to_kb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open text doc",
                                               "", "Text files (*.txt *.md)")
        if not path:
            return
        try:
            content = Path(path).read_text()
            self._kb.add_task_doc(Path(path).stem, content)
            self._refresh_kb_list()
        except Exception as e:
            self._status_label.setText(f"KB: error — {e}")

    def _snapshot_sensors(self) -> None:
        snap: dict = {"timestamp_snapshot": True}
        if self._get_model:
            try:
                m = self._get_model()
                if m is not None:
                    snap["n_dof"] = int(m.nq)
                    snap["n_bodies"] = int(m.nbody)
            except Exception:
                pass
        self._kb.add_sensor_snapshot(snap)
        self._refresh_kb_list()

    def _refresh_kb_list(self) -> None:
        self._kb_list.clear()
        for c in self._kb._chunks:
            item = QListWidgetItem(
                f"[{c.source}] {c.text[:80].replace(chr(10), ' ')}…")
            self._kb_list.addItem(item)

    # ── sim2real ──────────────────────────────────────────────────────────────

    def _browse_policy(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select policy ZIP",
                                               "", "ZIP files (*.zip)")
        if path:
            self._policy_zip_edit.setText(path)
            out = str(Path(path).with_suffix(".onnx"))
            self._onnx_out_edit.setText(out)

    def _export_onnx(self) -> None:
        src = self._policy_zip_edit.text().strip()
        dst = self._onnx_out_edit.text().strip()
        if not src or not dst:
            return
        self._s2r_status.setPlainText("Exporting to ONNX…")

        def _do():
            ok = Sim2RealAdapter.export_onnx(src, dst)
            msg = f"Export {'OK → ' + dst if ok else 'FAILED (install torch, stable-baselines3)'}"
            self._s2r_status.setPlainText(msg)
        threading.Thread(target=_do, daemon=True).start()

    def _load_onnx(self) -> None:
        path = self._onnx_out_edit.text().strip()
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select ONNX", "", "ONNX (*.onnx)")
        if path:
            ok = self._s2r.load_onnx(path)
            self._s2r_status.setPlainText(
                f"ONNX {'loaded: ' + path if ok else 'load failed (install onnxruntime)'}")

    def _save_calib(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", "", "JSON (*.json)")
        if path:
            self._s2r.save_calibration(path)

    def _load_calib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if path:
            ok = self._s2r.load_calibration(path)
            self._s2r_status.setPlainText(
                f"Calibration {'loaded' if ok else 'failed'}: {path}")

    # ── real2sim ──────────────────────────────────────────────────────────────

    def _start_r2s(self) -> None:
        n = self._n_joints_spin.value()
        self._r2s = Real2SimAdapter(n_joints=n)
        self._r2s_display.setPlainText(
            f"Real2Sim ID started for {n} joints.\n"
            "Feed joint data by connecting /joint_states in the ROS2 node,\n"
            "or call adapter.update() with real sensor data.")

    def _apply_r2s(self) -> None:
        if self._r2s is None:
            self._r2s_display.setPlainText("Start Real2Sim ID first.")
            return
        if self._get_model:
            try:
                m = self._get_model()
                if m is not None:
                    self._r2s.apply(m)
                    self._r2s_display.setPlainText(
                        "Applied to sim:\n" + json.dumps(self._r2s.get_status(), indent=2))
                    return
            except Exception as e:
                self._r2s_display.setPlainText(f"Apply failed: {e}")
                return
        self._r2s_display.setPlainText("No live MuJoCo model available.")

    def _save_r2s(self) -> None:
        if self._r2s is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save R2S params", "", "JSON (*.json)")
        if path:
            self._r2s.save(path)

    def _load_r2s(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load R2S params", "", "JSON (*.json)")
        if path:
            if self._r2s is None:
                self._r2s = Real2SimAdapter(n_joints=self._n_joints_spin.value())
            ok = self._r2s.load(path)
            self._r2s_display.setPlainText(
                f"Params {'loaded' if ok else 'failed'}: {path}")

    # ── status poll ───────────────────────────────────────────────────────────

    def _poll_status(self) -> None:
        # Update gap gauge
        gap   = self._s2r.domain_gap
        trend = self._s2r.domain_gap_trend
        self._gap_gauge.set_gap(gap, trend)

        # Update S2R status text
        dr = self._s2r.suggest_domain_randomization()
        self._s2r_status.setPlainText(
            f"Domain Gap: {gap:.4f}  Trend: {trend}\n"
            f"Suggested DR friction: {dr['friction_range']}\n"
            f"Suggested DR damping:  {dr['damping_range']}\n"
            f"Suggested noise std:   {dr['noise_std']:.4f}\n"
            f"Suggested delay steps: {dr['delay_steps_max']}"
        )

        # Update R2S display
        if self._r2s is not None:
            self._r2s_display.setPlainText(
                json.dumps(self._r2s.get_status(), indent=2))

        # KB list (refresh if changed)
        if self._kb_list.count() != len(self._kb._chunks):
            self._refresh_kb_list()

    # ── defaults ──────────────────────────────────────────────────────────────

    def _init_default_kb(self) -> None:
        self._kb.add_task_doc("p_roboai_system",
            "P_RoboAI is a robotics simulation and control platform built on MuJoCo and ROS2. "
            "It includes AMR navigation and robot arm control with reinforcement learning (PPO/SAC/TD3), "
            "YOLO object detection, and Gemini LLM integration for natural language robot control. "
            "The simulation uses MuJoCo physics engine with EGL offscreen rendering.")
        self._kb.add_task_doc("robot_control_interface",
            "Robot control topics: /cmd_vel (AMR), /arm/position_commands (arm joints). "
            "Sensor topics: /joint_states, /odom, /scan, /amr/camera/image. "
            "RL topics: /amr_rl/status, /arm_rl/status. "
            "YOLO topics: /yolo/detections, /yolo/image. "
            "Gemini topics: /gemini/query (input), /gemini/response (output), "
            "/gemini/robot_command (structured commands).")
        self._refresh_kb_list()
