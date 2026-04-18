"""
yolo_panel.py  —  P_RoboAI Studio

Qt panel that grabs the current MuJoCo rendered frame, runs YOLO detection,
and shows the annotated image alongside a list of detected objects.

The panel polls on a timer (default 10 Hz) so it never blocks the 200 Hz
physics loop.
"""
from __future__ import annotations

from typing import Optional, Callable

import numpy as np

from PyQt6.QtCore    import Qt, QTimer
from PyQt6.QtGui     import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QListWidget, QListWidgetItem, QFrame,
    QComboBox, QSizePolicy,
)

from yolo_detector import YOLODetector, Detection


class YOLOPanel(QWidget):
    """
    Parameters
    ----------
    frame_getter : callable → np.ndarray | None
        Returns the latest (H, W, 3) uint8 RGB frame from the viewport,
        or None if no model is loaded.
    """

    _GRP = ("QGroupBox{color:#aaa;font-size:11px;border:1px solid #444;"
            "border-radius:4px;margin-top:8px;padding:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}")

    def __init__(self, frame_getter: Callable[[], Optional[np.ndarray]],
                 parent=None) -> None:
        super().__init__(parent)
        self._get_frame  = frame_getter
        self._detector   = YOLODetector()
        self._running    = False
        self._last_dets: list[Detection] = []

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)   # 10 Hz
        self._poll_timer.timeout.connect(self._poll)

        self._build_ui()

        if not self._detector.ready:
            self._status_lbl.setText(
                f"⚠ YOLO unavailable\n{self._detector.error}")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title = QLabel("  YOLO Vision")
        title.setFixedHeight(32)
        title.setStyleSheet(
            "background:#1e1e1e;color:#ddd;font-size:13px;font-weight:bold;"
            "border-bottom:1px solid #444;")
        outer.addWidget(title)

        inner = QWidget()
        inner.setStyleSheet("background:#1e1e1e;")
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # ── Camera view ────────────────────────────────────────────────────
        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setMinimumHeight(180)
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)
        self._cam_label.setStyleSheet("background:#111;border:1px solid #333;")
        self._cam_label.setText(
            "<span style='color:#555;'>Camera feed appears here</span>")
        vbox.addWidget(self._cam_label, 3)

        # ── Controls ───────────────────────────────────────────────────────
        ctrl_grp = QGroupBox("Detection Settings")
        ctrl_grp.setStyleSheet(self._GRP)
        cg = QVBoxLayout(ctrl_grp)

        # Model selector
        model_row = QHBoxLayout()
        ml = QLabel("Model:")
        ml.setStyleSheet("color:#aaa;font-size:11px;")
        self._model_combo = QComboBox()
        self._model_combo.addItems(
            ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"])
        self._model_combo.setStyleSheet(
            "QComboBox{background:#2a2a2a;color:#ccc;border:1px solid #555;"
            "border-radius:3px;padding:2px;font-size:11px;}")
        self._model_combo.currentTextChanged.connect(self._reload_model)
        model_row.addWidget(ml)
        model_row.addWidget(self._model_combo, 1)
        cg.addLayout(model_row)

        # Confidence slider
        conf_row = QHBoxLayout()
        cl = QLabel("Confidence:")
        cl.setStyleSheet("color:#aaa;font-size:11px;")
        self._conf_val = QLabel("0.35")
        self._conf_val.setStyleSheet(
            "color:#00e5ff;font-family:monospace;font-size:11px;"
            "background:#1a1a1a;border:1px solid #333;padding:1px 4px;"
            "border-radius:3px;")
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(35)
        self._conf_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#444;border-radius:2px;}"
            "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
            "background:#00e5ff;border-radius:6px;}"
            "QSlider::sub-page:horizontal{background:#00e5ff;border-radius:2px;}")
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(cl)
        conf_row.addWidget(self._conf_slider, 1)
        conf_row.addWidget(self._conf_val)
        cg.addLayout(conf_row)

        # Poll rate
        rate_row = QHBoxLayout()
        rl = QLabel("Rate (Hz):")
        rl.setStyleSheet("color:#aaa;font-size:11px;")
        self._rate_combo = QComboBox()
        self._rate_combo.addItems(["2", "5", "10", "15", "20"])
        self._rate_combo.setCurrentText("10")
        self._rate_combo.setStyleSheet(
            "QComboBox{background:#2a2a2a;color:#ccc;border:1px solid #555;"
            "border-radius:3px;padding:2px;font-size:11px;}")
        self._rate_combo.currentTextChanged.connect(self._on_rate_changed)
        rate_row.addWidget(rl)
        rate_row.addWidget(self._rate_combo)
        cg.addLayout(rate_row)

        vbox.addWidget(ctrl_grp)

        # ── Detected objects list ──────────────────────────────────────────
        det_grp = QGroupBox("Detected Objects")
        det_grp.setStyleSheet(self._GRP)
        dg = QVBoxLayout(det_grp)
        self._det_list = QListWidget()
        self._det_list.setMaximumHeight(120)
        self._det_list.setStyleSheet(
            "QListWidget{background:#1a1a1a;color:#ccc;border:1px solid #333;"
            "font-family:monospace;font-size:11px;}"
            "QListWidget::item:selected{background:#2a4a2a;}")
        dg.addWidget(self._det_list)
        vbox.addWidget(det_grp)

        # ── Status ─────────────────────────────────────────────────────────
        self._status_lbl = QLabel(
            "YOLO ready" if self._detector.ready else "YOLO unavailable")
        self._status_lbl.setStyleSheet(
            "color:#4a9;font-size:10px;font-family:monospace;"
            "background:#1a1a1a;padding:3px;border-radius:3px;")
        self._status_lbl.setWordWrap(True)
        vbox.addWidget(self._status_lbl)

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Detection")
        self._stop_btn  = QPushButton("⏹  Stop")
        for b, c in ((self._start_btn, "#1a4a6b"), (self._stop_btn, "#6b1a1a")):
            b.setStyleSheet(
                f"QPushButton{{background:{c};color:#fff;border-radius:4px;"
                f"padding:5px;font-size:11px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{c}cc;}}"
                f"QPushButton:disabled{{background:#333;color:#666;}}")
        self._stop_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        vbox.addLayout(btn_row)

        outer.addWidget(inner, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_conf_changed(self, v: int) -> None:
        thresh = v / 100.0
        self._conf_val.setText(f"{thresh:.2f}")
        self._detector.set_conf(thresh)

    def _on_rate_changed(self, text: str) -> None:
        try:
            hz = int(text)
            self._poll_timer.setInterval(max(50, 1000 // hz))
        except ValueError:
            pass

    def _reload_model(self, name: str) -> None:
        self._poll_timer.stop()
        self._status_lbl.setText(f"Loading {name} …")
        self._detector = YOLODetector(
            model_name=name,
            conf_thresh=self._conf_slider.value() / 100.0)
        if self._detector.ready:
            self._status_lbl.setText(f"Model: {name}")
            if self._running:
                self._poll_timer.start()
        else:
            self._status_lbl.setText(f"⚠ {self._detector.error}")

    def _on_start(self) -> None:
        if not self._detector.ready:
            self._status_lbl.setText(
                "Install ultralytics: pip install ultralytics")
            return
        self._running = True
        self._poll_timer.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("Detection running …")

    def _on_stop(self) -> None:
        self._running = False
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText("Detection stopped")

    # ── Detection loop ─────────────────────────────────────────────────────

    def _poll(self) -> None:
        frame = self._get_frame()
        if frame is None or frame.size == 0:
            return

        dets = self._detector.detect(frame)
        annotated = self._detector.annotate_qt(frame, dets)
        self._last_dets = dets

        # ── Display annotated frame ─────────────────────────────────────
        H, W, _ = annotated.shape
        img = QImage(annotated.data, W, H, 3 * W, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._cam_label.setPixmap(pix)

        # ── Update detection list ───────────────────────────────────────
        self._det_list.clear()
        if not dets:
            item = QListWidgetItem("— no detections —")
            item.setForeground(QColor("#555"))
            self._det_list.addItem(item)
        else:
            for d in sorted(dets, key=lambda x: -x.confidence):
                w, h  = d.wh
                text  = (f"{d.label:<18} {d.confidence:.2f}  "
                         f"[{w}×{h}]  @ ({d.center[0]},{d.center[1]})")
                item  = QListWidgetItem(text)
                h_col = hash(d.label) & 0xFFFFFF
                item.setForeground(QColor(
                    (h_col >> 16) & 0xFF,
                    (h_col >>  8) & 0xFF,
                    h_col & 0xFF))
                self._det_list.addItem(item)

        n = len(dets)
        self._status_lbl.setText(
            f"{'No' if n == 0 else n} object{'s' if n != 1 else ''} detected")

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def detections(self) -> list[Detection]:
        return self._last_dets
