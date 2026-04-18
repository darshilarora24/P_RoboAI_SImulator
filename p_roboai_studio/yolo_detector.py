"""
yolo_detector.py  —  P_RoboAI Studio

YOLO object detection on MuJoCo rendered frames.
Uses ultralytics (YOLOv8 / YOLO11) when available; degrades gracefully.

Usage
-----
    det = YOLODetector()              # loads yolo11n.pt or yolov8n.pt
    results = det.detect(frame_rgb)   # numpy (H, W, 3) uint8
    # results: list[Detection]
    annotated = det.annotate(frame_rgb, results)  # draws boxes on a copy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class Detection:
    label:      str
    confidence: float
    x1: int; y1: int; x2: int; y2: int   # pixel bounding box

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def wh(self) -> tuple[int, int]:
        return self.x2 - self.x1, self.y2 - self.y1


class YOLODetector:
    """
    Parameters
    ----------
    model_name   : str   — 'yolo11n.pt', 'yolov8n.pt', or a local path
    conf_thresh  : float — minimum confidence to report a detection
    device       : str   — 'cpu', 'cuda', 'mps', or '' for auto
    """

    AVAILABLE = _ULTRALYTICS_AVAILABLE

    def __init__(self, model_name: str = "yolo11n.pt",
                 conf_thresh: float = 0.35,
                 device: str = "") -> None:
        self._conf   = conf_thresh
        self._model  = None
        self._error  = ""
        self._names: dict[int, str] = {}

        if not _ULTRALYTICS_AVAILABLE:
            self._error = "ultralytics not installed — pip install ultralytics"
            return

        try:
            self._model = _YOLO(model_name)
            if device:
                self._model.to(device)
            # Warm up
            dummy = np.zeros((64, 64, 3), np.uint8)
            self._model(dummy, verbose=False)
            self._names = self._model.names or {}
        except Exception as exc:
            self._error = str(exc)

    @property
    def ready(self) -> bool:
        return self._model is not None and not self._error

    @property
    def error(self) -> str:
        return self._error

    def set_conf(self, threshold: float) -> None:
        self._conf = float(np.clip(threshold, 0.01, 1.0))

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """
        Run detection on an RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray  shape (H, W, 3) uint8

        Returns
        -------
        list[Detection]  — empty list if ultralytics not available
        """
        if not self.ready:
            return []
        try:
            results = self._model(frame_rgb, conf=self._conf, verbose=False)
            detections: list[Detection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    label  = self._names.get(cls_id, str(cls_id))
                    detections.append(Detection(label, conf, x1, y1, x2, y2))
            return detections
        except Exception:
            return []

    def annotate(self, frame_rgb: np.ndarray,
                 detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes + labels onto a copy of the frame.
        Returns a new (H, W, 3) uint8 array.
        """
        import cv2  # optional; only used here
        out = frame_rgb.copy()
        for d in detections:
            # colour by class hash
            h    = hash(d.label) & 0xFFFFFF
            col  = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), col, 2)
            text = f"{d.label} {d.confidence:.2f}"
            ty   = max(d.y1 - 4, 14)
            cv2.putText(out, text, (d.x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return out

    def annotate_qt(self, frame_rgb: np.ndarray,
                    detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes using only numpy (no OpenCV dependency).
        Draws coloured 2-px borders directly into the pixel array.
        """
        out = frame_rgb.copy()
        H, W, _ = out.shape
        for d in detections:
            h   = hash(d.label) & 0xFFFFFF
            col = np.array([(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF],
                           np.uint8)
            x1, y1 = max(0, d.x1), max(0, d.y1)
            x2, y2 = min(W - 1, d.x2), min(H - 1, d.y2)
            for t in range(2):   # 2-pixel border
                if y1 + t < H: out[y1 + t, x1:x2] = col
                if y2 - t >= 0: out[y2 - t, x1:x2] = col
                if x1 + t < W: out[y1:y2, x1 + t] = col
                if x2 - t >= 0: out[y1:y2, x2 - t] = col
        return out
