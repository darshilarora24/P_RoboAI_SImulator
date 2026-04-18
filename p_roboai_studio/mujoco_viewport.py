"""
mujoco_viewport.py  —  P_RoboAI Studio

QWidget that renders a MuJoCo scene via the offscreen Renderer API,
then blits the pixel array as a QImage.  Works on Windows/Linux/macOS
without requiring a Qt OpenGL context — MuJoCo manages its own GL/EGL context.

Mouse controls
--------------
  Left-drag          orbit (azimuth / elevation)
  Right-drag         zoom (distance)
  Middle-drag        pan (lookat point)
  Scroll             zoom
  Double-click       reset camera to default
"""
from __future__ import annotations

import numpy as np
import mujoco

from PyQt6.QtCore  import Qt, QTimer, QPoint
from PyQt6.QtGui   import QImage, QPainter, QCursor, QFont, QColor
from PyQt6.QtWidgets import QWidget


class _SafeRenderer(mujoco.Renderer):
    """
    Thin wrapper that swallows the AttributeError MuJoCo raises in __del__
    when __init__ failed before _gl_context was assigned.  This is a MuJoCo
    bug (missing try/except in __del__); the subclass is the least-invasive
    workaround.
    """
    def __del__(self) -> None:
        try:
            super().__del__()
        except AttributeError:
            pass

    def close(self) -> None:
        if not hasattr(self, "_gl_context"):
            return
        try:
            super().close()
        except Exception:
            pass


class MuJoCoViewport(QWidget):
    """
    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    parent : QWidget | None
    """

    # Default camera pose
    _DEF_DISTANCE  = 2.5
    _DEF_AZIMUTH   = 135.0
    _DEF_ELEVATION = -30.0
    _DEF_LOOKAT    = (0.0, 0.0, 0.4)


    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStyleSheet("background:#111;")

        self._model = model
        self._data  = data

        # MuJoCo rendering objects
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()
        self._scn = mujoco.MjvScene(model, maxgeom=2000)
        mujoco.mjv_defaultCamera(self._cam)
        mujoco.mjv_defaultOption(self._opt)
        self._reset_camera()

        # Renderer — created at first resize so size is correct
        self._renderer: mujoco.Renderer | None = None
        self._renderer_error: str = ""   # non-empty = don't retry

        # Mouse state
        self._drag_start  = QPoint()
        self._drag_button = Qt.MouseButton.NoButton
        self._drag_az     = self._cam.azimuth
        self._drag_el     = self._cam.elevation
        self._drag_dist   = self._cam.distance
        self._drag_lookat = list(self._cam.lookat)

        # Render refresh timer (~30 fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

        # Overlay message (shown briefly after load)
        self._overlay: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def replace_model(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Hot-swap model when user loads a new URDF."""
        self._model = model
        self._data  = data
        self._scn   = mujoco.MjvScene(model, maxgeom=2000)
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._renderer_error = ""   # allow retry with new model
        self._reset_camera()
        self.update()

    def show_overlay(self, text: str, ms: int = 3000) -> None:
        self._overlay = text
        QTimer.singleShot(ms, lambda: self._clear_overlay())

    def _clear_overlay(self) -> None:
        self._overlay = ""
        self.update()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _reset_camera(self) -> None:
        self._cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.distance  = self._DEF_DISTANCE
        self._cam.azimuth   = self._DEF_AZIMUTH
        self._cam.elevation = self._DEF_ELEVATION
        self._cam.lookat[:] = self._DEF_LOOKAT

    # ── Qt event overrides ────────────────────────────────────────────────────

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._renderer = None        # recreate at new widget size
        self._renderer_error = ""    # allow retry

    def mouseDoubleClickEvent(self, ev) -> None:
        self._reset_camera()
        self.update()

    def mousePressEvent(self, ev) -> None:
        self._drag_start  = ev.pos()
        self._drag_button = ev.button()
        self._drag_az     = float(self._cam.azimuth)
        self._drag_el     = float(self._cam.elevation)
        self._drag_dist   = float(self._cam.distance)
        self._drag_lookat = list(self._cam.lookat)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev) -> None:
        if self._drag_button == Qt.MouseButton.NoButton:
            return
        dx = ev.pos().x() - self._drag_start.x()
        dy = ev.pos().y() - self._drag_start.y()
        W, H = max(self.width(), 1), max(self.height(), 1)

        if self._drag_button == Qt.MouseButton.LeftButton:
            # Orbit
            self._cam.azimuth   = self._drag_az  - dx * 180.0 / W
            self._cam.elevation = float(np.clip(
                self._drag_el + dy * 90.0 / H, -89.0, -1.0))

        elif self._drag_button == Qt.MouseButton.RightButton:
            # Zoom via drag
            self._cam.distance = float(np.clip(
                self._drag_dist * (1.0 + dy / H * 2.0), 0.2, 30.0))

        elif self._drag_button == Qt.MouseButton.MiddleButton:
            # Pan lookat
            scale = self._cam.distance * 0.001
            az    = np.radians(self._cam.azimuth)
            el    = np.radians(self._cam.elevation)
            right = np.array([ np.cos(az), np.sin(az), 0.0])
            up    = np.array([-np.sin(az)*np.sin(el),
                               np.cos(az)*np.sin(el),
                               np.cos(el)])
            delta = -dx * scale * right + dy * scale * up
            for i in range(3):
                self._cam.lookat[i] = self._drag_lookat[i] + float(delta[i])

        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._drag_button = Qt.MouseButton.NoButton
        self.setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, ev) -> None:
        delta = ev.angleDelta().y()
        factor = 0.87 if delta > 0 else 1.0 / 0.87
        self._cam.distance = float(np.clip(
            self._cam.distance * factor, 0.2, 30.0))
        self.update()

    # ── Rendering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _try_backends(model, H: int, W: int) -> "_SafeRenderer":
        """Try EGL → osmesa in order; raise if both fail."""
        import os
        for backend in ("egl", "osmesa"):
            os.environ["MUJOCO_GL"] = backend
            try:
                return _SafeRenderer(model, height=H, width=W)
            except Exception:
                pass
        raise RuntimeError(
            "MuJoCo renderer failed with both EGL and osmesa.\n"
            "Install libegl1 (EGL) or libosmesa6 (software) and retry.")

    def _ensure_renderer(self) -> bool:
        if self._renderer_error:
            return False
        W, H = max(self.width(), 64), max(self.height(), 64)
        if self._renderer is None or \
           self._renderer.width != W or self._renderer.height != H:
            try:
                if self._renderer is not None:
                    self._renderer.close()
                import os
                if "MUJOCO_GL" in os.environ:
                    self._renderer = _SafeRenderer(self._model, height=H, width=W)
                else:
                    self._renderer = self._try_backends(self._model, H, W)
            except Exception as exc:
                self._renderer = None
                self._renderer_error = str(exc) or "GL context failed"
                return False
        return True

    def paintEvent(self, ev) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not self._ensure_renderer():
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.setPen(QColor(200, 80, 80))
            hint = ("Set MUJOCO_GL=osmesa for software rendering"
                    if "egl" in self._renderer_error.lower() else "")
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             f"MuJoCo renderer unavailable\n{self._renderer_error}\n{hint}")
            return

        # Update scene and render
        try:
            mujoco.mjv_updateScene(
                self._model, self._data, self._opt, None,
                self._cam, mujoco.mjtCatBit.mjCAT_ALL, self._scn)
            self._renderer.update_scene(self._data, self._cam, self._opt)
            pixels = self._renderer.render()          # (H, W, 3) uint8 RGB
            pixels = np.ascontiguousarray(np.flipud(pixels))  # MuJoCo is bottom-up
        except Exception as exc:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.setPen(QColor(255, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             f"Render error:\n{exc}")
            return

        H, W, _ = pixels.shape
        img = QImage(pixels.data, W, H, 3 * W, QImage.Format.Format_RGB888)
        # Scale fixed-size buffer to fill the widget (maintains aspect ratio is
        # not required here — we fill the full rect for an immersive view)
        painter.drawImage(0, 0, img)

        # HUD — camera info
        self._draw_hud(painter)

        # Overlay message
        if self._overlay:
            self._draw_overlay(painter)

    def _draw_hud(self, p: QPainter) -> None:
        txt = (f"Az {self._cam.azimuth:.1f}°  "
               f"El {self._cam.elevation:.1f}°  "
               f"Dist {self._cam.distance:.2f} m")
        p.setPen(QColor(160, 160, 160))
        f = QFont("Monospace", 8)
        p.setFont(f)
        p.drawText(8, self.height() - 6, txt)

    def _draw_overlay(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.width(), 36, QColor(0, 0, 0, 160))
        p.setPen(QColor(0, 230, 118))
        f = QFont("Sans Serif", 11)
        f.setBold(True)
        p.setFont(f)
        p.drawText(12, 24, self._overlay)
