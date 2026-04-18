"""
main.py  —  P_RoboAI Studio entry point
"""
import os
import sys

# MuJoCo's default GL backend (GLFW) creates its own window which conflicts
# with Qt.  EGL gives a headless offscreen context that works alongside Qt.
# Fall back to osmesa (software) if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

from main_window import MainWindow


def _dark_palette() -> QPalette:
    p = QPalette()
    base   = QColor(30,  30,  30)
    alt    = QColor(40,  40,  40)
    text   = QColor(220, 220, 220)
    mid    = QColor(60,  60,  60)
    hi     = QColor(42,  130, 218)
    hi_txt = QColor(255, 255, 255)

    p.setColor(QPalette.ColorRole.Window,          QColor(45, 45, 45))
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            base)
    p.setColor(QPalette.ColorRole.AlternateBase,   alt)
    p.setColor(QPalette.ColorRole.ToolTipBase,     base)
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          mid)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Highlight,       hi)
    p.setColor(QPalette.ColorRole.HighlightedText, hi_txt)
    p.setColor(QPalette.ColorRole.Link,            hi)
    p.setColor(QPalette.ColorRole.Mid,             mid)
    p.setColor(QPalette.ColorRole.Shadow,          QColor(20, 20, 20))
    return p


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("P_RoboAI Studio")
    app.setOrganizationName("P_RoboAI")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
