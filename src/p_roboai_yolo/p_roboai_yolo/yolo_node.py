"""
yolo_node.py  —  P_RoboAI YOLO

ROS2 node that runs YOLO object detection on camera images.

Subscribes:   /camera/image_raw    sensor_msgs/Image   (BGR or RGB)
              /amr/camera/image    sensor_msgs/Image   (AMR camera fallback)

Publishes:    /yolo/image          sensor_msgs/Image   (annotated, RGB)
              /yolo/detections     std_msgs/String     (JSON detection list)
              /yolo/status         std_msgs/String

Parameters
----------
  image_topic : str    — topic to subscribe to (default: /camera/image_raw)
  model_name  : str    — YOLO model ('yolo11n.pt', 'yolov8n.pt', …)
  conf_thresh : float  — confidence threshold (0.01 – 0.99)
  detect_hz   : float  — max detection rate in Hz (default 10)
  publish_annotated : bool — also publish /yolo/image (default True)

If the subscribed image topic receives no messages for 3 s and MuJoCo is
available, the node renders its own camera frame from the AMR warehouse model
and runs detection on that instead.
"""
from __future__ import annotations

import json
import os
import site
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ── venv bootstrap ────────────────────────────────────────────────────────────

def _bootstrap_venv() -> None:
    for root in Path(__file__).resolve().parents:
        venv = root / ".venv"
        if venv.exists():
            for sp in venv.glob("lib/python*/site-packages"):
                if str(sp) not in sys.path:
                    site.addsitedir(str(sp))
                    sys.path.insert(0, str(sp))
            break

_bootstrap_venv()

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    import mujoco
    _MUJOCO_OK = True
except ImportError:
    _MUJOCO_OK = False


# ── YOLO wrapper ──────────────────────────────────────────────────────────────

class _Detector:
    def __init__(self, model_name: str, conf: float) -> None:
        self._model = None
        self._names: dict[int, str] = {}
        self._conf  = conf
        if _YOLO_OK:
            try:
                self._model = _YOLO(model_name)
                dummy = np.zeros((64, 64, 3), np.uint8)
                self._model(dummy, verbose=False)
                self._names = self._model.names or {}
            except Exception as exc:
                self._model = None
                self._err = str(exc)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, frame_rgb: np.ndarray) -> list[dict]:
        if not self.ready:
            return []
        try:
            results = self._model(frame_rgb, conf=self._conf, verbose=False)
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    dets.append({
                        "label": self._names.get(cls_id, str(cls_id)),
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                    })
            return dets
        except Exception:
            return []

    def annotate(self, frame_rgb: np.ndarray, dets: list[dict]) -> np.ndarray:
        out = frame_rgb.copy()
        H, W, _ = out.shape
        for d in dets:
            h   = hash(d["label"]) & 0xFFFFFF
            col = np.array([(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF], np.uint8)
            x1, y1, x2, y2 = d["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            for t in range(2):
                if y1 + t < H: out[y1 + t, x1:x2] = col
                if y2 - t >= 0: out[y2 - t, x1:x2] = col
                if x1 + t < W: out[y1:y2, x1 + t] = col
                if x2 - t >= 0: out[y1:y2, x2 - t] = col
        return out


# ── MuJoCo fallback camera renderer ──────────────────────────────────────────

class _MuJoCoCamera:
    """Renders a fixed overhead camera view from the warehouse XML."""

    def __init__(self, xml_path: str) -> None:
        self._ok = False
        if not _MUJOCO_OK:
            return
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            self._model = mujoco.MjModel.from_xml_path(xml_path)
            self._data  = mujoco.MjData(self._model)
            mujoco.mj_forward(self._model, self._data)
            self._cam = mujoco.MjvCamera()
            self._opt = mujoco.MjvOption()
            self._scn = mujoco.MjvScene(self._model, maxgeom=2000)
            mujoco.mjv_defaultCamera(self._cam)
            mujoco.mjv_defaultOption(self._opt)
            self._cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            self._cam.distance  = 8.0
            self._cam.azimuth   = 90.0
            self._cam.elevation = -70.0
            self._cam.lookat[:] = [0.0, 0.0, 0.0]
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._ok = True
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return self._ok

    def render(self) -> np.ndarray:
        mujoco.mjv_updateScene(self._model, self._data, self._opt, None,
                               self._cam, mujoco.mjtCatBit.mjCAT_ALL, self._scn)
        self._renderer.update_scene(self._data, self._cam, self._opt)
        pixels = self._renderer.render()
        return np.ascontiguousarray(np.flipud(pixels))


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class YOLONode(Node):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("image_topic",       "/camera/image_raw")
        self.declare_parameter("model_name",        "yolo11n.pt")
        self.declare_parameter("conf_thresh",       0.35)
        self.declare_parameter("detect_hz",         10.0)
        self.declare_parameter("publish_annotated", True)

        img_topic  = self.get_parameter("image_topic").value
        model_name = self.get_parameter("model_name").value
        conf       = float(self.get_parameter("conf_thresh").value)
        hz         = float(self.get_parameter("detect_hz").value)
        self._pub_ann = self.get_parameter("publish_annotated").value

        self._pub_img  = self.create_publisher(Image,  "/yolo/image",      10)
        self._pub_dets = self.create_publisher(String, "/yolo/detections",  10)
        self._pub_stat = self.create_publisher(String, "/yolo/status",      10)

        if not _NP_OK:
            self._pub_stat.publish(String(data="ERROR: numpy not available"))
            return

        self._detector  = _Detector(model_name, conf)
        self._last_frame: np.ndarray | None = None
        self._last_img_t = 0.0
        self._mj_cam: _MuJoCoCamera | None = None
        self._interval  = 1.0 / max(1.0, hz)
        self._last_det_t = 0.0

        if self._detector.ready:
            self._pub_stat.publish(String(data=f"YOLO ready — model: {model_name}"))
        else:
            self._pub_stat.publish(String(data=(
                f"⚠ YOLO unavailable — pip install ultralytics\n"
                "Detection disabled; annotated frames not published")))

        # Subscribe to camera image
        self.create_subscription(Image, img_topic,          self._on_image, 10)
        self.create_subscription(Image, "/amr/camera/image", self._on_image, 10)

        # Poll timer — runs detection at configured rate
        self.create_timer(self._interval, self._tick)

        # Fallback: try to build MuJoCo renderer after 3 s if no image arrives
        self.create_timer(3.0, self._check_fallback)

        self.get_logger().info(
            f"YOLO node started  model={model_name}  topic={img_topic}"
            f"  {hz:.0f} Hz")

    # ── Image callback ────────────────────────────────────────────────────────

    def _on_image(self, msg: Image) -> None:
        self._last_img_t = time.time()
        try:
            frame = np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width, -1)
            # Convert BGR→RGB if needed (ROS images are typically BGR)
            if msg.encoding in ("bgr8", "bgr"):
                frame = frame[:, :, ::-1].copy()
            elif msg.encoding in ("rgb8", "rgb"):
                frame = frame.copy()
            else:
                frame = frame[:, :, :3].copy()
            self._last_frame = frame
        except Exception as exc:
            self.get_logger().warn(f"Image decode error: {exc}")

    # ── Fallback: MuJoCo overhead view ───────────────────────────────────────

    def _check_fallback(self) -> None:
        if time.time() - self._last_img_t > 2.9:
            self._try_mujoco_fallback()

    def _try_mujoco_fallback(self) -> None:
        if self._mj_cam is not None or not _MUJOCO_OK:
            return
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory("robot_amr_mujoco_sim")
            xml_path = os.path.join(share, "models", "amr_warehouse.xml")
        except Exception:
            for p in Path("/home").rglob("amr_warehouse.xml"):
                xml_path = str(p)
                break
            else:
                return

        cam = _MuJoCoCamera(xml_path)
        if cam.ready:
            self._mj_cam = cam
            self._pub_stat.publish(String(data=(
                "No camera topic received — using MuJoCo overhead view")))
            self.get_logger().info("Using MuJoCo fallback camera")

    # ── Detection tick ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.time()

        # Get frame: live topic or MuJoCo fallback
        frame = self._last_frame
        if frame is None and self._mj_cam and self._mj_cam.ready:
            try:
                frame = self._mj_cam.render()
            except Exception:
                return

        if frame is None or not self._detector.ready:
            return

        dets = self._detector.detect(frame)

        # Publish JSON detections
        self._pub_dets.publish(String(data=json.dumps({
            "timestamp": now,
            "count":     len(dets),
            "detections": dets,
        })))

        # Publish annotated image
        if self._pub_ann:
            annotated = self._detector.annotate(frame, dets)
            H, W, _   = annotated.shape
            img_msg   = Image()
            img_msg.header.stamp    = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera"
            img_msg.height   = H
            img_msg.width    = W
            img_msg.encoding = "rgb8"
            img_msg.step     = W * 3
            img_msg.data     = annotated.tobytes()
            self._pub_img.publish(img_msg)

        if dets:
            labels = ", ".join(f"{d['label']}({d['confidence']:.2f})" for d in dets)
            self._pub_stat.publish(String(data=f"Detected: {labels}"))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
