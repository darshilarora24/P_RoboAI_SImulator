"""
AMR MuJoCo simulation node  (P_RoboAI_Simulator).

Runs a full 3-D MuJoCo physics simulation of a differential-drive AMR
inside the warehouse XML model.  All sensor data is derived from the
MuJoCo state rather than a purely kinematic approximation.

Topics
------
Subscribes:   /amr/cmd_vel    (geometry_msgs/Twist)
Publishes:    /amr/odom       (nav_msgs/Odometry)        — wheel-encoder dead-reckoning
              /amr/scan       (sensor_msgs/LaserScan)    — mj_ray() 360-ray lidar
              /amr/imu        (sensor_msgs/Imu)          — MuJoCo gyro + acc sensors
              /joint_states   (sensor_msgs/JointState)   — wheel angles for RSP

TF:           odom → amr_base_link
"""
from __future__ import annotations

import math
import os

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from tf2_ros import TransformBroadcaster

# ── Lidar parameters ─────────────────────────────────────────────────────────
LIDAR_RAYS   = 360
LIDAR_RANGE  = 8.0      # metres — matches XML sensor range_max
LIDAR_HZ     = 10.0     # scan publish rate

# ── Drive parameters ──────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.10     # m
WHEEL_SEP    = 0.50     # m  centre-to-centre
MAX_VEL      = 1.2      # m/s  (linear)
MAX_OMG      = 2.5      # rad/s (angular)

# Geomgroup bitmask: only cast rays against group 0 (world/obstacles).
# Robot geoms are in group 1 (chassis/wheels) or group 2 (visual) → excluded.
_GEOMGROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class AMRMuJoCoNode(Node):
    def __init__(self) -> None:
        super().__init__("amr_mujoco_sim")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("start_x",     1.0)
        self.declare_parameter("start_y",     1.0)
        self.declare_parameter("start_theta", 0.0)
        self.declare_parameter("sim_rate_hz", 200.0)

        sx     = float(self.get_parameter("start_x").value)
        sy     = float(self.get_parameter("start_y").value)
        stheta = float(self.get_parameter("start_theta").value)
        sim_hz = float(self.get_parameter("sim_rate_hz").value)

        # ── Load MuJoCo model ─────────────────────────────────────────────────
        pkg_share = get_package_share_directory("robot_amr_mujoco_sim")
        xml_path  = os.path.join(pkg_share, "models", "amr_warehouse.xml")
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data  = mujoco.MjData(self._model)

        # Set start pose via freejoint qpos (pos=3 + quat=4)
        # qpos layout for freejoint: [x, y, z, qw, qx, qy, qz, left_wheel, right_wheel, caster(4)]
        root_id      = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        root_qpos_adr = self._model.jnt_qposadr[root_id]
        self._data.qpos[root_qpos_adr + 0] = sx
        self._data.qpos[root_qpos_adr + 1] = sy
        self._data.qpos[root_qpos_adr + 2] = 0.10            # chassis height
        half = stheta / 2.0
        self._data.qpos[root_qpos_adr + 3] = math.cos(half)  # qw
        self._data.qpos[root_qpos_adr + 4] = 0.0
        self._data.qpos[root_qpos_adr + 5] = 0.0
        self._data.qpos[root_qpos_adr + 6] = math.sin(half)  # qz
        mujoco.mj_forward(self._model, self._data)

        # ── Cache body / joint / site / actuator IDs ──────────────────────────
        self._base_body_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY,      "amr_base")
        self._lidar_site_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE,      "lidar_site")
        self._imu_site_id    = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE,      "imu_site")
        self._left_act_id    = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR,  "left_vel")
        self._right_act_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR,  "right_vel")
        self._left_jnt_id    = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT,     "left_wheel_joint")
        self._right_jnt_id   = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT,     "right_wheel_joint")
        self._root_jnt_id    = root_id

        # Sensor data addresses
        self._s_quat = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        self._s_gyro = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self._s_acc  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")

        # ── Odometry state (dead-reckoning from wheel joints) ─────────────────
        self._odom_x     = sx
        self._odom_y     = sy
        self._odom_theta = stheta

        self._prev_left_angle  = float(
            self._data.qpos[self._model.jnt_qposadr[self._left_jnt_id]])
        self._prev_right_angle = float(
            self._data.qpos[self._model.jnt_qposadr[self._right_jnt_id]])

        # ── Simulation timing ─────────────────────────────────────────────────
        self._sim_dt       = 1.0 / sim_hz
        self._mj_dt        = float(self._model.opt.timestep)
        self._steps_per_tick = max(1, round(self._sim_dt / self._mj_dt))

        # ── Velocity command ──────────────────────────────────────────────────
        self._cmd_v  = 0.0
        self._cmd_w  = 0.0

        # ── Lidar scan rate ───────────────────────────────────────────────────
        self._scan_period   = 1.0 / LIDAR_HZ
        self._scan_accum    = 0.0
        self._angle_inc     = 2.0 * math.pi / LIDAR_RAYS
        self._ray_geomid    = np.zeros(1, dtype=np.int32)

        # Pre-compute ray direction table (360 unit vectors in lidar frame)
        angles = np.linspace(-math.pi, math.pi, LIDAR_RAYS, endpoint=False)
        self._ray_dirs_local = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(LIDAR_RAYS)], axis=1
        )  # shape (360, 3)

        # ── ROS2 interfaces ───────────────────────────────────────────────────
        self._cmd_sub   = self.create_subscription(
            Twist, "/amr/cmd_vel", self._cmd_cb, 10)
        self._odom_pub  = self.create_publisher(Odometry,   "/amr/odom",      10)
        self._scan_pub  = self.create_publisher(LaserScan,  "/amr/scan",      10)
        self._imu_pub   = self.create_publisher(Imu,        "/amr/imu",       10)
        self._js_pub    = self.create_publisher(JointState, "/joint_states",  10)
        self._tf_broad  = TransformBroadcaster(self)

        self._timer = self.create_timer(self._sim_dt, self._step)
        self.get_logger().info(
            f"AMR MuJoCo sim ready — start ({sx:.2f}, {sy:.2f}, "
            f"{math.degrees(stheta):.1f}°)  "
            f"mj_dt={self._mj_dt*1e3:.1f} ms  steps/tick={self._steps_per_tick}")

    # ── Command callback ──────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist) -> None:
        self._cmd_v = float(np.clip(msg.linear.x,  -MAX_VEL, MAX_VEL))
        self._cmd_w = float(np.clip(msg.angular.z, -MAX_OMG, MAX_OMG))

    # ── Main simulation tick ──────────────────────────────────────────────────

    def _step(self) -> None:
        # Convert (v, ω) → wheel angular-velocity targets [rad/s]
        wl = (self._cmd_v - self._cmd_w * WHEEL_SEP / 2.0) / WHEEL_RADIUS
        wr = (self._cmd_v + self._cmd_w * WHEEL_SEP / 2.0) / WHEEL_RADIUS
        self._data.ctrl[self._left_act_id]  = wl
        self._data.ctrl[self._right_act_id] = wr

        # Step MuJoCo physics N times
        for _ in range(self._steps_per_tick):
            mujoco.mj_step(self._model, self._data)

        # ── Dead-reckoning odometry ───────────────────────────────────────────
        la_adr = self._model.jnt_qposadr[self._left_jnt_id]
        ra_adr = self._model.jnt_qposadr[self._right_jnt_id]
        lv_adr = self._model.jnt_dofadr[self._left_jnt_id]
        rv_adr = self._model.jnt_dofadr[self._right_jnt_id]

        # Wheel angular velocities → linear velocities
        vl = float(self._data.qvel[lv_adr]) * WHEEL_RADIUS
        vr = float(self._data.qvel[rv_adr]) * WHEEL_RADIUS
        v  = (vl + vr) / 2.0
        w  = (vr - vl) / WHEEL_SEP

        dt = self._sim_dt
        self._odom_theta += w * dt
        self._odom_theta = (self._odom_theta + math.pi) % (2 * math.pi) - math.pi
        self._odom_x    += v * math.cos(self._odom_theta) * dt
        self._odom_y    += v * math.sin(self._odom_theta) * dt

        now = self.get_clock().now().to_msg()
        self._publish_odom(now, v, w)
        self._publish_imu(now)
        self._publish_joint_states(now)

        # Lidar at lower rate
        self._scan_accum += self._sim_dt
        if self._scan_accum >= self._scan_period:
            self._scan_accum -= self._scan_period
            self._publish_scan()

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_odom(self, now, v: float, w: float) -> None:
        θ  = self._odom_theta
        qw = math.cos(θ / 2.0)
        qz = math.sin(θ / 2.0)

        # TF odom → amr_base_link
        tf = TransformStamped()
        tf.header.stamp    = now
        tf.header.frame_id = "odom"
        tf.child_frame_id  = "amr_base_link"
        tf.transform.translation.x = self._odom_x
        tf.transform.translation.y = self._odom_y
        tf.transform.rotation.w    = qw
        tf.transform.rotation.z    = qz
        self._tf_broad.sendTransform(tf)

        # Odometry message
        odom = Odometry()
        odom.header.stamp            = now
        odom.header.frame_id         = "odom"
        odom.child_frame_id          = "amr_base_link"
        odom.pose.pose.position.x    = self._odom_x
        odom.pose.pose.position.y    = self._odom_y
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.z = qz
        odom.twist.twist.linear.x    = v
        odom.twist.twist.angular.z   = w
        self._odom_pub.publish(odom)

    def _publish_scan(self) -> None:
        """Cast 360 rays with mj_ray() from the lidar site."""
        lidar_pos = self._data.site_xpos[self._lidar_site_id].copy()
        lidar_mat = self._data.site_xmat[self._lidar_site_id].reshape(3, 3).copy()

        # Transform all ray directions from lidar-local to world frame at once
        ray_dirs_world = (lidar_mat @ self._ray_dirs_local.T).T  # (360, 3)

        ranges: list[float] = []
        for i in range(LIDAR_RAYS):
            d = mujoco.mj_ray(
                self._model, self._data,
                lidar_pos,
                ray_dirs_world[i],
                _GEOMGROUP,
                1,                     # include static geoms
                self._base_body_id,    # exclude chassis geom
                self._ray_geomid,
            )
            ranges.append(LIDAR_RANGE if d < 0 else min(d, LIDAR_RANGE))

        scan = LaserScan()
        scan.header.stamp    = self.get_clock().now().to_msg()
        scan.header.frame_id = "lidar_link"
        scan.angle_min       = -math.pi
        scan.angle_max       =  math.pi
        scan.angle_increment = self._angle_inc
        scan.range_min       = 0.12
        scan.range_max       = LIDAR_RANGE
        scan.ranges          = [float(r) for r in ranges]
        self._scan_pub.publish(scan)

    def _publish_imu(self, now) -> None:
        # Read sensor sensordata
        qw_a = self._model.sensor_adr[self._s_quat]
        gw_a = self._model.sensor_adr[self._s_gyro]
        aw_a = self._model.sensor_adr[self._s_acc]

        quat = self._data.sensordata[qw_a: qw_a + 4]  # w, x, y, z
        gyro = self._data.sensordata[gw_a: gw_a + 3]
        acc  = self._data.sensordata[aw_a: aw_a + 3]

        imu = Imu()
        imu.header.stamp    = now
        imu.header.frame_id = "amr_base_link"
        imu.orientation.w   = float(quat[0])
        imu.orientation.x   = float(quat[1])
        imu.orientation.y   = float(quat[2])
        imu.orientation.z   = float(quat[3])
        imu.angular_velocity.x    = float(gyro[0])
        imu.angular_velocity.y    = float(gyro[1])
        imu.angular_velocity.z    = float(gyro[2])
        imu.linear_acceleration.x = float(acc[0])
        imu.linear_acceleration.y = float(acc[1])
        imu.linear_acceleration.z = float(acc[2])
        self._imu_pub.publish(imu)

    def _publish_joint_states(self, now) -> None:
        la_adr = self._model.jnt_qposadr[self._left_jnt_id]
        ra_adr = self._model.jnt_qposadr[self._right_jnt_id]

        js = JointState()
        js.header.stamp = now
        js.name     = ["left_wheel_joint", "right_wheel_joint"]
        js.position = [float(self._data.qpos[la_adr]),
                       float(self._data.qpos[ra_adr])]
        js.velocity = [float(self._data.qvel[self._model.jnt_dofadr[self._left_jnt_id]]),
                       float(self._data.qvel[self._model.jnt_dofadr[self._right_jnt_id]])]
        self._js_pub.publish(js)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node: AMRMuJoCoNode | None = None
    try:
        node = AMRMuJoCoNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
