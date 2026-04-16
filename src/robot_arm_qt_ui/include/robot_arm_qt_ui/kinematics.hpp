#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

// ---------------------------------------------------------------------------
// Analytical forward kinematics and numerical (damped-least-squares) inverse
// kinematics for the 4-DOF robot_arm.xml arm.
//
// Joint order:  [0] shoulder_yaw  (Z axis)
//               [1] shoulder_pitch (Y axis)
//               [2] elbow_pitch   (Y axis)
//               [3] wrist_pitch   (Y axis)
//
// Link chain (all offsets along local Z):
//   world → base_link             : +0.08 m  (base_link body pos)
//   base_link → shoulder_yaw_link : +0.08 m  (shoulder_yaw_link body pos)
//   shoulder_yaw_link → upper_arm : +0.28 m
//   upper_arm → forearm           : +0.32 m
//   forearm → wrist_link          : +0.26 m
//   wrist_link → ee_site (tool0)  : +0.18 m
// ---------------------------------------------------------------------------

namespace robot_arm_qt_ui
{
namespace kinematics
{

// 4×4 homogeneous transformation matrix (row-major)
using Mat4 = std::array<std::array<double, 4>, 4>;
using Vec3 = std::array<double, 3>;
using JointAngles = std::array<double, 4>;  // [yaw, pitch1, pitch2, pitch3]

// ---------------- Robot geometry constants ---------------------------------

constexpr double L0 = 0.16;   // world to shoulder_yaw joint (0.08+0.08)
constexpr double L1 = 0.28;   // shoulder_yaw to shoulder_pitch
constexpr double L2 = 0.32;   // shoulder_pitch to elbow_pitch
constexpr double L3 = 0.26;   // elbow_pitch to wrist_pitch
constexpr double L4 = 0.18;   // wrist_pitch to ee_site

constexpr JointAngles Q_MIN = {-3.14159, -1.8, -2.2, -1.8};
constexpr JointAngles Q_MAX = { 3.14159,  1.8,  2.2,  1.8};

// ---------------- Basic matrix primitives ----------------------------------

inline Mat4 identity()
{
  Mat4 m{};
  m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
  return m;
}

inline Mat4 matmul(const Mat4 & A, const Mat4 & B)
{
  Mat4 C{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

// Translation along Z by d
inline Mat4 transZ(double d)
{
  auto T = identity();
  T[2][3] = d;
  return T;
}

// Rotation around Z by theta
inline Mat4 rotZ(double theta)
{
  auto T = identity();
  T[0][0] =  std::cos(theta); T[0][1] = -std::sin(theta);
  T[1][0] =  std::sin(theta); T[1][1] =  std::cos(theta);
  return T;
}

// Rotation around Y by theta
inline Mat4 rotY(double theta)
{
  auto T = identity();
  T[0][0] =  std::cos(theta); T[0][2] = std::sin(theta);
  T[2][0] = -std::sin(theta); T[2][2] = std::cos(theta);
  return T;
}

// ---------------- Forward kinematics ---------------------------------------

// Returns T_world_ee for joint angles q = [yaw, pitch1, pitch2, pitch3].
// T = Tz(L0)·Rz(q0)·Tz(L1)·Ry(q1)·Tz(L2)·Ry(q2)·Tz(L3)·Ry(q3)·Tz(L4)
inline Mat4 computeFK(const JointAngles & q)
{
  return matmul(
    matmul(
      matmul(
        matmul(
          matmul(
            matmul(
              matmul(
                matmul(
                  transZ(L0),
                  rotZ(q[0])),
                transZ(L1)),
              rotY(q[1])),
            transZ(L2)),
          rotY(q[2])),
        transZ(L3)),
      rotY(q[3])),
    transZ(L4));
}

// Returns the 5 intermediate world frames:
//   [0] T_world_shoulderYaw_frame   (after Tz(L0)·Rz(q0))
//   [1] T_world_shoulderPitch_frame (above + Tz(L1)·Ry(q1))
//   [2] T_world_elbowPitch_frame    (above + Tz(L2)·Ry(q2))
//   [3] T_world_wristPitch_frame    (above + Tz(L3)·Ry(q3))
//   [4] T_world_ee                  (above + Tz(L4))
inline std::array<Mat4, 5> computeAllFrames(const JointAngles & q)
{
  std::array<Mat4, 5> F;
  F[0] = matmul(transZ(L0), rotZ(q[0]));
  F[1] = matmul(F[0], matmul(transZ(L1), rotY(q[1])));
  F[2] = matmul(F[1], matmul(transZ(L2), rotY(q[2])));
  F[3] = matmul(F[2], matmul(transZ(L3), rotY(q[3])));
  F[4] = matmul(F[3], transZ(L4));
  return F;
}

// Extract position [x, y, z] from a 4×4 homogeneous matrix
inline Vec3 getPosition(const Mat4 & T)
{
  return {T[0][3], T[1][3], T[2][3]};
}

// Extract the 3×3 rotation submatrix (row-major, 9 elements)
inline std::array<double, 9> getRotation(const Mat4 & T)
{
  return {
    T[0][0], T[0][1], T[0][2],
    T[1][0], T[1][1], T[1][2],
    T[2][0], T[2][1], T[2][2]
  };
}

// Convert rotation matrix (row-major, 9 elements) to RPY Euler angles
// using ZYX (yaw-pitch-roll) convention.  Returns [roll, pitch, yaw] in rad.
inline Vec3 rotationToRPY(const std::array<double, 9> & R)
{
  // R[row*3 + col]
  const double pitch = std::atan2(-R[6], std::sqrt(R[7] * R[7] + R[8] * R[8]));
  double roll  = 0.0;
  double yaw   = 0.0;
  if (std::abs(std::cos(pitch)) > 1e-6) {
    roll = std::atan2(R[7], R[8]);
    yaw  = std::atan2(R[3], R[0]);
  } else {
    // Gimbal lock: pitch = ±90°
    roll = std::atan2(-R[5], R[4]);
  }
  return {roll, pitch, yaw};
}

// Convert rotation matrix to quaternion [x, y, z, w]
inline std::array<double, 4> rotationToQuat(const std::array<double, 9> & R)
{
  // Shepperd's method
  const double trace = R[0] + R[4] + R[8];
  double qw, qx, qy, qz;
  if (trace > 0.0) {
    const double s = 0.5 / std::sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (R[7] - R[5]) * s;
    qy = (R[2] - R[6]) * s;
    qz = (R[3] - R[1]) * s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = 2.0 * std::sqrt(1.0 + R[0] - R[4] - R[8]);
    qw = (R[7] - R[5]) / s;
    qx = 0.25 * s;
    qy = (R[1] + R[3]) / s;
    qz = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = 2.0 * std::sqrt(1.0 + R[4] - R[0] - R[8]);
    qw = (R[2] - R[6]) / s;
    qx = (R[1] + R[3]) / s;
    qy = 0.25 * s;
    qz = (R[5] + R[7]) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + R[8] - R[0] - R[4]);
    qw = (R[3] - R[1]) / s;
    qx = (R[2] + R[6]) / s;
    qy = (R[5] + R[7]) / s;
    qz = 0.25 * s;
  }
  return {qx, qy, qz, qw};
}

// ---------------- Numerical Jacobian (3×4, position only) -----------------

// J[row][col]: d(position[row]) / d(q[col]),  rows=xyz, cols=joints
using Jacobian3x4 = std::array<std::array<double, 4>, 3>;

inline Jacobian3x4 computeJacobian(const JointAngles & q)
{
  constexpr double kDelta = 1e-5;
  const Vec3 p0 = getPosition(computeFK(q));
  Jacobian3x4 J{};
  for (int col = 0; col < 4; ++col) {
    JointAngles q_plus = q;
    q_plus[col] += kDelta;
    const Vec3 p_plus = getPosition(computeFK(q_plus));
    for (int row = 0; row < 3; ++row) {
      J[row][col] = (p_plus[row] - p0[row]) / kDelta;
    }
  }
  return J;
}

// ---------------- Inverse kinematics (damped least squares) ----------------

struct IKResult
{
  bool success{false};
  std::string message;
  JointAngles joint_angles{};
  double position_error{0.0};
  int iterations{0};
};

// Solves position-only IK for target = [x, y, z] using damped least squares.
// initial_q: seed joint angles (defaults to home pose if not provided)
inline IKResult computeIK(
  const Vec3 & target,
  const JointAngles & initial_q = {0.0, 0.45, -0.95, 0.6},
  int max_iterations = 300,
  double tolerance = 5e-4,   // 0.5 mm convergence threshold
  double alpha = 0.6,         // step size
  double lambda = 0.05        // damping factor (regularisation)
)
{
  JointAngles q = initial_q;

  for (int iter = 0; iter < max_iterations; ++iter) {
    const Vec3 p = getPosition(computeFK(q));
    const double dp[3] = {target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    const double err = std::sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]);

    if (err < tolerance) {
      return {true, "Converged", q, err, iter};
    }

    const Jacobian3x4 J = computeJacobian(q);

    // A = J·J^T + λ²·I  (3×3 symmetric)
    double A[3][3]{};
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < 4; ++j) {
          A[i][k] += J[i][j] * J[k][j];
        }
      }
      A[i][i] += lambda * lambda;
    }

    // Solve A·x = dp  via 3×3 analytic inverse
    const double det =
      A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    double x[3]{};
    if (std::abs(det) > 1e-14) {
      const double inv[3][3] = {
        {(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det, -(A[0][1]*A[2][2]-A[0][2]*A[2][1])/det,  (A[0][1]*A[1][2]-A[0][2]*A[1][1])/det},
        {-(A[1][0]*A[2][2]-A[1][2]*A[2][0])/det,  (A[0][0]*A[2][2]-A[0][2]*A[2][0])/det, -(A[0][0]*A[1][2]-A[0][2]*A[1][0])/det},
        { (A[1][0]*A[2][1]-A[1][1]*A[2][0])/det, -(A[0][0]*A[2][1]-A[0][1]*A[2][0])/det,  (A[0][0]*A[1][1]-A[0][1]*A[1][0])/det}
      };
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          x[i] += inv[i][k] * dp[k];
        }
      }
    }

    // Δq = J^T · x; clamp to joint limits
    for (int j = 0; j < 4; ++j) {
      double dq = 0.0;
      for (int i = 0; i < 3; ++i) {
        dq += J[i][j] * x[i];
      }
      q[j] = std::clamp(q[j] + alpha * dq, Q_MIN[j], Q_MAX[j]);
    }
  }

  // Final error check with relaxed threshold
  const Vec3 pf = getPosition(computeFK(q));
  const double final_err = std::sqrt(
    std::pow(target[0] - pf[0], 2) +
    std::pow(target[1] - pf[1], 2) +
    std::pow(target[2] - pf[2], 2));

  if (final_err < 0.02) {
    return {true, "Converged (relaxed)", q, final_err, max_iterations};
  }
  return {false, "Did not converge — target may be outside workspace", q, final_err, max_iterations};
}

}  // namespace kinematics
}  // namespace robot_arm_qt_ui
