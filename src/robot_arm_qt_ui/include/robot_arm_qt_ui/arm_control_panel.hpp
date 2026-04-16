#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <QWidget>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "robot_arm_qt_ui/kinematics.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QTimer;

namespace robot_arm_qt_ui
{

class MujocoViewportWidget;

struct JointDefinition
{
  std::string name;
  double min_position_rad;
  double max_position_rad;
  double home_position_rad;
};

struct JointWidgets
{
  QSlider * slider{nullptr};
  QLabel * target_value_label{nullptr};
  QLabel * current_value_label{nullptr};
};

class ArmControlPanel : public QWidget
{
public:
  explicit ArmControlPanel(QWidget * parent = nullptr);
  ~ArmControlPanel() override = default;

private:
  // ── UI construction ───────────────────────────────────────────────────────
  void buildUi(const QString & model_path);
  void createRosInterfaces();
  void connectUi();

  // ── Simulation / joint control ────────────────────────────────────────────
  void setStatus(const QString & message);
  void syncSlidersToHome(bool apply_after_sync);
  void setSliderTargets(const std::vector<double> & positions);
  void refreshTargetValue(std::size_t index);
  void refreshAllTargetValues();
  void applySliderTargets(const QString & origin, bool update_status);
  void publishSimulationState();
  void requestSimulationReset();
  void handlePositionCommand(const std_msgs::msg::Float64MultiArray::SharedPtr message);
  void handleTrajectoryCommand(const trajectory_msgs::msg::JointTrajectory::SharedPtr message);
  builtin_interfaces::msg::Time simTimeToMsg(double time_seconds) const;
  int radiansToSliderValue(double radians) const;
  double sliderValueToRadians(int slider_value) const;
  QString formatAngle(double radians) const;

  // ── Kinematics ────────────────────────────────────────────────────────────
  // Called every time joint positions are updated; refreshes FK matrix display.
  void updateFKDisplay(const std::array<double, 4> & joint_angles);
  // Solves IK for the XYZ target entered by the user and displays the result.
  void solveIK();
  // Applies the last successful IK solution to the robot.
  void applyIKSolution();

  // ── ROS 2 interfaces ──────────────────────────────────────────────────────
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::executors::SingleThreadedExecutor ros_executor_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_publisher_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_publisher_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr position_command_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr position_command_alias_subscription_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_subscription_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_service_;

  // ── Qt timers ─────────────────────────────────────────────────────────────
  QTimer * ros_spin_timer_{nullptr};
  QTimer * auto_apply_timer_{nullptr};
  QTimer * state_publish_timer_{nullptr};

  // ── Existing UI widgets ───────────────────────────────────────────────────
  QCheckBox * auto_apply_checkbox_{nullptr};
  QLabel * status_label_{nullptr};
  QLabel * ee_position_label_{nullptr};
  QLabel * ee_orientation_label_{nullptr};
  MujocoViewportWidget * viewport_{nullptr};

  // ── Kinematics UI widgets ─────────────────────────────────────────────────
  // FK: 4×4 matrix cell labels + frame selector
  QLabel * fk_matrix_labels_[4][4]{};
  QLabel * fk_position_label_{nullptr};   // "p = [x, y, z]" summary
  QLabel * fk_rpy_label_{nullptr};        // "rpy = [r, p, y] deg" summary
  QComboBox * fk_frame_combo_{nullptr};   // which intermediate frame to display

  // IK: target input, result, status
  QDoubleSpinBox * ik_x_spin_{nullptr};
  QDoubleSpinBox * ik_y_spin_{nullptr};
  QDoubleSpinBox * ik_z_spin_{nullptr};
  QLabel * ik_result_label_{nullptr};     // joint angle results text
  QLabel * ik_status_label_{nullptr};     // "Converged / error: 0.2 mm"
  QPushButton * ik_apply_button_{nullptr};

  // Last IK solution (applied when user clicks Apply)
  kinematics::JointAngles ik_solution_{};
  bool ik_solution_valid_{false};

  // ── Configuration / state ─────────────────────────────────────────────────
  double state_pub_rate_hz_{50.0};
  bool publish_clock_{true};

  std::vector<JointDefinition> joint_definitions_;
  std::vector<JointWidgets> joint_widgets_;
  std::vector<double> latest_joint_positions_;
};

}  // namespace robot_arm_qt_ui
