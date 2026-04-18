#pragma once
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QWidget>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>

namespace p_roboai_viz {

class StatusPanel : public QWidget {
    Q_OBJECT
public:
    explicit StatusPanel(rclcpp::Node::SharedPtr node, QWidget* parent = nullptr);

    void onOdom     (const nav_msgs::msg::Odometry&     msg);
    void onNavStatus(const std_msgs::msg::String&        msg);
    void onImu      (const sensor_msgs::msg::Imu&        msg);

signals:
    void goalRequested(double x, double y, double theta);
    void cancelNav();

private slots:
    void sendGoal();

private:
    QLabel* makeValueLabel(const QString& init = "—");
    QWidget* makeGroup(const QString& title, QWidget* parent = nullptr);

    // ── Robot pose ─────────────────────────────────────────────────────────
    QLabel* _lbl_x{nullptr};
    QLabel* _lbl_y{nullptr};
    QLabel* _lbl_theta{nullptr};
    QLabel* _lbl_v{nullptr};
    QLabel* _lbl_w{nullptr};

    // ── IMU ────────────────────────────────────────────────────────────────
    QLabel* _lbl_roll{nullptr};
    QLabel* _lbl_pitch{nullptr};
    QLabel* _lbl_yaw{nullptr};

    // ── Navigation ─────────────────────────────────────────────────────────
    QLabel*        _lbl_nav_status{nullptr};
    QDoubleSpinBox* _goal_x{nullptr};
    QDoubleSpinBox* _goal_y{nullptr};
    QDoubleSpinBox* _goal_theta{nullptr};
    QPushButton*   _btn_send{nullptr};
    QPushButton*   _btn_cancel{nullptr};

    // ── ROS ────────────────────────────────────────────────────────────────
    rclcpp::Node::SharedPtr _node;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _goal_pub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       _odom_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         _status_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr         _imu_sub;
};

} // namespace p_roboai_viz
