#pragma once

#include <memory>
#include <vector>
#include <utility>   // std::pair

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QTimer>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

namespace robot_arm_qt_ui {

// Forward declaration
class AMRMapView;

// =============================================================================
//  AMRPanel  —  top-level widget: owns the ROS2 node, subscriptions, and layout
// =============================================================================
class AMRPanel : public QWidget
{
    Q_OBJECT
public:
    explicit AMRPanel(QWidget * parent = nullptr);
    ~AMRPanel() override;

private Q_SLOTS:
    void onGoClicked();

private:
    // ── ROS2 ────────────────────────────────────────────────────────────────
    rclcpp::Node::SharedPtr                                        node_;
    rclcpp::executors::SingleThreadedExecutor                      ros_exec_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr           path_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr   scan_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr  goal_pub_;

    // ── Shared state (updated in Qt main thread via ros_spin_timer_) ─────────
    double robot_x_     {1.0};
    double robot_y_     {1.0};
    double robot_theta_ {0.0};
    std::vector<std::pair<double, double>> path_pts_;
    float  scan_angle_min_  {0.f};
    float  scan_angle_inc_  {0.f};
    std::vector<float> scan_ranges_;

    // ── UI ──────────────────────────────────────────────────────────────────
    AMRMapView *     map_view_    {nullptr};
    QLabel *         status_label_{nullptr};
    QDoubleSpinBox * goal_x_spin_ {nullptr};
    QDoubleSpinBox * goal_y_spin_ {nullptr};
    QPushButton *    go_button_   {nullptr};

    QTimer * ros_spin_timer_{nullptr};
};

// =============================================================================
//  AMRMapView  —  QPainter-based 2-D occupancy-map widget
// =============================================================================
class AMRMapView : public QWidget
{
    Q_OBJECT
public:
    explicit AMRMapView(QWidget * parent = nullptr);

    // All setters are called from the Qt main thread only — no mutex needed.
    void setRobotPose(double x, double y, double theta);
    void setPath(const std::vector<std::pair<double, double>> & pts);
    void setScan(float angle_min, float angle_inc,
                 float robot_x, float robot_y, float robot_theta,
                 const std::vector<float> & ranges);

Q_SIGNALS:
    /// Emitted when the user left-clicks the map; world_x/y in metres.
    void goalClicked(double world_x, double world_y);

protected:
    void paintEvent(QPaintEvent * event) override;
    void mousePressEvent(QMouseEvent * event) override;
    QSize sizeHint()        const override { return {520, 520}; }
    QSize minimumSizeHint() const override { return {380, 380}; }

private:
    QPointF worldToScreen(double wx, double wy) const;

    double robot_x_    {1.0};
    double robot_y_    {1.0};
    double robot_theta_{0.0};

    std::vector<std::pair<double, double>> path_pts_;

    float scan_angle_min_{0.f};
    float scan_angle_inc_{0.f};
    float scan_rx_       {1.f};
    float scan_ry_       {1.f};
    float scan_rth_      {0.f};
    std::vector<float> scan_ranges_;

    static constexpr double kWorldSize = 10.0;   // metres
    static constexpr double kMargin    = 14.0;   // pixels
};

}  // namespace robot_arm_qt_ui
