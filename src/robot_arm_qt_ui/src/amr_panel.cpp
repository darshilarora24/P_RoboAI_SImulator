/**
 * amr_panel.cpp — AMR Studio: Qt 2-D map panel.
 *
 * Renders the 10 m × 10 m floor with obstacles, the A* path, lidar scan, and
 * the robot position. Left-clicking the map publishes a /amr/goal_pose.
 */
#include "robot_arm_qt_ui/amr_panel.hpp"

#include <array>
#include <cmath>
#include <chrono>

#include <QGroupBox>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QVBoxLayout>

namespace robot_arm_qt_ui {

// ---------------------------------------------------------------------------
// Obstacle table — must match amr_sim_node.py / amr_navigation_node.py
// Each entry: { x_min, x_max, y_min, y_max }
// ---------------------------------------------------------------------------
struct ObstRect { double x1, x2, y1, y2; };

static constexpr std::array<ObstRect, 10> kObstacles{{
    {0.0, 10.0,  0.0,  0.3},   // south wall
    {0.0, 10.0,  9.7, 10.0},   // north wall
    {0.0,  0.3,  0.0, 10.0},   // west wall
    {9.7, 10.0,  0.0, 10.0},   // east wall
    {3.0,  3.3,  1.5,  6.0},   // interior wall A
    {5.0,  8.2,  4.0,  4.3},   // interior wall B
    {1.5,  2.5,  6.5,  7.5},   // box 1
    {7.0,  8.0,  1.0,  2.0},   // box 2
    {6.0,  7.0,  6.5,  7.5},   // box 3
    {4.5,  5.5,  1.5,  2.5},   // box 4
}};

// ===========================================================================
//  AMRMapView
// ===========================================================================

AMRMapView::AMRMapView(QWidget * parent) : QWidget(parent)
{
    setMinimumSize(380, 380);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMouseTracking(false);
    setCursor(Qt::CrossCursor);
}

// ── State setters ────────────────────────────────────────────────────────────

void AMRMapView::setRobotPose(double x, double y, double theta)
{
    robot_x_ = x;  robot_y_ = y;  robot_theta_ = theta;
    update();
}

void AMRMapView::setPath(const std::vector<std::pair<double, double>> & pts)
{
    path_pts_ = pts;
    update();
}

void AMRMapView::setScan(float angle_min, float angle_inc,
                         float rx, float ry, float rth,
                         const std::vector<float> & ranges)
{
    scan_angle_min_ = angle_min;
    scan_angle_inc_ = angle_inc;
    scan_rx_ = rx;  scan_ry_ = ry;  scan_rth_ = rth;
    scan_ranges_    = ranges;
    update();
}

// ── Coordinate helpers ───────────────────────────────────────────────────────

QPointF AMRMapView::worldToScreen(double wx, double wy) const
{
    const double map_px = std::min(width(), height()) - 2.0 * kMargin;
    const double scale  = map_px / kWorldSize;
    return QPointF(kMargin + wx * scale,
                   height() - kMargin - wy * scale);
}

// ── Input ────────────────────────────────────────────────────────────────────

void AMRMapView::mousePressEvent(QMouseEvent * event)
{
    if (event->button() == Qt::LeftButton) {
        const double map_px = std::min(width(), height()) - 2.0 * kMargin;
        const double scale  = map_px / kWorldSize;
        const double wx = (event->pos().x() - kMargin) / scale;
        const double wy = (height() - kMargin - event->pos().y()) / scale;
        Q_EMIT goalClicked(
            std::max(0.0, std::min(kWorldSize, wx)),
            std::max(0.0, std::min(kWorldSize, wy)));
    }
}

// ── Paint ────────────────────────────────────────────────────────────────────

void AMRMapView::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const double map_px = std::min(width(), height()) - 2.0 * kMargin;
    const double scale  = map_px / kWorldSize;   // px / m

    // ── Background ──────────────────────────────────────────────────────────
    p.fillRect(rect(), QColor(248, 248, 250));

    // ── 1-metre grid ────────────────────────────────────────────────────────
    p.setPen(QPen(QColor(215, 215, 215), 0.5));
    for (int i = 0; i <= static_cast<int>(kWorldSize); ++i) {
        p.drawLine(worldToScreen(i, 0),         worldToScreen(i, kWorldSize));
        p.drawLine(worldToScreen(0, i),         worldToScreen(kWorldSize, i));
    }

    // ── Obstacles ────────────────────────────────────────────────────────────
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(65, 65, 75));
    for (const auto & o : kObstacles) {
        // world y_max → top of rect in screen coords
        p.drawRect(QRectF(worldToScreen(o.x1, o.y2),
                          worldToScreen(o.x2, o.y1)));
    }

    // ── Lidar scan ───────────────────────────────────────────────────────────
    if (!scan_ranges_.empty()) {
        // Rays (semi-transparent red)
        p.setPen(QPen(QColor(255, 80, 80, 45), 1.0));
        for (std::size_t i = 0; i < scan_ranges_.size(); ++i) {
            const float rng = scan_ranges_[i];
            if (rng <= 0.01f) continue;
            const double ang = scan_rth_ + scan_angle_min_ + i * scan_angle_inc_;
            const double hx  = scan_rx_ + rng * std::cos(ang);
            const double hy  = scan_ry_ + rng * std::sin(ang);
            p.drawLine(worldToScreen(scan_rx_, scan_ry_),
                       worldToScreen(hx, hy));
        }
        // Hit points (skip max-range rays — no obstacle hit)
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(220, 40, 40, 140));
        for (std::size_t i = 0; i < scan_ranges_.size(); ++i) {
            const float rng = scan_ranges_[i];
            if (rng >= 5.9f) continue;
            const double ang = scan_rth_ + scan_angle_min_ + i * scan_angle_inc_;
            const double hx  = scan_rx_ + rng * std::cos(ang);
            const double hy  = scan_ry_ + rng * std::sin(ang);
            p.drawEllipse(worldToScreen(hx, hy), 2.0, 2.0);
        }
    }

    // ── Planned path ─────────────────────────────────────────────────────────
    if (!path_pts_.empty()) {
        p.setPen(QPen(QColor(30, 110, 210), 2.0));
        p.setBrush(Qt::NoBrush);
        for (std::size_t i = 1; i < path_pts_.size(); ++i) {
            p.drawLine(
                worldToScreen(path_pts_[i - 1].first, path_pts_[i - 1].second),
                worldToScreen(path_pts_[i].first,     path_pts_[i].second));
        }

        // Goal marker (last point)
        const auto [gx, gy] = path_pts_.back();
        const QPointF gpt   = worldToScreen(gx, gy);
        p.setPen(QPen(QColor(230, 100, 0), 2.5));
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(gpt, 9.0, 9.0);
        p.drawLine(gpt + QPointF(-6, 0), gpt + QPointF(6, 0));
        p.drawLine(gpt + QPointF(0, -6), gpt + QPointF(0, 6));
    }

    // ── Robot ────────────────────────────────────────────────────────────────
    {
        const double  r      = 0.28 * scale;
        const QPointF centre = worldToScreen(robot_x_, robot_y_);

        // Body
        p.setBrush(QColor(0, 185, 85, 220));
        p.setPen(QPen(QColor(0, 100, 45), 2.0));
        p.drawEllipse(centre, r, r);

        // Heading arrow
        const double ax = robot_x_ + 0.36 * std::cos(robot_theta_);
        const double ay = robot_y_ + 0.36 * std::sin(robot_theta_);
        p.setPen(QPen(Qt::white, 2.5, Qt::SolidLine, Qt::RoundCap));
        p.drawLine(centre, worldToScreen(ax, ay));
    }

    // ── Border ───────────────────────────────────────────────────────────────
    p.setPen(QPen(QColor(160, 160, 170), 1.5));
    p.setBrush(Qt::NoBrush);
    p.drawRect(QRectF(worldToScreen(0, 0), worldToScreen(kWorldSize, kWorldSize)));
}

// ===========================================================================
//  AMRPanel
// ===========================================================================

AMRPanel::AMRPanel(QWidget * parent) : QWidget(parent)
{
    // ── ROS2 node ────────────────────────────────────────────────────────────
    node_ = std::make_shared<rclcpp::Node>("amr_studio");
    ros_exec_.add_node(node_);

    // Subscriptions — all callbacks run in the Qt main thread (via spin_some)

    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        "/amr/odom", 10,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {
            robot_x_ = msg->pose.pose.position.x;
            robot_y_ = msg->pose.pose.position.y;
            const auto & q = msg->pose.pose.orientation;
            const double siny = 2.0 * (q.w * q.z + q.x * q.y);
            const double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            robot_theta_ = std::atan2(siny, cosy);
            map_view_->setRobotPose(robot_x_, robot_y_, robot_theta_);
            status_label_->setText(
                QString("Pose   x = %1 m   y = %2 m   θ = %3°")
                    .arg(robot_x_,    0, 'f', 2)
                    .arg(robot_y_,    0, 'f', 2)
                    .arg(robot_theta_ * 180.0 / M_PI, 0, 'f', 1));
        });

    path_sub_ = node_->create_subscription<nav_msgs::msg::Path>(
        "/amr/path", 10,
        [this](nav_msgs::msg::Path::SharedPtr msg) {
            path_pts_.clear();
            path_pts_.reserve(msg->poses.size());
            for (const auto & ps : msg->poses) {
                path_pts_.emplace_back(ps.pose.position.x, ps.pose.position.y);
            }
            map_view_->setPath(path_pts_);
        });

    scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
        "/amr/scan", 10,
        [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
            map_view_->setScan(
                msg->angle_min, msg->angle_increment,
                static_cast<float>(robot_x_),
                static_cast<float>(robot_y_),
                static_cast<float>(robot_theta_),
                msg->ranges);
        });

    goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/amr/goal_pose", 10);

    // ── Map view ─────────────────────────────────────────────────────────────
    map_view_ = new AMRMapView(this);
    connect(map_view_, &AMRMapView::goalClicked, this,
            [this](double wx, double wy) {
                goal_x_spin_->setValue(wx);
                goal_y_spin_->setValue(wy);
                onGoClicked();
            });

    // ── Goal panel ───────────────────────────────────────────────────────────
    auto * goal_group = new QGroupBox("Navigate to Goal", this);
    auto * goal_row   = new QHBoxLayout(goal_group);
    goal_row->setSpacing(6);

    goal_row->addWidget(new QLabel("X (m):"));
    goal_x_spin_ = new QDoubleSpinBox;
    goal_x_spin_->setRange(0.5, 9.5);
    goal_x_spin_->setValue(8.0);
    goal_x_spin_->setDecimals(2);
    goal_x_spin_->setSingleStep(0.1);
    goal_x_spin_->setFixedWidth(75);
    goal_row->addWidget(goal_x_spin_);

    goal_row->addSpacing(8);
    goal_row->addWidget(new QLabel("Y (m):"));
    goal_y_spin_ = new QDoubleSpinBox;
    goal_y_spin_->setRange(0.5, 9.5);
    goal_y_spin_->setValue(8.0);
    goal_y_spin_->setDecimals(2);
    goal_y_spin_->setSingleStep(0.1);
    goal_y_spin_->setFixedWidth(75);
    goal_row->addWidget(goal_y_spin_);

    goal_row->addSpacing(8);
    go_button_ = new QPushButton("Go");
    go_button_->setFixedWidth(60);
    goal_row->addWidget(go_button_);
    goal_row->addStretch(1);

    connect(go_button_, &QPushButton::clicked, this, &AMRPanel::onGoClicked);

    // ── Status bar ───────────────────────────────────────────────────────────
    status_label_ = new QLabel("Waiting for odometry…", this);
    status_label_->setStyleSheet("color: #444; font-size: 11px;");
    status_label_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    auto * hint = new QLabel(
        "<i>Left-click the map to send a navigation goal.</i>", this);
    hint->setStyleSheet("color: gray; font-size: 10px;");

    // ── Main layout ──────────────────────────────────────────────────────────
    auto * layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);
    layout->addWidget(map_view_, 1);
    layout->addWidget(goal_group);
    layout->addWidget(hint);
    layout->addWidget(status_label_);

    // ── ROS2 spin timer (200 Hz, 4 ms budget) ────────────────────────────────
    ros_spin_timer_ = new QTimer(this);
    ros_spin_timer_->setInterval(5);
    connect(ros_spin_timer_, &QTimer::timeout, this, [this]() {
        if (rclcpp::ok()) {
            ros_exec_.spin_some(std::chrono::milliseconds(4));
        }
    });
    ros_spin_timer_->start();
}

AMRPanel::~AMRPanel()
{
    ros_spin_timer_->stop();
    ros_exec_.remove_node(node_);
}

void AMRPanel::onGoClicked()
{
    geometry_msgs::msg::PoseStamped goal;
    goal.header.stamp    = node_->now();
    goal.header.frame_id = "odom";
    goal.pose.position.x = goal_x_spin_->value();
    goal.pose.position.y = goal_y_spin_->value();
    goal.pose.orientation.w = 1.0;
    goal_pub_->publish(goal);
    RCLCPP_INFO(node_->get_logger(),
        "Goal published: (%.2f, %.2f)",
        goal.pose.position.x, goal.pose.position.y);
}

}  // namespace robot_arm_qt_ui
