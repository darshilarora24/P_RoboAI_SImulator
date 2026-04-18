#include "p_roboai_viz/status_panel.hpp"

#include <cmath>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

namespace p_roboai_viz {

// ── helpers ───────────────────────────────────────────────────────────────────

QLabel* StatusPanel::makeValueLabel(const QString& init)
{
    auto* l = new QLabel(init, this);
    l->setStyleSheet(
        "color:#00e5ff; font-family:monospace; font-size:12px;"
        "background:#1a1a1a; border:1px solid #333; padding:1px 4px;"
        "border-radius:3px;");
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    l->setMinimumWidth(80);
    return l;
}

// Create a titled group box with a QVBoxLayout; caller adds children.
static QGroupBox* makeGroupBox(const QString& title)
{
    auto* g = new QGroupBox(title);
    g->setStyleSheet(
        "QGroupBox { color:#aaa; font-size:11px; border:1px solid #444;"
        "  border-radius:4px; margin-top:8px; padding:4px; }"
        "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }");
    new QVBoxLayout(g);
    return g;
}

static void addRow(QGroupBox* g, const QString& label, QLabel*& val_out,
                   const QString& init = "—")
{
    auto* row  = new QWidget;
    auto* hbox = new QHBoxLayout(row);
    hbox->setContentsMargins(0, 1, 0, 1);

    auto* lbl = new QLabel(label);
    lbl->setStyleSheet("color:#999; font-size:11px;");
    lbl->setFixedWidth(68);

    val_out = new QLabel(init);
    val_out->setStyleSheet(
        "color:#00e5ff; font-family:monospace; font-size:12px;"
        "background:#1a1a1a; border:1px solid #333; padding:1px 4px;"
        "border-radius:3px;");
    val_out->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    val_out->setMinimumWidth(90);

    hbox->addWidget(lbl);
    hbox->addWidget(val_out, 1);
    qobject_cast<QVBoxLayout*>(g->layout())->addWidget(row);
}

// ── constructor ───────────────────────────────────────────────────────────────

StatusPanel::StatusPanel(rclcpp::Node::SharedPtr node, QWidget* parent)
    : QWidget(parent), _node(node)
{
    setFixedWidth(270);
    setStyleSheet("background:#262626;");

    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);

    // title
    auto* title = new QLabel("  Status & Control", this);
    title->setFixedHeight(32);
    title->setStyleSheet(
        "background:#1e1e1e; color:#ddd; font-size:13px; font-weight:bold;"
        "border-bottom:1px solid #444;");
    outer->addWidget(title);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setStyleSheet("QScrollArea { border:none; background:#262626; }");

    auto* container = new QWidget;
    container->setStyleSheet("background:#262626;");
    auto* vbox = new QVBoxLayout(container);
    vbox->setContentsMargins(8, 8, 8, 8);
    vbox->setSpacing(6);

    // ── Robot Pose ─────────────────────────────────────────────────────────
    auto* grp_pose = makeGroupBox("Robot Pose");
    addRow(grp_pose, "X (m):",     _lbl_x,     "1.000");
    addRow(grp_pose, "Y (m):",     _lbl_y,     "1.000");
    addRow(grp_pose, "θ (deg):",   _lbl_theta, "0.0");
    addRow(grp_pose, "v (m/s):",   _lbl_v,     "0.000");
    addRow(grp_pose, "ω (rad/s):", _lbl_w,     "0.000");
    vbox->addWidget(grp_pose);

    // ── IMU ────────────────────────────────────────────────────────────────
    auto* grp_imu = makeGroupBox("IMU");
    addRow(grp_imu, "Roll:",  _lbl_roll,  "0.0°");
    addRow(grp_imu, "Pitch:", _lbl_pitch, "0.0°");
    addRow(grp_imu, "Yaw:",   _lbl_yaw,   "0.0°");
    vbox->addWidget(grp_imu);

    // ── Navigation ─────────────────────────────────────────────────────────
    auto* grp_nav = makeGroupBox("Navigation");
    {
        auto* status_row  = new QWidget;
        auto* sr_hbox = new QHBoxLayout(status_row);
        sr_hbox->setContentsMargins(0, 1, 0, 1);
        auto* sl = new QLabel("Status:");
        sl->setStyleSheet("color:#999; font-size:11px;");
        sl->setFixedWidth(68);
        _lbl_nav_status = new QLabel("IDLE");
        _lbl_nav_status->setStyleSheet(
            "color:#ffa500; font-family:monospace; font-size:12px; font-weight:bold;"
            "background:#1a1a1a; border:1px solid #333; padding:1px 4px; border-radius:3px;");
        _lbl_nav_status->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        sr_hbox->addWidget(sl);
        sr_hbox->addWidget(_lbl_nav_status, 1);
        qobject_cast<QVBoxLayout*>(grp_nav->layout())->addWidget(status_row);
    }
    vbox->addWidget(grp_nav);

    // ── Send Goal ──────────────────────────────────────────────────────────
    auto* grp_goal = makeGroupBox("Send Goal");
    {
        auto* gl = qobject_cast<QVBoxLayout*>(grp_goal->layout());

        auto makeSpinBox = [](double lo, double hi, double val) {
            auto* s = new QDoubleSpinBox;
            s->setRange(lo, hi);
            s->setValue(val);
            s->setDecimals(2);
            s->setSingleStep(0.1);
            s->setStyleSheet(
                "QDoubleSpinBox { background:#1a1a1a; color:#eee; border:1px solid #555;"
                "  border-radius:3px; padding:2px; font-family:monospace; }"
                "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button"
                "  { background:#333; width:14px; border-radius:2px; }");
            return s;
        };

        auto addGoalRow = [&](const QString& lbl_txt, QDoubleSpinBox*& spin,
                               double lo, double hi, double val) {
            auto* row  = new QWidget;
            auto* hbox = new QHBoxLayout(row);
            hbox->setContentsMargins(0, 2, 0, 2);
            auto* lbl = new QLabel(lbl_txt);
            lbl->setStyleSheet("color:#999; font-size:11px;");
            lbl->setFixedWidth(50);
            spin = makeSpinBox(lo, hi, val);
            hbox->addWidget(lbl);
            hbox->addWidget(spin, 1);
            gl->addWidget(row);
        };

        addGoalRow("X (m):",   _goal_x,     -1.0, 12.0, 5.0);
        addGoalRow("Y (m):",   _goal_y,     -1.0, 12.0, 5.0);
        addGoalRow("θ (deg):", _goal_theta, -180.0, 180.0, 0.0);

        auto* btn_row  = new QWidget;
        auto* btn_hbox = new QHBoxLayout(btn_row);
        btn_hbox->setContentsMargins(0, 4, 0, 0);

        _btn_send = new QPushButton("Send Goal");
        _btn_send->setStyleSheet(
            "QPushButton { background:#1a6b3a; color:#fff; border-radius:4px; padding:5px; font-weight:bold; }"
            "QPushButton:hover { background:#22914e; }"
            "QPushButton:pressed { background:#155c30; }");

        _btn_cancel = new QPushButton("Cancel");
        _btn_cancel->setStyleSheet(
            "QPushButton { background:#6b1a1a; color:#fff; border-radius:4px; padding:5px; font-weight:bold; }"
            "QPushButton:hover { background:#912222; }"
            "QPushButton:pressed { background:#5c1515; }");

        btn_hbox->addWidget(_btn_send, 2);
        btn_hbox->addWidget(_btn_cancel, 1);
        gl->addWidget(btn_row);
    }
    vbox->addWidget(grp_goal);
    vbox->addStretch();

    scroll->setWidget(container);
    outer->addWidget(scroll, 1);

    // ── ROS publisher & subscriptions ─────────────────────────────────────
    _goal_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/amr/goal_pose", 10);

    _odom_sub = _node->create_subscription<nav_msgs::msg::Odometry>(
        "/amr/odom", 10,
        [this](nav_msgs::msg::Odometry::SharedPtr m){ onOdom(*m); });

    _status_sub = _node->create_subscription<std_msgs::msg::String>(
        "/p_roboai_nav2/status", 10,
        [this](std_msgs::msg::String::SharedPtr m){ onNavStatus(*m); });

    _imu_sub = _node->create_subscription<sensor_msgs::msg::Imu>(
        "/amr/imu", 10,
        [this](sensor_msgs::msg::Imu::SharedPtr m){ onImu(*m); });

    connect(_btn_send,   &QPushButton::clicked, this, &StatusPanel::sendGoal);
    connect(_btn_cancel, &QPushButton::clicked, this, &StatusPanel::cancelNav);
}

// ── slots ─────────────────────────────────────────────────────────────────────

void StatusPanel::onOdom(const nav_msgs::msg::Odometry& msg)
{
    double x = msg.pose.pose.position.x;
    double y = msg.pose.pose.position.y;
    double q_w = msg.pose.pose.orientation.w;
    double q_x = msg.pose.pose.orientation.x;
    double q_y = msg.pose.pose.orientation.y;
    double q_z = msg.pose.pose.orientation.z;
    double siny = 2.0*(q_w*q_z + q_x*q_y);
    double cosy = 1.0 - 2.0*(q_y*q_y + q_z*q_z);
    double theta = std::atan2(siny, cosy) * 180.0 / M_PI;

    double v = msg.twist.twist.linear.x;
    double w = msg.twist.twist.angular.z;

    _lbl_x    ->setText(QString::number(x,     'f', 3));
    _lbl_y    ->setText(QString::number(y,     'f', 3));
    _lbl_theta->setText(QString::number(theta, 'f', 1) + "°");
    _lbl_v    ->setText(QString::number(v,     'f', 3));
    _lbl_w    ->setText(QString::number(w,     'f', 3));
}

void StatusPanel::onNavStatus(const std_msgs::msg::String& msg)
{
    const auto& s = msg.data;
    QString style_base =
        "font-family:monospace; font-size:12px; font-weight:bold;"
        "background:#1a1a1a; border:1px solid #333; padding:1px 4px; border-radius:3px;";

    if (s == "GOAL_REACHED") {
        _lbl_nav_status->setStyleSheet("color:#00e676; " + style_base);
    } else if (s == "NAVIGATING") {
        _lbl_nav_status->setStyleSheet("color:#40c4ff; " + style_base);
    } else if (s == "NO_PATH") {
        _lbl_nav_status->setStyleSheet("color:#ff5252; " + style_base);
    } else {
        _lbl_nav_status->setStyleSheet("color:#ffa500; " + style_base);
    }
    _lbl_nav_status->setText(QString::fromStdString(s));
}

void StatusPanel::onImu(const sensor_msgs::msg::Imu& msg)
{
    double qw = msg.orientation.w;
    double qx = msg.orientation.x;
    double qy = msg.orientation.y;
    double qz = msg.orientation.z;

    // Roll
    double sinr = 2.0*(qw*qx + qy*qz);
    double cosr = 1.0 - 2.0*(qx*qx + qy*qy);
    double roll  = std::atan2(sinr, cosr) * 180.0 / M_PI;

    // Pitch
    double sinp = 2.0*(qw*qy - qz*qx);
    double pitch = std::asin(std::clamp(sinp, -1.0, 1.0)) * 180.0 / M_PI;

    // Yaw
    double siny = 2.0*(qw*qz + qx*qy);
    double cosy = 1.0 - 2.0*(qy*qy + qz*qz);
    double yaw   = std::atan2(siny, cosy) * 180.0 / M_PI;

    _lbl_roll ->setText(QString::number(roll,  'f', 1) + "°");
    _lbl_pitch->setText(QString::number(pitch, 'f', 1) + "°");
    _lbl_yaw  ->setText(QString::number(yaw,   'f', 1) + "°");
}

void StatusPanel::sendGoal()
{
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp    = _node->get_clock()->now();
    ps.header.frame_id = "map";
    ps.pose.position.x = _goal_x->value();
    ps.pose.position.y = _goal_y->value();
    ps.pose.position.z = 0.0;

    double theta_rad = _goal_theta->value() * M_PI / 180.0;
    ps.pose.orientation.w = std::cos(theta_rad / 2.0);
    ps.pose.orientation.z = std::sin(theta_rad / 2.0);

    _goal_pub->publish(ps);
    emit goalRequested(_goal_x->value(), _goal_y->value(), _goal_theta->value());
}

} // namespace p_roboai_viz
