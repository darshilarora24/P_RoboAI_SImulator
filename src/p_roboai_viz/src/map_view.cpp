#include "p_roboai_viz/map_view.hpp"
#include "p_roboai_viz/displays_panel.hpp"

#include <algorithm>
#include <cmath>

#include <QCursor>
#include <QFont>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QWheelEvent>

namespace p_roboai_viz {

// ── construction ──────────────────────────────────────────────────────────────

MapView::MapView(rclcpp::Node::SharedPtr node,
                 DisplaysPanel*          displays,
                 QWidget*                parent)
    : QWidget(parent), _node(node), _displays(displays)
{
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setMinimumSize(500, 400);
    setStyleSheet("background:#1a1a1a;");

    // ── ROS subscriptions ──────────────────────────────────────────────────
    _map_sub = _node->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/p_roboai_slam/map", rclcpp::QoS(1).transient_local(),
        [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ onMap(m); });

    _cost_sub = _node->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/p_roboai_nav2/costmap", 1,
        [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ onCostmap(m); });

    _scan_sub = _node->create_subscription<sensor_msgs::msg::LaserScan>(
        "/amr/scan", 10,
        [this](sensor_msgs::msg::LaserScan::SharedPtr m){ onScan(m); });

    _odom_sub = _node->create_subscription<nav_msgs::msg::Odometry>(
        "/amr/odom", 10,
        [this](nav_msgs::msg::Odometry::SharedPtr m){ onOdom(m); });

    _path_sub = _node->create_subscription<nav_msgs::msg::Path>(
        "/p_roboai_nav2/path", 1,
        [this](nav_msgs::msg::Path::SharedPtr m){ onPath(m); });

    _status_sub = _node->create_subscription<std_msgs::msg::String>(
        "/p_roboai_nav2/status", 10,
        [this](std_msgs::msg::String::SharedPtr m){ onNavStatus(m); });

    _goal_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/amr/goal_pose", 10);
}

// ── coordinate transforms ─────────────────────────────────────────────────────

QPointF MapView::worldToScreen(double wx, double wy) const
{
    double sx = (wx - _pan_x) * _zoom + width()  * 0.5;
    double sy = height() * 0.5 - (wy - _pan_y) * _zoom;   // flip Y
    return {sx, sy};
}

QPointF MapView::screenToWorld(double sx, double sy) const
{
    double wx = (sx - width()  * 0.5) / _zoom + _pan_x;
    double wy = (height() * 0.5 - sy) / _zoom + _pan_y;
    return {wx, wy};
}

// ── layer image builders ──────────────────────────────────────────────────────

void MapView::buildMapImage(const nav_msgs::msg::OccupancyGrid& msg)
{
    int w = static_cast<int>(msg.info.width);
    int h = static_cast<int>(msg.info.height);
    if (w <= 0 || h <= 0) return;

    _map_image = QImage(w, h, QImage::Format_RGB32);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int8_t val = static_cast<int8_t>(msg.data[static_cast<size_t>(row * w + col)]);
            QRgb color;
            if (val < 0) {
                color = qRgb(72, 72, 72);          // unknown  — mid grey
            } else if (val == 0) {
                color = qRgb(210, 210, 210);        // free     — light grey
            } else {
                int v = 210 - static_cast<int>(200.0 * val / 100.0);
                color = qRgb(v, v, v);              // occupied — dark
            }
            // Flip row so QImage row-0 = map top (high world-Y)
            _map_image.setPixel(col, h - 1 - row, color);
        }
    }
    _map_ready = true;
}

void MapView::buildCostImage(const nav_msgs::msg::OccupancyGrid& msg)
{
    int w = static_cast<int>(msg.info.width);
    int h = static_cast<int>(msg.info.height);
    if (w <= 0 || h <= 0) return;

    _cost_image = QImage(w, h, QImage::Format_ARGB32);
    _cost_image.fill(Qt::transparent);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int8_t raw = static_cast<int8_t>(msg.data[static_cast<size_t>(row * w + col)]);
            int cost = static_cast<int>(static_cast<uint8_t>(raw));
            if (cost <= 0) continue;

            QRgb color;
            if (cost >= 90) {
                color = qRgba(255, 30, 30, 200);   // lethal — red
            } else {
                float t = cost / 89.0f;
                int r = static_cast<int>(255 * std::min(2.0f * t, 1.0f));
                int g = static_cast<int>(255 * std::min(2.0f * (1.0f - t), 1.0f));
                color = qRgba(r, g, 0, 160);       // gradient green→yellow→orange
            }
            _cost_image.setPixel(col, h - 1 - row, color);
        }
    }
    _cost_ready = true;
}

// ── ROS callbacks ─────────────────────────────────────────────────────────────

void MapView::onMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    _map_origin_x  = msg->info.origin.position.x;
    _map_origin_y  = msg->info.origin.position.y;
    _map_resolution = msg->info.resolution;
    _map_width     = static_cast<int>(msg->info.width);
    _map_height    = static_cast<int>(msg->info.height);
    buildMapImage(*msg);
    update();
}

void MapView::onCostmap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    buildCostImage(*msg);
    update();
}

void MapView::onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    _scan_pts.clear();
    float angle = msg->angle_min;
    for (float r : msg->ranges) {
        if (r >= msg->range_min && r <= msg->range_max) {
            float world_angle = static_cast<float>(_robot.theta) + angle;
            _scan_pts.push_back({
                static_cast<float>(_robot.x) + r * std::cos(world_angle),
                static_cast<float>(_robot.y) + r * std::sin(world_angle),
            });
        }
        angle += msg->angle_increment;
    }
    update();
}

void MapView::onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    _robot.x = msg->pose.pose.position.x;
    _robot.y = msg->pose.pose.position.y;
    double q_w = msg->pose.pose.orientation.w;
    double q_x = msg->pose.pose.orientation.x;
    double q_y = msg->pose.pose.orientation.y;
    double q_z = msg->pose.pose.orientation.z;
    double siny = 2.0*(q_w*q_z + q_x*q_y);
    double cosy = 1.0 - 2.0*(q_y*q_y + q_z*q_z);
    _robot.theta = std::atan2(siny, cosy);
    _robot.v = msg->twist.twist.linear.x;
    _robot.w = msg->twist.twist.angular.z;

    // Pose trail — add if moved more than 0.05 m
    if (_trail.empty() ||
        std::hypot(_robot.x - _trail.back().x,
                   _robot.y - _trail.back().y) > 0.05)
    {
        _trail.push_back(_robot);
        if (static_cast<int>(_trail.size()) > MAX_TRAIL)
            _trail.erase(_trail.begin());
    }

    emit robotStateUpdated();
    update();
}

void MapView::onPath(const nav_msgs::msg::Path::SharedPtr msg)
{
    _path_pts.clear();
    for (const auto& ps : msg->poses) {
        _path_pts.push_back({
            static_cast<float>(ps.pose.position.x),
            static_cast<float>(ps.pose.position.y),
        });
    }
    update();
}

void MapView::onNavStatus(const std_msgs::msg::String::SharedPtr msg)
{
    _nav_status = msg->data;
    update();
}

// ── public helpers ────────────────────────────────────────────────────────────

void MapView::setGoalExternal(double wx, double wy)
{
    _goal = {wx, wy};
    update();
}

void MapView::resetView()
{
    _pan_x = 5.0;
    _pan_y = 5.0;
    _zoom  = 60.0;
    update();
}

// ── painting ──────────────────────────────────────────────────────────────────

void MapView::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // Background
    p.fillRect(rect(), QColor(26, 26, 26));

    drawGrid(p);

    if (_displays->isLayerVisible("map"))     drawMapLayer(p);
    if (_displays->isLayerVisible("costmap")) drawCostLayer(p);
    if (_displays->isLayerVisible("scan"))    drawScanLayer(p);
    if (_displays->isLayerVisible("trail"))   drawTrailLayer(p);
    if (_displays->isLayerVisible("path"))    drawPathLayer(p);

    drawGoalMarker(p);

    if (_displays->isLayerVisible("robot"))   drawRobot(p);

    drawScaleBar(p);
    drawCursorInfo(p);
    drawNavStatus(p);

    // Goal-tool cursor hint
    if (_goal_tool_active) {
        p.setPen(QPen(QColor(255, 160, 0, 180), 1, Qt::DashLine));
        QPointF c = mapFromGlobal(QCursor::pos());
        p.drawLine(QPointF(c.x(), 0), QPointF(c.x(), height()));
        p.drawLine(QPointF(0, c.y()), QPointF(width(), c.y()));
    }
}

// ── grid ──────────────────────────────────────────────────────────────────────

void MapView::drawGrid(QPainter& p) const
{
    if (!_displays->isLayerVisible("grid")) return;

    QColor gc = _displays->layerColor("grid");
    gc.setAlpha(60);
    p.setPen(QPen(gc, 1, Qt::SolidLine));

    // Determine grid spacing: 1 m if zoom > 40, else 2 m or 5 m
    double spacing = 1.0;
    if (_zoom < 20) spacing = 5.0;
    else if (_zoom < 40) spacing = 2.0;

    // World bounds visible
    QPointF wTL = screenToWorld(0, 0);
    QPointF wBR = screenToWorld(width(), height());
    double x0 = std::floor(wTL.x() / spacing) * spacing;
    double x1 = std::ceil (wBR.x() / spacing) * spacing;
    double y0 = std::floor(wBR.y() / spacing) * spacing;
    double y1 = std::ceil (wTL.y() / spacing) * spacing;

    for (double gx = x0; gx <= x1; gx += spacing) {
        QPointF s0 = worldToScreen(gx, y0);
        QPointF s1 = worldToScreen(gx, y1);
        p.drawLine(s0, s1);
    }
    for (double gy = y0; gy <= y1; gy += spacing) {
        QPointF s0 = worldToScreen(x0, gy);
        QPointF s1 = worldToScreen(x1, gy);
        p.drawLine(s0, s1);
    }

    // Axes at world origin
    p.setPen(QPen(QColor(255,100,100,120), 1.5));
    QPointF ox = worldToScreen(0, 0);
    p.drawLine(ox, worldToScreen(1.0, 0));    // +X red
    p.setPen(QPen(QColor(100,255,100,120), 1.5));
    p.drawLine(ox, worldToScreen(0, 1.0));    // +Y green

    // Grid labels
    if (_zoom > 35) {
        p.setPen(QColor(120, 120, 120));
        QFont f; f.setPointSize(8); p.setFont(f);
        for (double gx = x0; gx <= x1; gx += spacing) {
            QPointF s = worldToScreen(gx, wBR.y());
            p.drawText(QPointF(s.x() + 2, s.y() - 4),
                       QString::number(gx, 'f', gx == std::floor(gx) ? 0 : 1));
        }
        for (double gy = y0; gy <= y1; gy += spacing) {
            QPointF s = worldToScreen(wTL.x(), gy);
            p.drawText(QPointF(s.x() + 2, s.y() - 4),
                       QString::number(gy, 'f', gy == std::floor(gy) ? 0 : 1));
        }
    }
}

// ── map layer ─────────────────────────────────────────────────────────────────

void MapView::drawMapLayer(QPainter& p) const
{
    if (!_map_ready || _map_image.isNull()) return;

    float op = _displays->layerOpacity("map");
    p.setOpacity(op);

    // Map top-left world coord = (origin_x, origin_y + height*res)
    double map_world_w = _map_width  * _map_resolution;
    double map_world_h = _map_height * _map_resolution;

    QPointF tl = worldToScreen(_map_origin_x,               _map_origin_y + map_world_h);
    QPointF br = worldToScreen(_map_origin_x + map_world_w, _map_origin_y);

    QRectF dst(tl, br);
    p.drawImage(dst, _map_image);
    p.setOpacity(1.0);
}

// ── costmap layer ─────────────────────────────────────────────────────────────

void MapView::drawCostLayer(QPainter& p) const
{
    if (!_cost_ready || _cost_image.isNull()) return;

    float op = _displays->layerOpacity("costmap");
    p.setOpacity(op * 0.75f);

    double map_world_w = _cost_image.width()  * _map_resolution;
    double map_world_h = _cost_image.height() * _map_resolution;
    QPointF tl = worldToScreen(_map_origin_x,               _map_origin_y + map_world_h);
    QPointF br = worldToScreen(_map_origin_x + map_world_w, _map_origin_y);

    p.drawImage(QRectF(tl, br), _cost_image);
    p.setOpacity(1.0);
}

// ── scan layer ────────────────────────────────────────────────────────────────

void MapView::drawScanLayer(QPainter& p) const
{
    if (_scan_pts.empty()) return;

    QColor sc = _displays->layerColor("scan");
    sc.setAlpha(static_cast<int>(_displays->layerOpacity("scan") * 200));

    // Draw line from robot to each scan hit (faint beam), then dot
    QPointF robot_s = worldToScreen(_robot.x, _robot.y);
    p.setPen(QPen(QColor(sc.red(), sc.green(), sc.blue(), 25), 0.7));
    for (const auto& pt : _scan_pts) {
        p.drawLine(robot_s, worldToScreen(pt.x, pt.y));
    }

    double dot_r = std::max(1.5, _zoom * 0.015);
    p.setPen(Qt::NoPen);
    p.setBrush(sc);
    for (const auto& pt : _scan_pts) {
        QPointF s = worldToScreen(pt.x, pt.y);
        p.drawEllipse(s, dot_r, dot_r);
    }
}

// ── trail layer ───────────────────────────────────────────────────────────────

void MapView::drawTrailLayer(QPainter& p) const
{
    if (_trail.empty()) return;

    int n = static_cast<int>(_trail.size());
    double dot_r = std::max(1.5, _zoom * 0.012);
    p.setPen(Qt::NoPen);

    for (int i = 0; i < n; ++i) {
        float alpha = 30.0f + 170.0f * (float(i) / float(n));
        QColor c = _displays->layerColor("trail");
        c.setAlpha(static_cast<int>(alpha * _displays->layerOpacity("trail")));
        p.setBrush(c);
        QPointF s = worldToScreen(_trail[i].x, _trail[i].y);
        p.drawEllipse(s, dot_r, dot_r);
    }
}

// ── path layer ────────────────────────────────────────────────────────────────

void MapView::drawPathLayer(QPainter& p) const
{
    if (_path_pts.empty()) return;

    QColor pc = _displays->layerColor("path");
    float op = _displays->layerOpacity("path");

    // Line
    pc.setAlpha(static_cast<int>(op * 200));
    p.setPen(QPen(pc, 2.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);

    QPainterPath pp;
    bool first = true;
    for (const auto& [wx, wy] : _path_pts) {
        QPointF s = worldToScreen(wx, wy);
        if (first) { pp.moveTo(s); first = false; }
        else         pp.lineTo(s);
    }
    p.drawPath(pp);

    // Waypoint circles
    double dot_r = std::max(2.0, _zoom * 0.02);
    pc.setAlpha(static_cast<int>(op * 255));
    p.setBrush(pc);
    p.setPen(Qt::NoPen);
    for (const auto& [wx, wy] : _path_pts) {
        p.drawEllipse(worldToScreen(wx, wy), dot_r, dot_r);
    }
}

// ── goal marker ───────────────────────────────────────────────────────────────

void MapView::drawGoalMarker(QPainter& p) const
{
    if (!_goal.has_value()) return;

    QPointF s = worldToScreen(_goal->first, _goal->second);
    double r  = 14.0;

    // Outer ring
    p.setPen(QPen(QColor(255, 140, 0), 2.5));
    p.setBrush(QColor(255, 140, 0, 50));
    p.drawEllipse(s, r, r);

    // Inner dot
    p.setBrush(QColor(255, 140, 0));
    p.setPen(Qt::NoPen);
    p.drawEllipse(s, 4.0, 4.0);

    // Cross lines
    p.setPen(QPen(QColor(255, 140, 0), 2.0));
    p.drawLine(QPointF(s.x() - r*1.4, s.y()), QPointF(s.x() - r*0.7, s.y()));
    p.drawLine(QPointF(s.x() + r*0.7, s.y()), QPointF(s.x() + r*1.4, s.y()));
    p.drawLine(QPointF(s.x(), s.y() - r*1.4), QPointF(s.x(), s.y() - r*0.7));
    p.drawLine(QPointF(s.x(), s.y() + r*0.7), QPointF(s.x(), s.y() + r*1.4));

    // Label
    p.setPen(QColor(255, 180, 60));
    QFont f; f.setPointSize(9); f.setBold(true); p.setFont(f);
    p.drawText(s.x() + r + 4, s.y() - 4,
               QString("Goal (%1, %2)")
                   .arg(_goal->first,  0, 'f', 2)
                   .arg(_goal->second, 0, 'f', 2));
}

// ── robot drawing ─────────────────────────────────────────────────────────────

void MapView::drawRobot(QPainter& p) const
{
    // Robot chassis: 0.55 × 0.45 m
    static const double HL = 0.275;   // half-length
    static const double HW = 0.225;   // half-width

    double ct = std::cos(_robot.theta);
    double st = std::sin(_robot.theta);

    auto rot = [&](double dx, double dy) -> QPointF {
        double wx = _robot.x + dx*ct - dy*st;
        double wy = _robot.y + dx*st + dy*ct;
        return worldToScreen(wx, wy);
    };

    QPolygonF chassis;
    chassis << rot( HL,  HW) << rot( HL, -HW)
            << rot(-HL, -HW) << rot(-HL,  HW);

    QColor rc = _displays->layerColor("robot");
    rc.setAlpha(static_cast<int>(_displays->layerOpacity("robot") * 220));

    // Body fill
    p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), 100));
    p.setPen(QPen(rc, 2.0));
    p.drawPolygon(chassis);

    // Heading arrow (to front centre)
    QPointF origin = worldToScreen(_robot.x, _robot.y);
    QPointF front  = rot(HL, 0);
    p.setPen(QPen(rc, 3.0, Qt::SolidLine, Qt::RoundCap));
    p.drawLine(origin, front);

    // Arrow head
    double ax = front.x() - origin.x();
    double ay = front.y() - origin.y();
    double len = std::sqrt(ax*ax + ay*ay);
    if (len > 0) {
        ax /= len; ay /= len;
        double perp_x = -ay, perp_y = ax;
        double head = 8.0;
        QPolygonF arrowHead;
        arrowHead << front
                  << QPointF(front.x() - ax*head + perp_x*head*0.5,
                             front.y() - ay*head + perp_y*head*0.5)
                  << QPointF(front.x() - ax*head - perp_x*head*0.5,
                             front.y() - ay*head - perp_y*head*0.5);
        p.setBrush(rc);
        p.setPen(Qt::NoPen);
        p.drawPolygon(arrowHead);
    }

    // Centre dot
    p.setBrush(Qt::white);
    p.setPen(Qt::NoPen);
    p.drawEllipse(origin, 3.0, 3.0);
}

// ── scale bar ─────────────────────────────────────────────────────────────────

void MapView::drawScaleBar(QPainter& p) const
{
    // Choose a nice scale
    double world_m = 1.0;
    if      (_zoom < 15) world_m = 5.0;
    else if (_zoom < 30) world_m = 2.0;
    else if (_zoom > 120) world_m = 0.5;

    int bar_px = static_cast<int>(world_m * _zoom);
    int bx = 20, by = height() - 22;

    p.setPen(QPen(QColor(200,200,200), 2));
    p.drawLine(bx, by, bx + bar_px, by);
    p.drawLine(bx,      by - 5, bx,      by + 5);
    p.drawLine(bx+bar_px, by - 5, bx+bar_px, by + 5);

    p.setPen(QColor(220, 220, 220));
    QFont f; f.setPointSize(9); p.setFont(f);
    QString txt = world_m < 1.0
        ? QString("%1 cm").arg(static_cast<int>(world_m * 100))
        : (world_m == 1.0 ? "1 m" : QString("%1 m").arg(static_cast<int>(world_m)));
    p.drawText(bx + bar_px / 2 - 18, by - 8, txt);
}

// ── cursor info ───────────────────────────────────────────────────────────────

void MapView::drawCursorInfo(QPainter& p) const
{
    QString txt = QString("(%1, %2) m")
        .arg(_cursor_wx, 0, 'f', 2)
        .arg(_cursor_wy, 0, 'f', 2);
    QFont f; f.setPointSize(9); p.setFont(f);
    QFontMetrics fm(f);
    int tw = fm.horizontalAdvance(txt) + 10;
    p.fillRect(width() - tw - 4, height() - 20, tw + 4, 18,
               QColor(0, 0, 0, 130));
    p.setPen(QColor(180, 180, 180));
    p.drawText(width() - tw, height() - 6, txt);
}

// ── nav status overlay ────────────────────────────────────────────────────────

void MapView::drawNavStatus(QPainter& p) const
{
    if (_nav_status.empty()) return;

    QColor tc;
    if      (_nav_status == "GOAL_REACHED") tc = QColor(0, 230, 118);
    else if (_nav_status == "NAVIGATING")   tc = QColor(64, 196, 255);
    else if (_nav_status == "NO_PATH")      tc = QColor(255, 82, 82);
    else                                    tc = QColor(255, 165, 0);

    QFont f; f.setPointSize(10); f.setBold(true); p.setFont(f);
    QString txt = QString::fromStdString(_nav_status);
    QFontMetrics fm(f);
    int tw = fm.horizontalAdvance(txt) + 16;
    int th = 22;
    int tx = (width() - tw) / 2;
    int ty = 12;

    p.setBrush(QColor(0, 0, 0, 160));
    p.setPen(QPen(tc, 1.5));
    p.drawRoundedRect(tx, ty, tw, th, 4, 4);
    p.setPen(tc);
    p.drawText(QRect(tx, ty, tw, th), Qt::AlignCenter, txt);
}

// ── mouse events ──────────────────────────────────────────────────────────────

void MapView::mousePressEvent(QMouseEvent* ev)
{
    if (ev->button() == Qt::LeftButton && _goal_tool_active) {
        QPointF w = screenToWorld(ev->position().x(), ev->position().y());
        _goal = {w.x(), w.y()};

        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp    = _node->get_clock()->now();
        ps.header.frame_id = "map";
        ps.pose.position.x = w.x();
        ps.pose.position.y = w.y();
        ps.pose.orientation.w = 1.0;
        _goal_pub->publish(ps);
        emit goalSet(w.x(), w.y());
        update();
        return;
    }

    if (ev->button() == Qt::MiddleButton ||
        (ev->button() == Qt::LeftButton && !_goal_tool_active))
    {
        _panning   = true;
        _drag_start = ev->pos();
        _drag_pan_x = _pan_x;
        _drag_pan_y = _pan_y;
        setCursor(Qt::ClosedHandCursor);
    }
}

void MapView::mouseMoveEvent(QMouseEvent* ev)
{
    QPointF w = screenToWorld(ev->position().x(), ev->position().y());
    _cursor_wx = w.x();
    _cursor_wy = w.y();

    if (_panning) {
        double dx = (ev->pos().x() - _drag_start.x()) / _zoom;
        double dy = (ev->pos().y() - _drag_start.y()) / _zoom;
        _pan_x = _drag_pan_x - dx;
        _pan_y = _drag_pan_y + dy;   // +dy because screen-Y is flipped
    }
    update();
}

void MapView::mouseReleaseEvent(QMouseEvent*)
{
    _panning = false;
    setCursor(_goal_tool_active ? Qt::CrossCursor : Qt::ArrowCursor);
}

void MapView::wheelEvent(QWheelEvent* ev)
{
    // Zoom centred on cursor position
    QPointF before = screenToWorld(ev->position().x(), ev->position().y());

    double factor = ev->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    _zoom = std::clamp(_zoom * factor, 8.0, 400.0);

    QPointF after = screenToWorld(ev->position().x(), ev->position().y());
    _pan_x += before.x() - after.x();
    _pan_y += before.y() - after.y();
    update();
}

void MapView::keyPressEvent(QKeyEvent* ev)
{
    if (ev->key() == Qt::Key_R) {
        resetView();
    } else if (ev->key() == Qt::Key_G) {
        setGoalToolActive(!_goal_tool_active);
        setCursor(_goal_tool_active ? Qt::CrossCursor : Qt::ArrowCursor);
    }
    QWidget::keyPressEvent(ev);
}

} // namespace p_roboai_viz
