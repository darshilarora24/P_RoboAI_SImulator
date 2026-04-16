#pragma once
#include <QImage>
#include <QMouseEvent>
#include <QPointF>
#include <QWheelEvent>
#include <QWidget>
#include <optional>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>

namespace p_roboai_viz {

class DisplaysPanel;

// ── Shared robot state (written by MapView, read by MuJoCo3DView) ─────────────
struct RobotState {
    double x{1.0}, y{1.0}, theta{0.0};
    double v{0.0}, w{0.0};
};

// ── 2-D map + sensor visualisation viewport ───────────────────────────────────
class MapView : public QWidget {
    Q_OBJECT
public:
    explicit MapView(rclcpp::Node::SharedPtr node,
                     DisplaysPanel*          displays,
                     QWidget*                parent = nullptr);

    // Live robot state (used by MuJoCo3DView)
    const RobotState& robotState() const { return _robot; }

    // Called by main window when user sets goal from status panel
    void setGoalExternal(double wx, double wy);

    // Toggle goal-placement mode
    void setGoalToolActive(bool active) { _goal_tool_active = active; update(); }

signals:
    void goalSet(double wx, double wy);
    void robotStateUpdated();

public slots:
    void onMap      (const nav_msgs::msg::OccupancyGrid::SharedPtr  msg);
    void onCostmap  (const nav_msgs::msg::OccupancyGrid::SharedPtr  msg);
    void onScan     (const sensor_msgs::msg::LaserScan::SharedPtr   msg);
    void onOdom     (const nav_msgs::msg::Odometry::SharedPtr       msg);
    void onPath     (const nav_msgs::msg::Path::SharedPtr           msg);
    void onNavStatus(const std_msgs::msg::String::SharedPtr         msg);

    void resetView();

protected:
    void paintEvent       (QPaintEvent*)  override;
    void mousePressEvent  (QMouseEvent*)  override;
    void mouseMoveEvent   (QMouseEvent*)  override;
    void mouseReleaseEvent(QMouseEvent*)  override;
    void wheelEvent       (QWheelEvent*)  override;
    void keyPressEvent    (QKeyEvent*)    override;

private:
    // ── coordinate transforms ──────────────────────────────────────────────
    QPointF worldToScreen(double wx, double wy) const;
    QPointF screenToWorld(double sx, double sy) const;

    // ── layer builders ─────────────────────────────────────────────────────
    void buildMapImage (const nav_msgs::msg::OccupancyGrid& msg);
    void buildCostImage(const nav_msgs::msg::OccupancyGrid& msg);

    // ── drawing helpers ────────────────────────────────────────────────────
    void drawGrid       (QPainter& p) const;
    void drawMapLayer   (QPainter& p) const;
    void drawCostLayer  (QPainter& p) const;
    void drawScanLayer  (QPainter& p) const;
    void drawTrailLayer (QPainter& p) const;
    void drawPathLayer  (QPainter& p) const;
    void drawGoalMarker (QPainter& p) const;
    void drawRobot      (QPainter& p) const;
    void drawScaleBar   (QPainter& p) const;
    void drawCursorInfo (QPainter& p) const;
    void drawNavStatus  (QPainter& p) const;

    // ── map metadata ───────────────────────────────────────────────────────
    QImage _map_image;
    QImage _cost_image;
    double _map_origin_x{0.0}, _map_origin_y{0.0};
    double _map_resolution{0.05};
    int    _map_width{0}, _map_height{0};
    bool   _map_ready{false};
    bool   _cost_ready{false};

    // ── sensor data ────────────────────────────────────────────────────────
    struct ScanPt { float x, y; };
    std::vector<ScanPt>              _scan_pts;
    std::vector<std::pair<float,float>> _path_pts;
    std::vector<RobotState>          _trail;       // pose history
    static constexpr int MAX_TRAIL = 500;

    // ── robot & goal ───────────────────────────────────────────────────────
    RobotState _robot;
    std::optional<std::pair<double,double>> _goal;
    std::string _nav_status;

    // ── view state ─────────────────────────────────────────────────────────
    double _zoom{60.0};    // pixels per metre
    double _pan_x{5.0};   // world X at screen centre
    double _pan_y{5.0};   // world Y at screen centre
    QPoint _drag_start;
    double _drag_pan_x{5.0};
    double _drag_pan_y{5.0};
    bool   _panning{false};
    bool   _goal_tool_active{false};
    double _cursor_wx{0.0}, _cursor_wy{0.0};

    // ── ROS ────────────────────────────────────────────────────────────────
    rclcpp::Node::SharedPtr _node;
    DisplaysPanel*          _displays{nullptr};

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr  _map_sub;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr  _cost_sub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr   _scan_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       _odom_sub;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr           _path_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         _status_sub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr  _goal_pub;
};

} // namespace p_roboai_viz
