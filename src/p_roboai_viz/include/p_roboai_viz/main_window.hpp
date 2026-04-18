#pragma once
#include <QAction>
#include <QDockWidget>
#include <QLabel>
#include <QMainWindow>
#include <QTimer>

#include <rclcpp/rclcpp.hpp>

namespace p_roboai_viz {

class MapView;
class MuJoCo3DView;
class DisplaysPanel;
class StatusPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(rclcpp::Node::SharedPtr node,
                        QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void spinRos();
    void onGoalToolToggled(bool checked);
    void onGoalRequested(double x, double y, double theta);
    void onResetView();

private:
    void buildUI();
    void buildMenuBar();
    void buildToolBar();
    void buildStatusBar();

    rclcpp::Node::SharedPtr _node;

    MapView*       _map_view   {nullptr};
    MuJoCo3DView*  _mujoco_view{nullptr};
    DisplaysPanel* _displays   {nullptr};
    StatusPanel*   _status     {nullptr};

    QDockWidget*   _dock_displays{nullptr};
    QDockWidget*   _dock_status  {nullptr};
    QDockWidget*   _dock_3d      {nullptr};

    QAction*       _act_goal_tool{nullptr};
    QLabel*        _sb_ros_status{nullptr};
    QLabel*        _sb_coords    {nullptr};

    QTimer*        _ros_timer{nullptr};
    int            _spin_count{0};
};

} // namespace p_roboai_viz
