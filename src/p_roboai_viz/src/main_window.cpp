#include "p_roboai_viz/main_window.hpp"
#include "p_roboai_viz/displays_panel.hpp"
#include "p_roboai_viz/map_view.hpp"
#include "p_roboai_viz/mujoco_3d_view.hpp"
#include "p_roboai_viz/status_panel.hpp"

#include <QAction>
#include <QDockWidget>
#include <QLabel>
#include <QMenuBar>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace p_roboai_viz {

MainWindow::MainWindow(rclcpp::Node::SharedPtr node, QWidget* parent)
    : QMainWindow(parent), _node(node)
{
    setWindowTitle("P_RoboAI Visualizer");
    setMinimumSize(1100, 720);
    resize(1400, 860);

    buildMenuBar();
    buildUI();
    buildToolBar();   // must come after buildUI so _map_view exists
    buildStatusBar();

    // ── ROS spin timer ─────────────────────────────────────────────────────
    _ros_timer = new QTimer(this);
    connect(_ros_timer, &QTimer::timeout, this, &MainWindow::spinRos);
    _ros_timer->start(20);   // 50 Hz ROS spin
}

MainWindow::~MainWindow() = default;

// ── UI construction ───────────────────────────────────────────────────────────

void MainWindow::buildUI()
{
    // ── Displays dock (left) ───────────────────────────────────────────────
    _displays = new DisplaysPanel(this);
    _dock_displays = new QDockWidget("Displays", this);
    _dock_displays->setObjectName("dock_displays");
    _dock_displays->setFeatures(QDockWidget::DockWidgetMovable |
                                QDockWidget::DockWidgetFloatable);
    _dock_displays->setWidget(_displays);
    _dock_displays->setTitleBarWidget(new QWidget);   // hide default title (panel has its own)
    addDockWidget(Qt::LeftDockWidgetArea, _dock_displays);

    // ── Status dock (right) ────────────────────────────────────────────────
    _status = new StatusPanel(_node, this);
    _dock_status = new QDockWidget("Status & Control", this);
    _dock_status->setObjectName("dock_status");
    _dock_status->setFeatures(QDockWidget::DockWidgetMovable |
                               QDockWidget::DockWidgetFloatable);
    _dock_status->setWidget(_status);
    _dock_status->setTitleBarWidget(new QWidget);
    addDockWidget(Qt::RightDockWidgetArea, _dock_status);

    // ── 2-D map view (central widget) ─────────────────────────────────────
    _map_view = new MapView(_node, _displays, this);
    setCentralWidget(_map_view);

    // ── 3-D MuJoCo dock (bottom) ──────────────────────────────────────────
    std::string model_path;
    try {
        model_path = ament_index_cpp::get_package_share_directory(
                         "robot_amr_mujoco_sim") + "/models/amr_warehouse.xml";
    } catch (...) {
        model_path = "";
    }

    _mujoco_view = new MuJoCo3DView(_map_view, model_path, this);
    _mujoco_view->setMinimumHeight(260);

    _dock_3d = new QDockWidget("3D View  (drag / scroll to orbit / zoom)", this);
    _dock_3d->setObjectName("dock_3d");
    _dock_3d->setFeatures(QDockWidget::DockWidgetMovable |
                           QDockWidget::DockWidgetFloatable |
                           QDockWidget::DockWidgetClosable);
    _dock_3d->setWidget(_mujoco_view);
    addDockWidget(Qt::BottomDockWidgetArea, _dock_3d);

    // ── connections ────────────────────────────────────────────────────────
    // Displays panel → repaint map
    connect(_displays, &DisplaysPanel::layerChanged,
            _map_view, qOverload<>(&QWidget::update));

    // Map view → 3D view robot position
    connect(_map_view, &MapView::robotStateUpdated,
            _mujoco_view, &MuJoCo3DView::onRobotUpdated);

    // Map view goal → status panel spinboxes (sync display)
    connect(_map_view, &MapView::goalSet, this, [this](double x, double y) {
        // Status panel will show it as an IDLE→NAVIGATING transition
        (void)x; (void)y;
    });

    // Status panel goal → map view marker
    connect(_status, &StatusPanel::goalRequested,
            this, &MainWindow::onGoalRequested);

    // Status panel odom/imu/status — wired via ROS callbacks in StatusPanel
    // We need to connect StatusPanel's update methods to map-view subscriptions.
    // Since both have independent subscriptions on the same node this is automatic.
}

void MainWindow::buildMenuBar()
{
    auto* file_menu = menuBar()->addMenu("&File");
    auto* act_quit  = file_menu->addAction("&Quit");
    act_quit->setShortcut(Qt::CTRL | Qt::Key_Q);
    connect(act_quit, &QAction::triggered, this, &QMainWindow::close);

    auto* view_menu = menuBar()->addMenu("&View");
    auto* act_displays = view_menu->addAction("Displays Panel");
    act_displays->setCheckable(true);
    act_displays->setChecked(true);
    connect(act_displays, &QAction::toggled, this, [this](bool v){
        _dock_displays->setVisible(v);
    });

    auto* act_status = view_menu->addAction("Status Panel");
    act_status->setCheckable(true);
    act_status->setChecked(true);
    connect(act_status, &QAction::toggled, this, [this](bool v){
        _dock_status->setVisible(v);
    });

    auto* act_3d = view_menu->addAction("3D View");
    act_3d->setCheckable(true);
    act_3d->setChecked(true);
    connect(act_3d, &QAction::toggled, this, [this](bool v){
        _dock_3d->setVisible(v);
    });

    view_menu->addSeparator();
    auto* act_reset = view_menu->addAction("Reset Map View\tR");
    connect(act_reset, &QAction::triggered, this, &MainWindow::onResetView);
}

void MainWindow::buildToolBar()
{
    auto* tb = addToolBar("Tools");
    tb->setMovable(false);
    tb->setIconSize(QSize(20, 20));
    tb->setStyleSheet(
        "QToolBar { background:#1e1e1e; border-bottom:1px solid #333; spacing:4px; }"
        "QToolButton { color:#ccc; padding:4px 8px; border-radius:4px; font-size:12px; }"
        "QToolButton:hover   { background:#333; }"
        "QToolButton:checked { background:#1a5c3a; color:#7fff9a; }");

    // Logo / title label
    auto* title_lbl = new QLabel("  P_RoboAI Visualizer  |  ");
    title_lbl->setStyleSheet("color:#888; font-size:12px;");
    tb->addWidget(title_lbl);

    // Goal placement tool
    _act_goal_tool = tb->addAction("[ G ] Set Goal");
    _act_goal_tool->setCheckable(true);
    _act_goal_tool->setToolTip(
        "Click on the map to send a navigation goal\n"
        "Shortcut: press G while the map is focused");
    connect(_act_goal_tool, &QAction::toggled,
            this, &MainWindow::onGoalToolToggled);

    tb->addSeparator();

    // Reset view
    auto* act_reset = tb->addAction("[ R ] Reset View");
    act_reset->setToolTip("Reset map zoom and pan  (R)");
    connect(act_reset, &QAction::triggered, this, &MainWindow::onResetView);

    tb->addSeparator();

    // Help label
    auto* help_lbl = new QLabel(
        "  Drag: pan map   |   Scroll: zoom   |   Drag 3D: orbit   |   Shift+Drag 3D: pan  ");
    help_lbl->setStyleSheet("color:#555; font-size:11px;");
    tb->addWidget(help_lbl);
}

void MainWindow::buildStatusBar()
{
    _sb_ros_status = new QLabel(" ROS: connecting… ");
    _sb_ros_status->setStyleSheet("color:#ffa500; font-size:11px;");

    _sb_coords = new QLabel("");
    _sb_coords->setStyleSheet("color:#777; font-size:11px;");

    statusBar()->addWidget(_sb_ros_status);
    statusBar()->addPermanentWidget(_sb_coords);
    statusBar()->setStyleSheet(
        "QStatusBar { background:#1a1a1a; border-top:1px solid #333; }");
}

// ── slots ─────────────────────────────────────────────────────────────────────

void MainWindow::spinRos()
{
    rclcpp::spin_some(_node);
    ++_spin_count;

    // Update ROS status indicator every ~1 s
    if (_spin_count % 50 == 0) {
        if (rclcpp::ok()) {
            _sb_ros_status->setText(" ROS: connected ");
            _sb_ros_status->setStyleSheet("color:#00e676; font-size:11px;");
        } else {
            _sb_ros_status->setText(" ROS: disconnected ");
            _sb_ros_status->setStyleSheet("color:#ff5252; font-size:11px;");
        }
    }
}

void MainWindow::onGoalToolToggled(bool checked)
{
    _map_view->setGoalToolActive(checked);
    _map_view->setCursor(checked ? Qt::CrossCursor : Qt::ArrowCursor);
    _map_view->setFocus();
}

void MainWindow::onGoalRequested(double x, double y, double /*theta*/)
{
    _map_view->setGoalExternal(x, y);
}

void MainWindow::onResetView()
{
    _map_view->resetView();
}

} // namespace p_roboai_viz
