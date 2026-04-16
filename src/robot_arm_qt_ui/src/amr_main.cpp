/**
 * amr_main.cpp — entry point for the standalone AMR Studio Qt window.
 *
 * Launches:
 *   - rclcpp node (embedded inside AMRPanel)
 *   - Qt6 window showing the 2-D AMR map panel
 *
 * Run via:
 *   ros2 launch robot_amr_sim amr_sim.launch.py
 * or standalone:
 *   ros2 run robot_arm_qt_ui amr_studio
 */
#include <QApplication>

#include "rclcpp/rclcpp.hpp"
#include "robot_arm_qt_ui/amr_panel.hpp"

int main(int argc, char ** argv)
{
    // Initialise ROS2 before QApplication so rclcpp can strip its own args.
    rclcpp::init(argc, argv);

    QApplication app(argc, argv);
    app.setApplicationName("AMR Studio");
    app.setOrganizationName("P RoboAI");

    robot_arm_qt_ui::AMRPanel panel;
    panel.setWindowTitle("P RoboAI — AMR Studio");
    panel.resize(580, 700);
    panel.show();

    const int ret = app.exec();
    rclcpp::shutdown();
    return ret;
}
