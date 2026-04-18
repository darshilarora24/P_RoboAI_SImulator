#include <QApplication>
#include <QFont>
#include <QPalette>
#include <QStyleFactory>

#include <rclcpp/rclcpp.hpp>

#include "p_roboai_viz/main_window.hpp"

// ── Dark Fusion palette ───────────────────────────────────────────────────────

static void applyDarkTheme(QApplication& app)
{
    app.setStyle(QStyleFactory::create("Fusion"));

    QPalette p;
    p.setColor(QPalette::Window,          QColor( 40,  40,  40));
    p.setColor(QPalette::WindowText,      QColor(220, 220, 220));
    p.setColor(QPalette::Base,            QColor( 28,  28,  28));
    p.setColor(QPalette::AlternateBase,   QColor( 50,  50,  50));
    p.setColor(QPalette::ToolTipBase,     QColor( 50,  50,  50));
    p.setColor(QPalette::ToolTipText,     QColor(220, 220, 220));
    p.setColor(QPalette::Text,            QColor(210, 210, 210));
    p.setColor(QPalette::Button,          QColor( 55,  55,  55));
    p.setColor(QPalette::ButtonText,      QColor(220, 220, 220));
    p.setColor(QPalette::BrightText,      Qt::red);
    p.setColor(QPalette::Link,            QColor( 42, 130, 218));
    p.setColor(QPalette::Highlight,       QColor( 42, 130, 218));
    p.setColor(QPalette::HighlightedText, Qt::black);
    p.setColor(QPalette::Disabled, QPalette::Text,       QColor(100, 100, 100));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(100, 100, 100));
    app.setPalette(p);

    app.setStyleSheet(
        "QMenuBar { background:#1e1e1e; color:#ccc; border-bottom:1px solid #333; }"
        "QMenuBar::item:selected { background:#333; }"
        "QMenu { background:#2a2a2a; color:#ccc; border:1px solid #444; }"
        "QMenu::item:selected { background:#3a6fa8; }"
        "QDockWidget::title { background:#252525; color:#bbb; padding:4px;"
        "  border-bottom:1px solid #444; font-size:11px; }"
        "QScrollBar:vertical { background:#2a2a2a; width:8px; }"
        "QScrollBar::handle:vertical { background:#555; border-radius:4px; }"
        "QScrollBar:horizontal { background:#2a2a2a; height:8px; }"
        "QScrollBar::handle:horizontal { background:#555; border-radius:4px; }"
        "QToolTip { background:#333; color:#ddd; border:1px solid #555; padding:4px; }");
}

// ── entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    // Init ROS2 before QApplication so ROS args are stripped
    rclcpp::init(argc, argv);

    QApplication app(argc, argv);
    app.setApplicationName("P_RoboAI Visualizer");
    app.setOrganizationName("P_RoboAI");

    QFont f = app.font();
    f.setFamily("Sans Serif");
    f.setPointSize(10);
    app.setFont(f);

    applyDarkTheme(app);

    auto node = std::make_shared<rclcpp::Node>("p_roboai_viz");

    p_roboai_viz::MainWindow win(node);
    win.show();

    int ret = app.exec();
    rclcpp::shutdown();
    return ret;
}
