#pragma once
#include <QOpenGLWidget>
#include <QTimer>
#include <mutex>

#include <mujoco/mujoco.h>

#include "map_view.hpp"  // for RobotState

namespace p_roboai_viz {

// ── 3-D MuJoCo viewport (visual-only — tracks /amr/odom pose) ─────────────────
class MuJoCo3DView : public QOpenGLWidget {
    Q_OBJECT
public:
    explicit MuJoCo3DView(const MapView* map_view,
                          const std::string& model_path,
                          QWidget* parent = nullptr);
    ~MuJoCo3DView() override;

public slots:
    void onRobotUpdated();   // called when MapView emits robotStateUpdated

protected:
    void initializeGL() override;
    void paintGL()      override;
    void resizeGL(int w, int h) override;

    void mousePressEvent  (QMouseEvent*)  override;
    void mouseMoveEvent   (QMouseEvent*)  override;
    void mouseReleaseEvent(QMouseEvent*)  override;
    void wheelEvent       (QWheelEvent*)  override;

private:
    void positionRobot();

    const MapView*  _map_view{nullptr};
    std::string     _model_path;

    mjModel* _model{nullptr};
    mjData*  _data{nullptr};
    mjvScene    _scn;
    mjvCamera   _cam;
    mjvOption   _opt;
    mjrContext  _con;

    int _base_body_id{-1};
    int _root_joint_id{-1};
    int _root_qpos_adr{-1};

    // Camera drag
    QPoint _drag_start;
    double _drag_az{135.0};
    double _drag_el{-35.0};
    bool   _dragging{false};
    bool   _shift_drag{false};

    bool _mujoco_ready{false};
};

} // namespace p_roboai_viz
