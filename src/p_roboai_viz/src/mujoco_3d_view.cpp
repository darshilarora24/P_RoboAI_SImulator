#include "p_roboai_viz/mujoco_3d_view.hpp"

#include <cmath>
#include <cstring>

#include <QMouseEvent>
#include <QOpenGLContext>
#include <QWheelEvent>

namespace p_roboai_viz {

// ── construction / destruction ────────────────────────────────────────────────

MuJoCo3DView::MuJoCo3DView(const MapView*     map_view,
                             const std::string& model_path,
                             QWidget*           parent)
    : QOpenGLWidget(parent), _map_view(map_view), _model_path(model_path)
{
    setMinimumSize(320, 240);
    setStyleSheet("background:#111;");

    // Initialise MuJoCo structs to safe defaults
    mjv_defaultScene(&_scn);
    mjv_defaultCamera(&_cam);
    mjv_defaultOption(&_opt);
    mjr_defaultContext(&_con);

    // Free camera
    _cam.type     = mjCAMERA_FREE;
    _cam.distance = 9.0;
    _cam.azimuth  = 135.0;
    _cam.elevation = -35.0;
    _cam.lookat[0] = 5.0;
    _cam.lookat[1] = 5.0;
    _cam.lookat[2] = 0.3;
}

MuJoCo3DView::~MuJoCo3DView()
{
    makeCurrent();
    if (_mujoco_ready) {
        mjv_freeScene(&_scn);
        mjr_freeContext(&_con);
    }
    if (_data)  mj_deleteData(_data);
    if (_model) mj_deleteModel(_model);
    doneCurrent();
}

// ── OpenGL lifecycle ──────────────────────────────────────────────────────────

void MuJoCo3DView::initializeGL()
{
    // Load the warehouse model
    char err[1000] = "";
    _model = mj_loadXML(_model_path.c_str(), nullptr, err, sizeof(err));
    if (!_model) {
        // Graceful degradation — show blank viewport
        return;
    }
    _data = mj_makeData(_model);

    // Find robot body and freejoint
    _base_body_id  = mj_name2id(_model, mjOBJ_BODY, "amr_base");
    _root_joint_id = mj_name2id(_model, mjOBJ_JOINT, "root");
    if (_root_joint_id >= 0)
        _root_qpos_adr = _model->jnt_qposadr[_root_joint_id];

    // MuJoCo rendering context (requires active GL context)
    mjr_makeContext(_model, &_con, mjFONTSCALE_100);
    mjv_makeScene(_model, &_scn, 2000);

    // Visual options — render flags live in mjvScene.flags (mjtRndFlag)
    _scn.flags[mjRND_SHADOW]    = 1;
    _scn.flags[mjRND_WIREFRAME] = 0;
    // Disable transparent overlay on dynamic geoms
    _opt.flags[mjVIS_TRANSPARENT] = 0;

    // Run one forward kinematics pass so the robot is in its starting pose
    mj_forward(_model, _data);
    _mujoco_ready = true;
}

void MuJoCo3DView::resizeGL(int w, int h)
{
    // MuJoCo reads the framebuffer size from the context; nothing extra needed.
    (void)w; (void)h;
}

void MuJoCo3DView::paintGL()
{
    if (!_mujoco_ready) {
        glClearColor(0.07f, 0.07f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

    // Position the robot from the latest odometry
    positionRobot();

    // Update the scene graph
    mjv_updateScene(_model, _data, &_opt, nullptr, &_cam,
                    mjCAT_ALL, &_scn);

    // Get actual framebuffer size (handles HiDPI)
    int fbw = static_cast<int>(width()  * devicePixelRatio());
    int fbh = static_cast<int>(height() * devicePixelRatio());
    mjrRect viewport = {0, 0, fbw, fbh};

    mjr_render(viewport, &_scn, &_con);
}

// ── robot positioning ─────────────────────────────────────────────────────────

void MuJoCo3DView::positionRobot()
{
    if (_root_qpos_adr < 0) return;

    const RobotState& rs = _map_view->robotState();
    double* qp = _data->qpos + _root_qpos_adr;

    qp[0] = rs.x;
    qp[1] = rs.y;
    qp[2] = 0.10;                           // fixed floor height

    // MuJoCo quaternion order: w, x, y, z
    double half = rs.theta * 0.5;
    qp[3] = std::cos(half);                 // w
    qp[4] = 0.0;                            // x
    qp[5] = 0.0;                            // y
    qp[6] = std::sin(half);                 // z  (rotation about world Z)

    // Update kinematics only (no physics step)
    mj_kinematics(_model, _data);
}

// ── slot: called when map_view emits robotStateUpdated ────────────────────────

void MuJoCo3DView::onRobotUpdated()
{
    update();   // triggers paintGL()
}

// ── camera mouse control ──────────────────────────────────────────────────────

void MuJoCo3DView::mousePressEvent(QMouseEvent* ev)
{
    _drag_start  = ev->pos();
    _drag_az     = static_cast<double>(_cam.azimuth);
    _drag_el     = static_cast<double>(_cam.elevation);
    _dragging    = true;
    _shift_drag  = (ev->modifiers() & Qt::ShiftModifier) != 0;
}

void MuJoCo3DView::mouseMoveEvent(QMouseEvent* ev)
{
    if (!_dragging) return;

    int dx = ev->pos().x() - _drag_start.x();
    int dy = ev->pos().y() - _drag_start.y();

    if (_shift_drag) {
        // Pan: move lookat point
        double scale = _cam.distance * 0.001;
        double az_rad = _cam.azimuth * M_PI / 180.0;
        _cam.lookat[0] -= static_cast<float>(
            (-std::sin(az_rad)*dx + std::cos(az_rad)*dy) * scale);
        _cam.lookat[1] -= static_cast<float>(
            ( std::cos(az_rad)*dx + std::sin(az_rad)*dy) * scale);
    } else {
        // Orbit: rotate azimuth and elevation
        _cam.azimuth  = static_cast<float>(
            _drag_az - dx * 0.4);
        _cam.elevation = static_cast<float>(
            std::clamp(_drag_el + dy * 0.4, -89.0, -5.0));
    }
    update();
}

void MuJoCo3DView::mouseReleaseEvent(QMouseEvent*)
{
    _dragging = false;
}

void MuJoCo3DView::wheelEvent(QWheelEvent* ev)
{
    double factor = ev->angleDelta().y() > 0 ? 0.88 : 1.0 / 0.88;
    _cam.distance = static_cast<float>(
        std::clamp(static_cast<double>(_cam.distance) * factor, 1.5, 25.0));
    update();
}

} // namespace p_roboai_viz
