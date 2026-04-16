#include "robot_arm_qt_ui/arm_control_panel.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QPixmap>
#include <QPoint>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

using namespace std::chrono_literals;

namespace
{

constexpr double kPi = 3.14159265358979323846;

double radiansToDegrees(double radians)
{
  return radians * 180.0 / kPi;
}

QPoint eventPosition(const QMouseEvent * event)
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
  return event->position().toPoint();
#else
  return event->pos();
#endif
}

QFrame * makeHeroCard()
{
  auto * frame = new QFrame();
  frame->setObjectName("heroCard");
  return frame;
}

QLabel * makeValueLabel(const QString & text = QStringLiteral("--"))
{
  auto * label = new QLabel(text);
  label->setObjectName("telemetryValue");
  label->setMinimumWidth(160);
  return label;
}

QString resolveLogoPath()
{
  try {
    const auto share_directory =
      ament_index_cpp::get_package_share_directory("robot_arm_qt_ui");
    const QString installed_logo_path =
      QString::fromStdString(share_directory + "/resource/proboai_logo.jpeg");
    if (QFileInfo::exists(installed_logo_path)) {
      return installed_logo_path;
    }
  } catch (const std::exception &) {
  }

  const QStringList fallback_paths = {
    QDir::current().filePath("src/robot_arm_qt_ui/resource/proboai_logo.jpeg"),
    QDir::current().filePath("proboai_logo.jpeg"),
  };

  for (const auto & path : fallback_paths) {
    if (QFileInfo::exists(path)) {
      return path;
    }
  }

  return {};
}

QString resolveDefaultModelPath()
{
  try {
    const auto share_directory =
      ament_index_cpp::get_package_share_directory("robot_arm_mujoco_sim");
    const QString installed_model_path =
      QString::fromStdString(share_directory + "/models/robot_arm.xml");
    if (QFileInfo::exists(installed_model_path)) {
      return installed_model_path;
    }
  } catch (const std::exception &) {
  }

  const QStringList fallback_paths = {
    QDir::current().filePath("src/robot_arm_mujoco_sim/models/robot_arm.xml"),
    QDir::current().filePath("install/robot_arm_mujoco_sim/share/robot_arm_mujoco_sim/models/robot_arm.xml"),
  };

  for (const auto & path : fallback_paths) {
    if (QFileInfo::exists(path)) {
      return path;
    }
  }

  return {};
}

struct SimulationSnapshot
{
  bool valid{false};
  double sim_time_sec{0.0};
  std::vector<double> positions;
  std::vector<double> velocities;
  std::vector<double> efforts;
  std::array<double, 3> ee_position{0.0, 0.0, 0.0};
  std::array<double, 4> ee_orientation_xyzw{0.0, 0.0, 0.0, 1.0};
};

// Global flag used by the MuJoCo context error guard in initializeGL().
// Must be non-local so it can be referenced from a plain C function pointer.
bool g_mujoco_context_error{false};

void mujocoContextErrorCapture(const char *) noexcept
{
  g_mujoco_context_error = true;
}

}  // namespace

namespace robot_arm_qt_ui
{

class MujocoViewportWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
  explicit MujocoViewportWidget(
    const QString & model_path,
    const std::vector<JointDefinition> & joint_definitions,
    QWidget * parent = nullptr)
  : QOpenGLWidget(parent),
    model_path_(model_path),
    joint_definitions_(joint_definitions),
    target_positions_(joint_definitions.size(), 0.0)
  {
    setMinimumSize(720, 480);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);

    mjv_defaultCamera(&camera_);
    mjv_defaultOption(&option_);
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);

    loadModel();

    simulation_timer_ = new QTimer(this);
    simulation_timer_->setInterval(16);
    connect(
      simulation_timer_,
      &QTimer::timeout,
      this,
      [this]() {
        advanceSimulation();
      });
    simulation_timer_->start();
  }

  ~MujocoViewportWidget() override
  {
    if (context() != nullptr) {
      makeCurrent();
      if (render_context_ready_) {
        mjr_freeContext(&context_);
        mjv_freeScene(&scene_);
      }
      doneCurrent();
    }

    if (data_ != nullptr) {
      mj_deleteData(data_);
    }
    if (model_ != nullptr) {
      mj_deleteModel(model_);
    }
  }

  bool isReady() const
  {
    return model_ != nullptr && data_ != nullptr;
  }

  QString lastError() const
  {
    return last_error_;
  }

  void setTargetPositions(const std::vector<double> & positions)
  {
    if (!isReady()) {
      return;
    }

    const std::size_t count = std::min(positions.size(), target_positions_.size());
    for (std::size_t index = 0; index < count; ++index) {
      const auto & limits = joint_limits_[index];
      target_positions_[index] = std::clamp(positions[index], limits.first, limits.second);
    }
  }

  void resetSimulation()
  {
    if (!isReady()) {
      return;
    }

    const int home_keyframe_id = mj_name2id(model_, mjOBJ_KEY, "home");
    if (home_keyframe_id != -1) {
      mj_resetDataKeyframe(model_, data_, home_keyframe_id);
    } else {
      mj_resetData(model_, data_);
    }

    mj_forward(model_, data_);
    for (std::size_t index = 0; index < qpos_indices_.size(); ++index) {
      target_positions_[index] = data_->qpos[qpos_indices_[index]];
    }

    update();
  }

  SimulationSnapshot snapshot() const
  {
    SimulationSnapshot snapshot;
    if (!isReady()) {
      return snapshot;
    }

    snapshot.valid = true;
    snapshot.sim_time_sec = data_->time;
    snapshot.positions.reserve(qpos_indices_.size());
    snapshot.velocities.reserve(qvel_indices_.size());
    snapshot.efforts.reserve(qvel_indices_.size());

    for (std::size_t index = 0; index < qpos_indices_.size(); ++index) {
      snapshot.positions.push_back(data_->qpos[qpos_indices_[index]]);
      snapshot.velocities.push_back(data_->qvel[qvel_indices_[index]]);
      snapshot.efforts.push_back(data_->qfrc_actuator[qvel_indices_[index]]);
    }

    const mjtNum * site_position = data_->site_xpos + (3 * ee_site_id_);
    snapshot.ee_position = {
      static_cast<double>(site_position[0]),
      static_cast<double>(site_position[1]),
      static_cast<double>(site_position[2]),
    };

    const mjtNum * site_rotation_matrix = data_->site_xmat + (9 * ee_site_id_);
    mjtNum quaternion_wxyz[4] = {0.0, 0.0, 0.0, 1.0};
    mju_mat2Quat(quaternion_wxyz, site_rotation_matrix);
    snapshot.ee_orientation_xyzw = {
      static_cast<double>(quaternion_wxyz[1]),
      static_cast<double>(quaternion_wxyz[2]),
      static_cast<double>(quaternion_wxyz[3]),
      static_cast<double>(quaternion_wxyz[0]),
    };

    return snapshot;
  }

protected:
  void initializeGL() override
  {
    initializeOpenGLFunctions();

    if (!isReady()) {
      return;
    }

    // Log which OpenGL context the window system actually provided.
    qInfo(
      "[MuJoCo viewport] OpenGL %s  |  renderer: %s",
      reinterpret_cast<const char *>(glGetString(GL_VERSION)),
      reinterpret_cast<const char *>(glGetString(GL_RENDERER)));

    mjv_makeScene(model_, &scene_, 4000);

    // Guard: intercept MuJoCo errors from mjr_makeContext so a missing
    // OpenGL extension does not call exit() and terminate the process.
    // This can still happen if QT_XCB_GL_INTEGRATION is not "xcb_egl" and
    // the GLX driver omits GL_ARB_framebuffer_object from the extension
    // string (the extension is core since OpenGL 3.0 but MuJoCo checks it
    // by name in glGetString(GL_EXTENSIONS)).
    g_mujoco_context_error = false;
    auto * saved_error_handler = mju_user_error;
    mju_user_error = mujocoContextErrorCapture;
    mjr_makeContext(model_, &context_, mjFONTSCALE_150);
    mju_user_error = saved_error_handler;

    if (g_mujoco_context_error) {
      last_error_ =
        "MuJoCo OpenGL renderer failed.\n"
        "Set QT_XCB_GL_INTEGRATION=xcb_egl or LIBGL_ALWAYS_SOFTWARE=1 "
        "and re-launch.";
      qWarning("[MuJoCo viewport] mjr_makeContext failed: %s",
        last_error_.toStdString().c_str());
      mjv_freeScene(&scene_);
      return;
    }

    qInfo("[MuJoCo viewport] mjr_makeContext succeeded — renderer ready.");
    render_context_ready_ = true;

    camera_.type = mjCAMERA_FREE;
    camera_.azimuth = 140.0;
    camera_.elevation = -25.0;
    camera_.distance = 1.7;
    camera_.lookat[0] = 0.0;
    camera_.lookat[1] = 0.0;
    camera_.lookat[2] = 0.45;
  }

  void paintGL() override
  {
    glClearColor(0.92f, 0.96f, 0.98f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!isReady() || !render_context_ready_) {
      return;
    }

    mjv_updateScene(model_, data_, &option_, nullptr, &camera_, mjCAT_ALL, &scene_);

    const auto pixel_ratio = devicePixelRatioF();
    mjrRect viewport = {
      0,
      0,
      static_cast<int>(width() * pixel_ratio),
      static_cast<int>(height() * pixel_ratio),
    };
    mjr_render(viewport, &scene_, &context_);
  }

  void mousePressEvent(QMouseEvent * event) override
  {
    last_mouse_position_ = eventPosition(event);
    QOpenGLWidget::mousePressEvent(event);
  }

  void mouseMoveEvent(QMouseEvent * event) override
  {
    if (!isReady() || !render_context_ready_) {
      QOpenGLWidget::mouseMoveEvent(event);
      return;
    }

    const QPoint current_position = eventPosition(event);
    const double dx =
      static_cast<double>(current_position.x() - last_mouse_position_.x()) /
      std::max(1, height());
    const double dy =
      static_cast<double>(current_position.y() - last_mouse_position_.y()) /
      std::max(1, height());

    mjtMouse action = mjMOUSE_ROTATE_V;
    if (event->buttons() & Qt::LeftButton) {
      action = (event->modifiers() & Qt::ShiftModifier) ? mjMOUSE_MOVE_H : mjMOUSE_ROTATE_V;
    } else if (event->buttons() & Qt::RightButton) {
      action = (event->modifiers() & Qt::ShiftModifier) ? mjMOUSE_MOVE_V : mjMOUSE_ROTATE_H;
    } else {
      last_mouse_position_ = current_position;
      QOpenGLWidget::mouseMoveEvent(event);
      return;
    }

    mjv_moveCamera(model_, action, dx, dy, &scene_, &camera_);
    last_mouse_position_ = current_position;
    update();
    QOpenGLWidget::mouseMoveEvent(event);
  }

  void wheelEvent(QWheelEvent * event) override
  {
    if (isReady() && render_context_ready_) {
      const double zoom_delta =
        -0.08 * static_cast<double>(event->angleDelta().y()) / 120.0;
      mjv_moveCamera(model_, mjMOUSE_ZOOM, 0.0, zoom_delta, &scene_, &camera_);
      update();
    }
    QOpenGLWidget::wheelEvent(event);
  }

private:
  void loadModel()
  {
    auto fail = [this](const QString & message) {
        last_error_ = message;
        if (data_ != nullptr) {
          mj_deleteData(data_);
          data_ = nullptr;
        }
        if (model_ != nullptr) {
          mj_deleteModel(model_);
          model_ = nullptr;
        }
      };

    if (model_path_.isEmpty()) {
      fail("MuJoCo model path is empty.");
      return;
    }

    std::array<char, 1024> error_buffer{};
    model_ = mj_loadXML(model_path_.toStdString().c_str(), nullptr, error_buffer.data(), error_buffer.size());
    if (model_ == nullptr) {
      fail(QString("Failed to load MuJoCo model: %1").arg(error_buffer.data()));
      return;
    }

    data_ = mj_makeData(model_);
    if (data_ == nullptr) {
      fail("Failed to allocate MuJoCo simulation data.");
      return;
    }

    actuator_ids_.reserve(joint_definitions_.size());
    qpos_indices_.reserve(joint_definitions_.size());
    qvel_indices_.reserve(joint_definitions_.size());
    joint_limits_.reserve(joint_definitions_.size());

    for (const auto & joint_definition : joint_definitions_) {
      const int joint_id = mj_name2id(model_, mjOBJ_JOINT, joint_definition.name.c_str());
      const std::string actuator_name = joint_definition.name + "_servo";
      const int actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, actuator_name.c_str());

      if (joint_id == -1 || actuator_id == -1) {
        fail(QString(
          "MuJoCo model is missing joint or actuator for '%1'.")
          .arg(QString::fromStdString(joint_definition.name)));
        return;
      }

      actuator_ids_.push_back(actuator_id);
      qpos_indices_.push_back(model_->jnt_qposadr[joint_id]);
      qvel_indices_.push_back(model_->jnt_dofadr[joint_id]);

      const mjtNum * joint_range = model_->jnt_range + (2 * joint_id);
      joint_limits_.push_back({
        static_cast<double>(joint_range[0]),
        static_cast<double>(joint_range[1]),
      });
    }

    ee_site_id_ = mj_name2id(model_, mjOBJ_SITE, "ee_site");
    if (ee_site_id_ == -1) {
      fail("MuJoCo model is missing site 'ee_site'.");
      return;
    }

    resetSimulation();
  }

  void advanceSimulation()
  {
    if (!isReady()) {
      return;
    }

    // Compute how many MuJoCo steps to take this tick.
    // The simulation timer fires every TIMER_INTERVAL_MS milliseconds.
    // Rather than tracking wall-clock elapsed time (which can be 0 on some
    // platforms/GL configurations), take a fixed number of steps per tick so
    // the simulation always advances regardless of timer accuracy.
    const double sim_timestep = model_->opt.timestep;  // e.g. 0.002 s
    constexpr double timer_interval_sec = 0.016;        // matches setInterval(16)
    const int steps_per_tick =
      std::max(1, static_cast<int>(std::round(timer_interval_sec / sim_timestep)));

    for (int i = 0; i < steps_per_tick; ++i) {
      for (std::size_t index = 0; index < actuator_ids_.size(); ++index) {
        data_->ctrl[actuator_ids_[index]] = target_positions_[index];
      }
      mj_step(model_, data_);
    }

    // Periodic diagnostic: log sim time and first joint position every ~5 s.
    static int advance_count = 0;
    ++advance_count;
    if (advance_count % 300 == 1) {
      qInfo(
        "[sim] t=%.3f  qpos[0]=%.4f  ctrl[0]=%.4f  target[0]=%.4f  steps_per_tick=%d",
        data_->time,
        static_cast<double>(data_->qpos[qpos_indices_.empty() ? 0 : qpos_indices_[0]]),
        static_cast<double>(data_->ctrl[actuator_ids_.empty() ? 0 : actuator_ids_[0]]),
        target_positions_.empty() ? 0.0 : target_positions_[0],
        steps_per_tick);
    }

    update();
  }

  QString model_path_;
  QString last_error_;
  std::vector<JointDefinition> joint_definitions_;
  std::vector<double> target_positions_;
  std::vector<int> actuator_ids_;
  std::vector<int> qpos_indices_;
  std::vector<int> qvel_indices_;
  std::vector<std::pair<double, double>> joint_limits_;

  mjModel * model_{nullptr};
  mjData * data_{nullptr};
  int ee_site_id_{-1};

  mjvCamera camera_;
  mjvOption option_;
  mjvScene scene_;
  mjrContext context_;

  bool render_context_ready_{false};
  QTimer * simulation_timer_{nullptr};
  QPoint last_mouse_position_;
};

ArmControlPanel::ArmControlPanel(QWidget * parent)
: QWidget(parent),
  joint_definitions_(
{
  {"shoulder_yaw", -3.14159, 3.14159, 0.0},
  {"shoulder_pitch", -1.8, 1.8, 0.45},
  {"elbow_pitch", -2.2, 2.2, -0.95},
  {"wrist_pitch", -1.8, 1.8, 0.6},
}),
  latest_joint_positions_(joint_definitions_.size(), 0.0)
{
  node_ = rclcpp::Node::make_shared("robot_arm_qt_panel");

  const QString default_model_path = resolveDefaultModelPath();
  node_->declare_parameter<std::string>("model_path", default_model_path.toStdString());
  node_->declare_parameter<double>("state_pub_rate_hz", 50.0);
  node_->declare_parameter<bool>("publish_clock", true);

  const QString model_path =
    QString::fromStdString(node_->get_parameter("model_path").as_string());
  state_pub_rate_hz_ = node_->get_parameter("state_pub_rate_hz").as_double();
  publish_clock_ = node_->get_parameter("publish_clock").as_bool();

  setWindowTitle("P RoboAI MuJoCo Studio");
  setMinimumSize(1400, 860);
  setStyleSheet(
    "QWidget {"
    "  background: #eef4f8;"
    "  color: #163041;"
    "  font-family: 'Noto Sans', 'DejaVu Sans', sans-serif;"
    "  font-size: 13px;"
    "}"
    "QFrame#heroCard {"
    "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8fbfd, stop:1 #d8e8f2);"
    "  border: 1px solid #ccdae4;"
    "  border-radius: 20px;"
    "}"
    "QGroupBox {"
    "  background: rgba(255, 255, 255, 0.96);"
    "  border: 1px solid #d4e0e8;"
    "  border-radius: 16px;"
    "  margin-top: 18px;"
    "  padding: 18px 16px 16px 16px;"
    "  font-weight: 600;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: margin;"
    "  left: 14px;"
    "  padding: 0 6px;"
    "  color: #0f5f7d;"
    "}"
    "QPushButton {"
    "  background: #0d6f8c;"
    "  border: none;"
    "  border-radius: 10px;"
    "  color: white;"
    "  font-weight: 600;"
    "  min-height: 18px;"
    "  padding: 10px 16px;"
    "}"
    "QPushButton:hover {"
    "  background: #0a6280;"
    "}"
    "QPushButton#secondaryButton {"
    "  background: #ddeaf2;"
    "  color: #163041;"
    "}"
    "QPushButton#secondaryButton:hover {"
    "  background: #d2e4ee;"
    "}"
    "QCheckBox {"
    "  spacing: 8px;"
    "  color: #355364;"
    "}"
    "QSlider::groove:horizontal {"
    "  background: #d7e4ec;"
    "  height: 8px;"
    "  border-radius: 4px;"
    "}"
    "QSlider::sub-page:horizontal {"
    "  background: #0f6d8c;"
    "  border-radius: 4px;"
    "}"
    "QSlider::handle:horizontal {"
    "  background: #ef6c2f;"
    "  width: 18px;"
    "  margin: -6px 0;"
    "  border-radius: 9px;"
    "}"
    "QLabel#heroTitle {"
    "  font-size: 27px;"
    "  font-weight: 700;"
    "  color: #0a2740;"
    "}"
    "QLabel#heroSubtitle {"
    "  color: #567284;"
    "  font-size: 14px;"
    "}"
    "QLabel#logoBadge {"
    "  background: rgba(255, 255, 255, 0.72);"
    "  border: 1px solid #cfe0ea;"
    "  border-radius: 22px;"
    "  padding: 10px;"
    "}"
    "QLabel#viewportHint {"
    "  color: #5b7788;"
    "  font-size: 12px;"
    "}"
    "QLabel#telemetryValue {"
    "  color: #183446;"
    "  font-family: 'DejaVu Sans Mono', monospace;"
    "}"
    "QLabel#statusLabel {"
    "  background: rgba(255, 255, 255, 0.86);"
    "  border: 1px solid #d2dee7;"
    "  border-radius: 12px;"
    "  color: #355364;"
    "  padding: 10px 12px;"
    "}"
  );

  buildUi(model_path);
  createRosInterfaces();
  connectUi();
  syncSlidersToHome(true);
  refreshAllTargetValues();
  publishSimulationState();

  if (viewport_ != nullptr && viewport_->isReady()) {
    setStatus("Single-window MuJoCo desktop simulator is ready.");
  } else if (viewport_ != nullptr) {
    setStatus(viewport_->lastError());
  } else {
    setStatus("MuJoCo viewport failed to initialize.");
  }
}

void ArmControlPanel::buildUi(const QString & model_path)
{
  auto * root_layout = new QVBoxLayout(this);
  root_layout->setContentsMargins(22, 22, 22, 22);
  root_layout->setSpacing(18);

  auto * hero_card = makeHeroCard();
  auto * hero_layout = new QHBoxLayout(hero_card);
  hero_layout->setContentsMargins(22, 20, 22, 20);
  hero_layout->setSpacing(18);

  auto * hero_text_layout = new QVBoxLayout();
  hero_text_layout->setSpacing(8);

  auto * hero_title = new QLabel("P RoboAI MuJoCo Studio");
  hero_title->setObjectName("heroTitle");
  auto * hero_subtitle = new QLabel(
    "One Qt window for embedded MuJoCo rendering, joint control, live ROS 2 telemetry, and simulator reset."
  );
  hero_subtitle->setObjectName("heroSubtitle");
  hero_subtitle->setWordWrap(true);

  auto * logo_label = new QLabel();
  logo_label->setObjectName("logoBadge");
  logo_label->setAlignment(Qt::AlignCenter);
  logo_label->setFixedSize(156, 156);

  const QString logo_path = resolveLogoPath();
  const QPixmap logo_pixmap(logo_path);
  if (!logo_pixmap.isNull()) {
    logo_label->setPixmap(
      logo_pixmap.scaled(122, 122, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    logo_label->setToolTip(logo_path);
  } else {
    logo_label->setText("P RoboAI");
  }

  hero_text_layout->addWidget(hero_title);
  hero_text_layout->addWidget(hero_subtitle);
  hero_text_layout->addStretch(1);

  hero_layout->addLayout(hero_text_layout, 1);
  hero_layout->addWidget(logo_label, 0, Qt::AlignTop);
  root_layout->addWidget(hero_card);

  auto * body_layout = new QHBoxLayout();
  body_layout->setSpacing(18);
  root_layout->addLayout(body_layout, 1);

  auto * viewport_box = new QGroupBox("Embedded MuJoCo View");
  auto * viewport_layout = new QVBoxLayout(viewport_box);
  viewport_layout->setSpacing(10);
  viewport_ = new MujocoViewportWidget(model_path, joint_definitions_, viewport_box);
  viewport_layout->addWidget(viewport_, 1);

  auto * viewport_hint = new QLabel(
    "Left drag rotates, Shift plus drag pans, mouse wheel zooms. The viewport and controls are running the same MuJoCo simulation."
  );
  viewport_hint->setObjectName("viewportHint");
  viewport_hint->setWordWrap(true);
  viewport_layout->addWidget(viewport_hint);

  // Wrap the right column in a scroll area so the Kinematics panel is always
  // reachable even if the window is shorter than the total content height.
  auto * right_scroll = new QScrollArea();
  right_scroll->setWidgetResizable(true);
  right_scroll->setFrameShape(QFrame::NoFrame);
  right_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  right_scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

  auto * right_scroll_contents = new QWidget();
  auto * right_column_layout = new QVBoxLayout(right_scroll_contents);
  right_scroll->setWidget(right_scroll_contents);
  right_column_layout->setSpacing(18);

  auto * controls_box = new QGroupBox("Target Control");
  auto * controls_layout = new QGridLayout(controls_box);
  controls_layout->setHorizontalSpacing(14);
  controls_layout->setVerticalSpacing(12);
  controls_layout->addWidget(new QLabel("Joint"), 0, 0);
  controls_layout->addWidget(new QLabel("Target"), 0, 1);
  controls_layout->addWidget(new QLabel("Selected"), 0, 2);
  controls_layout->addWidget(new QLabel("Current"), 0, 3);

  const QFont mono_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);

  joint_widgets_.reserve(joint_definitions_.size());
  for (std::size_t index = 0; index < joint_definitions_.size(); ++index) {
    const auto & definition = joint_definitions_[index];
    const int row = static_cast<int>(index + 1);

    auto * name_label = new QLabel(QString::fromStdString(definition.name));
    auto * slider = new QSlider(Qt::Horizontal);
    slider->setRange(
      radiansToSliderValue(definition.min_position_rad),
      radiansToSliderValue(definition.max_position_rad));
    slider->setSingleStep(5);
    slider->setPageStep(25);
    slider->setTracking(true);

    auto * target_value_label = makeValueLabel();
    target_value_label->setFont(mono_font);

    auto * current_value_label = makeValueLabel();
    current_value_label->setFont(mono_font);

    controls_layout->addWidget(name_label, row, 0);
    controls_layout->addWidget(slider, row, 1);
    controls_layout->addWidget(target_value_label, row, 2);
    controls_layout->addWidget(current_value_label, row, 3);

    joint_widgets_.push_back(JointWidgets{slider, target_value_label, current_value_label});
  }

  auto * button_row = new QHBoxLayout();
  button_row->setSpacing(10);

  auto * send_button = new QPushButton("Apply Targets");
  auto * home_button = new QPushButton("Home Pose");
  auto * reset_button = new QPushButton("Reset Simulation");
  home_button->setObjectName("secondaryButton");
  reset_button->setObjectName("secondaryButton");

  button_row->addWidget(send_button);
  button_row->addWidget(home_button);
  button_row->addWidget(reset_button);
  button_row->addStretch(1);

  auto_apply_checkbox_ = new QCheckBox("Live apply at 10 Hz");
  auto_apply_checkbox_->setChecked(true);
  button_row->addWidget(auto_apply_checkbox_);

  controls_layout->addLayout(
    button_row, static_cast<int>(joint_definitions_.size() + 1), 0, 1, 4);

  auto * controls_note = new QLabel(
    "The desktop app is the simulator itself. It still listens to /joint_position_cmd and /joint_trajectory so external ROS tools can command the arm."
  );
  controls_note->setWordWrap(true);
  controls_note->setStyleSheet("color: #5b7788;");
  controls_layout->addWidget(
    controls_note, static_cast<int>(joint_definitions_.size() + 2), 0, 1, 4);

  auto * telemetry_box = new QGroupBox("Live Telemetry");
  auto * telemetry_layout = new QGridLayout(telemetry_box);
  telemetry_layout->setHorizontalSpacing(14);
  telemetry_layout->setVerticalSpacing(12);

  auto * ee_position_title = new QLabel("End-effector position");
  auto * ee_orientation_title = new QLabel("End-effector orientation");
  ee_position_label_ = makeValueLabel("x --  y --  z --");
  ee_orientation_label_ = makeValueLabel("qx --  qy --  qz --  qw --");
  ee_position_label_->setFont(mono_font);
  ee_orientation_label_->setFont(mono_font);

  auto * telemetry_note = new QLabel(
    "Current joint positions come directly from the embedded MuJoCo simulation and are also published to /joint_states for the rest of ROS 2."
  );
  telemetry_note->setWordWrap(true);
  telemetry_note->setStyleSheet("color: #5b7788;");

  telemetry_layout->addWidget(telemetry_note, 0, 0, 1, 2);
  telemetry_layout->addWidget(ee_position_title, 1, 0);
  telemetry_layout->addWidget(ee_position_label_, 1, 1);
  telemetry_layout->addWidget(ee_orientation_title, 2, 0);
  telemetry_layout->addWidget(ee_orientation_label_, 2, 1);

  // ── Kinematics panel ──────────────────────────────────────────────────────
  auto * kin_box = new QGroupBox("Kinematics");
  auto * kin_layout = new QVBoxLayout(kin_box);
  kin_layout->setSpacing(10);

  // --- FK section ---
  auto * fk_title = new QLabel("<b>Forward Kinematics</b>");
  kin_layout->addWidget(fk_title);

  // Frame selector
  auto * fk_frame_row = new QHBoxLayout();
  fk_frame_row->addWidget(new QLabel("Show frame:"));
  fk_frame_combo_ = new QComboBox();
  fk_frame_combo_->addItem("World → End-Effector");
  fk_frame_combo_->addItem("World → Shoulder Yaw");
  fk_frame_combo_->addItem("World → Shoulder Pitch");
  fk_frame_combo_->addItem("World → Elbow");
  fk_frame_combo_->addItem("World → Wrist");
  fk_frame_combo_->setCurrentIndex(0);
  fk_frame_row->addWidget(fk_frame_combo_);
  fk_frame_row->addStretch(1);
  kin_layout->addLayout(fk_frame_row);

  // 4×4 matrix grid
  auto * mat_grid = new QGridLayout();
  mat_grid->setHorizontalSpacing(6);
  mat_grid->setVerticalSpacing(3);
  const QFont mat_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      auto * cell = new QLabel("0.000");
      cell->setFont(mat_font);
      cell->setObjectName("telemetryValue");
      cell->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
      cell->setMinimumWidth(62);
      fk_matrix_labels_[row][col] = cell;
      mat_grid->addWidget(cell, row, col);
    }
  }
  kin_layout->addLayout(mat_grid);

  // Position & RPY summary
  fk_position_label_ = makeValueLabel("p = [ -- ]");
  fk_rpy_label_      = makeValueLabel("rpy = [ -- ]");
  fk_position_label_->setFont(mat_font);
  fk_rpy_label_->setFont(mat_font);
  kin_layout->addWidget(fk_position_label_);
  kin_layout->addWidget(fk_rpy_label_);

  // --- IK section ---
  auto * ik_separator = new QFrame();
  ik_separator->setFrameShape(QFrame::HLine);
  ik_separator->setFrameShadow(QFrame::Sunken);
  kin_layout->addWidget(ik_separator);

  auto * ik_title = new QLabel("<b>Inverse Kinematics</b>");
  kin_layout->addWidget(ik_title);

  // Target XYZ spinboxes — two rows for readability
  auto make_ik_spin = [](double lo, double hi, double val) {
    auto * s = new QDoubleSpinBox();
    s->setRange(lo, hi);
    s->setDecimals(3);
    s->setSingleStep(0.01);
    s->setValue(val);
    s->setSuffix(" m");
    s->setMinimumWidth(95);
    return s;
  };

  ik_x_spin_ = make_ik_spin(-1.5, 1.5, 0.35);
  ik_y_spin_ = make_ik_spin(-1.5, 1.5, 0.0);
  ik_z_spin_ = make_ik_spin( 0.0, 1.5, 0.85);

  auto * ik_xyz_grid = new QGridLayout();
  ik_xyz_grid->setHorizontalSpacing(8);
  ik_xyz_grid->setVerticalSpacing(4);
  ik_xyz_grid->addWidget(new QLabel("X (m):"), 0, 0);
  ik_xyz_grid->addWidget(ik_x_spin_,           0, 1);
  ik_xyz_grid->addWidget(new QLabel("Y (m):"), 0, 2);
  ik_xyz_grid->addWidget(ik_y_spin_,           0, 3);
  ik_xyz_grid->addWidget(new QLabel("Z (m):"), 1, 0);
  ik_xyz_grid->addWidget(ik_z_spin_,           1, 1);
  kin_layout->addLayout(ik_xyz_grid);

  // IK buttons
  auto * ik_button_row = new QHBoxLayout();
  ik_button_row->setSpacing(8);
  auto * ik_from_ee_button = new QPushButton("From EE");
  ik_from_ee_button->setObjectName("secondaryButton");
  ik_from_ee_button->setToolTip("Fill X/Y/Z from current end-effector position");
  auto * ik_solve_button = new QPushButton("Solve IK");
  ik_apply_button_ = new QPushButton("Apply Solution");
  ik_apply_button_->setEnabled(false);
  ik_apply_button_->setObjectName("secondaryButton");

  ik_button_row->addWidget(ik_from_ee_button);
  ik_button_row->addWidget(ik_solve_button);
  ik_button_row->addWidget(ik_apply_button_);
  ik_button_row->addStretch(1);
  kin_layout->addLayout(ik_button_row);

  // IK result display
  ik_result_label_ = new QLabel("—");
  ik_result_label_->setFont(mat_font);
  ik_result_label_->setObjectName("telemetryValue");
  ik_result_label_->setWordWrap(true);
  kin_layout->addWidget(ik_result_label_);

  ik_status_label_ = new QLabel("No solution computed yet.");
  ik_status_label_->setStyleSheet("color: #5b7788;");
  ik_status_label_->setWordWrap(true);
  kin_layout->addWidget(ik_status_label_);

  // Wire up IK buttons
  connect(ik_from_ee_button, &QPushButton::clicked, this, [this]() {
    if (!latest_joint_positions_.empty()) {
      kinematics::JointAngles q{};
      for (int i = 0; i < 4 && i < static_cast<int>(latest_joint_positions_.size()); ++i) {
        q[i] = latest_joint_positions_[static_cast<std::size_t>(i)];
      }
      const auto T = kinematics::computeFK(q);
      const auto p = kinematics::getPosition(T);
      ik_x_spin_->setValue(p[0]);
      ik_y_spin_->setValue(p[1]);
      ik_z_spin_->setValue(p[2]);
    }
  });
  connect(ik_solve_button, &QPushButton::clicked, this, &ArmControlPanel::solveIK);
  connect(ik_apply_button_, &QPushButton::clicked, this, &ArmControlPanel::applyIKSolution);

  // Wire up FK frame combo
  connect(fk_frame_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
    this, [this](int) {
      if (!latest_joint_positions_.empty()) {
        kinematics::JointAngles q{};
        for (int i = 0; i < 4 && i < static_cast<int>(latest_joint_positions_.size()); ++i) {
          q[i] = latest_joint_positions_[static_cast<std::size_t>(i)];
        }
        updateFKDisplay(q);
      }
    });

  right_column_layout->addWidget(controls_box);
  right_column_layout->addWidget(telemetry_box);
  right_column_layout->addWidget(kin_box);
  right_column_layout->addStretch(1);

  body_layout->addWidget(viewport_box, 5);
  body_layout->addWidget(right_scroll, 3);

  status_label_ = new QLabel();
  status_label_->setObjectName("statusLabel");
  status_label_->setWordWrap(true);
  root_layout->addWidget(status_label_);

  send_button->setProperty("actionId", "send");
  home_button->setProperty("actionId", "home");
  reset_button->setProperty("actionId", "reset");
}

void ArmControlPanel::createRosInterfaces()
{
  joint_state_publisher_ =
    node_->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  ee_pose_publisher_ =
    node_->create_publisher<geometry_msgs::msg::PoseStamped>("/end_effector_pose", 10);

  if (publish_clock_) {
    clock_publisher_ = node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  }

  position_command_subscription_ =
    node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/joint_position_cmd",
    10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr message) {
      handlePositionCommand(message);
    });

  position_command_alias_subscription_ =
    node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/joint_group_position_controller/commands",
    10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr message) {
      handlePositionCommand(message);
    });

  trajectory_subscription_ =
    node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "/joint_trajectory",
    10,
    [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr message) {
      handleTrajectoryCommand(message);
    });

  reset_service_ =
    node_->create_service<std_srvs::srv::Empty>(
    "/reset_simulation",
    [this](
      const std::shared_ptr<std_srvs::srv::Empty::Request> request,
      std::shared_ptr<std_srvs::srv::Empty::Response> response) {
      (void)request;
      (void)response;
      requestSimulationReset();
    });

  // Add the node to a persistent executor. Calling spin_some() on a
  // persistent executor is correct; rclcpp::spin_some(node_) internally
  // creates and destroys a temporary executor on every call, which can
  // drop pending callbacks and has high overhead.
  ros_executor_.add_node(node_);

  ros_spin_timer_ = new QTimer(this);
  ros_spin_timer_->setInterval(5);  // 200 Hz — low latency for incoming commands
  connect(
    ros_spin_timer_,
    &QTimer::timeout,
    this,
    [this]() {
      if (rclcpp::ok()) {
        ros_executor_.spin_some(std::chrono::milliseconds(4));
      }
    });
  ros_spin_timer_->start();

  state_publish_timer_ = new QTimer(this);
  state_publish_timer_->setInterval(
    std::max(1, static_cast<int>(std::round(1000.0 / state_pub_rate_hz_))));
  connect(
    state_publish_timer_,
    &QTimer::timeout,
    this,
    [this]() {
      publishSimulationState();
    });
  state_publish_timer_->start();

  auto_apply_timer_ = new QTimer(this);
  auto_apply_timer_->setInterval(100);
}

void ArmControlPanel::connectUi()
{
  const auto buttons = findChildren<QPushButton *>();
  for (auto * button : buttons) {
    const auto action_id = button->property("actionId").toString();
    if (action_id == "send") {
      connect(
        button,
        &QPushButton::clicked,
        this,
        [this]() {
          applySliderTargets("manual apply", true);
        });
    } else if (action_id == "home") {
      connect(
        button,
        &QPushButton::clicked,
        this,
        [this]() {
          syncSlidersToHome(true);
        });
    } else if (action_id == "reset") {
      connect(
        button,
        &QPushButton::clicked,
        this,
        [this]() {
          requestSimulationReset();
        });
    }
  }

  for (std::size_t index = 0; index < joint_widgets_.size(); ++index) {
    auto * slider = joint_widgets_[index].slider;
    connect(
      slider,
      &QSlider::valueChanged,
      this,
      [this, index](int) {
        refreshTargetValue(index);
      });
  }

  connect(
    auto_apply_checkbox_,
    &QCheckBox::toggled,
    this,
    [this](bool enabled) {
      if (enabled) {
        auto_apply_timer_->start();
        setStatus("Live apply enabled. Slider targets are streamed into the embedded simulator.");
      } else {
        auto_apply_timer_->stop();
        setStatus("Live apply disabled. Use 'Apply Targets' to push the selected pose.");
      }
    });

  connect(
    auto_apply_timer_,
    &QTimer::timeout,
    this,
    [this]() {
      applySliderTargets("live apply", false);
    });

  if (auto_apply_checkbox_->isChecked()) {
    auto_apply_timer_->start();
  }
}

void ArmControlPanel::setStatus(const QString & message)
{
  status_label_->setText(message);
}

void ArmControlPanel::syncSlidersToHome(bool apply_after_sync)
{
  std::vector<double> home_positions;
  home_positions.reserve(joint_definitions_.size());
  for (const auto & definition : joint_definitions_) {
    home_positions.push_back(definition.home_position_rad);
  }

  setSliderTargets(home_positions);

  if (apply_after_sync) {
    applySliderTargets("home pose", true);
  } else {
    setStatus("Home pose loaded into the control panel.");
  }
}

void ArmControlPanel::setSliderTargets(const std::vector<double> & positions)
{
  const std::size_t count = std::min(positions.size(), joint_widgets_.size());
  for (std::size_t index = 0; index < count; ++index) {
    QSignalBlocker blocker(joint_widgets_[index].slider);
    joint_widgets_[index].slider->setValue(radiansToSliderValue(positions[index]));
  }

  refreshAllTargetValues();
}

void ArmControlPanel::refreshTargetValue(std::size_t index)
{
  const double radians = sliderValueToRadians(joint_widgets_[index].slider->value());
  joint_widgets_[index].target_value_label->setText(formatAngle(radians));
}

void ArmControlPanel::refreshAllTargetValues()
{
  for (std::size_t index = 0; index < joint_widgets_.size(); ++index) {
    refreshTargetValue(index);
  }
}

void ArmControlPanel::applySliderTargets(const QString & origin, bool update_status)
{
  if (viewport_ == nullptr || !viewport_->isReady()) {
    setStatus("MuJoCo viewport is not ready, so target updates cannot be applied.");
    return;
  }

  std::vector<double> positions;
  positions.reserve(joint_widgets_.size());

  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(2);
  stream << "Applied " << joint_widgets_.size() << " joint targets via "
         << origin.toStdString() << ": ";

  for (std::size_t index = 0; index < joint_widgets_.size(); ++index) {
    const double radians = sliderValueToRadians(joint_widgets_[index].slider->value());
    positions.push_back(radians);
    stream << joint_definitions_[index].name << "=" << radians;
    if (index + 1 < joint_widgets_.size()) {
      stream << ", ";
    }
  }

  viewport_->setTargetPositions(positions);
  if (update_status) {
    setStatus(QString::fromStdString(stream.str()));
  }
}

void ArmControlPanel::publishSimulationState()
{
  if (viewport_ == nullptr) {
    return;
  }

  const SimulationSnapshot snapshot = viewport_->snapshot();
  if (!snapshot.valid) {
    return;
  }

  latest_joint_positions_ = snapshot.positions;
  for (std::size_t index = 0; index < joint_widgets_.size() && index < snapshot.positions.size(); ++index) {
    joint_widgets_[index].current_value_label->setText(formatAngle(snapshot.positions[index]));
  }

  // Refresh FK display with the new joint angles.
  if (snapshot.positions.size() >= 4) {
    kinematics::JointAngles q{
      snapshot.positions[0], snapshot.positions[1],
      snapshot.positions[2], snapshot.positions[3]
    };
    updateFKDisplay(q);
  }

  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << "x " << snapshot.ee_position[0]
           << "  y " << snapshot.ee_position[1]
           << "  z " << snapshot.ee_position[2];
    ee_position_label_->setText(QString::fromStdString(stream.str()));
  }

  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(3);
    stream << "qx " << snapshot.ee_orientation_xyzw[0]
           << "  qy " << snapshot.ee_orientation_xyzw[1]
           << "  qz " << snapshot.ee_orientation_xyzw[2]
           << "  qw " << snapshot.ee_orientation_xyzw[3];
    ee_orientation_label_->setText(QString::fromStdString(stream.str()));
  }

  const auto stamp = simTimeToMsg(snapshot.sim_time_sec);

  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp = stamp;
  for (const auto & definition : joint_definitions_) {
    joint_state.name.push_back(definition.name);
  }
  joint_state.position = snapshot.positions;
  joint_state.velocity = snapshot.velocities;
  joint_state.effort = snapshot.efforts;
  joint_state_publisher_->publish(joint_state);

  geometry_msgs::msg::PoseStamped ee_pose;
  ee_pose.header.stamp = stamp;
  ee_pose.header.frame_id = "world";
  ee_pose.pose.position.x = snapshot.ee_position[0];
  ee_pose.pose.position.y = snapshot.ee_position[1];
  ee_pose.pose.position.z = snapshot.ee_position[2];
  ee_pose.pose.orientation.x = snapshot.ee_orientation_xyzw[0];
  ee_pose.pose.orientation.y = snapshot.ee_orientation_xyzw[1];
  ee_pose.pose.orientation.z = snapshot.ee_orientation_xyzw[2];
  ee_pose.pose.orientation.w = snapshot.ee_orientation_xyzw[3];
  ee_pose_publisher_->publish(ee_pose);

  if (publish_clock_ && clock_publisher_ != nullptr) {
    rosgraph_msgs::msg::Clock clock_message;
    clock_message.clock = stamp;
    clock_publisher_->publish(clock_message);
  }
}

void ArmControlPanel::requestSimulationReset()
{
  if (viewport_ == nullptr || !viewport_->isReady()) {
    setStatus("MuJoCo viewport is not ready, so reset is unavailable.");
    return;
  }

  viewport_->resetSimulation();
  syncSlidersToHome(false);
  publishSimulationState();
  setStatus("Simulation reset to the home pose inside the Qt desktop window.");
}

void ArmControlPanel::handlePositionCommand(
  const std_msgs::msg::Float64MultiArray::SharedPtr message)
{
  if (message->data.size() != joint_definitions_.size()) {
    return;
  }

  std::vector<double> positions(message->data.begin(), message->data.end());
  setSliderTargets(positions);
  if (viewport_ != nullptr) {
    viewport_->setTargetPositions(positions);
  }
}

void ArmControlPanel::handleTrajectoryCommand(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr message)
{
  static int traj_count = 0;
  if (message->points.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[trajectory] received message with no points");
    return;
  }

  ++traj_count;
  if (traj_count <= 5 || traj_count % 100 == 0) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[trajectory] #%d: %zu joints, %zu points",
      traj_count,
      message->joint_names.size(),
      message->points.size());
  }

  std::vector<double> updated_positions;
  updated_positions.reserve(joint_widgets_.size());
  for (const auto & widget : joint_widgets_) {
    updated_positions.push_back(sliderValueToRadians(widget.slider->value()));
  }

  const auto & point = message->points.back();
  std::vector<std::string> joint_names;
  if (message->joint_names.empty()) {
    joint_names = {
      "shoulder_yaw",
      "shoulder_pitch",
      "elbow_pitch",
      "wrist_pitch",
    };
  } else {
    joint_names.assign(message->joint_names.begin(), message->joint_names.end());
  }

  if (joint_names.size() != point.positions.size()) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[trajectory] joint_names size %zu != positions size %zu — ignoring",
      joint_names.size(),
      point.positions.size());
    return;
  }

  for (std::size_t incoming_index = 0; incoming_index < joint_names.size(); ++incoming_index) {
    for (std::size_t joint_index = 0; joint_index < joint_definitions_.size(); ++joint_index) {
      if (joint_definitions_[joint_index].name == joint_names[incoming_index]) {
        updated_positions[joint_index] = point.positions[incoming_index];
      }
    }
  }

  if (traj_count <= 5) {
    std::string pos_str;
    for (std::size_t i = 0; i < updated_positions.size(); ++i) {
      pos_str += joint_definitions_[i].name + "=" +
        std::to_string(updated_positions[i]) + " ";
    }
    RCLCPP_INFO(node_->get_logger(), "[trajectory] targets: %s", pos_str.c_str());
  }

  setSliderTargets(updated_positions);
  if (viewport_ != nullptr) {
    viewport_->setTargetPositions(updated_positions);
  }
}

builtin_interfaces::msg::Time ArmControlPanel::simTimeToMsg(double time_seconds) const
{
  builtin_interfaces::msg::Time time_message;
  int32_t seconds = static_cast<int32_t>(std::floor(time_seconds));
  int64_t nanoseconds =
    static_cast<int64_t>((time_seconds - static_cast<double>(seconds)) * 1e9);
  if (nanoseconds >= 1'000'000'000LL) {
    ++seconds;
    nanoseconds -= 1'000'000'000LL;
  }

  time_message.sec = seconds;
  time_message.nanosec = static_cast<uint32_t>(nanoseconds);
  return time_message;
}

int ArmControlPanel::radiansToSliderValue(double radians) const
{
  return static_cast<int>(std::round(radiansToDegrees(radians) * 10.0));
}

double ArmControlPanel::sliderValueToRadians(int slider_value) const
{
  return (static_cast<double>(slider_value) / 10.0) * kPi / 180.0;
}

QString ArmControlPanel::formatAngle(double radians) const
{
  return QString("%1 deg | %2 rad")
         .arg(radiansToDegrees(radians), 0, 'f', 1)
         .arg(radians, 0, 'f', 3);
}

// ── Kinematics methods ────────────────────────────────────────────────────

void ArmControlPanel::updateFKDisplay(const std::array<double, 4> & q)
{
  // Pick the requested frame from the combo box.
  //   index 0 → T_world_EE (frame[4])
  //   index 1..4 → intermediate frames [0..3]
  const int combo_idx = fk_frame_combo_ ? fk_frame_combo_->currentIndex() : 0;
  const auto frames = kinematics::computeAllFrames(q);
  const kinematics::Mat4 & T =
    (combo_idx == 0) ? frames[4] :
    (combo_idx == 1) ? frames[0] :
    (combo_idx == 2) ? frames[1] :
    (combo_idx == 3) ? frames[2] : frames[3];

  // Fill the 4×4 label grid.
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (fk_matrix_labels_[row][col]) {
        fk_matrix_labels_[row][col]->setText(
          QString::number(T[row][col], 'f', 3));
      }
    }
  }

  // Position summary.
  const auto p = kinematics::getPosition(T);
  if (fk_position_label_) {
    fk_position_label_->setText(
      QString("p = [%1,  %2,  %3]  m")
      .arg(p[0], 0, 'f', 4)
      .arg(p[1], 0, 'f', 4)
      .arg(p[2], 0, 'f', 4));
  }

  // RPY summary (ZYX convention).
  if (fk_rpy_label_) {
    const auto rot = kinematics::getRotation(T);
    const auto rpy = kinematics::rotationToRPY(rot);
    fk_rpy_label_->setText(
      QString("rpy = [%1,  %2,  %3]  deg")
      .arg(rpy[0] * 180.0 / kPi, 0, 'f', 2)
      .arg(rpy[1] * 180.0 / kPi, 0, 'f', 2)
      .arg(rpy[2] * 180.0 / kPi, 0, 'f', 2));
  }
}

void ArmControlPanel::solveIK()
{
  if (!ik_x_spin_ || !ik_y_spin_ || !ik_z_spin_) {
    return;
  }

  const kinematics::Vec3 target = {
    ik_x_spin_->value(),
    ik_y_spin_->value(),
    ik_z_spin_->value()
  };

  // Use current joint angles as seed for faster convergence.
  kinematics::JointAngles seed{0.0, 0.45, -0.95, 0.6};
  if (latest_joint_positions_.size() >= 4) {
    for (int i = 0; i < 4; ++i) {
      seed[i] = latest_joint_positions_[static_cast<std::size_t>(i)];
    }
  }

  const kinematics::IKResult result = kinematics::computeIK(target, seed);
  ik_solution_ = result.joint_angles;
  ik_solution_valid_ = result.success;

  // Build result text.
  const auto & names = joint_definitions_;
  QString result_text;
  for (int i = 0; i < 4 && i < static_cast<int>(names.size()); ++i) {
    result_text += QString("%1:  %2 deg  |  %3 rad\n")
      .arg(QString::fromStdString(names[static_cast<std::size_t>(i)].name))
      .arg(result.joint_angles[i] * 180.0 / kPi, 0, 'f', 2)
      .arg(result.joint_angles[i], 0, 'f', 4);
  }
  if (ik_result_label_) {
    ik_result_label_->setText(result_text.trimmed());
  }

  // Status line.
  const QString status_text = QString("%1  |  error: %2 mm  |  %3 iters")
    .arg(QString::fromStdString(result.message))
    .arg(result.position_error * 1000.0, 0, 'f', 2)
    .arg(result.iterations);

  if (ik_status_label_) {
    ik_status_label_->setText(status_text);
    ik_status_label_->setStyleSheet(
      result.success ? "color: #1a7a3c;" : "color: #c0392b;");
  }

  if (ik_apply_button_) {
    ik_apply_button_->setEnabled(result.success);
  }
}

void ArmControlPanel::applyIKSolution()
{
  if (!ik_solution_valid_ || viewport_ == nullptr) {
    return;
  }

  std::vector<double> positions(
    ik_solution_.begin(), ik_solution_.end());
  setSliderTargets(positions);
  viewport_->setTargetPositions(positions);
  setStatus(QString("IK solution applied: %1° %2° %3° %4°")
    .arg(ik_solution_[0] * 180.0 / kPi, 0, 'f', 1)
    .arg(ik_solution_[1] * 180.0 / kPi, 0, 'f', 1)
    .arg(ik_solution_[2] * 180.0 / kPi, 0, 'f', 1)
    .arg(ik_solution_[3] * 180.0 / kPi, 0, 'f', 1));
}

}  // namespace robot_arm_qt_ui
