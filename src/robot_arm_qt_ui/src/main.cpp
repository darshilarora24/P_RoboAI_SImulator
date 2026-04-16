#include <QApplication>
#include <QCoreApplication>
#include <QSurfaceFormat>

#include <rclcpp/rclcpp.hpp>

#include "robot_arm_qt_ui/arm_control_panel.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Switch Qt's XCB OpenGL backend from GLX to EGL before QApplication is
  // created.  With GLX, requesting an explicit OpenGL version (e.g. 3.3)
  // causes NVIDIA and some Mesa drivers to omit GL_ARB_framebuffer_object
  // from glGetString(GL_EXTENSIONS) because FBOs are core since OpenGL 3.0.
  // MuJoCo's mjr_makeContext() checks the extension string directly and calls
  // exit() when it does not find the entry.  The EGL backend returns a
  // complete extension table unconditionally, so MuJoCo's check always passes.
  // libqxcb-egl-integration.so is present on this system; the env-var must be
  // set before QApplication so the xcb platform plugin picks it up.
  qputenv("QT_XCB_GL_INTEGRATION", "xcb_egl");
  QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);

  QSurfaceFormat surface_format;
  surface_format.setDepthBufferSize(24);
  surface_format.setStencilBufferSize(8);
  // Do not pin an explicit version.  The NVIDIA driver (and Mesa) will return
  // the highest supported compatibility context (typically OpenGL 4.6).
  // A 4.x compatibility context keeps GL_ARB_framebuffer_object in the
  // extension string, which MuJoCo's renderer requires.  Pinning to 3.3
  // caused NVIDIA to omit that extension string (FBOs are core since 3.0)
  // and MuJoCo would call exit() with "ARB_framebuffer_object required".
  surface_format.setProfile(QSurfaceFormat::CompatibilityProfile);
  QSurfaceFormat::setDefaultFormat(surface_format);

  QApplication application(argc, argv);

  robot_arm_qt_ui::ArmControlPanel panel;
  panel.show();

  const int exit_code = application.exec();
  rclcpp::shutdown();
  return exit_code;
}
