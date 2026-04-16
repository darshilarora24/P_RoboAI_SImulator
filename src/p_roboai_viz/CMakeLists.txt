cmake_minimum_required(VERSION 3.16)
project(p_roboai_viz)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ── ROS2 dependencies ──────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenGL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)

# ── Qt6 (with Qt5 fallback) ────────────────────────────────────────────────────
find_package(Qt6 COMPONENTS Widgets OpenGL OpenGLWidgets Gui QUIET)
if(Qt6_FOUND)
  set(QT_WIDGETS_TARGET          Qt6::Widgets)
  set(QT_OPENGL_TARGET           Qt6::OpenGL)
  set(QT_OPENGL_WIDGETS_TARGET   Qt6::OpenGLWidgets)
  set(QT_GUI_TARGET              Qt6::Gui)
  set(QT_VERSION_USED 6)
else()
  find_package(Qt5 COMPONENTS Widgets OpenGL Gui REQUIRED)
  set(QT_WIDGETS_TARGET          Qt5::Widgets)
  set(QT_OPENGL_TARGET           Qt5::OpenGL)
  set(QT_OPENGL_WIDGETS_TARGET   "")
  set(QT_GUI_TARGET              Qt5::Gui)
  set(QT_VERSION_USED 5)
endif()
message(STATUS "Building ${PROJECT_NAME} with Qt${QT_VERSION_USED}")

# ── MuJoCo — located in workspace .venv ───────────────────────────────────────
get_filename_component(MUJOCO_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
file(GLOB MUJOCO_PACKAGE_DIR
     "${MUJOCO_WORKSPACE_ROOT}/.venv/lib/python*/site-packages/mujoco")
list(LENGTH MUJOCO_PACKAGE_DIR _MJ_COUNT)
if(_MJ_COUNT EQUAL 0)
  message(FATAL_ERROR
    "MuJoCo not found under ${MUJOCO_WORKSPACE_ROOT}/.venv — "
    "run 'pip install -r requirements.txt' inside the venv first.")
endif()
list(GET MUJOCO_PACKAGE_DIR 0 MUJOCO_PACKAGE_DIR)

file(GLOB MUJOCO_LIBRARY "${MUJOCO_PACKAGE_DIR}/libmujoco.so*")
list(LENGTH MUJOCO_LIBRARY _MJ_LIB_COUNT)
if(_MJ_LIB_COUNT EQUAL 0)
  message(FATAL_ERROR "libmujoco.so not found under ${MUJOCO_PACKAGE_DIR}")
endif()
list(GET MUJOCO_LIBRARY 0 MUJOCO_LIBRARY)
message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")

# ── Sources ────────────────────────────────────────────────────────────────────
set(VIZ_HEADERS
  include/p_roboai_viz/main_window.hpp
  include/p_roboai_viz/map_view.hpp
  include/p_roboai_viz/mujoco_3d_view.hpp
  include/p_roboai_viz/displays_panel.hpp
  include/p_roboai_viz/status_panel.hpp
)

add_executable(p_roboai_viz
  src/main.cpp
  src/main_window.cpp
  src/map_view.cpp
  src/mujoco_3d_view.cpp
  src/displays_panel.cpp
  src/status_panel.cpp
  ${VIZ_HEADERS}          # listed so AUTOMOC picks up Q_OBJECT macros
)

target_include_directories(p_roboai_viz PRIVATE
  include
  "${MUJOCO_PACKAGE_DIR}/include"
)

ament_target_dependencies(p_roboai_viz
  ament_index_cpp
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2_msgs
)

target_link_libraries(p_roboai_viz
  ${QT_WIDGETS_TARGET}
  ${QT_OPENGL_TARGET}
  ${QT_OPENGL_WIDGETS_TARGET}
  ${QT_GUI_TARGET}
  OpenGL::GL
  ${MUJOCO_LIBRARY}
)

set_target_properties(p_roboai_viz PROPERTIES
  BUILD_RPATH   "${MUJOCO_PACKAGE_DIR}"
  INSTALL_RPATH "${MUJOCO_PACKAGE_DIR}"
)

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS p_roboai_viz
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
