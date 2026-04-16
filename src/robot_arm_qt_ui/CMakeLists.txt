cmake_minimum_required(VERSION 3.16)
project(robot_arm_qt_ui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenGL REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)

find_package(Qt6 COMPONENTS Widgets OpenGL OpenGLWidgets QUIET)
if(Qt6_FOUND)
  set(QT_WIDGETS_TARGET Qt6::Widgets)
  set(QT_OPENGL_TARGET Qt6::OpenGL)
  set(QT_OPENGL_WIDGETS_TARGET Qt6::OpenGLWidgets)
  set(QT_VERSION_USED 6)
else()
  find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
  set(QT_WIDGETS_TARGET Qt5::Widgets)
  set(QT_OPENGL_TARGET Qt5::OpenGL)
  set(QT_OPENGL_WIDGETS_TARGET "")
  set(QT_VERSION_USED 5)
endif()

message(STATUS "Building ${PROJECT_NAME} with Qt${QT_VERSION_USED} Widgets")

get_filename_component(MUJOCO_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
file(GLOB MUJOCO_PACKAGE_DIR "${MUJOCO_WORKSPACE_ROOT}/.venv/lib/python*/site-packages/mujoco")
list(LENGTH MUJOCO_PACKAGE_DIR MUJOCO_PACKAGE_DIR_COUNT)
if(MUJOCO_PACKAGE_DIR_COUNT EQUAL 0)
  message(FATAL_ERROR "MuJoCo package not found under ${MUJOCO_WORKSPACE_ROOT}/.venv. Install requirements.txt first.")
endif()
list(GET MUJOCO_PACKAGE_DIR 0 MUJOCO_PACKAGE_DIR)

file(GLOB MUJOCO_LIBRARY "${MUJOCO_PACKAGE_DIR}/libmujoco.so*")
list(LENGTH MUJOCO_LIBRARY MUJOCO_LIBRARY_COUNT)
if(MUJOCO_LIBRARY_COUNT EQUAL 0)
  message(FATAL_ERROR "libmujoco.so was not found under ${MUJOCO_PACKAGE_DIR}.")
endif()
list(GET MUJOCO_LIBRARY 0 MUJOCO_LIBRARY)

add_executable(robot_arm_qt_panel
  src/main.cpp
  src/arm_control_panel.cpp
)

target_include_directories(
  robot_arm_qt_panel PRIVATE
  include
  "${MUJOCO_PACKAGE_DIR}/include"
)
ament_target_dependencies(
  robot_arm_qt_panel
  ament_index_cpp
  builtin_interfaces
  geometry_msgs
  rclcpp
  rosgraph_msgs
  sensor_msgs
  std_msgs
  std_srvs
  trajectory_msgs
)
target_link_libraries(
  robot_arm_qt_panel
  ${QT_WIDGETS_TARGET}
  ${QT_OPENGL_TARGET}
  ${QT_OPENGL_WIDGETS_TARGET}
  OpenGL::GL
  ${MUJOCO_LIBRARY}
)

set_target_properties(
  robot_arm_qt_panel PROPERTIES
  BUILD_RPATH "${MUJOCO_PACKAGE_DIR}"
  INSTALL_RPATH "${MUJOCO_PACKAGE_DIR}"
)

install(
  TARGETS robot_arm_qt_panel
  DESTINATION lib/${PROJECT_NAME}
)

# ─────────────────────────────────────────────────────────────────────────────
# amr_studio — standalone AMR map visualisation window (no MuJoCo dependency)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(amr_studio
  src/amr_main.cpp
  src/amr_panel.cpp
  include/robot_arm_qt_ui/amr_panel.hpp   # listed so AUTOMOC processes Q_OBJECT
)

target_include_directories(amr_studio PRIVATE include)

ament_target_dependencies(amr_studio
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
)

target_link_libraries(amr_studio
  ${QT_WIDGETS_TARGET}
)

install(
  TARGETS amr_studio
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch resource
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
