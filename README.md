# ROS 2 + MuJoCo Robot Arm Simulator

This workspace contains a minimal 3D robot-arm simulator built with ROS 2 and MuJoCo. It ships with:

- A 4-DOF articulated arm model in MuJoCo XML
- A matching URDF for `robot_state_publisher`
- A ROS 2 simulator node that steps MuJoCo and publishes `/joint_states`, `/clock`, and `/end_effector_pose`
- Command interfaces for `trajectory_msgs/JointTrajectory` and `std_msgs/Float64MultiArray`
- A Qt C++ desktop app with an embedded MuJoCo viewport, control panel, and live telemetry in one window
- A small sine-wave commander node for a quick motion demo

## Workspace Layout

```text
src/robot_arm_mujoco_sim/
  launch/
  models/
  robot_arm_mujoco_sim/
  urdf/
src/robot_arm_qt_ui/
  include/
  launch/
  src/
```

## Prerequisites

- ROS 2 installed and sourced, such as Humble or Jazzy
- MuJoCo Python bindings installed in a workspace-local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

- Qt Widgets development files installed. Either Qt 6 or Qt 5 works:

```bash
sudo apt install qt6-base-dev
```

- Common ROS 2 runtime packages:

```bash
sudo apt install ros-$ROS_DISTRO-robot-state-publisher
```

## Build

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
```

The simulator node automatically checks for `mujoco` inside `./.venv`, so you do not need to activate the virtual environment before launching.

## Run The Simulator

Launch the simulator with the MuJoCo viewer:

```bash
ros2 launch robot_arm_mujoco_sim simulator.launch.py
```

Run headless:

```bash
ros2 launch robot_arm_mujoco_sim simulator.launch.py use_viewer:=false
```

Run the built-in motion demo:

```bash
ros2 launch robot_arm_mujoco_sim demo.launch.py
```

Run the single-window Qt + MuJoCo desktop app:

```bash
ros2 launch robot_arm_qt_ui desktop.launch.py
```

Run just the Qt simulator window:

```bash
ros2 run robot_arm_qt_ui robot_arm_qt_panel
```

## Manual Joint Commands

Send direct joint targets in joint order:

```bash
ros2 topic pub --once /joint_position_cmd std_msgs/msg/Float64MultiArray "{data: [0.0, 0.45, -0.95, 0.6]}"
```

Or send a trajectory command:

```bash
ros2 topic pub --once /joint_trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: ['shoulder_yaw', 'shoulder_pitch', 'elbow_pitch', 'wrist_pitch'], points: [{positions: [0.2, 0.6, -1.1, 0.4]}]}"
```

Reset the simulation:

```bash
ros2 service call /reset_simulation std_srvs/srv/Empty
```

## Published Topics

- `/joint_states`
- `/clock`
- `/end_effector_pose`

## Command Topics

- `/joint_position_cmd`
- `/joint_trajectory`
- `/joint_group_position_controller/commands`

## Desktop UI

- The desktop app is written in C++ and embeds the MuJoCo simulation directly inside the Qt window.
- It provides a live 3D MuJoCo viewport, joint sliders, a home-pose shortcut, a simulator reset button, and end-effector readouts.
- The app still listens on `/joint_position_cmd` and `/joint_trajectory`, so ROS 2 tools can command the integrated simulator.
- Live joint state, end-effector pose, and `/clock` are published from the Qt app itself.
- The company logo is loaded from `src/robot_arm_qt_ui/resource/proboai_logo.jpeg` and installed with the UI package.

## Notes

- `ros2 launch robot_arm_mujoco_sim simulator.launch.py` still starts the standalone Python simulator with the separate MuJoCo viewer.
- `robot_state_publisher` is launched by default so the simulated joints can also be consumed by the rest of the ROS 2 stack.
- The provided arm is intended as a clean starter model that you can later replace with your own URDF and MuJoCo dynamics.
