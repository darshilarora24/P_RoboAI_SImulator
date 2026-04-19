# P_RoboAI Simulator

A full-stack robotics simulation and AI platform built on **MuJoCo**, **ROS2**, and **PyQt6**.  
Combines physics simulation, reinforcement learning, YOLO object detection, and Gemini Robotics LLM — all in one unified environment.

---

## Features

| Module | Description |
|--------|-------------|
| **P_RoboAI Studio** | Standalone Qt GUI — load URDF, visualize in 3D, control joints, train RL, run YOLO and Gemini |
| **MuJoCo Simulation** | High-fidelity physics for AMR and robot arm with EGL offscreen rendering |
| **Reinforcement Learning** | PPO / SAC / TD3 via Stable Baselines 3 — train and deploy policies for AMR navigation and arm reaching |
| **YOLO Vision** | Real-time object detection (YOLO11 / YOLOv8) on camera feeds with MuJoCo fallback renderer |
| **Gemini Robotics** | Gemini Robotics ER 1.6 LLM with RAG, Sim2Real policy transfer, and Real2Sim system identification |
| **ROS2 Integration** | Full ROS2 node graph — all modules publish/subscribe standard message types |

---

## Repository Structure

```
P_RoboAI_SImulator/
├── p_roboai_studio/          # Standalone Qt GUI application
│   ├── main.py               # Entry point
│   ├── main_window.py        # QMainWindow — wires all panels
│   ├── mujoco_viewport.py    # 3D MuJoCo renderer widget
│   ├── urdf_loader.py        # URDF → MJCF converter
│   ├── arm_panel.py          # Joint slider control (arm)
│   ├── amr_panel.py          # WASD keyboard drive (AMR)
│   ├── rl_panel.py           # RL train / eval panel
│   ├── rl_trainer.py         # SB3 training in QThread
│   ├── rl_env.py             # MuJoCo Gymnasium environments
│   ├── yolo_panel.py         # YOLO live detection panel
│   ├── yolo_detector.py      # Ultralytics YOLO wrapper
│   ├── gemini_panel.py       # Gemini LLM chat + RAG + Sim2Real panel
│   ├── rag_engine.py         # Vector knowledge base (Gemini embeddings / TF-IDF)
│   ├── sim2real_bridge.py    # SB3 → ONNX export, calibration, domain gap
│   ├── real2sim_bridge.py    # Online system ID via RLS
│   └── requirements.txt
│
└── src/                      # ROS2 packages
    ├── robot_amr_mujoco_sim/ # AMR MuJoCo simulation node + camera publisher
    ├── robot_arm_mujoco_sim/ # Robot arm MuJoCo simulation node
    ├── robot_amr_description/# AMR URDF / meshes
    ├── robot_amr_sim/        # Gazebo AMR sim (alternative)
    ├── robot_arm_qt_ui/      # Arm Qt UI ROS2 node
    ├── p_roboai_rl/          # RL nodes (AMR + Arm)
    ├── p_roboai_yolo/        # YOLO detection node
    ├── p_roboai_gemini/      # Gemini Robotics LLM node
    ├── p_roboai_nav2/        # Nav2 integration
    ├── p_roboai_slam/        # SLAM integration
    └── p_roboai_viz/         # Visualization utilities
```

---

## Requirements

### System
- Ubuntu 22.04 or 24.04
- ROS2 Humble or Jazzy
- Python 3.10+
- GPU recommended for YOLO and RL training (CPU works too)

### Python dependencies
```bash
pip install -r p_roboai_studio/requirements.txt
```

| Package | Purpose |
|---------|---------|
| `mujoco>=3.1.0` | Physics simulation and rendering |
| `PyQt6>=6.6.0` | Qt GUI |
| `numpy>=1.26.0` | Numerical computation |
| `gymnasium>=0.29.0` | RL environment interface |
| `stable-baselines3>=2.3.0` | PPO / SAC / TD3 algorithms |
| `ultralytics>=8.3.0` | YOLO11 / YOLOv8 detection |
| `google-generativeai>=0.8.0` | Gemini API |
| `onnxruntime>=1.18.0` | Sim2Real ONNX inference |
| `torch>=2.2.0` | Policy ONNX export |
| `Pillow>=10.0.0` | Image encoding for Gemini multimodal |

---

## Installation

### 1. Clone and set up Python environment

```bash
git clone <repo-url> P_RoboAI_SImulator
cd P_RoboAI_SImulator

python3 -m venv .venv
source .venv/bin/activate
pip install -r p_roboai_studio/requirements.txt
```

### 2. Build ROS2 workspace

```bash
source /opt/ros/humble/setup.bash   # or jazzy
colcon build --symlink-install
source install/setup.bash
```

### 3. Set MuJoCo rendering backend

```bash
export MUJOCO_GL=egl     # headless / SSH (recommended)
# or
export MUJOCO_GL=osmesa  # software fallback if EGL unavailable
```

---

## Launch Guide

### P_RoboAI Studio (standalone — no ROS2 needed)

```bash
cd p_roboai_studio
export MUJOCO_GL=egl
python3 main.py
```

- **File → Open URDF** — load any robot URDF or MJCF
- **View → Reinforcement Learning Panel** — train / run RL policy
- **View → YOLO Vision Panel** — live object detection
- **View → Gemini Robotics Panel** — LLM chat, knowledge base, Sim2Real, Real2Sim

---

### AMR MuJoCo Simulation

```bash
ros2 launch robot_amr_mujoco_sim amr_mujoco.launch.py
```

**Published topics:**

| Topic | Type | Description |
|-------|------|-------------|
| `/amr/camera/image` | `sensor_msgs/Image` | Overhead camera at 10 Hz |
| `/odom` | `nav_msgs/Odometry` | AMR odometry |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan |

**Subscribed topics:**

| Topic | Type |
|-------|------|
| `/cmd_vel` | `geometry_msgs/Twist` |

---

### YOLO Object Detection

```bash
# Default (subscribes to /amr/camera/image):
ros2 launch p_roboai_yolo yolo.launch.py

# Custom model and threshold:
ros2 launch p_roboai_yolo yolo.launch.py model_name:=yolov8s.pt conf_thresh:=0.4

# Custom image topic:
ros2 launch p_roboai_yolo yolo.launch.py image_topic:=/camera/image_raw
```

**Published topics:**

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/detections` | `std_msgs/String` | JSON list of detections |
| `/yolo/image` | `sensor_msgs/Image` | Annotated RGB image |
| `/yolo/status` | `std_msgs/String` | Node status |

---

### Reinforcement Learning

```bash
# Train AMR navigation (PPO, 200k steps):
ros2 launch p_roboai_rl amr_rl.launch.py

# Train with custom steps:
ros2 launch p_roboai_rl amr_rl.launch.py total_steps:=500000

# Run trained AMR policy:
ros2 launch p_roboai_rl amr_rl.launch.py mode:=run

# Train robot arm (end-effector reaching):
ros2 launch p_roboai_rl arm_rl.launch.py

# Run trained arm policy:
ros2 launch p_roboai_rl arm_rl.launch.py mode:=run
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `train` | `train` or `run` |
| `policy_path` | `~/p_roboai_rl/amr_policy` | Save / load path |
| `total_steps` | `200000` | Training timesteps |

**Published topics (during training):**

| Topic | Description |
|-------|-------------|
| `/amr_rl/status` | Episode reward and step count |
| `/arm_rl/status` | Arm training progress |

---

### Gemini Robotics LLM

```bash
export GOOGLE_API_KEY=AIza...

# Default — Gemini Robotics ER 1.6:
ros2 launch p_roboai_gemini gemini.launch.py

# Fallback if not on ER allowlist:
ros2 launch p_roboai_gemini gemini.launch.py model_name:=gemini-2.0-flash

# AMR with Sim2Real (export SB3 policy to ONNX):
ros2 launch p_roboai_gemini gemini.launch.py \
    enable_sim2real:=true \
    policy_zip:=~/p_roboai_rl/amr_policy.zip

# Arm with Real2Sim system identification:
ros2 launch p_roboai_gemini gemini.launch.py \
    robot_type:=arm  n_joints:=6  enable_real2sim:=true
```

**Send a natural language query:**
```bash
ros2 topic pub --once /gemini/query std_msgs/String \
    "data: 'navigate to the charging station'"

ros2 topic echo /gemini/response
ros2 topic echo /gemini/robot_command
```

**Published topics:**

| Topic | Type | Description |
|-------|------|-------------|
| `/gemini/response` | `std_msgs/String` | Gemini NL response |
| `/gemini/robot_command` | `std_msgs/String` | Structured JSON command |
| `/gemini/domain_gap` | `std_msgs/String` | Sim2Real gap + DR suggestions |
| `/gemini/real2sim_params` | `std_msgs/String` | Identified dynamic parameters |
| `/gemini/status` | `std_msgs/String` | Node health JSON |

---

### Full Stack

```bash
# Terminal 1 — physics sim
ros2 launch robot_amr_mujoco_sim amr_mujoco.launch.py

# Terminal 2 — YOLO vision
ros2 launch p_roboai_yolo yolo.launch.py

# Terminal 3 — RL policy execution
ros2 launch p_roboai_rl amr_rl.launch.py mode:=run

# Terminal 4 — Gemini LLM
export GOOGLE_API_KEY=AIza...
ros2 launch p_roboai_gemini gemini.launch.py
```

---

## Gemini Robotics ER 1.6

This platform integrates **Gemini Robotics ER 1.6** (Embodied Reasoning) — Google DeepMind's robotics-specialized multimodal model.

Capabilities used in P_RoboAI:

- **Spatial reasoning** over camera frames, LiDAR, and joint states simultaneously
- **Manipulation planning** — generates joint-space and Cartesian motion commands
- **RAG grounding** — retrieves URDF specs, task docs, and sensor history before each response
- **Sim2Real awareness** — advises on domain gap and calibration steps

> **Access:** Gemini Robotics ER 1.6 requires allowlist access via Google DeepMind's robotics program.  
> If not yet approved, set `model_name:=gemini-2.0-flash` — all other features work identically.

### RAG Knowledge Base

The `RobotKnowledgeBase` automatically embeds:
- URDF link and joint definitions
- Task documentation
- Sensor state snapshots
- Trajectory logs

Retrieval uses **Gemini text-embedding-004** (768-dim cosine similarity) with **TF-IDF fallback** when offline or without an API key.

### Sim2Real Pipeline

```
Train in MuJoCo  →  Export to ONNX  →  Calibration layer  →  Real robot
```

- Per-joint scale/offset calibration from paired sim/real rollouts
- Action delay buffer compensation
- Domain gap tracked as RMS residual; auto-suggests friction/damping DR ranges

### Real2Sim Pipeline

```
Real sensors  →  RLS system identification  →  Update MuJoCo model params
```

- Recursive least squares identifies joint damping, Coulomb friction, actuator gain
- Writes directly into live `mujoco.MjModel` (`dof_damping`, `dof_frictionloss`, `actuator_gainprm`)
- Confidence score increases as estimate stabilizes over time

---

## RL Environments

### MuJoCoAMREnv

| Property | Value |
|----------|-------|
| Observation | `[dx_r, dy_r, dist, sin(θ_err), cos(θ_err), v_left, v_right]` — 7-dim |
| Action | `[v_left, v_right]` continuous ±3 m/s |
| Reward | `-0.1 × dist  +3.0 on reach  −0.5 on collision` |
| Algorithms | PPO, SAC, TD3 |

### MuJoCoArmEnv

| Property | Value |
|----------|-------|
| Observation | `joint_pos(n) + joint_vel(n) + ee_pos(3) + target(3)` |
| Action | Joint torques / velocities (n-dim) |
| Reward | `-dist_to_target  +2.0 if dist < 5 cm` |
| Algorithms | PPO, SAC, TD3 |

---

## Keyboard Shortcuts (Studio)

| Key | Action |
|-----|--------|
| `Ctrl+O` | Open URDF |
| `Ctrl+R` | Reload URDF |
| `P` | Pause / Resume physics |
| `W A S D` | Drive AMR (AMR panel focused) |
| `Dbl-click 3D` | Reset camera |
| `Drag` | Orbit camera |
| `Right-drag` | Zoom |
| `Mid-drag` | Pan |

---

## Troubleshooting

**MuJoCo renderer unavailable / black viewport**
```bash
export MUJOCO_GL=egl      # try EGL first
export MUJOCO_GL=osmesa   # software fallback
```

**"Image width > framebuffer width" error**  
Rebuild after reloading the URDF. The loader sets `<global offwidth="1920" offheight="1080"/>` automatically.

**Gemini Robotics ER 1.6 returns 403**  
Model requires allowlist access. Use `model_name:=gemini-2.0-flash` until approved.

**YOLO model not found**  
First run downloads the model automatically. Pre-download manually if offline:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

**SB3 policy not loading in ROS2 arm node**  
Set `urdf_path` explicitly to skip auto-detection:
```bash
ros2 launch p_roboai_rl arm_rl.launch.py mode:=run urdf_path:=/path/to/robot.urdf
```

**ROS2 node can't find ultralytics / stable-baselines3**  
Nodes auto-bootstrap the `.venv` in the workspace root. Ensure packages are installed there:
```bash
.venv/bin/pip install ultralytics stable-baselines3 google-generativeai
```
