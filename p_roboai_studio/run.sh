#!/usr/bin/env bash
# Launch P_RoboAI Studio — sets up MUJOCO_GL and uses the venv if present.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pick the right Python (venv > system)
VENV_PY="$SCRIPT_DIR/../.venv/bin/python3"
if [ -x "$VENV_PY" ]; then
    PY="$VENV_PY"
else
    PY="python3"
fi

# Let mujoco_viewport auto-detect EGL/osmesa; only set if not already in env
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "$SCRIPT_DIR"
exec "$PY" main.py "$@"
