@echo off
REM ──────────────────────────────────────────────────────────────────────────
REM  P_RoboAI Studio — Windows build script
REM  Run this from the p_roboai_studio\ directory in a regular cmd prompt.
REM
REM  Requirements:
REM    Python 3.10-3.12  (64-bit)
REM    pip install mujoco PyQt6 numpy pyinstaller
REM ──────────────────────────────────────────────────────────────────────────

setlocal

echo [1/3] Checking dependencies...
python -c "import mujoco, PyQt6, numpy" || (
    echo ERROR: Missing dependencies. Run:
    echo   pip install mujoco PyQt6 numpy pyinstaller
    exit /b 1
)

echo [2/3] Running PyInstaller...
pyinstaller --clean p_roboai_studio.spec

if errorlevel 1 (
    echo BUILD FAILED.
    exit /b 1
)

echo [3/3] Done.
echo Executable folder: dist\P_RoboAI_Studio\
echo Run: dist\P_RoboAI_Studio\P_RoboAI_Studio.exe
endlocal
