# PyInstaller spec for P_RoboAI Studio
# Build: pyinstaller p_roboai_studio.spec
import sys
import os
from pathlib import Path
import mujoco

block_cipher = None

# Locate MuJoCo data (DLLs, default meshes, shaders, etc.)
mujoco_pkg_dir = Path(mujoco.__file__).parent

added_files = [
    # MuJoCo runtime data
    (str(mujoco_pkg_dir / "*.dll"),   "mujoco") if sys.platform == "win32" else
    (str(mujoco_pkg_dir / "*.so*"),   "mujoco"),
    (str(mujoco_pkg_dir / "model"),   "mujoco/model"),
    (str(mujoco_pkg_dir / "include"), "mujoco/include"),
]

# Filter out tuples where the source glob matches nothing (safe fallback)
import glob as _glob
added_files = [(src, dst) for src, dst in added_files
               if _glob.glob(src)]

a = Analysis(
    ["main.py"],
    pathex=[str(Path(__file__).parent)],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        "mujoco",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "numpy",
        "xml.etree.ElementTree",
        "urdf_loader",
        "mujoco_viewport",
        "arm_panel",
        "amr_panel",
        "main_window",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="P_RoboAI_Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # no terminal window on Windows
    icon=None,                # set to "icon.ico" if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="P_RoboAI_Studio",
)
