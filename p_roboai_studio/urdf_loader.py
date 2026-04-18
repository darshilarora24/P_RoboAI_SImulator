"""
urdf_loader.py  —  P_RoboAI Studio

Parses a URDF file and produces:
  1. An MJCF XML string ready for mujoco.MjModel.from_xml_string()
  2. A list of JointInfo describing every controllable joint for UI generation
  3. A RobotKind enum (ARM or AMR) detected from the joint topology

URDF features handled
---------------------
  Links      : inertial (mass, inertia), visual / collision geometry
  Geometry   : box, cylinder, sphere, mesh (.stl / .obj)
  Joints     : fixed, revolute, continuous, prismatic
  Limits     : effort, velocity, lower/upper
  Origins    : xyz + rpy on joints and geometry
  Meshes     : resolved relative to the URDF file; assets injected into MJCF
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


# ── Public data types ─────────────────────────────────────────────────────────

class RobotKind(Enum):
    ARM = auto()   # fixed-base manipulator
    AMR = auto()   # mobile differential-drive robot


@dataclass
class JointInfo:
    name:         str
    joint_type:   str          # 'revolute' | 'continuous' | 'prismatic'
    lower:        float = -3.14159
    upper:        float =  3.14159
    ctrl_index:   int   = 0    # index into MuJoCo data.ctrl[]
    axis:         list  = field(default_factory=lambda: [0.0, 0.0, 1.0])
    is_wheel:     bool  = False


@dataclass
class LoadResult:
    mjcf_xml:    str
    joints:      list[JointInfo]
    kind:        RobotKind
    model_name:  str


# ── Internal URDF data classes ────────────────────────────────────────────────

@dataclass
class _Geometry:
    gtype:    str                          # 'box'|'cylinder'|'sphere'|'mesh'
    size:     list[float] = field(default_factory=list)  # box half-extents or [r,l]
    radius:   float = 0.05
    filename: str   = ""


@dataclass
class _Link:
    name:          str
    mass:          float = 1.0
    com:           list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inertia_diag:  list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    col_geom:      Optional[_Geometry] = None
    col_xyz:       list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    col_rpy:       list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class _Joint:
    name:        str
    jtype:       str          # 'fixed'|'revolute'|'continuous'|'prismatic'
    parent:      str
    child:       str
    xyz:         list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy:         list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis:        list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    lower:       float = -3.14159
    upper:       float =  3.14159
    vel_limit:   float =  5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _xyz_rpy(el: Optional[ET.Element]) -> tuple[list[float], list[float]]:
    if el is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz_s = el.get("xyz", "0 0 0")
    rpy_s = el.get("rpy", "0 0 0")
    return _floats(xyz_s), _floats(rpy_s)


def _fmt(vals: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _rpy_to_euler_mujoco(rpy: list[float]) -> list[float]:
    """URDF rpy = roll-pitch-yaw (XYZ extrinsic). MuJoCo euler = same convention."""
    return rpy   # MuJoCo 'euler' attribute uses XYZ extrinsic by default (same as URDF)


_WHEEL_HINTS = re.compile(r"wheel|caster|drive|roue", re.IGNORECASE)


# ── URDF parser ───────────────────────────────────────────────────────────────

def _parse_urdf(urdf_path: Path) -> tuple[dict[str, _Link], dict[str, _Joint], str]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    model_name = root.get("name", urdf_path.stem)

    links:  dict[str, _Link]  = {}
    joints: dict[str, _Joint] = {}

    for lel in root.findall("link"):
        lname = lel.get("name", "")
        link  = _Link(name=lname)

        inel = lel.find("inertial")
        if inel is not None:
            mel = inel.find("mass")
            if mel is not None:
                link.mass = max(float(mel.get("value", 1.0)), 1e-4)
            com_el = inel.find("origin")
            if com_el is not None:
                link.com, _ = _xyz_rpy(com_el)
            iel = inel.find("inertia")
            if iel is not None:
                ixx = float(iel.get("ixx", 0.01))
                iyy = float(iel.get("iyy", 0.01))
                izz = float(iel.get("izz", 0.01))
                link.inertia_diag = [max(ixx, 1e-6), max(iyy, 1e-6), max(izz, 1e-6)]

        # Prefer collision geometry; fall back to visual
        for tag in ("collision", "visual"):
            gel = lel.find(tag)
            if gel is None:
                continue
            orig = gel.find("origin")
            link.col_xyz, link.col_rpy = _xyz_rpy(orig)
            geom_el = gel.find("geometry")
            if geom_el is None:
                continue
            box  = geom_el.find("box")
            cyl  = geom_el.find("cylinder")
            sph  = geom_el.find("sphere")
            mesh = geom_el.find("mesh")
            if box is not None:
                sz = _floats(box.get("size", "0.1 0.1 0.1"))
                link.col_geom = _Geometry("box", size=[s / 2.0 for s in sz])
                break
            elif cyl is not None:
                r = float(cyl.get("radius", 0.05))
                l = float(cyl.get("length", 0.1))
                link.col_geom = _Geometry("cylinder", size=[r, l / 2.0])
                break
            elif sph is not None:
                r = float(sph.get("radius", 0.05))
                link.col_geom = _Geometry("sphere", radius=r)
                break
            elif mesh is not None:
                fn = mesh.get("filename", "")
                fn = fn.replace("package://", "").replace("file://", "")
                link.col_geom = _Geometry("mesh", filename=fn)
                break

        links[lname] = link

    for jel in root.findall("joint"):
        jname  = jel.get("name", "")
        jtype  = jel.get("type", "fixed")
        parent = jel.find("parent").get("link", "")  # type: ignore
        child  = jel.find("child").get("link",  "")  # type: ignore
        orig   = jel.find("origin")
        xyz, rpy = _xyz_rpy(orig)

        axis_el = jel.find("axis")
        axis = _floats(axis_el.get("xyz", "0 0 1")) if axis_el is not None else [0.0, 0.0, 1.0]

        lower = -3.14159
        upper =  3.14159
        vel   =  5.0
        lim   = jel.find("limit")
        if lim is not None:
            lower = float(lim.get("lower", lower))
            upper = float(lim.get("upper", upper))
            vel   = float(lim.get("velocity", vel))

        joints[jname] = _Joint(jname, jtype, parent, child, xyz, rpy, axis, lower, upper, vel)

    return links, joints, model_name


# ── Robot type detection ──────────────────────────────────────────────────────

def _detect_kind(joints: dict[str, _Joint]) -> tuple[RobotKind, list[str]]:
    """Return (kind, list_of_wheel_joint_names)."""
    wheel_joints = [
        j.name for j in joints.values()
        if j.jtype == "continuous" and _WHEEL_HINTS.search(j.name)
    ]
    if not wheel_joints:
        # Check for two opposing continuous joints regardless of name
        cont = [j for j in joints.values() if j.jtype == "continuous"]
        if len(cont) >= 2:
            wheel_joints = [c.name for c in cont]

    kind = RobotKind.AMR if wheel_joints else RobotKind.ARM
    return kind, wheel_joints


# ── Kinematic tree ────────────────────────────────────────────────────────────

def _find_root(links: dict[str, _Link], joints: dict[str, _Joint]) -> str:
    children = {j.child for j in joints.values()}
    candidates = [l for l in links if l not in children]
    return candidates[0] if candidates else next(iter(links))


def _children_of(parent_link: str, joints: dict[str, _Joint]) -> list[_Joint]:
    return [j for j in joints.values() if j.parent == parent_link]


# ── Geometry → MJCF snippet ───────────────────────────────────────────────────

def _geom_xml(link: _Link, urdf_dir: Path,
              mesh_assets: dict[str, str],
              rgba: str = "0.45 0.55 0.65 1") -> str:
    g = link.col_geom
    if g is None:
        # Default tiny sphere so the link is visible
        return f'<geom type="sphere" size="0.02" rgba="{rgba}" mass="{link.mass:.4g}"/>'

    xyz_s   = _fmt(link.col_xyz)
    euler_s = _fmt(_rpy_to_euler_mujoco(link.col_rpy))
    pos_attr = f'pos="{xyz_s}"' if any(v != 0 for v in link.col_xyz) else ""
    euler_attr = f'euler="{euler_s}"' if any(v != 0 for v in link.col_rpy) else ""

    if g.gtype == "box":
        sz = _fmt(g.size)
        return (f'<geom type="box" size="{sz}" {pos_attr} {euler_attr} '
                f'rgba="{rgba}" mass="{link.mass:.4g}"/>')

    elif g.gtype == "cylinder":
        r, hl = g.size[0], g.size[1]
        return (f'<geom type="cylinder" size="{r:.4g} {hl:.4g}" {pos_attr} {euler_attr} '
                f'rgba="{rgba}" mass="{link.mass:.4g}"/>')

    elif g.gtype == "sphere":
        return (f'<geom type="sphere" size="{g.radius:.4g}" {pos_attr} {euler_attr} '
                f'rgba="{rgba}" mass="{link.mass:.4g}"/>')

    elif g.gtype == "mesh":
        # Resolve mesh path relative to URDF dir
        mesh_path = urdf_dir / g.filename
        if not mesh_path.exists():
            # Try just the filename part
            mesh_path = urdf_dir / Path(g.filename).name
        if not mesh_path.exists():
            # Fall back to a box placeholder
            return f'<geom type="box" size="0.05 0.05 0.05" {pos_attr} rgba="{rgba}" mass="{link.mass:.4g}"/>'

        asset_name = f"mesh_{link.name}"
        mesh_assets[asset_name] = str(mesh_path)
        return (f'<geom type="mesh" mesh="{asset_name}" {pos_attr} {euler_attr} '
                f'rgba="{rgba}" mass="{link.mass:.4g}"/>')

    return ""


# ── Recursive MJCF body builder ───────────────────────────────────────────────

def _build_body(link_name: str,
                links: dict[str, _Link],
                joints: dict[str, _Joint],
                urdf_dir: Path,
                mesh_assets: dict[str, str],
                joint_infos: list[JointInfo],
                wheel_joints: list[str],
                ctrl_counter: list[int],   # mutable int wrapped in list
                depth: int = 0,
                parent_joint: Optional[_Joint] = None,
                extra_inner_xml: str = "",
                pos_override: str = "") -> str:

    link = links.get(link_name)
    if link is None:
        return ""

    indent = "  " * (depth + 1)

    # Body position/orientation from parent joint
    if parent_joint:
        xyz_s   = _fmt(parent_joint.xyz)
        euler_s = _fmt(_rpy_to_euler_mujoco(parent_joint.rpy))
        pos_attr   = f'pos="{xyz_s}"'
        euler_attr = f'euler="{euler_s}"' if any(v != 0 for v in parent_joint.rpy) else ""
    else:
        pos_attr   = pos_override if pos_override else 'pos="0 0 0"'
        euler_attr = ""

    # Joint XML (for non-fixed joints)
    joint_xml = ""
    if parent_joint and parent_joint.jtype in ("revolute", "continuous", "prismatic"):
        ax  = _fmt(parent_joint.axis)
        is_continuous = parent_joint.jtype == "continuous"
        lo  = parent_joint.lower if not is_continuous else -1e9
        hi  = parent_joint.upper if not is_continuous else  1e9
        lim_attr = f'range="{lo:.4g} {hi:.4g}"' if not is_continuous else 'limited="false"'

        joint_xml = (
            f'<joint name="{parent_joint.name}" type="hinge" '
            f'axis="{ax}" {lim_attr} '
            f'damping="0.2" armature="0.01"/>'
            if parent_joint.jtype in ("revolute", "continuous") else
            f'<joint name="{parent_joint.name}" type="slide" '
            f'axis="{ax}" {lim_attr} damping="50"/>'
        )

        is_wheel = parent_joint.name in wheel_joints
        joint_infos.append(JointInfo(
            name       = parent_joint.name,
            joint_type = parent_joint.jtype,
            lower      = lo,
            upper      = hi,
            ctrl_index = ctrl_counter[0],
            axis       = parent_joint.axis,
            is_wheel   = is_wheel,
        ))
        ctrl_counter[0] += 1

    # Geometry
    is_base = (depth == 0)
    rgba = "0.18 0.40 0.78 1" if is_base else "0.45 0.55 0.65 1"
    if link_name.lower().find("wheel") >= 0 or (parent_joint and _WHEEL_HINTS.search(parent_joint.name or "")):
        rgba = "0.10 0.10 0.10 1"
    geom_xml = _geom_xml(link, urdf_dir, mesh_assets, rgba)

    # Children
    children_xml = ""
    for cj in _children_of(link_name, joints):
        children_xml += "\n" + _build_body(
            cj.child, links, joints, urdf_dir, mesh_assets,
            joint_infos, wheel_joints, ctrl_counter, depth + 1, cj)

    # MuJoCo reserves the name "world" for the implicit worldbody
    safe_name = "robot_base" if link_name == "world" else link_name
    return (
        f"\n{indent}<body name=\"{safe_name}\" {pos_attr} {euler_attr}>"
        f"\n{indent}  {extra_inner_xml}"
        f"\n{indent}  {joint_xml}"
        f"\n{indent}  {geom_xml}"
        f"{children_xml}"
        f"\n{indent}</body>"
    )


# ── Top-level loader ──────────────────────────────────────────────────────────

def load(urdf_path: str | Path) -> LoadResult:
    """
    Parse the URDF and return a LoadResult containing the MJCF XML string,
    joint metadata, and detected robot kind.
    """
    urdf_path = Path(urdf_path).resolve()
    urdf_dir  = urdf_path.parent

    links, joints, model_name = _parse_urdf(urdf_path)
    kind, wheel_joints        = _detect_kind(joints)
    root_link                 = _find_root(links, joints)

    mesh_assets:  dict[str, str]  = {}
    joint_infos:  list[JointInfo] = []
    ctrl_counter: list[int]       = [0]

    # For AMR: give the base a freejoint so it can move
    freejoint_xml = ""
    if kind == RobotKind.AMR:
        freejoint_xml = '<freejoint name="root"/>'

    base_pos = 'pos="0 0 0.12"' if kind == RobotKind.AMR else 'pos="0 0 0"'
    body_xml = _build_body(
        root_link, links, joints, urdf_dir, mesh_assets,
        joint_infos, wheel_joints, ctrl_counter,
        extra_inner_xml=freejoint_xml,
        pos_override=base_pos)

    # ── Assets ────────────────────────────────────────────────────────────────
    asset_lines = []
    for aname, apath in mesh_assets.items():
        asset_lines.append(f'<mesh name="{aname}" file="{apath}"/>')
    assets_xml = "\n    ".join(asset_lines)

    # ── Actuators ─────────────────────────────────────────────────────────────
    actuator_lines = []
    for ji in joint_infos:
        if ji.joint_type in ("revolute", "prismatic"):
            lo, hi = ji.lower, ji.upper
            actuator_lines.append(
                f'<position name="{ji.name}_act" joint="{ji.name}" '
                f'kp="120" kv="8" ctrlrange="{lo:.4g} {hi:.4g}"/>'
            )
        elif ji.joint_type == "continuous":
            # Wheels use velocity control
            actuator_lines.append(
                f'<velocity name="{ji.name}_act" joint="{ji.name}" '
                f'kv="5.0" ctrlrange="-20 20"/>'
            )
    actuators_xml = "\n    ".join(actuator_lines)

    # ── Full MJCF ─────────────────────────────────────────────────────────────
    mjcf = f"""<mujoco model="{model_name}">
  <compiler angle="radian" balanceinertia="true"
            boundmass="0.001" boundinertia="1e-6"
            meshdir="{urdf_dir}"/>

  <option timestep="0.004" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.12 0.16 0.22 1"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.70 0.70 0.70" rgb2="0.80 0.80 0.80"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.05"/>
    {assets_xml}
  </asset>

  <worldbody>
    <light name="sun" pos="1 1 4" dir="-0.2 -0.2 -1" directional="true"
           diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="-2 -2 3" directional="true"
           diffuse="0.3 0.3 0.3" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"
          friction="0.8 0.005 0.0001" contype="1" conaffinity="1"/>
    {body_xml}
  </worldbody>

  <actuator>
    {actuators_xml}
  </actuator>
</mujoco>
"""
    return LoadResult(mjcf_xml=mjcf, joints=joint_infos,
                      kind=kind, model_name=model_name)
