"""Author the Box2D walker's mass model as a local planar URDF and USD.

Run with the Isaac Lab Python from the repository root:
    ~/env_isaaclab/bin/python -m isaaclab_port.assets.create_box2d_walker
The hull has no collider: touching the ground ends the episode, and the environment
tests that geometrically. Passive carriages add only 2 g to the hull's x and z motion.
Joint axes are +y, so joint values are the negative of Box2D's counter-clockwise angles.
"""
import math
from pathlib import Path

from pxr import Usd, UsdPhysics

from isaaclab_port.assets.generate_planar_usd import generate, validate
from isaaclab_port.walker import spec

ASSET_DIR = Path(__file__).resolve().parent
URDF_PATH = ASSET_DIR / "box2d_walker.urdf"
USD_PATH = URDF_PATH.with_suffix(".usda")
SIDE_OFFSET = 0.2
DRIVE_DAMPING = 300.0


def _inertial(mass, iyy, com=(0.0, 0.0, 0.0)):
    # Only pitch inertia matters in this planar model; the others just stay positive.
    other = max(iyy, 1e-6)
    return (f'<inertial><origin xyz="{com[0]} {com[1]} {com[2]}" rpy="0 0 0"/><mass value="{mass}"/>'
            f'<inertia ixx="{other}" iyy="{iyy}" izz="{other}" ixy="0" ixz="0" iyz="0"/></inertial>')


def _box(kind, size, center, material=""):
    tag = f'<material name="{material}"/>' if material else ""
    return (f'<{kind}><origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>'
            f'<geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry>{tag}</{kind}>')


def _joint(name, kind, parent, child, origin, axis, limits=None, effort=0):
    limit = f'lower="{limits[0]}" upper="{limits[1]}"' if limits else ""
    return (f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/><axis xyz="{axis}"/>'
            f'<limit effort="{effort}" velocity="1000" {limit}/><dynamics damping="0" friction="0"/></joint>')


def urdf_text():
    links, joints = [], []
    links.append(f'<link name="anchor">{_inertial(1.0, 0.01)}</link>')
    links.append(f'<link name="x_carriage">{_inertial(0.001, 1e-6)}</link>')
    links.append(f'<link name="z_carriage">{_inertial(0.001, 1e-6)}</link>')
    xs, ys = [p[0] for p in spec.HULL_POLY], [p[1] for p in spec.HULL_POLY]
    size = (max(xs) - min(xs), 0.2, max(ys) - min(ys))
    center = ((max(xs) + min(xs)) / 2, 0, (max(ys) + min(ys)) / 2)
    links.append(f'<link name="hull">{_inertial(spec.HULL_MASS, spec.HULL_INERTIA, (spec.HULL_COM[0], 0, spec.HULL_COM[1]))}'
                 f'{_box("visual", size, center, "hull_color")}</link>')
    joints.append(_joint("root_x", "prismatic", "anchor", "x_carriage", (0, 0, 0), "1 0 0", (-1000, 1000)))
    joints.append(_joint("root_z", "prismatic", "x_carriage", "z_carriage", (0, 0, 0), "0 0 1", (-1000, 1000)))
    joints.append(_joint("root_pitch", "continuous", "z_carriage", "hull", (0, 0, 0), "0 1 0"))
    for side, offset in (("left", SIDE_OFFSET), ("right", -SIDE_OFFSET)):
        upper, lower = f"{side}_upper", f"{side}_lower"
        leg_center = (0, 0, -spec.LEG_H / 2)
        links.append(f'<link name="{upper}">{_inertial(spec.UPPER_MASS, spec.UPPER_INERTIA, leg_center)}'
                     f'{_box("visual", (spec.LEG_W, 0.1, spec.LEG_H), leg_center, "leg_color")}'
                     f'{_box("collision", (spec.LEG_W, 0.1, spec.LEG_H), leg_center)}</link>')
        links.append(f'<link name="{lower}">{_inertial(spec.LOWER_MASS, spec.LOWER_INERTIA, leg_center)}'
                     f'{_box("visual", (spec.LOWER_W, 0.1, spec.LEG_H), leg_center, "leg_color")}'
                     f'{_box("collision", (spec.LOWER_W, 0.1, spec.LEG_H), leg_center)}</link>')
        hip = (-spec.HIP_LIMITS[1], -spec.HIP_LIMITS[0])
        knee = (-spec.KNEE_LIMITS[1], -spec.KNEE_LIMITS[0])
        joints.append(_joint(f"{side}_hip", "revolute", "hull", upper, (0, offset, spec.LEG_DOWN), "0 1 0", hip, spec.MOTORS_TORQUE))
        joints.append(_joint(f"{side}_knee", "revolute", upper, lower, (0, 0, -spec.LEG_H), "0 1 0", knee, spec.MOTORS_TORQUE))
    colors = ('<material name="hull_color"><color rgba="0.5 0.4 0.9 1"/></material>'
              '<material name="leg_color"><color rgba="0.6 0.3 0.5 1"/></material>')
    return f'<?xml version="1.0" encoding="utf-8"?>\n<robot name="box2d_walker">{colors}{"".join(links)}{"".join(joints)}</robot>\n'


def add_velocity_drives():
    """A damping-only drive makes Newton create a velocity actuator for each leg joint."""
    stage = Usd.Stage.Open(str(USD_PATH))
    for side in ("left", "right"):
        for joint in ("hip", "knee"):
            prim = stage.GetPrimAtPath(f"/planar_biped/joints/{side}_{joint}")
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            # USD angular gains are per degree; the environment sets the actual gain.
            drive.CreateDampingAttr(math.radians(DRIVE_DAMPING))
            drive.CreateMaxForceAttr(spec.MOTORS_TORQUE)
    stage.GetRootLayer().Save()


def main():
    URDF_PATH.write_text(urdf_text())
    generate(URDF_PATH, USD_PATH)
    add_velocity_drives()
    print(validate(USD_PATH, URDF_PATH))


if __name__ == "__main__":
    main()
