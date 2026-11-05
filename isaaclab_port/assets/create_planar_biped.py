"""Generate a local, primitive-only URDF for the Isaac Lab POET port.

The fixed anchor and three unactuated root joints leave the robot free to move
in x/z and rotate about y. Four hip/knee joints provide the policy's actions.
Planarity is a mechanical constraint of the articulation, not a state reset.
This is an original small test robot, not the original Box2D body or a validated
replacement benchmark. All dimensions use SI units.

No simulator, network access, mesh files, or third-party packages are required
to generate and structurally validate this file. PhysX import and locomotion
still require an actual Isaac Lab/Isaac Sim runtime.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_OUTPUT = Path(__file__).with_name("planar_biped.urdf")
PASSIVE_JOINTS = ("root_x", "root_z", "root_pitch")
ACTUATED_JOINTS = ("left_hip", "left_knee", "right_hip", "right_knee")
FOOT_BODIES = ("left_foot", "right_foot")
TORSO_BODY = "torso"
DEFAULT_ROOT_HEIGHT = 0.98  # Hip height [m]; straight feet start 0.03 m above z=0.
TORQUE_LIMITS = (40.0, 40.0, 40.0, 40.0)  # Joint torques [N m].
LEG_SEGMENT_LENGTH = 0.46  # [m].


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _origin(parent: ET.Element, xyz: tuple[float, float, float]) -> None:
    ET.SubElement(parent, "origin", xyz=_numbers(xyz), rpy="0 0 0")


def _link(
    robot: ET.Element,
    name: str,
    mass: float,
    size: tuple[float, float, float],
    center: tuple[float, float, float] = (0., 0., 0.),
    material: str | None = None,
) -> None:
    """Add a box link with analytic positive-definite inertia [kg m^2]."""
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, center)
    ET.SubElement(inertial, "mass", value=f"{mass:.12g}")
    x, y, z = size
    ET.SubElement(
        inertial, "inertia", ixx=f"{mass * (y * y + z * z) / 12:.12g}",
        iyy=f"{mass * (x * x + z * z) / 12:.12g}",
        izz=f"{mass * (x * x + y * y) / 12:.12g}", ixy="0", ixz="0", iyz="0",
    )
    # Anchor/carriage links deliberately have neither visuals nor collisions.
    if material is not None:
        for kind in ("visual", "collision"):
            element = ET.SubElement(link, kind)
            _origin(element, center)
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", size=_numbers(size))
            if kind == "visual":
                ET.SubElement(element, "material", name=material)


def _joint(
    robot: ET.Element,
    name: str,
    kind: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float] = (0., 0., 0.),
    axis: tuple[float, float, float] = (0., 1., 0.),
    limits: tuple[float, float] = (-math.pi, math.pi),
    effort: float = 0.,
    velocity: float = 100.,
) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    _origin(joint, xyz)
    if kind != "fixed":
        ET.SubElement(joint, "axis", xyz=_numbers(axis))
        attributes = {"effort": f"{effort:.12g}", "velocity": f"{velocity:.12g}"}
        if kind != "continuous":
            attributes.update(lower=f"{limits[0]:.12g}", upper=f"{limits[1]:.12g}")
        ET.SubElement(joint, "limit", **attributes)
        ET.SubElement(joint, "dynamics", damping="0", friction="0")


def build_planar_biped() -> ET.Element:
    """Build the robot XML with dimensions [m], masses [kg], torques [N m]."""
    robot = ET.Element("robot", name="poet_planar_biped")
    robot.append(ET.Comment(
        " Locally generated primitive geometry. Import with fix_base=True and "
        "joint_drive.target_type='none'. root_x/root_z/root_pitch are passive. "
    ))
    for name, rgba in (("torso_gold", "0.92 0.65 0.16 1"),
                       ("left_blue", "0.16 0.43 0.82 1"),
                       ("right_red", "0.82 0.25 0.21 1"),
                       ("foot_dark", "0.15 0.18 0.22 1")):
        material = ET.SubElement(robot, "material", name=name)
        ET.SubElement(material, "color", rgba=rgba)

    _link(robot, "anchor", 1., (.1, .1, .1))
    # Small but finite inertias avoid massless dynamic articulation links.
    # The sliders add 0.04 kg to this test robot's translating support chain.
    _link(robot, "x_carriage", .02, (.1, .1, .1))
    _link(robot, "z_carriage", .02, (.1, .1, .1))
    _link(robot, TORSO_BODY, 5., (.28, .20, .36), (0., 0., .18), "torso_gold")
    _joint(robot, "root_x", "prismatic", "anchor", "x_carriage",
           axis=(1., 0., 0.), limits=(-1000., 1000.))
    _joint(robot, "root_z", "prismatic", "x_carriage", "z_carriage",
           axis=(0., 0., 1.), limits=(-10., 20.))
    _joint(robot, "root_pitch", "continuous", "z_carriage", TORSO_BODY)

    for side, y, material in (("left", .14, "left_blue"), ("right", -.14, "right_red")):
        thigh, shin, foot = f"{side}_thigh", f"{side}_shin", f"{side}_foot"
        _link(robot, thigh, 1., (.09, .09, LEG_SEGMENT_LENGTH),
              (0., 0., -LEG_SEGMENT_LENGTH / 2), material)
        _link(robot, shin, .8, (.075, .085, LEG_SEGMENT_LENGTH),
              (0., 0., -LEG_SEGMENT_LENGTH / 2), material)
        _link(robot, foot, .15, (.22, .12, .06), (.05, 0., 0.), "foot_dark")
        _joint(robot, f"{side}_hip", "revolute", TORSO_BODY, thigh,
               xyz=(0., y, 0.), limits=(-1.2, 1.2), effort=40., velocity=20.)
        _joint(robot, f"{side}_knee", "revolute", thigh, shin,
               xyz=(0., 0., -LEG_SEGMENT_LENGTH), limits=(0., 2.6), effort=40., velocity=20.)
        _joint(robot, f"{side}_ankle_fixed", "fixed", shin, foot,
               xyz=(0., 0., -LEG_SEGMENT_LENGTH))
    validate_planar_biped(robot)
    return robot


def validate_planar_biped(robot: ET.Element) -> dict[str, int | float]:
    """Check joint topology, planar axes, inertias, and local geometry.

    Returns counts and total link mass [kg], including the fixed anchor.
    These checks establish structural consistency, not dynamic stability.
    """
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
    if len(links) != len(robot.findall("link")) or len(joints) != len(robot.findall("joint")):
        raise ValueError("Link and joint names must be unique")
    movable = {name for name, joint in joints.items() if joint.attrib["type"] != "fixed"}
    if movable != set(PASSIVE_JOINTS + ACTUATED_JOINTS):
        raise ValueError("Expected three passive root joints and four actuated leg joints")
    parent_of = {}
    total_mass = 0.
    for name, link in links.items():
        mass = float(link.find("inertial/mass").attrib["value"])
        inertia = link.find("inertial/inertia").attrib
        diagonal = [float(inertia[key]) for key in ("ixx", "iyy", "izz")]
        if not math.isfinite(mass) or mass <= 0 or not all(
            math.isfinite(value) and value > 0 for value in diagonal
        ):
            raise ValueError(f"Invalid mass/inertia in {name}")
        if any(diagonal[i] > sum(diagonal) - diagonal[i] + 1e-10 for i in range(3)):
            raise ValueError(f"Inertia triangle inequality violated in {name}")
        total_mass += mass
        for geometry in link.findall("*/geometry"):
            box = geometry.find("box")
            if box is None or any(float(value) <= 0 for value in box.attrib["size"].split()):
                raise ValueError("Only local positive-size primitive boxes are expected")
    for name, joint in joints.items():
        parent, child = joint.find("parent").attrib["link"], joint.find("child").attrib["link"]
        if parent not in links or child not in links or child in parent_of:
            raise ValueError(f"Invalid parent/child topology in {name}")
        parent_of[child] = parent
        if joint.find("origin").attrib["rpy"] != "0 0 0":
            raise ValueError(f"Unexpected rotated joint frame in {name}")
        if name in movable:
            expected_type = "prismatic" if name in ("root_x", "root_z") else (
                "continuous" if name == "root_pitch" else "revolute"
            )
            if joint.attrib["type"] != expected_type:
                raise ValueError(f"Unexpected joint type in {name}")
            expected = "1 0 0" if name == "root_x" else "0 0 1" if name == "root_z" else "0 1 0"
            if joint.find("axis").attrib["xyz"] != expected:
                raise ValueError(f"Out-of-plane motion introduced by {name}")
            limit = joint.find("limit").attrib
            if name in ACTUATED_JOINTS and float(limit["effort"]) <= 0:
                raise ValueError(f"Actuated joint {name} needs a positive effort limit")
            if name in PASSIVE_JOINTS and float(limit["effort"]) != 0:
                raise ValueError(f"Passive joint {name} must have no drive effort")
    if set(links) - set(parent_of) != {"anchor"}:
        raise ValueError("Expected the anchor to be the only root link")
    for child in parent_of:
        visited = set()
        node = child
        while node in parent_of:
            if node in visited:
                raise ValueError("Cycle in articulation graph")
            visited.add(node)
            node = parent_of[node]
        if node != "anchor":
            raise ValueError("Disconnected articulation graph")
    clearance = DEFAULT_ROOT_HEIGHT - 2 * LEG_SEGMENT_LENGTH - .03
    if clearance <= 0:
        raise ValueError("Default straight-leg pose would intersect the ground")
    return {"links": len(links), "movable_joints": len(movable),
            "actuated_joints": len(ACTUATED_JOINTS), "total_link_mass_kg": total_mass,
            "default_foot_clearance_m": clearance}


def urdf_text() -> str:
    """Return deterministic URDF XML for the planar biped."""
    robot = build_planar_biped()
    ET.indent(robot, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"


def write_planar_biped(path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write a self-contained URDF to the requested local path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(urdf_text(), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate existing file without changing it")
    args = parser.parse_args()
    if args.check:
        robot = ET.parse(args.output).getroot()
        result = validate_planar_biped(robot)
        if args.output.read_text(encoding="utf-8") != urdf_text():
            raise ValueError("Committed asset differs from deterministic generator output")
        print(f"Validated {args.output}: {result}")
    else:
        print(f"Generated {write_planar_biped(args.output)}")
        print(validate_planar_biped(build_planar_biped()))


if __name__ == "__main__":
    main()
