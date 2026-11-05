"""Author the primitive planar-biped URDF as self-contained USD with pxr.

This deliberately uses only core OpenUSD, not Kit's URDF importer or usdex.
The conversion supports the exact local asset's axis-aligned box geometry,
diagonal inertias and zero static joint rotations. Unsupported URDF features
raise errors rather than being silently dropped. Dimensions are metres,
masses kilograms, inertias kg*m^2 and efforts N or N*m.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ASSET_DIR / "planar_biped.urdf"
DEFAULT_OUTPUT = ASSET_DIR / "planar_biped.usda"
ROOT = Sdf.Path("/planar_biped")


def _vector(text: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected a three-vector, got {text}")
    return values


def _origin(element: ET.Element) -> tuple[float, float, float]:
    origin = element.find("origin")
    if origin is None:
        return (0., 0., 0.)
    if any(abs(v) > 1e-12 for v in _vector(origin.get("rpy", "0 0 0"))):
        raise ValueError("This local converter requires zero static origin rotations")
    return _vector(origin.get("xyz", "0 0 0"))


def generate(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Convert local boxes, rigid bodies, inertias and joints to USD."""
    robot = ET.parse(input_path).getroot()
    links = {element.attrib["name"]: element for element in robot.findall("link")}
    joints = robot.findall("joint")
    materials = {
        element.attrib["name"]: tuple(float(v) for v in element.find("color").attrib["rgba"].split())
        for element in robot.findall("material")
    }
    child_joints = {j.find("child").attrib["link"]: j for j in joints}
    if set(links) - set(child_joints) != {"anchor"}:
        raise ValueError("Expected the original asset's unique anchor link")

    def position(name):
        if name == "anchor":
            return (0., 0., 0.)
        joint = child_joints[name]
        parent = joint.find("parent").attrib["link"]
        return tuple(a + b for a, b in zip(position(parent), _origin(joint)))

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)
    root.SetDocumentation("Local planar biped: passive x/z/pitch base and four hip/knee torque joints.")
    UsdPhysics.ArticulationRootAPI.Apply(root)

    material = UsdShade.Material.Define(stage, ROOT.AppendPath("Materials/contact"))
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(.9)
    physics_material.CreateDynamicFrictionAttr(.9)
    physics_material.CreateRestitutionAttr(0.)

    for name, link in links.items():
        body_path = ROOT.AppendChild(name)
        body = UsdGeom.Xform.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(*position(name)))
        rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(False)
        inertial = link.find("inertial")
        inertia = inertial.find("inertia").attrib
        if any(abs(float(inertia[k])) > 1e-12 for k in ("ixy", "ixz", "iyz")):
            raise ValueError("This local converter requires diagonal link inertias")
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(float(inertial.find("mass").attrib["value"]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*_origin(inertial)))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(inertia[k]) for k in ("ixx", "iyy", "izz"))))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))

        for kind in ("visual", "collision"):
            for index, element in enumerate(link.findall(kind)):
                shape = element.find("geometry/box")
                if shape is None:
                    raise ValueError("This local converter requires explicit box geometry")
                cube = UsdGeom.Cube.Define(stage, body_path.AppendChild(f"{kind}_{index}"))
                cube.CreateSizeAttr(1.)
                cube.AddTranslateOp().Set(Gf.Vec3d(*_origin(element)))
                cube.AddScaleOp().Set(Gf.Vec3f(*_vector(shape.attrib["size"])))
                if kind == "visual":
                    color = materials[element.find("material").attrib["name"]]
                    cube.CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
                else:
                    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")

    identity = Gf.Quatf(1.)
    world_fix = UsdPhysics.FixedJoint.Define(stage, ROOT.AppendPath("joints/anchor_world_fixed"))
    world_fix.CreateBody1Rel().SetTargets([ROOT.AppendChild("anchor")])
    world_fix.CreateLocalPos0Attr(Gf.Vec3f(0.))
    world_fix.CreateLocalPos1Attr(Gf.Vec3f(0.))
    world_fix.CreateLocalRot0Attr(identity)
    world_fix.CreateLocalRot1Attr(identity)

    for element in joints:
        name, kind = element.attrib["name"], element.attrib["type"]
        path = ROOT.AppendPath(f"joints/{name}")
        if kind == "fixed":
            joint = UsdPhysics.FixedJoint.Define(stage, path)
        elif kind == "prismatic":
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        elif kind in ("revolute", "continuous"):
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
        else:
            raise ValueError(f"Unsupported joint type {kind}")
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        joint.CreateBody0Rel().SetTargets([ROOT.AppendChild(parent)])
        joint.CreateBody1Rel().SetTargets([ROOT.AppendChild(child)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*_origin(element)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateLocalRot0Attr(identity)
        joint.CreateLocalRot1Attr(identity)
        joint.CreateCollisionEnabledAttr(False)
        if kind != "fixed":
            axis = _vector(element.find("axis").attrib["xyz"])
            axes = {(1., 0., 0.): "X", (0., 1., 0.): "Y", (0., 0., 1.): "Z"}
            if axis not in axes:
                raise ValueError(f"Unsupported joint axis {axis}")
            joint.CreateAxisAttr(axes[axis])
            limit = element.find("limit").attrib
            if kind != "continuous":
                scale = 1. if kind == "prismatic" else 180. / math.pi
                joint.CreateLowerLimitAttr(float(limit["lower"]) * scale)
                joint.CreateUpperLimitAttr(float(limit["upper"]) * scale)
            effort = float(limit["effort"])
            if effort > 0:
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if kind == "prismatic" else "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(0.)
                drive.CreateDampingAttr(0.)
                drive.CreateMaxForceAttr(effort)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output_path))
    validate(output_path, input_path)
    return output_path


def validate(path: Path = DEFAULT_OUTPUT, urdf_path: Path = DEFAULT_INPUT) -> dict[str, int]:
    """Verify all source bodies/joints and self-contained USD physics assets."""
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI):
        raise ValueError("Missing articulation root")
    robot = ET.parse(urdf_path).getroot()
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if {p.GetName() for p in bodies} != {e.attrib["name"] for e in robot.findall("link")}:
        raise ValueError("USD rigid bodies differ from URDF links")
    expected = {e.attrib["name"] for e in robot.findall("joint")} | {"anchor_world_fixed"}
    if {p.GetName() for p in joints} != expected:
        raise ValueError("USD joints differ from URDF plus fixed world anchor")
    for element in robot.findall("joint"):
        prim = stage.GetPrimAtPath(ROOT.AppendPath("joints/" + element.attrib["name"]))
        joint = UsdPhysics.Joint(prim)
        for body_number, role in ((0, "parent"), (1, "child")):
            target = ROOT.AppendChild(element.find(role).attrib["link"])
            relation = joint.GetBody0Rel() if body_number == 0 else joint.GetBody1Rel()
            if relation.GetTargets() != [target]:
                raise ValueError("Joint parent/child relationship changed")
        kind = element.attrib["type"]
        schema = UsdPhysics.FixedJoint if kind == "fixed" else (
            UsdPhysics.PrismaticJoint if kind == "prismatic" else UsdPhysics.RevoluteJoint
        )
        if not prim.IsA(schema):
            raise ValueError("Joint type changed during USD conversion")
        if not Gf.IsClose(joint.GetLocalPos0Attr().Get(), Gf.Vec3f(*_origin(element)), 1e-6):
            raise ValueError("Joint origin changed during USD conversion")
        if kind != "fixed":
            typed = schema(prim)
            expected_axis = {(1., 0., 0.): "X", (0., 1., 0.): "Y", (0., 0., 1.): "Z"}[
                _vector(element.find("axis").attrib["xyz"])
            ]
            if typed.GetAxisAttr().Get() != expected_axis:
                raise ValueError("Joint axis changed during USD conversion")
            limit = element.find("limit").attrib
            if kind != "continuous":
                scale = 1. if kind == "prismatic" else 180. / math.pi
                if not math.isclose(typed.GetLowerLimitAttr().Get(), float(limit["lower"]) * scale, abs_tol=1e-4):
                    raise ValueError("Incorrect lower joint limit or angle units")
                if not math.isclose(typed.GetUpperLimitAttr().Get(), float(limit["upper"]) * scale, abs_tol=1e-4):
                    raise ValueError("Incorrect upper joint limit or angle units")
            drive_type = "linear" if kind == "prismatic" else "angular"
            drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if float(limit["effort"]) == 0.:
                if prim.HasAPI(UsdPhysics.DriveAPI, drive_type):
                    raise ValueError("Passive root unexpectedly acquired a motor drive")
            elif not drive or drive.GetMaxForceAttr().Get() != float(limit["effort"]):
                raise ValueError("Active joint lost its torque limit")
    for link in robot.findall("link"):
        prim = stage.GetPrimAtPath(ROOT.AppendChild(link.attrib["name"]))
        mass = UsdPhysics.MassAPI(prim)
        inertial = link.find("inertial")
        expected_mass = float(inertial.find("mass").attrib["value"])
        if not math.isclose(mass.GetMassAttr().Get(), expected_mass, rel_tol=1e-6):
            raise ValueError("Link mass changed during USD conversion")
        expected_inertia = Gf.Vec3f(*(float(inertial.find("inertia").attrib[k]) for k in ("ixx", "iyy", "izz")))
        if not Gf.IsClose(mass.GetDiagonalInertiaAttr().Get(), expected_inertia, 1e-8):
            raise ValueError("Link inertia changed during USD conversion")
    world = UsdPhysics.FixedJoint(stage.GetPrimAtPath(ROOT.AppendPath("joints/anchor_world_fixed")))
    if world.GetBody0Rel().GetTargets() or world.GetBody1Rel().GetTargets() != [ROOT.AppendChild("anchor")]:
        raise ValueError("Anchor must be fixed to world")
    if len(colliders) != sum(len(link.findall("collision")) for link in robot.findall("link")):
        raise ValueError("Collision geometry count changed")
    if stage.GetRootLayer().subLayerPaths or stage.GetRootLayer().GetExternalReferences():
        raise ValueError("USD asset must be self-contained")
    return {"rigid_bodies": len(bodies), "joints_including_world_fix": len(joints), "colliders": len(colliders)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        generate(args.input, args.output)
    print(f"Validated {args.output}: {validate(args.output, args.input)}")


if __name__ == "__main__":
    main()
