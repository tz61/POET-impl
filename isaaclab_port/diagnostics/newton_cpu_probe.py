"""Minimal offline Isaac Lab 3/Newton CPU physics probe.

Run with ~/env_isaaclab/bin/python. Creates only a local USD cube. All
runtime caches are redirected to /tmp; the installed environment is unchanged.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback
from types import SimpleNamespace

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/poet-isaaclab-probe/cache")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/poet-isaaclab-probe/warp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/poet-isaaclab-probe/matplotlib")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--solver", choices=("featherstone", "xpbd"), default="featherstone")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--floor", action="store_true")
parser.add_argument("--result", type=Path)
args = parser.parse_args()
result = {"solver": args.solver, "device": "cpu", "steps": args.steps,
          "floor": args.floor,
          "python": sys.version, "phase": "imports", "ok": False,
          "versions": {name: importlib.metadata.version(name) for name in
                       ("isaaclab", "isaaclab_newton", "newton", "warp-lang", "torch")}}
source_root = Path.home() / "IsaacLab"
if (source_root / "VERSION").is_file():
    result["isaaclab_source"] = {
        "path": str(source_root),
        "product_version": (source_root / "VERSION").read_text().strip(),
        "commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
    }
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics import NewtonCfg, FeatherstoneSolverCfg, XPBDSolverCfg
    from isaaclab_tasks.utils.sim_launcher import launch_simulation

    solver = FeatherstoneSolverCfg() if args.solver == "featherstone" else XPBDSolverCfg()
    cfg = sim_utils.SimulationCfg(device="cpu", dt=1.0 / 240.0,
        physics=NewtonCfg(solver_cfg=solver, use_cuda_graph=False),
        visualizer_cfgs=[], create_stage_in_memory=True,
        save_logs_to_file=False)
    result["phase"] = "launch_simulation"
    print("PHASE", result["phase"], flush=True)
    with launch_simulation(SimpleNamespace(sim=cfg)):
        result["phase"] = "simulation_context"
        print("PHASE", result["phase"], flush=True)
        sim = SimulationContext(cfg)
        if args.floor:
            floor_cfg = sim_utils.CuboidCfg(size=(5.0, 5.0, 0.2),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9, dynamic_friction=0.8, restitution=0.0))
            floor_cfg.func("/World/Floor", floor_cfg, translation=(0.0, 0.0, -0.1))
        result["phase"] = "spawn_local_cube"
        print("PHASE", result["phase"], flush=True)
        cube = RigidObject(RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        ))
        result["phase"] = "reset_compile"
        print("PHASE", result["phase"], flush=True)
        sim.reset()
        def positions():
            val = cube.data.root_pos_w
            val = val.torch if hasattr(val, "torch") else val
            return val.detach().cpu().clone()
        initial = positions()
        result["initial_position"] = initial.tolist()
        result["phase"] = "stepping"
        print("PHASE", result["phase"], flush=True)
        for _ in range(args.steps):
            cube.write_data_to_sim()
            sim.step(render=False)
            cube.update(cfg.dt)
        final = positions()
        result["final_position"] = final.tolist()
        result["elapsed_sim_seconds"] = args.steps * cfg.dt
        expected_z = 1.0 - 0.5 * 9.81 * (args.steps * cfg.dt)**2
        result["continuous_freefall_expected_z"] = expected_z
        if not torch.isfinite(final).all() or not (final[:, 2] < initial[:, 2]).all():
            raise AssertionError("Cube did not fall to finite lower position")
        if args.floor and args.steps >= 240:
            if not 0.08 < float(final[0, 2]) < 0.2:
                raise AssertionError("Cube final center is outside the expected floor-contact region")
        elif abs(float(final[0, 2]) - expected_z) > 0.01:
            raise AssertionError("Cube gravity motion differs from freefall by >1 cm")
        result["phase"] = "cleanup"
        sim.stop()
        SimulationContext.clear_instance()
        result.update(ok=True, phase="complete")
except Exception as exc:
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
    result["traceback"] = traceback.format_exc()
    traceback.print_exc()
finally:
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.result:
        args.result.write_text(text + "\n", encoding="utf-8")
sys.exit(0 if result["ok"] else 1)
