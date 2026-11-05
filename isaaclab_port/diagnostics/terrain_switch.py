"""Check two-lane evaluation and flat -> rough/gap -> flat rebuilding.

Uses the actual Isaac Lab task and evaluator. Small random and zero policies
are duplicated across lanes. Ten-step trials validate lifecycle/reproducibility;
the robot starts on the flat margin and need not reach the altered course.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache" / "warp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24582922)
    parser.add_argument("--spacing", type=float, default=None,
                        help="Diagnostic override only; never changes env.py defaults")
    parser.add_argument("--result", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = dict(ok=False, phase="imports", trials=[], substeps=args.substeps,
                  num_envs=2, episode_steps=args.steps, seed=args.seed,
                  spacing_override=args.spacing,
                  scope="Terrain lifecycle and duplicated-policy consistency only; not learned locomotion or evolution success.")
    evaluator = None
    try:
        import numpy as np
        import torch
        import warp as wp
        from pxr import UsdPhysics
        try:
            from isaaclab.app import launch_simulation
        except ImportError:
            from isaaclab_tasks.utils import launch_simulation
        from isaaclab_newton.physics import NewtonCfg, FeatherstoneSolverCfg
        from isaaclab_port.core import POET, Settings, Terrain, PARAMETER_COUNT
        from isaaclab_port.doctor import inspect
        from isaaclab_port.env import make_env_cfg
        from isaaclab_port.runtime import Evaluator
        wp.config.kernel_cache_dir = str(ROOT / ".cache" / "warp")
        result["environment"] = inspect()
        assert PARAMETER_COUNT == 2804

        def cfg_factory(terrain):
            cfg = make_env_cfg(asdict(terrain), seed=terrain.seed,
                num_envs=2, device="cpu", episode_steps=args.steps,
                physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(),
                                  use_cuda_graph=False, num_substeps=args.substeps))
            if args.spacing is not None:
                cfg.scene.env_spacing = args.spacing
            return cfg

        flat = Terrain(seed=args.seed)
        rough = Terrain(roughness=0.12, gap_width=0.4, seed=args.seed)
        evaluator = Evaluator(2, cfg_factory)
        core = POET(Settings(seed=args.seed, pairs=1), evaluator)
        policies = {"small_random": core.active[0].theta * 0.1,
                    "zero": np.zeros(PARAMETER_COUNT)}
        profiles = []
        with launch_simulation(cfg_factory(flat)):
            try:
                for stage_index, terrain in enumerate((flat, rough, flat)):
                    for policy_name, theta in policies.items():
                        result["phase"] = f"stage_{stage_index}_{policy_name}"
                        values = evaluator(terrain, np.repeat(theta[None, :], 2, axis=0))
                        env = evaluator.env
                        profile = env.course_heights.copy()
                        if policy_name == "small_random":
                            profiles.append(profile)
                        origin = env.scene.env_origins.detach().cpu().numpy()
                        anchor_id = env.robot.body_names.index("anchor")
                        anchor = env.robot.data.body_link_pos_w.torch[:, anchor_id].detach().cpu().numpy()
                        model = env.sim.physics_manager._model
                        profile_text = json.dumps([None if not np.isfinite(v) else float(v) for v in profile])
                        trial = dict(stage=stage_index, terrain=asdict(terrain), policy=policy_name,
                            returns=values.tolist(), lengths=evaluator.last_episode_lengths.copy(),
                            successes=evaluator.last_episode_successes.copy(),
                            profile_sha256=hashlib.sha256(profile_text.encode()).hexdigest(),
                            missing_tiles=int(np.isnan(profile).sum()),
                            max_tile_height=float(np.nanmax(profile)),
                            source_collision_tiles=sum(
                                p.HasAPI(UsdPhysics.CollisionAPI)
                                for p in env.sim.stage.Traverse()
                                if str(p.GetPath()).startswith("/World/envs/env_0/Course/Tile_")),
                            env_origins=origin.tolist(), anchors=anchor.tolist(),
                            newton_world_count=int(model.world_count),
                            body_world_ids=np.unique(model.body_world.numpy()).tolist(),
                            shape_world_ids=np.unique(model.shape_world.numpy()).tolist())
                        trial["lane_returns_match"] = bool(np.allclose(values[0], values[1], rtol=1e-4, atol=1e-4))
                        trial["lane_lengths_match"] = trial["lengths"][0] == trial["lengths"][1]
                        result["trials"].append(trial)
                        print(json.dumps(trial), flush=True)
                        assert np.isfinite(values).all(), "Nonfinite episode return"
                        assert all(1 <= length <= args.steps for length in trial["lengths"])
                        np.testing.assert_allclose(anchor, origin, rtol=0, atol=1e-5,
                                                   err_msg="Fixed anchors do not match lane origins")
                        assert trial["source_collision_tiles"] == int(np.isfinite(profile).sum())
                assert np.array_equal(profiles[0], profiles[2], equal_nan=True)
                assert not np.array_equal(profiles[0], profiles[1], equal_nan=True)
                assert np.isnan(profiles[1]).any() and np.nanmax(profiles[1]) > 0
                for policy_name in policies:
                    first = next(t for t in result["trials"] if t["stage"] == 0 and t["policy"] == policy_name)
                    last = next(t for t in result["trials"] if t["stage"] == 2 and t["policy"] == policy_name)
                    np.testing.assert_allclose(first["returns"], last["returns"], rtol=1e-6, atol=1e-5,
                                               err_msg="Flat-terrain rebuild changed fixed-policy return")
                    assert first["lengths"] == last["lengths"]
                matches = all(t["lane_returns_match"] and t["lane_lengths_match"] for t in result["trials"])
                result.update(ok=matches, phase="complete", diagnostic_completed=True,
                              rebuild_reproducible=True, lane_consistency=matches,
                              metrics=evaluator.metrics())
            finally:
                evaluator.close()
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.result.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
